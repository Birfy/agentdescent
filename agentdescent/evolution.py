"""The general evolution engine -- evolve *any* artifact.

This is the module. It is domain-agnostic: it knows nothing about "skills" or
"harnesses". You describe **what evolves** and **the rules of evolution**, and it
runs the parallel, merge-based loop (ledger + aggregator + staleness +
governance) for you.

You provide four things (all customizable, none built in):

* a :class:`Strategy` -- how the artifact is represented, how it renders into a
  prompt/config, and how a proposal becomes a :class:`~agentdescent.evolvable.Diff`;
* ``run(rendered, task) -> output`` -- apply the current artifact to a task;
* ``reward(task, output) -> [0, 1]`` -- score the output;
* ``propose(rendered, task, output, reward) -> str | None`` -- on a failure,
  propose one improvement.

Then :func:`evolve` drives it. The same engine evolves a **skill** (artifact =
a lesson playbook, run = an LLM using it) or a **harness / verifier** (artifact =
routing/context config, higher ``blast_radius`` -> L1 governance) -- see
``examples/skill_evolution.py`` and ``examples/harness_evolution.py``.

An :class:`Agent` (an object bundling ``solve`` + ``propose``) is a convenience
for the common case where the same actor both runs tasks and proposes changes;
:func:`evolve` also accepts ``run`` / ``propose`` callables directly.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .agents import Completion, claude
from .aggregator import (
    Aggregator, AggregatorConfig, AggregatorFactory, AggregatorContractError,
    check_reports,
)
from .evolvable import Contract, ContractError, Diff, EvidenceCard
from .governance import assert_mutable
from .ledger import Ledger, LedgerFailure
from .sampling import RoundRobin, TaskSampler
from .scheduler import AuditScheduler
from .staleness import StalenessPolicy


# ---------------------------------------------------------------------------
# Task + actor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One unit of work the artifact is evaluated on.

    ``frozen=True`` but the auto-generated ``__hash__`` hit the mutable ``meta``
    dict, so ``set(tasks)`` and ``{task: ...}`` raised ``TypeError``. Identity is
    the ``id`` (which the engine already requires to be unique), so hash and
    compare on that and leave ``meta`` out of both.
    """

    id: str
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __hash__(self) -> int:
        return hash(self.id)


Reward = Callable[["Task", str], float]        # (task, output) -> [0, 1]
Run = Callable[[str, "Task"], str]             # (rendered_artifact, task) -> output
Propose = Callable[[str, "Task", str, float], Optional[str]]  # -> a proposal or None


@runtime_checkable
class Agent(Protocol):
    """Convenience actor: bundles running a task and proposing an improvement."""

    def solve(self, rendered: str, task: Task) -> str: ...

    def propose(self, rendered: str, task: Task, output: str, reward: float) -> Optional[str]: ...


_SOLVE_TMPL = (
    "You are executing an artifact defined below.\n\n{artifact}\n\n"
    "Apply it to this input and output ONLY the result, nothing else.\n\nInput:\n{prompt}"
)
_PROPOSE_TMPL = (
    "The artifact just failed a task (score {reward:.2f} out of 1.0).\n\n"
    "Artifact so far:\n{artifact}\n\nTask input:\n{prompt}\n\n"
    "It produced:\n{output}\n{expected}\n"
    "Propose exactly ONE concise, general rule (a single imperative sentence) to "
    "improve the artifact for this and similar cases. State the rule in general "
    "terms -- it will be applied to other tasks, so do NOT mention this task's "
    "specific values or answer. Output only the rule text, or NONE if no rule "
    "would help."
)


@dataclass
class LLMAgent:
    """Adapt a ``Completion`` (from :mod:`agentdescent.agents`) into an :class:`Agent`."""

    complete: Completion
    solve_template: str = _SOLVE_TMPL
    propose_template: str = _PROPOSE_TMPL
    #: Show ``task.meta`` to the reflector. Without it the reflector sees only a
    #: score -- it is told it was wrong but not what right looks like, which makes
    #: any convention it cannot guess effectively unlearnable. Callers put the
    #: expected answer there (every shipped port does), so it is on by default;
    #: set ``False`` if your meta holds something you would rather not show.
    show_meta: bool = True
    #: Meta is rendered truncated: it can hold a whole document.
    meta_chars: int = 600
    _empty_replies: int = field(default=0, repr=False)

    def solve(self, rendered: str, task: Task) -> str:
        return self.complete(
            self.solve_template.format(artifact=rendered, prompt=task.prompt)).strip()

    def propose(self, rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        raw = self.complete(self.propose_template.format(
            artifact=rendered, prompt=task.prompt, output=output, reward=reward,
            expected=_expected_block(task, self.show_meta, self.meta_chars)))
        rule = raw.strip()
        if not rule:
            # An empty completion is almost never "no rule would help" -- that answer
            # is the literal string NONE. It is nearly always a starved reasoning
            # model: the token budget went to internal reasoning and no visible
            # content came back. Silently treating it as "no proposal" makes the run
            # look like the framework cannot learn, when the reflector never spoke.
            self._empty_replies += 1
            if self._empty_replies in (1, 10, 100):
                warnings.warn(
                    f"the reflector returned an empty completion "
                    f"({self._empty_replies} so far), so no improvement was proposed. "
                    "A reasoning model given too small a max_tokens spends it all on "
                    "reasoning and returns no visible text -- try raising max_tokens.",
                    RuntimeWarning, stacklevel=2)
            return None
        return None if rule.upper().startswith("NONE") else rule


def _expected_block(task: "Task", show_meta: bool, limit: int) -> str:
    """Render ``task.meta`` for the reflection prompt, bounded."""
    if not show_meta or not getattr(task, "meta", None):
        return ""
    text = ", ".join(f"{k}={v!r}" for k, v in task.meta.items())
    if len(text) > limit:
        text = text[:limit] + " ..."
    return f"\nWhat the scorer expected (task metadata):\n{text}\n"


def tasks_from(rows, prompt: str = "prompt", gold: str = "gold",
               id: Optional[str] = None, **meta_keys: str) -> List["Task"]:
    """Turn a list of dicts -- a dataset -- into :class:`Task` objects.

    The same six lines everyone writes after loading a dataset, including the
    ``enumerate`` for ids and the ``meta`` dict the scorers and the reflector both
    read.

        rows  = hf_rows("openai/gsm8k", config="main", split="train", limit=64)
        tasks = tasks_from(rows, prompt="question", gold="answer")

    ``prompt`` and ``gold`` name the columns. ``id`` names a column to use as the
    task id; without it rows are numbered. Extra keyword arguments map more
    columns into ``meta`` (``difficulty="level"`` puts ``row["level"]`` at
    ``meta["difficulty"]``), which is useful because the reflector sees ``meta``.
    """
    out: List[Task] = []
    for i, row in enumerate(rows):
        if prompt not in row:
            raise KeyError(
                f"row {i} has no {prompt!r} column; it has {sorted(row)}. "
                f"Pass prompt= to name the question column.")
        meta = {"gold": row[gold]} if gold in row else {}
        for name, column in meta_keys.items():
            if column in row:
                meta[name] = row[column]
        out.append(Task(id=str(row[id]) if id else str(i),
                        prompt=str(row[prompt]), meta=meta))
    if not out:
        raise ValueError("tasks_from() got no rows")
    return out


def reflector(complete: Completion, template: str = _PROPOSE_TMPL,
              show_meta: bool = True) -> Propose:
    """Use any model as the *reflector* for an agent you already have.

    :class:`LLMAgent` bundles solving and proposing, which only fits when the
    framework also drives the rollout. The common case is the other way round --
    you have an agent, you want it evolved, and you need something to look at a
    failure and say what to change:

        evolve(tasks, reward,
               run=lambda rendered, task: my_agent(rendered, task.prompt),
               propose=reflector(claude(model="claude-haiku-4-5")),
               strategy=SingleSlot())

    The model never has to be the same one the agent uses; a cheap model is often
    the right reflector for an expensive agent."""
    return LLMAgent(complete, propose_template=template, show_meta=show_meta).propose


def claude_agent(model: str = "claude-opus-4-8", max_tokens: int = 1024) -> LLMAgent:
    """Convenience: ``LLMAgent(claude(model))`` (provider code lives in :mod:`agentdescent.agents`)."""
    return LLMAgent(claude(model=model, max_tokens=max_tokens))


# ---------------------------------------------------------------------------
# Strategy: what evolves and how a proposal becomes a change
# ---------------------------------------------------------------------------


def rule_id(text: str) -> str:
    """Content-address a proposal so identical proposals dedupe automatically."""
    return "r" + hashlib.sha1(text.strip().lower().encode()).hexdigest()[:10]


@runtime_checkable
class Strategy(Protocol):
    """Defines *what evolves and how* -- the representation and the merge rule.

    An artifact's state is a flat ``{key: value}`` dict (the diff op-space the
    aggregator resolves conflicts and fusion over). A strategy decides the
    initial state, how it renders, and how a proposal becomes a :class:`Diff`."""

    def initial(self) -> Dict[str, str]: ...

    def render(self, state: Dict[str, str]) -> str: ...

    def to_diff(self, state: Dict[str, str], proposal: str, author: str,
                base_version: int, target: str) -> Optional[Diff]: ...

    # Optional. A strategy that knows, ahead of time, every key it can write should
    # say so: that declared space is what tensor parallelism partitions into
    # sections. A strategy that content-addresses its keys (AppendRules) has no
    # such space, so it simply does not implement this -- and `evolve()` refuses to
    # pair it with TP rather than silently dropping most of its proposals.
    # def keys(self) -> Sequence[str]: ...


@dataclass
class AppendRules:
    """Accumulate a deduped list of rules/lessons (append-only, content-addressed).

    Identical proposals from different workers collapse to one; complementary
    rules are *fused* by the aggregator."""

    title: str = "# Playbook"

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty)"
        return "\n".join([self.title] + [f"- {state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        rid = rule_id(proposal)
        if rid in state:
            return None
        return Diff(diff_id=f"{author}:{rid}:{base_version}", target=target,
                    ops={rid: proposal}, author=author)


@dataclass
class SingleSlot:
    """The artifact **is one value**, and each accepted proposal replaces it.

    The most common thing anyone evolves -- a system prompt, an instruction, one
    document -- and until now every caller wrote this themselves (three of the
    shipped algorithm ports each rolled their own variant). Competing proposals
    contradict on the same key, so the aggregator resolves them on held-out score
    and the best replacement wins:

        evolve(tasks, reward, agent=agent,
               strategy=SingleSlot(initial_value="Answer concisely."))

    ``key`` names the slot in the artifact state and ``initial_value`` seeds it.
    ``min_chars`` is the shortest proposal worth taking, which guards against a
    reflector that replies with a terse non-answer; ``empty_render`` is what the
    artifact renders as before anything has been accepted."""

    initial_value: str = ""
    key: str = "value"
    empty_render: str = "(no instruction yet)"
    min_chars: int = 1

    def keys(self) -> Sequence[str]:
        """The artifact is one slot, so the key space has exactly one member.

        Declared so ``evolve()`` can reject ``TensorParallel`` up front: a single
        key cannot be split into disjoint sections, so every worker but one would
        be authorised for nothing."""
        return [self.key]

    def initial(self) -> Dict[str, str]:
        return {self.key: self.initial_value} if self.initial_value else {}

    def render(self, state: Dict[str, str]) -> str:
        return state.get(self.key) or self.empty_render

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        text = (proposal or "").strip()
        if len(text) < self.min_chars or state.get(self.key) == text:
            return None
        return Diff(diff_id=f"{author}:{self.key}:{base_version}", target=target,
                    ops={self.key: text}, author=author)


@dataclass
class KeyedRules:
    """One entry per *category*: competing proposals contradict and are resolved.

    Proposals look like ``"category: text"``. A new proposal for an existing
    category **overwrites** it, so two workers proposing different text for the
    same category produce a contradiction the aggregator resolves (keeping the
    one that scores better). Unknown categories fall back to append behaviour."""

    categories: Sequence[str]
    title: str = "# Config (by category)"

    def keys(self) -> Sequence[str]:
        """The declared categories -- the key space tensor parallelism partitions.

        Note that an *unrecognised* proposal still falls back to a content-addressed
        key, which is outside this space; under TP those are reported as
        ``section-violation`` rather than silently dropped."""
        return list(self.categories)

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty)"
        return "\n".join([self.title] + [f"## {k}\n{state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        m = re.match(r"\s*([\w\- ]+?)\s*:\s*(.+)", proposal, re.DOTALL)
        if m and m.group(1).strip().lower() in {c.lower() for c in self.categories}:
            key, value = m.group(1).strip().lower(), m.group(2).strip()
        else:
            key, value = rule_id(proposal), proposal.strip()
        if state.get(key) == value:
            return None
        return Diff(diff_id=f"{author}:{key}:{base_version}", target=target,
                    ops={key: value}, author=author)


# ---------------------------------------------------------------------------
# Evaluation cache + the evolving artifact
# ---------------------------------------------------------------------------


# The engine treats a reward of >= 0.999 as a pass and never asks for a proposal,
# so a scorer on the wrong scale (0-100, say) silently means "everything already
# passes": nothing is ever learned, while the reported final_reward looks large and
# healthy. Catch that at the boundary instead.
_REWARD_TOL = 1e-6


class ProposalContractError(ContractError, TypeError):
    """``propose`` returned something that is not text (or ``None``).

    A strategy then fails deep inside ``to_diff`` with something like
    ``'int' object has no attribute 'strip'``, which reads as a framework bug
    rather than a caller one."""


class RewardContractError(ContractError, ValueError):
    """The caller's ``reward`` returned something outside the documented contract.

    A distinct type so the engine can tell a *caller* mistake (fail fast, the run
    is meaningless) from a *backend* failure (stop, keep partial results)."""


def _checked_proposal(value, task: "Task"):
    """``propose`` may return text or ``None``; anything else is a caller error."""
    if value is None or isinstance(value, str):
        return value
    raise ProposalContractError(
        f"propose(task={task.id!r}, ...) returned {type(value).__name__} "
        f"({value!r:.40}); it must return a string or None")


def _checked_reward(value, task: "Task") -> float:
    try:
        r = float(value)
    except (TypeError, ValueError):
        raise RewardContractError(
            f"reward(task={task.id!r}, ...) returned {value!r}; it must return a "
            "number in [0, 1] (1.0 = solved)") from None
    if not (0.0 - _REWARD_TOL) <= r <= (1.0 + _REWARD_TOL):
        raise RewardContractError(
            f"reward(task={task.id!r}, ...) returned {r}, outside [0, 1]. The engine "
            "treats >= 0.999 as solved, so an out-of-range scorer makes every task "
            "look solved and nothing is ever learned. Normalise your score "
            "(e.g. accuracy/100) before returning it.")
    return min(1.0, max(0.0, r))


class _EvalCache:
    def __init__(self) -> None:
        self._d: Dict[Any, float] = {}
        self._lock = threading.Lock()

    def get_or_eval(self, key: Any, fn: Callable[[], float]) -> float:
        with self._lock:
            if key in self._d:
                return self._d[key]
        value = fn()
        with self._lock:
            self._d[key] = value
        return value


class EvolvingArtifact:
    """An :class:`~agentdescent.evolvable.Evolvable`: flat state + a strategy.

    The strategy handles representation (``render``); this class handles the
    Evolvable plumbing and evaluation (``run`` the artifact on tasks, score)."""

    def __init__(self, id: str, state: Optional[Dict[str, str]] = None,
                 version: int = 1, blast_radius: float = 0.2,
                 runtime: Optional["_Runtime"] = None,
                 strategy: Optional[Strategy] = None) -> None:
        self.id = id
        self.state: Dict[str, str] = dict(state or {})
        self.version = version
        self.blast_radius = blast_radius
        self.contract = Contract(input_schema="task", output_schema="text", major=1)
        self._rt = runtime
        self._strategy = strategy or AppendRules()

    def render(self) -> str:
        return self._strategy.render(self.state)

    def diff(self, other: "EvolvingArtifact") -> Diff:
        ops = {k: v for k, v in other.state.items() if self.state.get(k) != v}
        return Diff(diff_id=f"{self.id}:diff", target=self.id, ops=ops)

    def apply(self, diff: Diff) -> "EvolvingArtifact":
        new_state = dict(self.state)
        new_state.update(diff.ops)
        return EvolvingArtifact(self.id, new_state, self.version + 1, self.blast_radius,
                                self._rt, self._strategy)

    def _signature(self):
        return tuple(sorted(self.state.items()))

    def score(self, tasks: Sequence[Task]) -> float:
        """Mean reward over ``tasks``, evaluated concurrently.

        This is the hot path and it used to be a sequential generator sum. Every
        gate in the system goes through it -- each round's held-out measurement and,
        far more often, the aggregator's per-candidate comparisons -- so with N
        candidates a round paid N x len(tasks) rollouts *one at a time*, while the
        workers that produced those candidates ran in parallel. Measured on
        HotpotQA with a reasoning model, that made the merge, not the rollouts,
        about 90% of a round's wall-clock.
        """
        if not tasks or self._rt is None:
            return 0.0
        if len(tasks) == 1 or self._rt.eval_concurrency <= 1:
            return sum(self._rt.eval_one(self, t) for t in tasks) / len(tasks)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(
                min(self._rt.eval_concurrency, len(tasks))) as pool:
            scores = list(pool.map(lambda t: self._rt.eval_one(self, t), tasks))
        return sum(scores) / len(scores)

    def cheap_eval(self, evidence: EvidenceCard) -> float:
        return self.score([t for t in evidence.trajectory_refs if isinstance(t, Task)])

    def full_eval(self, task_set: Sequence[Task]) -> Dict[str, float]:
        return {"reward": self.score(task_set)}


@dataclass
class _Runtime:
    run: Run
    reward: Reward
    cache: _EvalCache
    #: How many held-out tasks to evaluate at once. Memoised and lock-guarded, so
    #: this is safe; 1 restores the old sequential behaviour.
    eval_concurrency: int = 8

    #: Attempts per (artifact, task) evaluation before the failure is raised.
    #: Every evaluation the engine makes funnels through here -- a round's held-out
    #: score, the final score, and the aggregator's own accept/reject measurements
    #: (`cheap_eval`, `eval_counts`, `oracle_eval`) -- and each of those *runs the
    #: agent*, so each is a backend call that can hit a transient. Retrying at the
    #: single choke point covers all of them at once, and it is nearly free: the
    #: result is memoised, so a retry re-runs only what actually failed.
    ATTEMPTS = 3

    def eval_one(self, artifact: EvolvingArtifact, task: Task) -> float:
        key = (artifact._signature(), task.id)

        def _measure() -> float:
            for attempt in range(self.ATTEMPTS):
                try:
                    return _checked_reward(
                        self.reward(task, self.run(artifact.render(), task)), task)
                except ContractError:
                    raise            # a caller bug: retrying cannot help
                except Exception:  # noqa: BLE001 - a backend transient
                    if attempt == self.ATTEMPTS - 1:
                        raise
                    time.sleep(0.2 * (attempt + 1))
            raise AssertionError("unreachable")

        return self.cache.get_or_eval(key, _measure)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


#: Scratch ledgers are named so they can be recognised and reclaimed later.
_SCRATCH_PREFIX = "agentdescent-evolve-"
#: How long an orphaned scratch ledger may sit in $TMPDIR before the next run
#: collects it. Generous, because a *live* run's directory must never be touched.
_SCRATCH_MAX_AGE = 24 * 3600.0


def _reap_stale_scratch_repos(max_age: float = _SCRATCH_MAX_AGE) -> int:
    """Delete scratch ledgers left behind by processes that never exited cleanly.

    ``atexit`` does not run on SIGKILL, an OOM kill or a hard container stop, so
    every such death leaks a git repo into ``$TMPDIR``. Best-effort and silent:
    reclaiming disk must never be able to fail a run."""
    import tempfile
    removed = 0
    try:
        root = tempfile.gettempdir()
        cutoff = time.time() - max_age
        for name in os.listdir(root):
            if not name.startswith(_SCRATCH_PREFIX):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        return removed
    return removed


def _resolve_sections(parallel, strategy) -> Dict[str, int]:
    """Work out which artifact key belongs to which TP section, or refuse.

    Tensor parallelism promises that workers edit disjoint sections of one hot
    artifact, which is what makes the merge a conflict-free union. That is only
    true if the sections partition the keys the **strategy actually writes** --
    and the strategy is the only thing that knows them. A strategy that
    content-addresses its keys (:class:`AppendRules`, whose keys are hashes of the
    proposal text) has no fixed key space at all, so TP cannot constrain it: every
    proposal would land in an arbitrary section and the ~(n-1)/n that missed the
    worker's own would be discarded. That is what used to happen, silently.

    Returns ``{}`` for non-TP strategies (nothing to enforce)."""
    n_sections = getattr(parallel, "n_sections", None)
    if n_sections is None:
        return {}                       # not tensor-parallel
    if n_sections < 1:
        raise ValueError(f"n_sections must be >= 1, got {n_sections}")
    keys = list(getattr(parallel, "keys", None) or [])
    if not keys:
        declared = getattr(strategy, "keys", None)
        keys = list(declared()) if callable(declared) else []
    name = type(strategy).__name__
    if not keys:
        raise ValueError(
            f"TensorParallel needs the artifact's key space, and {name} does not "
            f"declare one (its keys are content-addressed, so a proposal's section "
            f"is unpredictable). Pass the keys explicitly -- "
            f"TensorParallel(n_sections={n_sections}, keys=[...]) -- or use a "
            f"strategy with a fixed key space (KeyedRules), or DataParallel.")
    if n_sections > len(keys):
        raise ValueError(
            f"TensorParallel(n_sections={n_sections}) but {name} has only "
            f"{len(keys)} key(s) ({sorted(keys)[:5]}), so {n_sections - len(keys)} "
            f"section(s) would own nothing and the workers holding them could never "
            f"commit. Lower n_sections to at most {len(keys)}, or use DataParallel.")
    from .parallel import assign_key_sections
    return assign_key_sections(keys, n_sections)


def _reject_pipeline_parallel(parallel) -> None:
    """PP is a multi-artifact paradigm; ``evolve()`` evolves exactly one artifact.

    ``WorkUnit.stage`` -- the only thing distinguishing PP's units, since it hands
    every worker the whole task list -- was never read by the driver, so passing
    ``parallel=PipelineParallel(...)`` silently degraded to n_workers all rolling
    out the same tasks: strictly worse than the default, with no signal. Say so."""
    if type(parallel).__name__ == "PipelineParallel" or getattr(parallel, "name", "") == "PP":
        raise ValueError(
            "evolve() cannot run PipelineParallel: it evolves a single artifact_id, "
            "while PP needs one artifact per stage. Passing it used to be accepted "
            "and quietly ignored (every worker got the whole task list and the "
            "stage was never read), which is worse than the DataParallel default. "
            "The PP primitives are still usable directly -- see "
            "agentdescent.parallel.PipelineChain for stage ordering and upstream "
            "blame attribution.")


def _safe_log(ledger: Ledger, limit: int = 40) -> List[str]:
    """``ledger.log()``, but never at the cost of the result.

    ``ledger_log`` is a diagnostic -- the last few commit subjects. It used to be
    fetched inside the ``return`` expression, so a git failure there discarded a
    run that had already completed every round and computed its final reward."""
    try:
        return ledger.log(Ledger.DEV, limit=limit)
    except LedgerFailure:
        return []


def _tally(reports) -> Dict[str, int]:
    """Count merge outcomes by stable category (see ``MergeReport.category``)."""
    out: Dict[str, int] = {}
    for rep in reports:
        key = getattr(rep, "category", "") or "unknown"
        out[key] = out.get(key, 0) + 1
    return out


@dataclass
class RoundInfo:
    round: int
    held_out_reward: float
    n_items: int
    committed: int
    rejected: int
    #: ``MergeReport.category -> count`` for this round. A run that commits
    #: nothing otherwise reports only ``rejected: 3``, leaving the caller with no
    #: way to tell "the gate says my proposals do not help" from "they never
    #: reached the gate" -- which need opposite fixes.
    reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    state: Dict[str, str]
    rendered: str
    final_reward: float
    history: List[RoundInfo]
    ledger_log: List[str]
    #: ``None`` on a clean run; otherwise a description of the failure that either
    #: ended the run early **or** made its final measurement unusable (in which case
    #: ``final_reward`` falls back to the last measured round, and the message says
    #: so). Covers both a *backend* failure (a rate limit, a dead endpoint) and a
    #: *ledger* failure (a held ``index.lock``, a full ``$TMPDIR``) -- neither is
    #: allowed to escape as an exception. A caller-contract violation is the one
    #: thing that still raises, because the run is meaningless either way. The
    #: artifact evolved so far is still returned -- check this to tell "converged"
    #: from "died".
    error: Optional[str] = None
    #: Workers that gave up after repeated backend failures (async path only). A
    #: run can finish *cleanly* at a fraction of its requested concurrency, so
    #: `error` stays `None` while throughput quietly drops -- check this to tell a
    #: fast run from a lucky one.
    retired_workers: int = 0

    def outcomes(self) -> Dict[str, int]:
        """Merge outcomes for the whole run, by category -- *why* it went as it did.

        The first question about a disappointing run is always "why did nothing
        commit?", and `committed`/`rejected` counts cannot answer it: the fixes are
        opposite. ``below-threshold`` means proposals reached the gate and failed to
        beat the baseline (the reflector is the problem). ``all-stale`` means they
        never reached it (the lag budget is). ``cas-conflict`` means workers raced.

        >>> result.outcomes()
        {'below-threshold': 7, 'committed': 2, 'all-stale': 1}
        """
        out: Dict[str, int] = {}
        for h in self.history:
            for k, v in h.reasons.items():
                out[k] = out.get(k, 0) + v
        return out

    def save(self, path: str) -> None:
        """Write the evolved artifact and its run summary to a JSON file.

        The point of a run is the artifact it produced; without this every caller
        hand-rolls the same serialisation to keep it."""
        import json

        payload = {
            "state": self.state,
            "rendered": self.rendered,
            "final_reward": self.final_reward,
            "error": self.error,
            "history": [
                {"round": h.round, "held_out_reward": h.held_out_reward,
                 "n_items": h.n_items, "committed": h.committed,
                 "rejected": h.rejected, "reasons": h.reasons}
                for h in self.history
            ],
            "retired_workers": self.retired_workers,
            "ledger_log": list(self.ledger_log),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EvolutionResult":
        """Read back a result written by :meth:`save`."""
        import json

        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            state=d["state"], rendered=d["rendered"],
            final_reward=d["final_reward"],
            history=[RoundInfo(**h) for h in d.get("history", [])],
            ledger_log=d.get("ledger_log", []), error=d.get("error"),
            retired_workers=d.get("retired_workers", 0),
        )


@dataclass
class _Engine:
    """Everything the sync and async drivers share: a ledger + runtime + verifier
    + aggregator, plus the resolved actor and the train/held-out split."""

    ledger: Ledger
    runtime: _Runtime
    verifier: Any
    aggregator: Any
    strategy: Strategy
    run: Run
    reward: Reward
    propose: Propose
    train: List[Task]
    held_out: List[Task]
    by_id: Dict[str, Task]
    train_ids: List[str]
    artifact_id: str
    blast_radius: float
    #: Set only when the ledger lives in a throwaway directory this call created;
    #: ``None`` when the caller passed ``repo_path`` (theirs to keep, and how a run
    #: is resumed).
    scratch_repo: Optional[str] = None

    def cleanup(self) -> None:
        """Remove the scratch ledger, if this call created one. Idempotent.

        Close before deleting: a rollout abandoned on ``round_timeout`` keeps
        running (Python cannot cancel a thread) and would otherwise commit into
        the deleted directory, recreating it."""
        if self.scratch_repo:
            self.ledger.close()
            shutil.rmtree(self.scratch_repo, ignore_errors=True)
            self.scratch_repo = None


def _check_callable(fn: Callable, n_args: int, sig_hint: str) -> None:
    """Fail fast if ``fn`` cannot accept ``n_args`` positional arguments.

    Signatures we cannot introspect (builtins, C callables, some partials) are
    left alone -- the check is a courtesy, never a restriction."""
    import inspect
    if not callable(fn):
        raise TypeError(f"expected a callable for {sig_hint}, got {type(fn).__name__}")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(*(None,) * n_args)
    except TypeError as e:
        raise TypeError(
            f"{getattr(fn, '__name__', fn)!r} does not match {sig_hint}: {e}") from None


def _build_engine(tasks, reward, *, agent, run, propose, strategy, initial_state,
                  blast_radius, artifact_id, held_out_frac, repo_path, agg_config,
                  staleness_policy, aggregator_factory, oracle_budget,
                  eval_concurrency: int = 8,
                  cheap_eval_tasks: Optional[int] = None) -> _Engine:
    """Wire the ledger, runtime, verifier and aggregator (shared by
    :func:`evolve` and :func:`~agentdescent.async_evolve.async_evolve`)."""
    import tempfile
    from .verifier import ThreeLayerVerifier, VerifierBudget

    if agent is not None:
        run = run or agent.solve
        propose = propose or agent.propose
    if run is None or propose is None:
        raise ValueError("provide agent=, or both run= and propose=")
    # Check the actor's signatures once, before any rollout. Otherwise a plain
    # typo (a `propose` missing the reward parameter, say) surfaces as a
    # TypeError inside the round body, where the backend-failure handler turns it
    # into an empty, clean-looking result with zero rounds run.
    _check_callable(run, 2, "run(rendered, task)")
    _check_callable(propose, 4, "propose(rendered, task, output, reward)")

    strategy = strategy or AppendRules()
    tasks = list(tasks)
    if len(tasks) < 4:
        raise ValueError("need at least 4 tasks to split train/held-out")
    # Fail loudly on inputs that would otherwise produce silent nonsense: a run
    # that does no work, a split with no training data, or tasks that vanish
    # because two of them share an id.
    if not 0.0 < held_out_frac < 1.0:
        raise ValueError(f"held_out_frac must be in (0, 1), got {held_out_frac}")
    if not 0.0 <= blast_radius <= 1.0:
        raise ValueError(f"blast_radius must be in [0, 1], got {blast_radius}")
    dupes = {t.id for t in tasks if sum(1 for o in tasks if o.id == t.id) > 1}
    if dupes:
        raise ValueError(f"task ids must be unique; duplicated: {sorted(dupes)[:5]}")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", artifact_id):
        raise ValueError("artifact_id must match [A-Za-z0-9_.-]+ (it becomes a filename), "
                         f"got {artifact_id!r}")
    # Governance is a caller-level constraint, so check it before any rollout.
    # Previously only the reference aggregator's per-merge guard caught an L0
    # target: nothing was mutated, but the async path burned its whole budget
    # first and then reported the violation as a *backend failure*.
    assert_mutable(EvolvingArtifact(artifact_id, {}, blast_radius=blast_radius))
    # round, not truncate: Dataset.val_frac promises "the engine's held-out split
    # is exactly this Dataset's val", and float truncation (13.9999 -> 13) quietly
    # pushed one train item into held-out for many dataset sizes.
    cut = max(1, round(len(tasks) * (1 - held_out_frac)))
    train, held_out = tasks[:cut], tasks[cut:]
    if not held_out:
        train, held_out = tasks[:-1], tasks[-1:]

    runtime = _Runtime(run=run, reward=reward, cache=_EvalCache(),
                       eval_concurrency=eval_concurrency)

    def serialize(a: EvolvingArtifact) -> dict:
        return {"state": a.state, "blast_radius": a.blast_radius}

    def deserialize(aid: str, version: int, state: dict) -> EvolvingArtifact:
        return EvolvingArtifact(aid, state.get("state", {}), version,
                                state.get("blast_radius", blast_radius), runtime, strategy)

    scratch: Optional[str] = None
    if repo_path:
        repo = repo_path              # caller-owned (and how a run is resumed): keep it
    else:
        # A scratch ledger per run would otherwise pile up in $TMPDIR forever --
        # one git repo per evolve() call, never reclaimed. atexit alone was not
        # enough: it does not run on SIGKILL/OOM, and inside a notebook or a
        # parameter sweep it fires only when the *interpreter* exits, so every run
        # in the process held a live git repo. The driver now removes its own
        # scratch repo on the way out; atexit stays as the belt-and-braces path for
        # an exception escaping the driver, and the reaper collects what earlier
        # killed processes left behind.
        _reap_stale_scratch_repos()
        repo = scratch = tempfile.mkdtemp(prefix=_SCRATCH_PREFIX)
        atexit.register(shutil.rmtree, repo, True)
    ledger = Ledger(repo, serialize, deserialize)
    # `register` is a no-op when the artifact already exists, which is what makes
    # re-using a repo_path resume the run -- but it also means a supplied
    # initial_state would be discarded without a word. Say so.
    resuming = artifact_id in ledger.head_version(Ledger.DEV)
    if resuming and initial_state:
        warnings.warn(
            f"resuming the existing ledger at {repo!r}: artifact {artifact_id!r} "
            "already has state, so initial_state is ignored. Use a fresh repo_path "
            "to start over.", RuntimeWarning, stacklevel=3)
    ledger.register(EvolvingArtifact(artifact_id, initial_state or strategy.initial(),
                                     blast_radius=blast_radius, runtime=runtime,
                                     strategy=strategy))

    # The cheap layer must actually be cheap. It used to be pinned to the full
    # held-out set (`rule_subset=len(held_out)`) with zero noise, on the reasoning
    # that `eval_fn` is deterministic ground truth -- true of the synthetic router
    # domain, and exactly backwards here, where `eval_fn` RUNS THE AGENT. That made
    # rule / learned / oracle compute the identical number, so the aggregator paid
    # a full held-out sweep for every candidate it merely wanted to *rank*, and
    # `oracle_budget` capped nothing (its documented fallback, `rule_eval`, returned
    # the same value it was trying to avoid buying).
    #
    # Ranking is what the cheap layer is for; committing is not. `eval_counts` --
    # the Beta-posterior acceptance test -- still uses the whole held-out set, so
    # sub-sampling trades tournament precision, never commit safety. Noise stays at
    # zero: `eval_fn` is deterministic, so the sub-sample is the only approximation
    # and inventing more would just make the ranking worse.
    cheap = (len(held_out) if cheap_eval_tasks is None
             else max(1, min(int(cheap_eval_tasks), len(held_out))))
    verifier = ThreeLayerVerifier(
        eval_fn=lambda a, ts: a.score(ts), held_out=held_out,
        rule_subset=cheap, learned_noise=0.0,
        budget=VerifierBudget(oracle_calls_remaining=oracle_budget))

    def _default_aggregator(ledger, verifier, audit, config, policy):
        return Aggregator(ledger, verifier, audit, config, staleness_policy=policy)

    aggregator = (aggregator_factory or _default_aggregator)(
        ledger, verifier, AuditScheduler(),
        agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        staleness_policy)
    # A custom aggregator is the main extension point and is user code. Check the
    # contract here rather than letting it fail three frames deep in the driver
    # with something like "'MissingMethods' object has no attribute 'ingest'".
    for method in ("ingest", "step"):
        if not callable(getattr(aggregator, method, None)):
            raise TypeError(
                f"aggregator_factory returned {type(aggregator).__name__}, which has "
                f"no callable {method}(). An aggregator needs ingest(card) and "
                "step() -> list[MergeReport] (see AggregatorProtocol).")

    return _Engine(ledger, runtime, verifier, aggregator, strategy, run, reward,
                   propose, train, held_out, {t.id: t for t in train},
                   [t.id for t in train], artifact_id, blast_radius, scratch)


def evolve(
    tasks: Sequence[Task],
    reward: Reward,
    *,
    agent: Optional[Agent] = None,
    run: Optional[Run] = None,
    propose: Optional[Propose] = None,
    strategy: Optional[Strategy] = None,
    parallel: Optional["ParallelStrategy"] = None,
    task_sampler: Optional["TaskSampler"] = None,
    initial_state: Optional[Dict[str, str]] = None,
    blast_radius: float = 0.2,
    artifact_id: str = "artifact",
    rounds: int = 15,
    n_workers: int = 4,
    max_concurrency: int = 1,
    round_timeout: Optional[float] = None,
    target_reward: Optional[float] = None,
    patience: Optional[int] = None,
    max_worker_errors: int = 3,
    eval_concurrency: int = 8,
    asynchronous: bool = False,
    async_ratio: int = 3,
    max_seconds: Optional[float] = None,
    self_verify: bool = True,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory: Optional[AggregatorFactory] = None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
    on_round: Optional[Callable[["RoundInfo"], None]] = None,
    verbose: bool = False,
) -> EvolutionResult:
    """Evolve an artifact. Provide either ``agent`` (with ``solve``/``propose``)
    or the ``run`` / ``propose`` callables directly.

    ``strategy`` (default :class:`AppendRules`) is the evolution rule. The
    aggregator dedupes, resolves contradictions, fuses complementary changes, and
    commits a change only if it improves held-out reward.

    ``blast_radius`` chooses governance: ``0.2`` is an L2 (fast, local) artifact
    like a skill; raise it (e.g. ``0.6``) for an **L1** artifact -- a harness,
    context policy, tool router, or learned verifier -- which the aggregator
    treats conservatively (every merge forced through the oracle, wider staleness
    tolerance; design spec §6).

    ``parallel`` (default :class:`~agentdescent.parallel.DataParallel`) is the
    parallelism method -- how each round's tasks are partitioned across the
    ``n_workers``. Swap in :class:`~agentdescent.parallel.TensorParallel` or your
    own :class:`~agentdescent.parallel.ParallelStrategy`. ``PipelineParallel`` is
    refused: it needs one artifact per stage and this function evolves one.

    ``max_concurrency`` runs a round's ``n_workers`` **concurrently** (a thread
    pool), then the single ``aggregator.step()`` is the round barrier -- this is
    *synchronous data-parallelism*: the rollout+propose stage of all workers
    overlaps (real wall-clock speedup for I/O-bound LLM rollouts, since Python
    releases the GIL during network I/O), and the merge is the sync point.
    ``1`` (default) keeps the loop sequential and deterministic; set it to
    ``n_workers`` to parallelise. Custom strategies/aggregators that mutate shared
    state from ``propose``/``to_diff`` must guard it (the async runtime's buffer,
    CAS and per-diff staleness already are). For the *barrier-free* async pipeline
    (``async_ratio`` lag budget, staleness policies overlapping the aggregator),
    see :class:`~agentdescent.async_runtime.AsyncAgentDescent`.

    Parameters
    ----------
    tasks:
        The work the artifact is evaluated on. Split into train / held-out by
        ``held_out_frac``; at least 4 are required and ids must be unique.
    reward:
        ``(task, output) -> [0, 1]``. Scores in ``[0, 1]``; the engine treats
        ``>= 0.999`` as a pass (no proposal is requested).
    agent:
        An object with ``solve`` + ``propose``. Provide this **or** ``run`` and
        ``propose``; both signatures are checked before the first rollout.
    run, propose:
        ``run(rendered, task) -> output`` and
        ``propose(rendered, task, output, reward) -> str | None``.
    strategy:
        How the artifact is represented and how a proposal becomes a ``Diff``.
    parallel:
        How a round's tasks are partitioned across workers. ``DataParallel``
        (default) shards them; ``TensorParallel(n_sections, keys=, route=)`` also
        gives each worker a disjoint **section of the artifact** and rejects
        out-of-section edits -- counted as ``section-violation`` in
        :meth:`EvolutionResult.outcomes`. The pairing is validated before the
        first rollout: a strategy with no declared key space (``AppendRules``) or
        fewer keys than sections is refused rather than silently dropping most of
        its proposals. ``PipelineParallel`` raises (see above).
    task_sampler:
        **Which** task a worker rolls out next, from its shard. Defaults to
        :class:`~agentdescent.sampling.RoundRobin`; use
        :class:`~agentdescent.sampling.DifficultyWeighted` to spend rollouts on
        tasks that still carry a learning signal.
    initial_state:
        Seed the artifact instead of starting from ``strategy.initial()``.
        Ignored when resuming an existing ``repo_path``.
    blast_radius:
        Governance layer, in ``[0, 1]`` (see above).
    artifact_id:
        Name of the evolving artifact; becomes a filename, so it must match
        ``[A-Za-z0-9_.-]+``.
    rounds:
        Number of round barriers to run. Under ``asynchronous=True`` this becomes
        a worker-rollout budget of ``rounds * n_workers`` instead.
    n_workers:
        Workers per round (``>= 1``).
    max_concurrency:
        How many of them actually run at once (see above).
    eval_concurrency:
        How many held-out tasks to score at once. Every gate goes through this --
        each round's measurement and, far more often, the aggregator's
        per-candidate comparisons -- so it is the merge half of the run's
        parallelism, independent of ``n_workers``. ``1`` restores the old
        sequential behaviour.
    max_worker_errors:
        How much total failure to tolerate before giving up -- and only while *no*
        worker has ever completed a rollout, which reads as a misconfiguration
        (wrong key, dead endpoint). Once any worker has succeeded the backend
        demonstrably works, so failures are treated as transient and the run
        continues on whatever evidence it did gather. Counts consecutive failed
        rollouts per worker on the async path (see ``result.retired_workers``) and
        consecutive rounds in which *every* worker failed on the sync path.
    target_reward:
        Stop as soon as held-out reward reaches this. Without it a run always
        spends all ``rounds``, including after it has converged -- measured at 43%
        of rollouts wasted on an artifact that had stopped changing.
    patience:
        Stop after this many consecutive rounds with no improvement in held-out
        reward. ``None`` disables it. Cheap insurance for a run that plateaus
        below ``target_reward``.
    round_timeout:
        Seconds a round will wait for its concurrent workers before giving up on
        the slow ones. ``None`` (default) waits forever, which is what you want
        when every rollout is bounded -- but a single hung rollout then stalls the
        run, because the aggregator is a barrier. Abandoned work keeps running in
        the background (Python cannot cancel a thread) and is simply not waited
        for; it is reported when ``verbose``. Only applies when
        ``max_concurrency > 1``.
    asynchronous, async_ratio:
        Delegate to :func:`~agentdescent.async_evolve.async_evolve` -- no round
        barrier, with ``async_ratio`` as the staleness lag budget.
    max_seconds:
        Wall-clock budget. ``None`` (default) means unbounded; the async path
        uses ``20.0`` when unset.
    self_verify:
        Re-run the trajectory with the diff applied to record a local
        before/after delta. Doubles the rollouts spent per proposal; ports that
        score candidates only on held-out should pass ``False``.
    held_out_frac:
        Fraction of ``tasks`` reserved for held-out scoring, in ``(0, 1)``.
    repo_path:
        Where the git-backed ledger lives. Omit for a throwaway repo that is
        removed when this call returns (not held until interpreter exit, so a
        sweep does not accumulate one git repo per run); **passing the same path
        again resumes** that ledger, and a caller-supplied path is never deleted.
        Git runs with an isolated config, so a personal ``~/.gitconfig``
        (``commit.gpgsign``, ``core.hooksPath``) cannot fail the ledger's own
        bookkeeping commits.
    agg_config:
        Tuning for the reference aggregator (batching, acceptance risk, trust
        region, staleness tolerance).
    staleness_policy:
        What to do with a diff proposed against an out-of-date version --
        ``full`` / ``guarded`` (default) / ``reflective``.
    aggregator_factory:
        Replace the optimizer entirely; receives
        ``(ledger, verifier, audit, config, staleness_policy)``.
    oracle_budget:
        Hard cap on full held-out oracle evaluations during audits. Once spent,
        the verifier falls back to its cheap layer -- which only saves anything
        when ``cheap_eval_tasks`` makes that layer genuinely cheaper, so the two
        knobs go together.
    cheap_eval_tasks:
        How many held-out tasks the *cheap* layer scores when the aggregator is
        merely **ranking** candidates -- conflict resolution and the fusion
        tournament, which run once per candidate. ``None`` (default) scores the
        whole held-out set, which is exact but means a full sweep of real agent
        calls per candidate. Set it to trade ranking precision for cost; the
        acceptance test always scores the full set, so this never affects whether
        a change is safe to commit, only which candidate is put forward. The
        sample is fixed for the run, so candidates are always compared
        like-for-like.
    on_round:
        Called with each :class:`RoundInfo` as the round completes -- progress
        for a long run, which otherwise reports nothing until it returns. An
        exception raised here is reported but does not abort the run.
    verbose:
        Print a line per round. Independent of the ``RuntimeWarning`` emitted
        when a run ends early -- that always fires.

    Returns
    -------
    EvolutionResult
        The evolved artifact plus ``history`` and ``error``. **Check ``error``**:
        it is ``None`` only on a clean run, and a run that died still returns a
        (partial) result rather than raising.
    """
    from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
    from .parallel import DataParallel

    if asynchronous:
        # barrier-free mode: hand the same plug-ins to the async runtime. `rounds`
        # becomes a worker-rollout budget (rounds x n_workers) alongside max_seconds.
        # Two sync-only knobs have no meaning here and were previously dropped in
        # silence, so `parallel=TensorParallel(...)` looked honoured while the run
        # was plain DP. Say so instead: the async runtime shards round-robin itself
        # and its concurrency *is* `n_workers`.
        for name, value, why in (
            ("parallel", parallel,
             f"the async runtime shards data-parallel across its own {n_workers} "
             f"workers"),
            ("max_concurrency", None if max_concurrency == 1 else max_concurrency,
             "async concurrency is n_workers"),
            ("round_timeout", round_timeout,
             "it bounds the round barrier, and the async path has no barrier -- "
             "bound a rollout with your backend's own timeout= instead, and the "
             "run with max_seconds"),
        ):
            if value is not None:
                warnings.warn(
                    f"evolve(asynchronous=True) ignores {name}=: {why}. Use the "
                    f"synchronous path if you need it.", RuntimeWarning,
                    stacklevel=2)
        from .async_evolve import async_evolve
        return async_evolve(
            tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
            initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
            n_workers=n_workers, async_ratio=async_ratio,
            max_seconds=20.0 if max_seconds is None else max_seconds,
            max_iters=rounds * max(1, n_workers), held_out_frac=held_out_frac,
            repo_path=repo_path, agg_config=agg_config, staleness_policy=staleness_policy,
            aggregator_factory=aggregator_factory, oracle_budget=oracle_budget,
            cheap_eval_tasks=cheap_eval_tasks,
            self_verify=self_verify, task_sampler=task_sampler,
            target_reward=target_reward, patience=patience,
            max_worker_errors=max_worker_errors,
            eval_concurrency=eval_concurrency,
            on_round=on_round, verbose=verbose)

    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    parallel = parallel or DataParallel()
    sampler = task_sampler or RoundRobin()
    strategy = strategy or AppendRules()
    # TP owns a *section of the artifact*, so it needs the artifact's key space --
    # not the task ids `plan()` is handed. Resolve and validate it here, before any
    # rollout: an incompatible pairing used to be discovered one diff at a time, by
    # silently discarding it.
    section_map = _resolve_sections(parallel, strategy)
    _reject_pipeline_parallel(parallel)
    section_violations = [0]
    tp_lock = threading.Lock()
    eng = _build_engine(
        tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
        initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
        held_out_frac=held_out_frac, repo_path=repo_path, agg_config=agg_config,
        staleness_policy=staleness_policy, aggregator_factory=aggregator_factory,
        oracle_budget=oracle_budget, eval_concurrency=eval_concurrency,
        cheap_eval_tasks=cheap_eval_tasks)
    ledger, aggregator, strategy = eng.ledger, eng.aggregator, eng.strategy
    run, propose, reward = eng.run, eng.propose, eng.reward
    held_out, by_id, train_ids = eng.held_out, eng.by_id, eng.train_ids

    history: List[RoundInfo] = []
    run_error: Optional[str] = None
    #: The most recent artifact successfully read from the ledger. If the final
    #: read fails there is still a real result to hand back instead of an exception.
    last_good: Optional[EvolvingArtifact] = None
    straggler_rounds = 0
    best_reward = float('-inf')
    stalled = 0
    unit_lock = threading.Lock()
    first_error: List[Optional[str]] = [None]
    contract_error: List[Optional[BaseException]] = [None]   # caller bug -> re-raise
    any_success = [False]      # has ANY worker ever completed a rollout?
    dead_rounds = 0            # consecutive rounds where every worker failed
    deadline = time.time() + max_seconds if max_seconds else None
    for r in range(rounds):
        if deadline is not None and time.time() >= deadline:
            if verbose:
                print(f"round {r:>3}  stopping: max_seconds={max_seconds} reached")
            break
        try:
            snap = ledger.snapshot(Ledger.DEV)
        except LedgerFailure as e:
            # The ledger is infrastructure, not a backend and not a caller bug, so
            # it fits neither existing category -- and letting it propagate broke
            # the one guarantee the result contract makes ("a run that died still
            # returns a partial result"). Treat it like an unmeasurable round: the
            # same tally decides whether to keep going or give up.
            if first_error[0] is None:
                first_error[0] = f"ledger read failed: {type(e).__name__}: {str(e)[:200]}"
            dead_rounds += 1
            if dead_rounds >= max_worker_errors:
                run_error = first_error[0]
                if verbose:
                    print(f"round {r:>3}  giving up: {run_error}")
                break
            if verbose:
                print(f"round {r:>3}  ledger unreadable, skipping: {str(e)[:100]}")
            continue
        artifact = snap.get(artifact_id)
        base_v = snap.version.get(artifact_id, 0)
        assert_mutable(artifact)
        ok_units, failed_units = [0], [0]      # this round's tally

        def _run_unit(unit) -> None:
            """One worker: rollout -> propose -> ingest evidence (against `snap`).

            A backend failure here is this worker's problem, not the round's. It
            used to be the round's: the first exception propagated out of
            `f.result()` and broke the loop, so a *single* transient ended the whole
            run -- measured, one 429 on call 5 turned a 20-round run into 0 rounds.
            """
            try:
                _run_unit_inner(unit)
            except ContractError as e:
                # A caller bug: the run is meaningless. It has to travel back to the
                # main thread by hand -- an exception raised in a plain worker thread
                # goes to the thread excepthook and is lost, not propagated.
                with unit_lock:
                    if contract_error[0] is None:
                        contract_error[0] = e
            except Exception as e:  # noqa: BLE001 - a backend failure
                with unit_lock:
                    if first_error[0] is None:
                        first_error[0] = f"{type(e).__name__}: {str(e)[:200]}"
                    failed_units[0] += 1
                if verbose:
                    print(f"round {r:>3}  worker {unit.worker} failed: "
                          f"{type(e).__name__}: {str(e)[:100]}")

        def _run_unit_inner(unit) -> None:
            if not unit.keys:
                return
            task = by_id[sampler.pick(unit.keys, r)]     # a task from this worker's shard
            output = run(artifact.render(), task)
            score = _checked_reward(reward(task, output), task)
            sampler.record(task.id, score)               # learn which tasks carry signal
            with unit_lock:
                ok_units[0] += 1
                any_success[0] = True
            if score >= 0.999:
                return
            proposal = _checked_proposal(
                propose(artifact.render(), task, output, score), task)
            if not proposal:
                return
            diff = strategy.to_diff(artifact.state, proposal, f"w{unit.worker}", base_v, artifact_id)
            if diff is None:
                return
            # Tensor parallelism means each worker owns a disjoint *section* of the
            # artifact, which is what makes the merge a conflict-free union. The
            # plan assigns the section; enforce it here, or the guarantee is only a
            # comment: without this every worker could edit the same hot key and TP
            # degenerated into differently-sharded DP.
            if unit.section is not None:
                outside = [k for k in diff.ops
                           if section_map.get(k) != unit.section]
                if outside:
                    # Counted, not swallowed. These never reach the aggregator, so
                    # no MergeReport can mention them: without this a TP run that
                    # dropped most of its proposals was indistinguishable from one
                    # whose reflector had nothing useful to say -- opposite fixes.
                    with tp_lock:
                        section_violations[0] += 1
                    if verbose:
                        print(f"round {r:>3}  worker {unit.worker} proposed "
                              f"{outside[0]!r}, outside its section {unit.section}")
                    return
            # The self-verify rollout doubles the cost of every proposal, so it is
            # opt-out here exactly as it is on the async path.
            if self_verify:
                after = _checked_reward(
                    reward(task, run(artifact.apply(diff).render(), task)), task)
                delta = after - score
            else:
                delta = 0.0
            aggregator.ingest(EvidenceCard(
                diff=diff, base_version={artifact_id: base_v}, touched=[artifact_id],
                before_after_delta=delta, trajectory_refs=[task]))

        try:
            # the parallel strategy assigns this round's tasks to workers; they run
            # concurrently (rollout+propose overlap) then the aggregator is the barrier.
            units = list(parallel.plan(n_workers, r, train_ids))
            if max_concurrency > 1 and len(units) > 1:
                # The aggregator is the round barrier, so without a bound the whole
                # round waits on its slowest worker for as long as that takes -- one
                # hung rollout stalls the run indefinitely. round_timeout caps the
                # wait; stragglers are abandoned (Python cannot kill a thread, so the
                # work continues in the background but no longer holds up the round).
                # Daemon threads rather than a ThreadPoolExecutor, because the
                # executor registers an atexit hook that JOINS its workers: with
                # `shutdown(wait=False)` the round moved on as documented, but the
                # abandoned straggler still held the interpreter open at exit.
                # Measured: a rollout wedged for 600s printed "evolve returned" and
                # then kept the process alive -- round_timeout bounded the round and
                # not the program.
                gate = threading.Semaphore(min(max_concurrency, len(units)))

                def _bounded(u=None):
                    with gate:              # preserve max_concurrency
                        _run_unit(u)

                threads = [threading.Thread(target=_bounded, args=(u,), daemon=True)
                           for u in units]
                for t in threads:
                    t.start()
                cutoff = None if round_timeout is None else time.time() + round_timeout
                for t in threads:
                    t.join(None if cutoff is None else max(0.0, cutoff - time.time()))
                pending = [t for t in threads if t.is_alive()]
                if pending:
                    straggler_rounds += 1
                    if verbose:
                        print(f"round {r:>3}  abandoned {len(pending)} straggler(s) "
                              f"after round_timeout={round_timeout}s")
            else:
                for unit in units:
                    _run_unit(unit)

            # Both paths funnel a caller bug through `contract_error` rather than
            # letting it propagate directly, because on the threaded path a raise
            # inside a worker never reaches here. Re-raise before any evidence is
            # read: a broken contract makes the round meaningless.
            if contract_error[0] is not None:
                raise contract_error[0]

            # Decide on the round's tally rather than on the first exception. The
            # same global signal the async path uses: while NO worker has ever
            # completed a rollout, repeated total failure means the backend is
            # misconfigured, so give up quickly and loudly. Once any worker has
            # succeeded the backend demonstrably works, so failures are transient
            # and the run keeps going on the evidence it did gather.
            if failed_units[0] and not ok_units[0]:
                dead_rounds += 1
                if not any_success[0] and dead_rounds >= max_worker_errors:
                    run_error = first_error[0]
                    if verbose:
                        print(f"round {r:>3}  giving up: {dead_rounds} rounds with no "
                              f"worker ever succeeding ({run_error})")
                    break
            else:
                dead_rounds = 0

            reports = check_reports(aggregator.step(), aggregator)
        except ContractError:
            raise            # a caller-contract violation: the run is meaningless
        except Exception as e:  # noqa: BLE001 - a rollout backend failure (e.g. an
            # API/credit error) shouldn't lose the run: stop and return partial results.
            run_error = f"{type(e).__name__}: {str(e)[:200]}"
            if verbose:
                print(f"round {r:>3}  stopped early: {run_error[:140]}")
            break
        committed = sum(1 for x in reports if x.committed_version is not None)
        rejected = sum(1 for x in reports if x.committed_version is None)
        try:
            dev = ledger.snapshot(Ledger.DEV).get(artifact_id)
        except LedgerFailure as e:      # as at the round head: skip, do not raise
            if first_error[0] is None:
                first_error[0] = f"ledger read failed: {type(e).__name__}: {str(e)[:200]}"
            dead_rounds += 1
            if dead_rounds >= max_worker_errors:
                run_error = first_error[0]
                break
            continue
        last_good = dev
        # Scoring held-out runs the agent, so it is a backend call like any other and
        # must not raise out of the driver -- that would discard everything already
        # committed. Treat an unmeasurable round like a failed one: keep the last
        # known reward so early stopping still has something to compare.
        # Retried like the final measurement, and for the same reason: scoring is
        # memoised per (artifact, task), so a retry re-runs only the tasks that
        # actually failed. Giving up after one try loses the whole round's
        # measurement to a single unlucky task -- on a 30-task held-out set with a
        # 1% per-call failure rate that is ~26% of rounds measuring nothing.
        round_reward, score_error = None, None
        for attempt in range(3):
            try:
                round_reward = dev.score(held_out)
                break
            except ContractError:
                raise
            except Exception as e:  # noqa: BLE001
                score_error = e
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
        if round_reward is None:
            e = score_error
            if first_error[0] is None:
                first_error[0] = f"{type(e).__name__}: {str(e)[:200]}"
            dead_rounds += 1
            if not any_success[0] and dead_rounds >= max_worker_errors:
                run_error = first_error[0]
                if verbose:
                    print(f"round {r:>3}  giving up: held-out unmeasurable "
                          f"({run_error})")
                break
            if verbose:
                print(f"round {r:>3}  held-out unmeasurable, carrying last reward: "
                      f"{type(e).__name__}: {str(e)[:100]}")
            continue
        reasons = _tally(reports)
        with tp_lock:
            if section_violations[0]:
                reasons["section-violation"] = section_violations[0]
                section_violations[0] = 0
        info = RoundInfo(r, round_reward, len(dev.state), committed, rejected,
                         reasons)
        history.append(info)
        # Early stopping: an LLM rollout costs money, so do not keep buying them
        # once the artifact has converged or clearly stalled.
        if info.held_out_reward > best_reward + 1e-9:
            best_reward, stalled = info.held_out_reward, 0
        else:
            stalled += 1
        if verbose:
            print(f"round {r:>3}  reward={info.held_out_reward:.3f}  "
                  f"items={info.n_items}  +{committed}/-{rejected}")
        if target_reward is not None and info.held_out_reward >= target_reward:
            if verbose:
                print(f"round {r:>3}  target_reward={target_reward} reached, stopping")
            if on_round is not None:
                try:
                    on_round(info)
                except Exception:  # noqa: BLE001 - reported below on the normal path
                    pass
            break
        if patience is not None and stalled >= patience:
            if verbose:
                print(f"round {r:>3}  no improvement for {stalled} rounds, stopping")
            break
        if on_round is not None:
            # A reporting callback must never take the run down with it.
            try:
                on_round(info)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"on_round callback raised: {type(e).__name__}: {e}",
                              RuntimeWarning, stacklevel=2)

    try:
        final = ledger.snapshot(Ledger.DEV).get(artifact_id)
    except LedgerFailure as e:
        # Fall back to the last artifact we did read. Raising here would throw away
        # a run that has already finished all its rounds, which is exactly what the
        # result contract promises not to do.
        final = last_good
        run_error = run_error or (
            f"the final ledger read failed, so the returned artifact is the last "
            f"one successfully read: {type(e).__name__}: {str(e)[:160]}")
    if final is None:                 # nothing was ever read: hand back the seed
        final = EvolvingArtifact(artifact_id, dict(initial_state or strategy.initial()),
                                 blast_radius=blast_radius, runtime=eng.runtime,
                                 strategy=strategy)
    # Scoring runs the agent, so a dead backend must not raise out of the driver
    # and discard everything already committed.
    try:
        final_reward = final.score(held_out)
    except ContractError:
        raise
    except Exception as e:  # noqa: BLE001 - report, keep the partial result
        run_error = run_error or f"{type(e).__name__}: {str(e)[:200]}"
        final_reward = history[-1].held_out_reward if history else 0.0
    if run_error:
        # Never end a run silently: verbose=False is the default, so a partial
        # result is otherwise indistinguishable from a converged one.
        warnings.warn(f"evolve() stopped early after {len(history)} round(s): "
                      f"{run_error}", RuntimeWarning, stacklevel=2)
    # Read the log before reclaiming the repo, then hand the scratch directory
    # back rather than holding it for the lifetime of the interpreter.
    result = EvolutionResult(state=dict(final.state), rendered=final.render(),
                             final_reward=final_reward, history=history,
                             ledger_log=_safe_log(ledger), error=run_error)
    eng.cleanup()
    return result
