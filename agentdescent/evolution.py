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
from dataclasses import asdict, dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple,
    runtime_checkable,
)

from .agents import Completion, Usage, claude
from .aggregator import (
    Aggregator, AggregatorConfig, AggregatorFactory, AggregatorContractError,
    check_reports,
)
from .evalcache import CacheProtocol, MemoryCache, cache_key
from .evaluator import EvaluatorGroup
from .evolvable import Contract, ContractError, Diff, EvidenceCard, stable_hash
from .governance import FROZEN_IDS, GovernanceError, assert_mutable
from .ledger import Ledger, LedgerFailure
from .metrics import Meter, measured
from .pipeline import EarlyStop, FirstError, WorkerHealth
from .policies import FusionTrial, Policies
from .sampling import RoundRobin, TaskSampler
from .selection import SingleHead
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
# Strategies -- what evolves. They live in their own modules, one per family:
# `strategies` (text) and `treestrategy` (a directory). Re-exported here because
# `from agentdescent.evolution import AppendRules` is a published import path.
# ---------------------------------------------------------------------------

from .strategies import (  # noqa: E402
    AppendRules,
    KeyedRules,
    SingleSlot,
    Strategy,
    rule_id,
)


# ---------------------------------------------------------------------------
# Evaluation cache + the evolving artifact
# ---------------------------------------------------------------------------


# The engine treats a reward of >= 0.999 as a pass and never asks for a proposal,
# so a scorer on the wrong scale (0-100, say) silently means "everything already
# passes": nothing is ever learned, while the reported final_reward looks large and
# healthy. Catch that at the boundary instead.
_REWARD_TOL = 1e-6

#: A reward at or above this counts as solved: the engine asks for no proposal and
#: the task sampler counts a pass. It was written out four times -- twice in the
#: drivers, once in `evolve`'s docstring, and once as `DifficultyWeighted`'s
#: default, whose own docstring says it "mirrors the engine" (exactly the coupling
#: a shared constant exists to express). Right for a binary scorer, and wrong in a
#: way that produces no error for a graded one: a ROUGE or LLM-judge score rarely
#: reaches 0.999, so *every* rollout requests a proposal, the reflector is asked to
#: fix an answer that scored 0.95, and the run reports `below-threshold` -- which
#: reads as "the reflector is useless" when the real cause is that nothing is ever
#: recognised as solved. `evolve(solved_threshold=)` overrides it.
SOLVED = 0.999


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


#: Kept as a name because tests and call sites use it; the implementation moved
#: to `evalcache` when a second process made single-flight and an environment-
#: aware key necessary. A plain dictionary is correct in one process and wasteful
#: in the exact case caching is for.
_EvalCache = MemoryCache


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
        ops: Dict[str, Optional[str]] = {
            k: v for k, v in other.state.items() if self.state.get(k) != v}
        # A key `other` no longer has is a *deletion*, and leaving it out made
        # `a.apply(a.diff(b))` differ from `b` -- silently, and only for the case
        # that matters most to a file tree, where a key is a path. `apply` learned
        # the `None` sentinel; this is the other half of it.
        ops.update({k: None for k in self.state if k not in other.state})
        return Diff(diff_id=f"{self.id}:diff", target=self.id, ops=ops)

    def apply(self, diff: Diff) -> "EvolvingArtifact":
        new_state = dict(self.state)
        # A ``None`` op *removes* the key rather than storing ``None``. Plain
        # ``update`` had no way to express deletion at all, which is invisible for
        # a rules playbook (a stale rule can be overwritten) and disqualifying for
        # a file tree (:class:`~agentdescent.treestrategy.FileTree`), where a key
        # is a path and "delete this file" is an ordinary edit. Popping rather
        # than storing the sentinel keeps every downstream consumer -- ``render``,
        # the ledger's JSON, the trust region -- working on ``Dict[str, str]``.
        for key, value in diff.ops.items():
            if value is None:
                new_state.pop(key, None)
            else:
                new_state[key] = value
        return EvolvingArtifact(self.id, new_state, self.version + 1, self.blast_radius,
                                self._rt, self._strategy)

    def _signature(self):
        """The evaluation-cache key: what the artifact *renders to*.

        It used to be the whole state, which is a finer key than evaluation
        actually depends on -- ``eval_one`` only ever passes ``render()`` to
        ``run``, so two states that render identically cannot score differently.
        Any state a strategy carries for bookkeeping rather than for rendering
        (ADAS keeps the design's name and rationale beside the design itself)
        therefore invalidated the cache for free: the aggregator scored a
        candidate on the full held-out set, the round committed it, and the
        driver's own held-out measurement re-ran every one of those rollouts
        because a label had changed. On an LLM workload that is a duplicate sweep
        of real model calls per committing round."""
        return self.render()

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
        scores = self._rt.evaluator().map(lambda t: self._rt.eval_one(self, t), tasks)
        return sum(scores) / len(scores)

    def evidence_eval(self, evidence: EvidenceCard) -> float:
        """Score this artifact on the trajectories an evidence card carries.

        Says so out loud when there is nothing to score. This is the one call
        behind the staleness policy's REBASE branch, which keeps a rebased diff
        iff ``before <= after`` -- and :meth:`score` returns ``0.0`` for an empty
        task list, so when the filter below empties it the branch compares
        ``0.0 <= 0.0``, keeps *every* rebased diff, and the cheap re-verification
        the policy is named for silently becomes a no-op. A diff that makes the
        artifact worse survives it.

        :class:`~agentdescent.evolvable.EvidenceCard.trajectory_refs` was
        annotated ``List[str]`` for a while, so a custom domain that followed the
        annotation and stored ids landed here with a non-empty list and nothing
        in it to score. The annotation is fixed; this is the part that would have
        made it visible at the time.
        """
        tasks = [t for t in evidence.trajectory_refs if isinstance(t, Task)]
        if not tasks:
            self._warn_unscorable(len(evidence.trajectory_refs))
        return self.score(tasks)

    def _warn_unscorable(self, n_refs: int) -> None:
        """Warn once per run. The caller is a per-card loop, so once is the point."""
        rt = self._rt
        if rt is None or rt.warned_unscorable:
            return
        rt.warned_unscorable = True
        detail = (f"its {n_refs} trajectory ref(s) are not Task objects"
                  if n_refs else "it carries no trajectory refs")
        warnings.warn(
            f"an evidence card reached the staleness filter's re-verification with "
            f"nothing to score ({detail}), so the check compares 0.0 <= 0.0 and "
            "keeps the diff whatever it does to the artifact. Put the failing "
            "Task objects in EvidenceCard.trajectory_refs -- ids are not enough, "
            "the engine scores the tasks themselves.",
            RuntimeWarning, stacklevel=3)

    def full_eval(self, task_set: Sequence[Task]) -> Dict[str, float]:
        """Score on a task set. No longer part of the `Evolvable` protocol -- the
        engine reaches ground truth through the verifier's `eval_fn` -- and kept
        because it is a convenient thing for a caller to have."""
        return {"reward": self.score(task_set)}

    #: Back-compatible alias for :meth:`evidence_eval`, which it was called until
    #: the name collided with the verifier's `cheap_eval(artifact)`.
    cheap_eval = evidence_eval



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

    #: Where evaluation time and cache hits are recorded. Optional so a
    #: hand-built `_Runtime` (several tests do this) needs no changes.
    meter: Optional["Meter"] = None
    #: The evaluation group. Built on first use and kept, because `score()` is
    #: called once per gate and used to build a fresh `ThreadPoolExecutor` every
    #: time -- 83 of them in a six-round run of twelve tasks. A pool created per
    #: call also has no identity: nothing can bound it, observe it, or hand it a
    #: different one.
    eval_group: Optional["EvaluatorGroup"] = None

    def evaluator(self) -> "EvaluatorGroup":
        if self.eval_group is None:
            self.eval_group = EvaluatorGroup(self.eval_concurrency)
        return self.eval_group

    #: Identifies the environment measurements are taken in. Empty while there is
    #: one, which is why a default run's keys are unchanged; the moment there are
    #: two, it is what stops one environment's score answering the other's
    #: question.
    env_fingerprint: str = ""

    #: Has this run already reported an evidence card with nothing to score?
    #: Lives here rather than on the artifact because artifacts are rebuilt from
    #: the ledger constantly, and rather than at module scope because "once" has
    #: to mean once per run, not once per interpreter.
    warned_unscorable: bool = False

    def eval_one(self, artifact: EvolvingArtifact, task: Task) -> float:
        key = cache_key(artifact._signature(), task.id, self.env_fingerprint)

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

        if self.meter is None:
            return self.cache.get_or_eval(key, _measure)
        t0 = time.time()
        try:
            return self.cache.get_or_eval(key, _measure)
        finally:
            # Includes hits (near-zero) on purpose: `eval_seconds` is what the
            # gate cost the run, and a hit costing nothing is the point.
            self.meter.add("eval_seconds", time.time() - t0)


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


def _publish_stable(aggregator) -> None:
    """Publish the run's dev head to ``stable``, if the aggregator knows how.

    Promotion is confirmation-based -- ``promote_after_k`` regression-free rounds
    -- and a run can legitimately end before that many elapse: ``target_reward``
    fires on the very commit that reaches it, and ``patience`` / ``rounds`` /
    ``max_seconds`` can all stop a converged run one round short. Both reference
    runtimes therefore call ``Aggregator.finalize()`` on the way out, and the two
    engines a real workload actually reaches -- :func:`evolve` and
    :func:`~agentdescent.async_evolve.async_evolve` -- did not, so a clean run that
    hit its target left ``stable`` holding the *seed* artifact while ``dev`` held
    the one the run was for. ``docs/ledger.md`` documented the call that was not
    being made.

    Optional on purpose: ``finalize`` is not part of
    :class:`~agentdescent.aggregator.AggregatorProtocol`, so a custom optimizer
    need not have one. And it is a courtesy like ``_safe_log`` -- a git failure
    while publishing must not discard a run that has already finished."""
    fn = getattr(aggregator, "finalize", None)
    if not callable(fn):
        return
    try:
        fn()
    except LedgerFailure:
        pass


#: What `evolve` can honour from a `Policies` bundle today. Everything else
#: raises rather than being accepted and ignored -- see
#: `Policies.require_supported`. The set grows as the implementations land.
_WIRED_POLICIES = ("task_sampler", "selection", "proposal", "conflict", "fusion",
                   "acceptance", "promotion", "staleness", "verifier", "ledger",
                   "aggregator_factory", "eval_cache", "sandbox_spec", "evaluator",
                   "executor")

#: The same bundle, minus the one field the barrier-free loop does not read.
#: `async_evolve`'s worker calls `eng.run` directly; there is no executor seam in
#: it yet, so a supplied one was accepted and then dropped -- the exact outcome
#: `require_supported` exists to make impossible, arriving through a *shared*
#: constant rather than a missing check. It matters more since a supplied
#: executor started working on the synchronous path: flipping
#: `asynchronous=True` would otherwise silently stop honouring it.
_ASYNC_WIRED_POLICIES = tuple(p for p in _WIRED_POLICIES if p != "executor")


def _propose_via_policy(policy):
    """Adapt a `ProposalPolicy` back to the engine's one-proposal contract.

    A policy that returns several is refused rather than truncated: the engine
    turns one proposal into one diff per rollout, so quietly keeping the first
    would discard work the policy did and make a k-sampling algorithm look like
    it ran when only a fraction of it did."""
    from .policies import ProposalContext

    def propose(rendered, task, output, reward):
        out = list(policy.propose(ProposalContext(
            rendered=rendered, task=task, output=output, reward=reward)))
        if len(out) > 1:
            # A caller-contract violation, not a backend failure: raised as one so
            # it travels the channel the engine already has for "this run is
            # meaningless" rather than being folded into `error` as a transient
            # and retried.
            raise ProposalContractError(
                f"the engine consumes one proposal per rollout; this policy "
                f"returned {len(out)}, and keeping the first would silently drop "
                f"{len(out) - 1}. Batched rollouts are what makes k > 1 usable.")
        return out[0] if out else None

    return propose


def _resolve_policies(policies: Optional[Policies], where: str, *,
                      supported: Optional[Sequence[str]] = None,
                      **shortcuts) -> Policies:
    """Fold the legacy keyword arguments into a bundle and check it is honourable.

    The keyword arguments are shortcuts onto bundle fields, so an explicit
    argument has to beat a bundle default -- being silently dropped is the one
    outcome a caller cannot detect.

    ``supported`` is keyword-only and defaults to `evolve`'s set. The two engines
    do not honour quite the same bundle, and sharing one constant meant the
    narrower of them silently ignored a field."""
    merged = (policies or Policies()).merged_with(**shortcuts)
    merged.require_supported(supported or _WIRED_POLICIES, where)
    return merged


def undescribable_actor(which: str = "run"):
    """The ``Ref`` a spec carries when ``evolve()`` cannot name the caller's actor.

    Module-level and public because a `Ref` is resolved *by name*, including on
    the far side of a process boundary -- which is exactly the case this exists
    to report.

    `evolve()` is handed `run` and `reward` as callables. In this process that is
    all an executor needs, and `ThreadExecutor` is given them directly. There is
    no way to turn a closure back into `Ref("module:factory", {...})`, so a spec
    built here cannot describe them for anybody else, and the honest thing for
    the spec to carry is a reference that says so when it is resolved.
    """
    from .workspec import RefError

    raise RefError(
        f"evolve() cannot describe its `{which}` argument as a Ref: it was passed "
        "as a callable, and a closure has no name to resolve on the other side. "
        "The built-in ThreadExecutor is handed the callables directly and never "
        "reaches this. An executor that resolves the spec instead -- anything "
        "across a process boundary -- needs the rollout described as data; build "
        "the RolloutSpecs yourself and drive the executor directly (see "
        "docs/execution.md).")


def _spec_for(engine: "_Engine", artifact, task: Task):
    """Describe one rollout, as completely as `evolve()`'s arguments allow.

    Everything the far side needs is here except the two things `evolve()` was
    given as closures. Those are carried as a reference that raises when it is
    resolved, rather than as a plausible-looking default: a spec that quietly
    names *some* actor turns "this executor cannot be driven from `evolve()`"
    into a finished run measuring the wrong thing.
    """
    from .policies import SandboxSpec
    from .workspec import Ref, RolloutSpec

    return RolloutSpec(
        rendered=artifact.render(), task=task,
        run=Ref("agentdescent.evolution:undescribable_actor", {"which": "run"}),
        reward=Ref("agentdescent.evolution:undescribable_actor", {"which": "reward"}),
        sandbox=engine.sandbox_spec or SandboxSpec())


def _rollout_failure(outcome) -> BaseException:
    """Turn a reported failure back into the exception the round body expects.

    The executor reports rather than raises, because one bad rollout is evidence
    and not the end of a run. The round body's error handling predates that and
    is written against exceptions -- and it distinguishes a caller's broken
    contract, which must stop the run, from a backend transient, which must not.
    That distinction is carried in `Result.kind`, so it survives the trip."""
    if outcome.kind == "caller":
        return ProposalContractError(outcome.error or "contract violation")
    return RuntimeError(outcome.error or "rollout failed")


def notify(on_round: Optional[Callable[["RoundInfo"], None]],
           info: "RoundInfo") -> None:
    """Run a reporting callback without letting it take the run down.

    One behaviour, because there were two: every site warned except the one on
    `evolve`'s target-reached path, which swallowed the exception with a comment
    saying the normal path would report it -- and that path `break`s immediately
    after, so nothing ever did. A callback that raises on the last round of a
    successful run was the one case where the user heard nothing.
    """
    if on_round is None:
        return
    try:
        on_round(info)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"on_round callback raised: {type(e).__name__}: {e}",
                      RuntimeWarning, stacklevel=2)


def _fusion_trials(aggregator) -> List[FusionTrial]:
    """Whatever the fusion policy recorded, if it recorded anything.

    ``trials`` is optional on :class:`~agentdescent.policies.FusionPolicy` -- it
    is instrumentation, not a decision, and a replacement policy is not obliged
    to keep one. Read through ``getattr`` for that reason, and
    :meth:`EvolutionResult.fusion_stats` reports the count so an uninstrumented
    policy cannot be misread as a mechanism that never fired."""
    return list(getattr(getattr(aggregator, "fusion_policy", None), "trials", ()))


def _check_selection(policy, artifact_id, artifact, base_v, round_index,
                     n_workers, history) -> None:
    """Ask the selection policy where the next batch starts, and check we can go.

    The engine has one live head. `SingleHead` -- the default -- returns it for
    every worker, so this is a no-op on the default path and the whole point is
    that it stays one: the seam exists before the ledger can express what a
    non-default policy would ask for.

    A policy that names a *different* starting point is refused, loudly, rather
    than collapsed to the head. Silently ignoring it is the failure mode
    `Policies.require_supported` was written against: a caller who passes a beam
    and watches a run finish has every reason to believe the beam ran. Refusing
    also means every policy in `agentdescent.selection` is usable today in its
    degenerate, single-candidate shape -- ``Beam(1)``, an archive of one -- and
    only genuinely multi-head requests fail.
    """
    from .selection import Candidate, SelectionContext

    head = Candidate(
        artifact_id=artifact_id, version=base_v,
        state=dict(getattr(artifact, "state", {}) or {}),
        # The last measured held-out reward, which is what any of these policies
        # would rank on. `None` before the first round has been scored -- and the
        # policies treat that as "unmeasured", not as zero.
        score=history[-1].held_out_reward if history else None,
        selected=round_index)
    ctx = SelectionContext(head=head, candidates=(head,), round=round_index,
                           n_workers=n_workers)
    chosen = list(policy.select(ctx, n_workers))
    if any(c.version != base_v or c.artifact_id != artifact_id for c in chosen):
        raise NotImplementedError(
            f"{type(policy).__name__}.select() asked to start from a candidate "
            f"other than the current head, and the ledger holds one live branch: "
            "`dev`, with staleness defined as eta = max(head - base). Multi-head "
            "support is a separate change; until then a selection policy may only "
            "return the head it was given.")


def _cost_fields(meter: Meter) -> Dict[str, Any]:
    """The meter's counters, keyed as :class:`EvolutionResult` fields.

    One mapping shared by both drivers: they assemble their results separately,
    and a field added to one and forgotten in the other is invisible -- the
    result still constructs, the number is just always zero."""
    m = meter.snapshot()
    return {
        "usage": meter.usage,
        "wallclock": m.elapsed_s,
        "rollouts": m.rollouts,
        "rollout_seconds": m.rollout_seconds,
        "eval_seconds": m.eval_seconds,
        "stale_considered": m.stale_considered,
        "stale_discarded": m.stale_discarded,
        "redispatched": m.redispatched,
        "duplicates_dropped": m.duplicates_dropped,
        "cas_conflicts": m.cas_conflicts,
        "cache_hits": m.cache_hits,
        "cache_misses": m.cache_misses,
        "sandbox_wait_s": m.sandbox_wait_s,
        "sandbox_setup_s": m.sandbox_setup_s,
        "sandboxes_created": m.sandboxes_created,
        "sandboxes_reused": m.sandboxes_reused,
        "sandbox_failures": m.sandbox_failures,
    }


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
    #: How many **keys the artifact holds** -- its size (rules for `AppendRules`,
    #: slots for `KeyedRules`), not how many tasks `held_out_reward` was measured
    #: on. The verbose line prints it as ``size=`` for that reason: sitting beside
    #: the reward under the name ``items`` it reads as the sample size, and a
    #: reader who takes it that way concludes a 108-item measurement rested on 3.
    n_items: int
    committed: int
    rejected: int
    #: ``MergeReport.category -> count`` for this round. A run that commits
    #: nothing otherwise reports only ``rejected: 3``, leaving the caller with no
    #: way to tell "the gate says my proposals do not help" from "they never
    #: reached the gate" -- which need opposite fixes.
    reasons: Dict[str, int] = field(default_factory=dict)
    #: Cumulative wall-clock at the end of this round, in seconds. This is the
    #: x-axis for time-to-quality: paired with ``held_out_reward`` it says when a
    #: quality bar was reached, which a round index cannot -- rounds are not the
    #: same length, and across parallel configurations they are not comparable.
    elapsed_s: float = 0.0
    #: Rollouts completed by the end of this round, cumulative. The other
    #: denominator: a configuration that reaches the same reward having spent
    #: twice the rollouts has not done as well.
    rollouts: int = 0
    #: Actor invocations by the end of this round, cumulative. Recorded per round
    #: because a budget fixed in *rounds* hands the wider configuration more
    #: model and then reports the extra model as a win for parallelism -- so a
    #: comparison has to be able to ask "where was each configuration after N
    #: calls", which needs the number at every round rather than only at the end.
    calls: int = 0

    # -- what the merge did, this round --------------------------------------
    #
    # `MergeReport` computes all four and the driver used to throw them away, so
    # a `evolve()` user could see *that* nothing committed and never *how* the
    # merge got there. The reference runtimes reported them all along
    # (`RoundStat.fused`, `AsyncStats.conflicts_dropped`), which is the shape of
    # gap this seam exists to close.

    #: Evidence cards this round's merge looked at, before any filter. The
    #: denominator: `discarded_stale=3` needs "out of how many" to mean anything,
    #: and shipping a numerator without one is a mistake this codebase has
    #: already made once, on the async path's stale rate.
    considered: int = 0
    #: Cards the staleness filter dropped. Rising here with a flat
    #: `held_out_reward` is the lag budget, not the reflector.
    discarded_stale: int = 0
    #: Diffs dropped because they contradicted a better-scoring one. Non-zero
    #: means the workers are genuinely disagreeing about the same key, which is
    #: the case `KeyedRules` and `SingleSlot` are for and `AppendRules` avoids.
    conflicts_dropped: int = 0
    #: Commits whose winning candidate was the **fusion** of several diffs rather
    #: than any single one -- the model-soup question, per round. Counted only
    #: when it committed: the tournament builds a fused candidate whenever the
    #: survivors are complementary, so counting the ones it built says nothing
    #: about whether combining them beat taking the best single diff.
    fused: int = 0


@dataclass(frozen=True)
class FusionStats:
    """The fusion tournament's record, with every denominator it needs.

    Four counters rather than one rate, because the rate alone is ambiguous in
    both directions. ``trials`` counts tournaments held; ``contested`` counts the
    ones where a fused candidate existed to compete at all. A run with
    ``trials=40, contested=0`` never tested fusion once -- every round either had
    a single survivor or had contradicting ones -- and reporting "win rate 0%"
    for it would be a claim about a mechanism that never ran.
    """

    trials: int = 0
    #: Tournaments where a fused candidate was built and scored.
    contested: int = 0
    #: Why the rest were not contested.
    single_candidate: int = 0
    contradiction: int = 0

    fused_wins: int = 0
    single_wins: int = 0
    #: Tournaments where nothing beat the artifact the round started from.
    neither: int = 0
    #: Contested tournaments where the fusion exactly tied the best single. High
    #: here with an empty negative tail means the cheap layer cannot separate the
    #: candidates, not that fusion is safe.
    ties: int = 0

    #: Mean of (fused - best single) over contested tournaments.
    mean_gain: float = 0.0
    #: The losing tail: how many contested tournaments the fusion lost, its mean
    #: loss, and the worst single one. This is the number the objection is about,
    #: and the tournament's job is to make sure it never commits.
    negative: int = 0
    mean_loss: float = 0.0
    worst_loss: float = 0.0
    #: Fusions that were worse than the *baseline* -- not merely ranked below the
    #: best single, but actively harmful. `negative` includes ranking noise;
    #: this is the failure mode "merging averages the improvements away" names.
    below_baseline: int = 0

    @property
    def win_rate(self) -> Optional[float]:
        """Fused wins over contested tournaments; ``None`` when none were.

        ``None`` rather than ``0.0`` on purpose: a rate with an empty denominator
        printed as zero reads as "fusion always lost"."""
        return self.fused_wins / self.contested if self.contested else None

    @classmethod
    def of(cls, trials: Sequence["FusionTrial"]) -> "FusionStats":
        gains = [t.gain for t in trials if t.gain is not None]
        losses = [g for g in gains if g < 0]
        return cls(
            trials=len(trials),
            contested=len(gains),
            single_candidate=sum(1 for t in trials if t.reason == "single-candidate"),
            contradiction=sum(1 for t in trials if t.reason == "contradiction"),
            fused_wins=sum(1 for t in trials if t.winner == "fused"),
            single_wins=sum(1 for t in trials if t.winner == "single"),
            neither=sum(1 for t in trials if t.winner == "neither"),
            ties=sum(1 for g in gains if g == 0.0),
            mean_gain=sum(gains) / len(gains) if gains else 0.0,
            negative=len(losses),
            mean_loss=sum(losses) / len(losses) if losses else 0.0,
            worst_loss=min(losses) if losses else 0.0,
            below_baseline=sum(
                1 for t in trials
                if t.fused_score is not None and t.fused_score < t.baseline_score))

    def summary(self) -> str:
        """One line, and it says when there is nothing to report."""
        if not self.contested:
            return (f"fusion: {self.trials} tournaments, none contested "
                    f"({self.single_candidate} single-candidate, "
                    f"{self.contradiction} contradicting) -- fusion never ran")
        return (f"fusion: won {self.fused_wins}/{self.contested} "
                f"({self.win_rate:.0%}), mean gain {self.mean_gain:+.3f}, "
                f"{self.negative} losses (worst {self.worst_loss:+.3f}, "
                f"{self.below_baseline} below baseline), {self.ties} ties")


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
    #: Why the run ended -- ``"target_reward"`` / ``"patience"`` / ``"rounds"`` /
    #: ``"max_seconds"`` / ``"max_iters"`` / ``"max_rollouts"`` / ``"max_calls"``
    #: / ``"error"``. Without it a budget
    #: expiry is indistinguishable from convergence: ``error`` is ``None`` for
    #: both, ``history`` has entries for both, and the only other clue is
    #: re-deriving ``len(history)`` against arguments whose meaning changes between
    #: the sync and async paths. The ``verbose`` print lines always knew the
    #: reason; this makes it available to a non-interactive caller.
    stop_reason: str = "rounds"
    #: Times a worker was forced to resync because the pipeline stalled -- cards
    #: arriving, nothing committing (async path only). A non-zero count means the
    #: lag budget and the staleness tolerance are mismatched.
    forced_refreshes: int = 0
    #: Rollouts that overran their own predicted duration by ``straggler_factor``
    #: (async path, and only when a ``duration_estimator`` was given). The design's
    #: L-traj signal; detection only, nothing is resumed.
    stragglers: int = 0
    #: Workers that gave up after repeated backend failures (async path only). A
    #: run can finish *cleanly* at a fraction of its requested concurrency, so
    #: `error` stays `None` while throughput quietly drops -- check this to tell a
    #: fast run from a lucky one.
    retired_workers: int = 0

    # -- what the run cost ----------------------------------------------------
    #
    # Every field below defaults, and `load` reads them with `.get`, so a result
    # written before they existed still loads.

    #: Model spend. ``calls`` counts actor invocations (``run`` and ``propose``);
    #: token counts appear only when the same :class:`~agentdescent.agents.Usage`
    #: was also passed to an adapter that reports them -- pass ``usage=`` to both
    #: ``evolve`` and ``claude``/``openai_compatible`` and they accumulate
    #: together.
    usage: Usage = field(default_factory=Usage)
    #: Total wall-clock of the run, in seconds.
    wallclock: float = 0.0
    #: Rollouts completed, and their summed duration. The sum exceeds
    #: ``wallclock`` whenever workers overlapped -- that is the point of it.
    rollouts: int = 0
    rollout_seconds: float = 0.0
    #: Wall-clock inside the gate (held-out scoring and every acceptance
    #: measurement). Measured against ``wallclock`` this is what says whether a
    #: run was spent exploring or verifying.
    eval_seconds: float = 0.0
    #: Staleness, with its denominator: ``discarded / considered``. Without the
    #: denominator a stale count cannot be read at all.
    stale_considered: int = 0
    stale_discarded: int = 0
    #: Task-level recovery: tasks sent out again after their worker was presumed
    #: lost, and results that arrived for a task already answered. The second
    #: being non-zero means re-dispatch is firing on workers that were still
    #: alive -- correct, and paid for twice.
    redispatched: int = 0
    duplicates_dropped: int = 0
    #: Commits that lost a compare-and-swap race and rebased. Zero with one
    #: writer by construction.
    cas_conflicts: int = 0
    #: Evaluation cache. ``misses`` is the number of evaluations actually
    #: performed; the ratio is the duplicate-computation figure a multi-process
    #: run has to report.
    cache_hits: int = 0
    cache_misses: int = 0
    #: Sandbox accounting. Zero on the default single-workspace path: there is
    #: no pool to wait for and no image to warm. They exist here so that when a
    #: pool does appear, "8 workers only bought 2x" can be attributed to queueing
    #: or cold starts rather than guessed at.
    sandbox_wait_s: float = 0.0
    sandbox_setup_s: float = 0.0
    sandboxes_created: int = 0
    sandboxes_reused: int = 0
    sandbox_failures: int = 0
    #: Every fusion tournament the run held, when the fusion policy recorded
    #: them (the shipped one does). Read it through :meth:`fusion_stats`.
    fusion_trials: List["FusionTrial"] = field(default_factory=list)

    def fusion_stats(self) -> "FusionStats":
        """How often merging beat the best single diff -- and how badly it lost.

        The strongest objection to this whole design is that two workers' local
        improvements might be worse together than either is alone. `RoundStat.fused`
        counted how often a fusion was *committed*, which cannot answer it: the
        tournament only ever commits a fusion that won, so the count is a tally of
        successes with the denominator missing.

        Three shapes of answer, all worth having:

        * **win rate well above 50%** -- merging recovers the N-1 proposals
          best-of-N throws away, which is the claim.
        * **win rate near 50%** -- fusion is noise, and the tournament's cost (an
          extra held-out pass per merge) has to be justified some other way.
        * **win rate below 50%, with the tournament catching it** -- the gate is
          doing real work, which is its own result: the optimizer audits itself.

        Read ``negative`` before the win rate. An empty negative tail does not mean
        fusion never hurts; on a small held-out set it usually means the cheap layer
        cannot separate the candidates at all, and ``ties`` is the tell.
        """
        return FusionStats.of(self.fusion_trials)

    def outcomes(self) -> Dict[str, int]:
        """Merge outcomes for the whole run, by category -- *why* it went as it did.

        The first question about a disappointing run is always "why did nothing
        commit?", and `committed`/`rejected` counts cannot answer it: the fixes are
        opposite. ``below-threshold`` means proposals reached the gate and failed to
        beat the baseline (the reflector is the problem). ``all-stale`` means they
        never reached it (the lag budget is). ``cas-conflict`` means workers raced.

        The keys are :class:`~agentdescent.aggregator.MergeOutcome` values, which
        subclass ``str`` -- so ``outcomes()["below-threshold"]`` works, and so does
        ``outcomes()[MergeOutcome.BELOW_THRESHOLD]``:

        ``committed`` · ``below-threshold`` · ``all-stale`` · ``oversized``
        (outside the trust region -- a runaway reflector, which used to be counted
        as ``all-stale`` and so pointed at the opposite fix) · ``oracle-rejected``
        · ``cas-conflict`` · ``unknown-artifact`` · ``section-violation``
        (tensor-parallel only).

        >>> result.outcomes()
        {'below-threshold': 7, 'committed': 2, 'all-stale': 1}
        """
        out: Dict[str, int] = {}
        for h in self.history:
            for k, v in h.reasons.items():
                out[k] = out.get(k, 0) + v
        return out

    def time_to_quality(self, target: float) -> Optional[float]:
        """Wall-clock at the first round whose held-out reward reached ``target``.

        ``None`` when the run never got there -- which is a result, not an error:
        a configuration that never reaches the bar has no time-to-quality, and
        reporting its total wall-clock instead would flatter it."""
        for h in self.history:
            if h.held_out_reward >= target:
                return h.elapsed_s
        return None

    def cost_to_quality(self, target: float) -> Optional[int]:
        """Rollouts spent up to the first round that reached ``target``.

        Rollouts rather than tokens, because tokens are only known when a
        token-reporting adapter shares this result's :class:`Usage`; rollouts are
        always counted. Use ``usage.total_tokens`` when you have them."""
        for h in self.history:
            if h.held_out_reward >= target:
                return h.rollouts
        return None

    def stale_rate(self) -> float:
        """Discarded evidence as a fraction of evidence considered; 0.0 if none."""
        return (self.stale_discarded / self.stale_considered
                if self.stale_considered else 0.0)

    def duplicate_rate(self) -> float:
        """Cache hits as a fraction of lookups -- work that did *not* have to be
        redone. In one process this is memoisation working; across processes it
        is the figure that says how much a shared cache would be worth."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def cost_summary(self) -> str:
        """One line: what the run cost. Complements ``outcomes()``, which says why
        it went as it did."""
        parts = [f"{self.rollouts} rollouts in {self.wallclock:.1f}s",
                 self.usage.summary()]
        if self.eval_seconds:
            parts.append(f"{self.eval_seconds:.1f}s in the gate")
        if self.stale_considered:
            parts.append(f"stale {self.stale_rate():.0%}")
        if self.cache_hits + self.cache_misses:
            parts.append(f"cache {self.duplicate_rate():.0%} hit")
        if self.sandboxes_created:
            parts.append(f"{self.sandboxes_created} sandboxes, "
                         f"{self.sandbox_wait_s:.1f}s waiting")
        return " | ".join(parts)

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
                 "rejected": h.rejected, "reasons": h.reasons,
                 "elapsed_s": h.elapsed_s, "rollouts": h.rollouts,
                 "calls": h.calls, "considered": h.considered,
                 "discarded_stale": h.discarded_stale,
                 "conflicts_dropped": h.conflicts_dropped, "fused": h.fused}
                for h in self.history
            ],
            "retired_workers": self.retired_workers,
            "forced_refreshes": self.forced_refreshes,
            "stragglers": self.stragglers,
            "stop_reason": self.stop_reason,
            "ledger_log": list(self.ledger_log),
            "usage": {"calls": self.usage.calls,
                      "prompt_tokens": self.usage.prompt_tokens,
                      "completion_tokens": self.usage.completion_tokens,
                      "seconds": self.usage.seconds,
                      "failures": self.usage.failures},
            "wallclock": self.wallclock,
            "rollouts": self.rollouts,
            "rollout_seconds": self.rollout_seconds,
            "eval_seconds": self.eval_seconds,
            "stale_considered": self.stale_considered,
            "stale_discarded": self.stale_discarded,
            "redispatched": self.redispatched,
            "duplicates_dropped": self.duplicates_dropped,
            "cas_conflicts": self.cas_conflicts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "sandbox_wait_s": self.sandbox_wait_s,
            "sandbox_setup_s": self.sandbox_setup_s,
            "sandboxes_created": self.sandboxes_created,
            "sandboxes_reused": self.sandboxes_reused,
            "sandbox_failures": self.sandbox_failures,
            "fusion_trials": [asdict(t) for t in self.fusion_trials],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def write_to(self, path: str, *, backup: bool = True, prune: bool = False,
                 dry_run: bool = False) -> Dict[str, List[str]]:
        """Install a file-tree artifact back into a real directory.

        Only meaningful when the artifact was evolved as a
        :class:`~agentdescent.treestrategy.FileTree` -- every state key must be a
        relative path. Returns the plan: ``{"written": [...], "extra": [...],
        "deleted": [...], "backup": [...]}``.

        Deliberately conservative, because this is the one call in the whole
        package that writes into a directory the user cares about:

        * ``backup=True`` (default) copies the existing directory to
          ``<path>.bak-N`` first;
        * files present in the target but **not** in the artifact are reported as
          ``extra`` and left alone unless ``prune=True`` -- the run only ever knew
          about the files its ``TreeSpec`` selected, so deleting by omission would
          take out anything the spec excluded;
        * ``dry_run=True`` returns the same plan without touching the disk.
        """
        import os
        import shutil

        from .filetree import TreeError, materialize, parse_tree, safe_relpath

        # `rendered` is the strategy's own serialisation, so it is the one
        # reliable signal that this artifact really is a tree. Without the check,
        # an `AppendRules` result would happily be written out as a directory of
        # files named after rule hashes -- every key of every strategy is a
        # *syntactically* valid relative path.
        try:
            parse_tree(self.rendered)
        except TreeError as e:
            raise TreeError(
                "write_to() only applies to an artifact evolved as a FileTree; "
                f"this result does not render as one ({e}). Use save() for a "
                "text artifact.") from None

        bad = []
        for key in self.state:
            try:
                safe_relpath(key)
            except TreeError as e:
                bad.append(str(e))
        if bad:
            raise TreeError(
                "write_to() needs a file-tree artifact (every state key a relative "
                "path); this result has keys that are not paths:\n  "
                + "\n  ".join(bad[:5]))

        dest = os.path.abspath(os.path.expanduser(path))
        planned = sorted(self.state)
        existing: List[str] = []
        walker = os.walk(dest) if os.path.isdir(dest) else ()
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), dest)
                existing.append(rel.replace(os.sep, "/"))
        extra = sorted(set(existing) - set(planned))
        plan = {"written": planned, "extra": [] if prune else extra,
                "deleted": extra if prune else [], "backup": []}
        if dry_run:
            return plan
        if backup and os.path.isdir(dest):
            n = 0
            while os.path.exists(f"{dest}.bak-{n}"):
                n += 1
            shutil.copytree(dest, f"{dest}.bak-{n}", symlinks=True)
            plan["backup"] = [f"{dest}.bak-{n}"]
        materialize(self.state, dest)
        if prune:
            for rel in extra:
                try:
                    os.remove(os.path.join(dest, rel.replace("/", os.sep)))
                except OSError:
                    pass
        return plan

    @classmethod
    def load(cls, path: str) -> "EvolutionResult":
        """Read back a result written by :meth:`save`."""
        import json

        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        u = d.get("usage") or {}
        return cls(
            state=d["state"], rendered=d["rendered"],
            final_reward=d["final_reward"],
            history=[RoundInfo(**h) for h in d.get("history", [])],
            ledger_log=d.get("ledger_log", []), error=d.get("error"),
            retired_workers=d.get("retired_workers", 0),
            forced_refreshes=d.get("forced_refreshes", 0),
            stragglers=d.get("stragglers", 0),
            stop_reason=d.get("stop_reason", "rounds"),
            fusion_trials=[FusionTrial(**t) for t in d.get("fusion_trials", [])],
            # Every cost field is `.get` with a default, so a file written before
            # they existed loads as a run that simply did not measure them.
            usage=Usage(calls=u.get("calls", 0),
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        seconds=u.get("seconds", 0.0),
                        failures=u.get("failures", 0)),
            wallclock=d.get("wallclock", 0.0),
            rollouts=d.get("rollouts", 0),
            rollout_seconds=d.get("rollout_seconds", 0.0),
            eval_seconds=d.get("eval_seconds", 0.0),
            stale_considered=d.get("stale_considered", 0),
            stale_discarded=d.get("stale_discarded", 0),
            redispatched=d.get("redispatched", 0),
            duplicates_dropped=d.get("duplicates_dropped", 0),
            cas_conflicts=d.get("cas_conflicts", 0),
            cache_hits=d.get("cache_hits", 0),
            cache_misses=d.get("cache_misses", 0),
            sandbox_wait_s=d.get("sandbox_wait_s", 0.0),
            sandbox_setup_s=d.get("sandbox_setup_s", 0.0),
            sandboxes_created=d.get("sandboxes_created", 0),
            sandboxes_reused=d.get("sandboxes_reused", 0),
            sandbox_failures=d.get("sandbox_failures", 0),
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
    #: Where rollouts run. Always present, defaulting to this process, so the
    #: round body has one path rather than one per substrate.
    executor: Any = None
    #: What environment a rollout asks for. Empty by default; it is what a
    #: cross-process executor hands its sandbox pool.
    sandbox_spec: Any = None
    #: Every counter this run accumulates. Always present: an optional meter
    #: would mean every recording site grows an `if`, and the sites are the hot
    #: path.
    meter: Meter = field(default_factory=Meter)
    #: Set only when the ledger lives in a throwaway directory this call created;
    #: ``None`` when the caller passed ``repo_path`` (theirs to keep, and how a run
    #: is resumed).
    scratch_repo: Optional[str] = None

    def record_round(self, *, index: int, reward: float, n_items: int,
                     reports: Sequence[Any],
                     history: List["RoundInfo"], early: "EarlyStop",
                     on_round: Optional[Callable[["RoundInfo"], None]],
                     extra_reasons: Optional[Dict[str, int]] = None,
                     ) -> Tuple["RoundInfo", Optional[str]]:
        """Close out one round: record it, ask whether to stop, tell the caller.

        The two loops disagree about what a round *is* -- a barrier sweep in one,
        a merger sweep in the other -- and agreed on everything that happens once
        one has finished. That agreement was written twice.

        It now takes the ``MergeReport``s rather than a pre-chewed
        ``committed`` / ``rejected`` / ``reasons``, because deriving those was the
        *other* thing both loops were doing separately -- and one of them counted
        ``rejected`` as ``len(reports) - committed`` while the other re-scanned for
        a missing ``committed_version``. Same answer, two spellings, and the
        numbers below could have been added to one loop and not the other in
        exactly the way this whole seam exists to stop.

        ``extra_reasons`` is for a category the aggregator cannot know about --
        today only tensor parallelism's ``section-violation``, which is counted in
        the round body because those diffs never reach a bucket.

        Returns the round and a stop reason, or `None` to continue. The caller
        stops; this does not, because "should we stop" and "how do we stop" are
        different questions and only the loop knows the second.
        """
        reports = list(reports)
        committed = sum(1 for x in reports if x.committed_version is not None)
        reasons = _tally(reports)
        if extra_reasons:
            reasons.update(extra_reasons)
        m = self.meter.snapshot()
        info = RoundInfo(
            index, reward, n_items, committed, len(reports) - committed, reasons,
            elapsed_s=m.elapsed_s, rollouts=m.rollouts, calls=m.calls,
            considered=sum(x.considered for x in reports),
            discarded_stale=sum(x.discarded_stale for x in reports),
            conflicts_dropped=sum(x.conflicts_dropped for x in reports),
            # A fusion that *won*, which is the question worth asking -- the
            # tournament builds a fused candidate whenever the survivors are
            # complementary, so counting the ones it built says nothing about
            # whether combining them beat taking the best single diff.
            fused=sum(1 for x in reports
                      if x.fused and x.committed_version is not None))
        history.append(info)
        stop_reason = early.observe(info.held_out_reward)
        notify(on_round, info)
        return info, stop_reason

    def cleanup(self) -> None:
        """Remove the scratch ledger, if this call created one. Idempotent.

        Close before deleting: a rollout abandoned on ``round_timeout`` keeps
        running (Python cannot cancel a thread) and would otherwise commit into
        the deleted directory, recreating it."""
        group = getattr(self.runtime, "eval_group", None)
        if group is not None:
            # Not waiting: an evaluation still in flight is holding a rollout
            # that cannot be cancelled, and the run is already over.
            group.shutdown(wait=False)
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
                  cheap_eval_tasks: Optional[int] = None,
                  shuffle: bool = False, seed: int = 0,
                  usage: Optional[Usage] = None,
                  verifier: Optional[Any] = None,
                  ledger_impl: Optional[Any] = None,
                  policies_bundle: Optional[Policies] = None) -> _Engine:
    """Wire the ledger, runtime, verifier and aggregator (shared by
    :func:`evolve` and :func:`~agentdescent.async_evolve.async_evolve`)."""
    import tempfile
    from .verifier import ThreeLayerVerifier, VerifierBudget

    if agent is not None:
        run = run or agent.solve
        propose = propose or agent.propose
    if run is None or propose is None:
        raise ValueError("provide agent=, or both run= and propose=")
    if policies_bundle is not None and policies_bundle.proposal is not None:
        propose = _propose_via_policy(policies_bundle.proposal)
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
    # A handful of names are reserved for the frozen layer, and they are ordinary
    # words -- "oracle" is a plausible name for an evolving judge prompt. Say so
    # here, where the other artifact_id rules live, rather than letting it surface
    # as a GovernanceError on the first round that names governance and not the
    # actual cause, which is the *name*.
    if artifact_id in FROZEN_IDS:
        # GovernanceError, not ValueError: refusing to mutate L0 is the safety
        # claim, and callers are told to catch that type. What changes is the
        # *message* -- it now names the cause, which is the name -- and the timing,
        # since this is checked beside the other artifact_id rules rather than
        # surfacing on the first round.
        raise GovernanceError(
            f"artifact_id={artifact_id!r} is reserved for the L0 frozen layer "
            f"({', '.join(sorted(FROZEN_IDS))}), which the evolution loop may only "
            "read -- these are ordinary words, so this is most likely just a name "
            "collision. Rename the artifact.")
    # Governance is a caller-level constraint, so check it before any rollout.
    # Previously only the reference aggregator's per-merge guard caught an L0
    # target: nothing was mutated, but the async path burned its whole budget
    # first and then reported the violation as a *backend failure*.
    assert_mutable(EvolvingArtifact(artifact_id, {}, blast_radius=blast_radius))
    if shuffle:
        # Off by default so `Dataset.val_frac` keeps its promise ("the engine's
        # held-out split is exactly this Dataset's val", which only holds because
        # `trainval` is train + val in that order and the split below is
        # positional) and so a seeded run stays reproducible.
        import random as _random
        tasks = list(tasks)
        _random.Random(seed).shuffle(tasks)
    # round, not truncate: Dataset.val_frac promises "the engine's held-out split
    # is exactly this Dataset's val", and float truncation (13.9999 -> 13) quietly
    # pushed one train item into held-out for many dataset sizes.
    cut = max(1, round(len(tasks) * (1 - held_out_frac)))
    train, held_out = tasks[:cut], tasks[cut:]
    if not held_out:
        train, held_out = tasks[:-1], tasks[-1:]
    # Every gate in the run -- the Beta acceptance test, target_reward, patience,
    # final_reward -- is measured on this set. At 3 items a single binary rollout
    # moves the reported reward by 0.33, and at 1 it is 0.0 or 1.0 and nothing
    # else. The engine already refuses fewer than 4 tasks; this is the same
    # argument one level down, where it decides whether the numbers mean anything.
    if len(held_out) < 4:
        warnings.warn(
            f"the held-out set has only {len(held_out)} task(s) "
            f"({len(tasks)} tasks x held_out_frac={held_out_frac}), so every "
            "acceptance decision and the reported final_reward rest on that many "
            "rollouts. Pass more tasks or raise held_out_frac.",
            RuntimeWarning, stacklevel=3)

    # Wrap the actors before anything else can capture them: `run` is closed over
    # by the runtime, by every worker and by the verifier's `eval_fn`, so a later
    # wrap would miss whichever reference was taken first.
    meter = Meter(usage=usage) if usage is not None else Meter()
    run, propose = measured(run, meter), measured(propose, meter)

    cache = (policies_bundle.eval_cache if policies_bundle is not None
             and policies_bundle.eval_cache is not None else MemoryCache(meter))
    # A supplied cache was built before this run existed, so it has no meter. Ask
    # it to take one; a cache that does not offer the hook simply reports nothing,
    # which is better than a zero that looks like a measurement.
    attach = getattr(cache, "attach_meter", None)
    if callable(attach):
        attach(meter)
    runtime = _Runtime(run=run, reward=reward, cache=cache,
                       eval_concurrency=eval_concurrency, meter=meter,
                       eval_group=(policies_bundle.evaluator
                                   if policies_bundle is not None
                                   and policies_bundle.evaluator is not None
                                   else EvaluatorGroup(eval_concurrency, meter=meter)),
                       env_fingerprint=(policies_bundle.sandbox_spec.fingerprint()
                                        if policies_bundle is not None
                                        and policies_bundle.sandbox_spec is not None
                                        else ""))

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
    if ledger_impl is not None:
        # A caller-supplied ledger owns its own storage, so the scratch repo this
        # call may have just made is not its home -- drop the claim to it rather
        # than deleting a directory the ledger is not using.
        ledger, scratch = ledger_impl, None
    else:
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
    # Ranking is what the cheap layer is for; committing is not. Both gates that
    # decide a commit -- the Beta-posterior test and the regression guard beside it
    # -- read the FULL held-out set via `eval_counts`, so sub-sampling trades
    # tournament precision and nothing else. (The guard used to read the cheap
    # layer, which made that claim false; see `Aggregator._process`.) Noise stays at
    # zero: `eval_fn` is deterministic, so the sub-sample is the only approximation
    # and inventing more would just make the ranking worse.
    cheap = (len(held_out) if cheap_eval_tasks is None
             else max(1, min(int(cheap_eval_tasks), len(held_out))))
    verifier = verifier if verifier is not None else ThreeLayerVerifier(
        eval_fn=lambda a, ts: a.score(ts), held_out=held_out,
        rule_subset=cheap, learned_noise=0.0,
        budget=VerifierBudget(oracle_calls_remaining=oracle_budget))

    pol = policies_bundle or Policies()

    def _default_aggregator(ledger, verifier, audit, config, policy):
        return Aggregator(ledger, verifier, audit, config, staleness_policy=policy,
                          meter=meter, conflict=pol.conflict, fusion=pol.fusion,
                          acceptance=pol.acceptance, promotion=pol.promotion)

    aggregator = (aggregator_factory or _default_aggregator)(
        ledger, verifier, AuditScheduler(),
        agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        staleness_policy)
    # A custom factory cannot know about the meter -- its signature predates it
    # and is part of the public surface. Attach it to whatever came back if that
    # object has the slot and left it empty, so an `aggregator_factory` returning
    # a plain `Aggregator` (or a subclass of one) still reports staleness. A
    # factory returning something else simply reports nothing, which is the
    # honest outcome.
    if getattr(aggregator, "meter", None) is None:
        try:
            aggregator.meter = meter
        except AttributeError:      # slots, or a read-only property
            pass
    # A custom aggregator is the main extension point and is user code. Check the
    # contract here rather than letting it fail three frames deep in the driver
    # with something like "'MissingMethods' object has no attribute 'ingest'".
    for method in ("ingest", "step"):
        if not callable(getattr(aggregator, method, None)):
            raise TypeError(
                f"aggregator_factory returned {type(aggregator).__name__}, which has "
                f"no callable {method}(). An aggregator needs ingest(card) and "
                "step() -> list[MergeReport] (see AggregatorProtocol).")

    # Imported here rather than at module scope: `executor` reaches `workspec`,
    # which reaches back here for `Task`.
    from .executor import ThreadExecutor

    # In-process by default, taking the actors directly: a closure crosses no
    # boundary here, and resolving a `Ref` per rollout would rebuild a model
    # client every time.
    supplied = (policies_bundle.executor
                if policies_bundle is not None else None)
    executor = supplied if supplied is not None else ThreadExecutor(
        max(1, eval_concurrency), meter=meter, run=run, reward=reward)
    if supplied is not None:
        # A supplied executor was built before this run existed, so it has
        # neither the meter nor the actors. The meter is optional -- an executor
        # that cannot report simply reports nothing. The actors are not: without
        # them the executor falls back to resolving the spec, and `evolve()`
        # cannot describe a closure as a `Ref`, so every rollout would fail. That
        # was the behaviour, and it produced a finished run with `rollouts=0` and
        # a plausible reward from the gate -- a wrong answer wearing the shape of
        # a right one. Refuse instead, and say what would work.
        if not callable(getattr(supplied, "attach_actors", None)):
            raise TypeError(
                f"policies.executor is a {type(supplied).__name__}, which cannot "
                "be driven by evolve(): it has no attach_actors(run, reward), so "
                "it would have to resolve the rollout spec -- and evolve() is "
                "given `run` and `reward` as callables, which have no name to "
                "resolve. Pass an in-process executor (ThreadExecutor), or "
                "describe the rollouts yourself as RolloutSpecs and drive the "
                "executor directly (see docs/execution.md).")
        supplied.attach_actors(run, reward)
    attach = getattr(executor, "attach_meter", None)
    if callable(attach):
        attach(meter)
    return _Engine(ledger, runtime, verifier, aggregator, strategy, run, reward,
                   propose, train, held_out, {t.id: t for t in train},
                   [t.id for t in train], artifact_id, blast_radius,
                   executor=executor, meter=meter, scratch_repo=scratch)


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
    refresh_interval: int = 1,
    round_timeout: Optional[float] = None,
    target_reward: Optional[float] = None,
    patience: Optional[int] = None,
    max_worker_errors: int = 3,
    eval_concurrency: int = 8,
    asynchronous: bool = False,
    async_ratio: int = 3,
    max_seconds: Optional[float] = None,
    max_rollouts: Optional[int] = None,
    max_calls: Optional[int] = None,
    self_verify: bool = True,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    aggregator_factory: Optional[AggregatorFactory] = None,
    oracle_budget: int = 200,
    cheap_eval_tasks: Optional[int] = None,
    solved_threshold: float = SOLVED,
    shuffle: bool = False,
    seed: int = 0,
    on_round: Optional[Callable[["RoundInfo"], None]] = None,
    verbose: bool = False,
    #: Share one `Usage` with your model adapters (`claude(usage=u)`) and the
    #: result's token counts become real; without it only calls and seconds
    #: are known, because an opaque `run` cannot report tokens.
    usage: Optional[Usage] = None,
    policies: Optional["Policies"] = None,
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
        The work the artifact is evaluated on. Split into train / held-out **by
        position** -- the last ``held_out_frac`` of the sequence is held out, in
        the order given. At least 4 are required and ids must be unique.
    solved_threshold:
        A reward at or above this counts as solved, so no proposal is requested
        and the task sampler counts a pass. The default (:data:`SOLVED`, 0.999) is
        right for a binary scorer. **Lower it for a graded one** -- a ROUGE score
        or an LLM judge rarely reaches 0.999, so every rollout would ask the
        reflector to "fix" an answer that scored 0.95, and the run reports
        ``below-threshold`` as if the reflector were the problem.
    shuffle, seed:
        Shuffle ``tasks`` before that positional split. Off by default, which
        keeps a run reproducible and keeps
        :attr:`~agentdescent.dataloader.Dataset.val_frac`'s promise that the
        engine's held-out split is exactly that ``Dataset``'s ``val``. Turn it on
        for **grouped** data -- anything ordered by category, source, difficulty
        or date -- where the tail of the file is a different distribution from
        the head, and every gate in the run (the acceptance test,
        ``target_reward``, ``final_reward``) would then be measured against it.
    reward:
        ``(task, output) -> [0, 1]``. Scores in ``[0, 1]``; the engine treats
        ``>= solved_threshold`` as a pass (no proposal is requested).
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
    refresh_interval:
        How many rounds a worker keeps its ledger snapshot before taking the
        round's fresh one. ``1`` (default) is what this loop always did: every
        worker proposes against the current head, so a diff's staleness ``eta``
        is **0 by construction** -- and that made ``staleness_policy=`` a knob
        with nothing to decide on this path (measured over an 8-round run: all
        15 staleness decisions saw ``eta=0`` and returned ACCEPT, so Full,
        Guarded and Reflective were indistinguishable).

        Above ``1``, workers hold a spread of versions -- the refresh is
        staggered by worker id -- so their diffs arrive with a spread of ``eta``
        and the staleness policy, the ``alpha`` tolerances in ``agg_config`` and
        the ``all-stale`` outcome all become reachable synchronously. Costs no
        extra ledger read: a worker either adopts the snapshot the round already
        took, or keeps the older one it has. Ignored under ``asynchronous=True``,
        where the lag budget is ``async_ratio``.
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
    max_rollouts, max_calls:
        The budget in the two units a comparison has to hold fixed: rollouts
        completed, and actor invocations (``run`` + ``propose``). ``rounds`` is
        not one of them -- configurations differ in how much model a round buys,
        so a budget fixed in rounds hands the wider configuration more model and
        then reports the extra model as a win for parallelism. Either bound stops
        the run with ``stop_reason`` ``"max_rollouts"`` / ``"max_calls"``.

        **Checked at the round barrier, so a run overshoots by up to one round.**
        A round is dispatched or it is not; stopping halfway would leave a
        half-merged round, and the states a comparison compares are the ones a
        merge produced. So a budget is a *bound on where to stop*, never the
        number to compare on: read the spend the run actually reported
        (``result.rollouts``, ``result.usage.calls``), which is what
        :mod:`agentdescent.baselines` does -- it refuses to call two arms
        equal-budget when their measured spends differ.

        The async path has no barrier and enforces both per rollout, so it
        overshoots by at most the rollouts already in flight.
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
        commit gates always score the full set, so this never affects whether a
        change is safe to commit, only which candidate is put forward. The
        sample is fixed for the run, so candidates are always compared
        like-for-like. Both commit gates -- the acceptance test and the regression
        guard -- score the full set, so this never decides whether a change is
        committed, only which candidate is put forward.
    on_round:
        Called with each :class:`RoundInfo` as the round completes -- progress
        for a long run, which otherwise reports nothing until it returns. An
        exception raised here is reported but does not abort the run.
    usage:
        Share one :class:`~agentdescent.agents.Usage` with your model adapters
        (``claude(usage=u)``, ``openai_compatible(usage=u)``) and the result's
        token counts become real. Without it the run still reports calls,
        seconds and failures -- ``run`` is ``(rendered, task) -> str``, so an
        opaque actor has no way to surface tokens, and inventing a number would
        be worse than reporting zero.
    verbose:
        Print a line per round. Independent of the ``RuntimeWarning`` emitted
        when a run ends early -- that always fires.
    policies:
        Bundle of replaceable pieces (:class:`~agentdescent.policies.Policies`).
        Every field defaults to ``None`` meaning "current behaviour", so
        ``Policies()`` and passing nothing are the same run. The individual
        keyword arguments -- ``task_sampler``, ``staleness_policy``,
        ``aggregator_factory`` -- are shortcuts onto its fields and keep working;
        an explicit argument wins over a bundle default rather than being
        silently ignored. Fields whose implementations have not landed yet raise
        rather than being accepted and ignored: a caller who passes a custom
        acceptance rule and sees a finished run would reasonably conclude it ran.
        New capabilities go here rather than adding another parameter to a
        function that already has thirty-five.

    Returns
    -------
    EvolutionResult
        The evolved artifact plus ``history``, ``error`` and ``stop_reason``.
        **Check ``error``**: it is ``None`` only on a clean run, and a run that
        died still returns a (partial) result rather than raising. **Check
        ``stop_reason``** to tell convergence (``"target_reward"``) from a budget
        expiry (``"max_seconds"`` / ``"rounds"`` / ``"max_iters"``) -- ``error``
        is ``None`` for both.
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
            ("refresh_interval", None if refresh_interval == 1 else refresh_interval,
             "it staggers snapshot refresh across round barriers, and the async "
             "path has none -- async_ratio is its lag budget"),
        ):
            if value is not None:
                warnings.warn(
                    f"evolve(asynchronous=True) ignores {name}=: {why}. Use the "
                    f"synchronous path if you need it.", RuntimeWarning,
                    stacklevel=2)
        # The two above that are *not* ignored but silently REDEFINED were the
        # sharper edge, because nothing said so. Flipping one boolean turned an
        # unbounded run into a 20-second one, and a partial artifact with
        # `error=None` and a populated `history` is indistinguishable from a
        # converged one.
        if max_seconds is None:
            warnings.warn(
                "evolve(asynchronous=True) has no unbounded mode: max_seconds=None "
                "becomes 20.0 seconds here, where it means 'no limit' on the "
                "synchronous path. Pass max_seconds= explicitly, and check "
                "result.stop_reason -- a budget expiry otherwise looks exactly "
                "like convergence.", RuntimeWarning, stacklevel=2)
        if rounds != 15 and max_rollouts is None:   # i.e. the caller chose a value
            warnings.warn(
                f"evolve(asynchronous=True) has no round barrier, so rounds={rounds} "
                f"is reinterpreted as a budget of {rounds * max(1, n_workers)} worker "
                "rollouts, and RoundInfo.round becomes a merger-sweep index -- "
                "len(result.history) is not comparable with the synchronous path. "
                "Pass max_rollouts= to say the budget outright.",
                RuntimeWarning, stacklevel=2)
        from .async_evolve import async_evolve
        return async_evolve(
            tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
            initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
            n_workers=n_workers, async_ratio=async_ratio,
            max_seconds=20.0 if max_seconds is None else max_seconds,
            max_iters=(max_rollouts if max_rollouts is not None
                       else rounds * max(1, n_workers)),
            max_calls=max_calls, held_out_frac=held_out_frac,
            repo_path=repo_path, agg_config=agg_config, staleness_policy=staleness_policy,
            aggregator_factory=aggregator_factory, oracle_budget=oracle_budget,
            cheap_eval_tasks=cheap_eval_tasks, shuffle=shuffle, seed=seed,
            solved_threshold=solved_threshold,
            self_verify=self_verify, task_sampler=task_sampler,
            target_reward=target_reward, patience=patience,
            max_worker_errors=max_worker_errors,
            eval_concurrency=eval_concurrency,
            on_round=on_round, verbose=verbose, usage=usage, policies=policies)

    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    parallel = parallel or DataParallel()
    _pol = _resolve_policies(policies, "evolve()", task_sampler=task_sampler,
                             staleness=staleness_policy,
                             aggregator_factory=aggregator_factory)
    task_sampler, staleness_policy = _pol.task_sampler, _pol.staleness
    aggregator_factory = _pol.aggregator_factory
    sampler = task_sampler or RoundRobin()
    # Where the next batch starts. `SingleHead` is the current head for every
    # worker, i.e. exactly what this loop has always done.
    selection = _pol.selection or SingleHead()
    strategy = strategy or AppendRules()
    # TP owns a *section of the artifact*, so it needs the artifact's key space --
    # not the task ids `plan()` is handed. Resolve and validate it here, before any
    # rollout: an incompatible pairing used to be discovered one diff at a time, by
    # silently discarding it.
    section_map = _resolve_sections(parallel, strategy)
    _reject_pipeline_parallel(parallel)
    # Resolved once: the round body is the hot path, and a strategy either has
    # the hook for the whole run or does not.
    observe_plan = getattr(parallel, "observe", None)
    if not callable(observe_plan):
        observe_plan = None
    section_violations = [0]
    tp_lock = threading.Lock()
    eng = _build_engine(
        tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
        initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
        held_out_frac=held_out_frac, repo_path=repo_path, agg_config=agg_config,
        staleness_policy=staleness_policy, aggregator_factory=aggregator_factory,
        oracle_budget=oracle_budget, eval_concurrency=eval_concurrency,
        cheap_eval_tasks=cheap_eval_tasks, shuffle=shuffle, seed=seed,
        usage=usage, verifier=_pol.verifier, ledger_impl=_pol.ledger,
        policies_bundle=_pol)
    # Start the clock after the wiring, before the first unit of work: setup
    # is not what a time-to-quality number is asking about.
    eng.meter.start()
    ledger, aggregator, strategy = eng.ledger, eng.aggregator, eng.strategy
    run, propose, reward = eng.run, eng.propose, eng.reward
    held_out, by_id, train_ids = eng.held_out, eng.by_id, eng.train_ids

    history: List[RoundInfo] = []
    run_error: Optional[str] = None
    #: The most recent artifact successfully read from the ledger. If the final
    #: read fails there is still a real result to hand back instead of an exception.
    last_good: Optional[EvolvingArtifact] = None
    straggler_rounds = 0
    # Shared with the barrier-free loop: the same two questions, the same
    # tracker, and now the same epsilon (they had two).
    early = EarlyStop(target_reward=target_reward, patience=patience)
    unit_lock = threading.Lock()
    # Per-worker snapshots, when `refresh_interval > 1`. See `_snapshot_for`.
    worker_snaps: Dict[int, Tuple["EvolvingArtifact", int]] = {}
    snap_lock = threading.Lock()
    first_error: List[Optional[str]] = [None]
    contract_error = FirstError()          # caller bug -> re-raise on this thread
    # The retirement rule lives in `pipeline.WorkerHealth`, shared with the
    # barrier-free runtimes. It used to be re-implemented here from the same
    # description, which is the arrangement `pipeline.py`'s docstring records as
    # having already cost one hand-ported fix.
    health = WorkerHealth(max_errors=max_worker_errors)
    dead_rounds = 0            # consecutive rounds where every worker failed
    deadline = time.time() + max_seconds if max_seconds else None
    stop_reason = "rounds"
    for r in range(rounds):
        if deadline is not None and time.time() >= deadline:
            stop_reason = "max_seconds"
            if verbose:
                print(f"round {r:>3}  stopping: max_seconds={max_seconds} reached")
            break
        # The cost budgets, checked where the wall-clock one already is: at the
        # barrier, before a round is dispatched. Mid-round would mean a partial
        # merge, and the state a budget comparison compares is one a merge
        # produced -- so the run overshoots by up to a round and reports the
        # spend it actually incurred instead of the one it was asked for.
        spent = eng.meter.snapshot()
        over = ((max_rollouts is not None and spent.rollouts >= max_rollouts
                 and "max_rollouts") or
                (max_calls is not None and spent.calls >= max_calls
                 and "max_calls"))
        if over:
            stop_reason = over
            if verbose:
                print(f"round {r:>3}  stopping: {over} reached "
                      f"({spent.rollouts} rollouts / {spent.calls} calls)")
            break
        try:
            snap = ledger.snapshot(Ledger.DEV)
        except LedgerFailure as e:
            # The ledger is infrastructure, not a backend and not a caller bug, so
            # it fits neither existing category -- and letting it propagate broke
            # the one guarantee the result contract makes ("a run that died still
            # returns a partial result"). Treat it like an unmeasurable round: the
            # same tally decides whether to keep going or give up.
            #
            # Note the tally is shared but the *rule* is not: a repeatedly failing
            # ledger gives up after `max_worker_errors` rounds whether or not
            # workers have succeeded, while a repeatedly failing worker gives up
            # only while nothing ever has (`health.should_retire`). Defensible --
            # a ledger that cannot be read is not going to start working because a
            # rollout succeeded -- but it is a second rule, and this comment is
            # here because the sentence above claims there is one.
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
        _check_selection(selection, artifact_id, artifact, base_v, r, n_workers,
                         history)
        ok_units, failed_units = [0], [0]      # this round's tally

        def _snapshot_for(worker: int):
            """The artifact this worker proposes against, and the version it read.

            With ``refresh_interval=1`` -- the default, and what this loop always
            did -- every worker takes the round's fresh snapshot, so a diff is
            always proposed against the current head and ``eta`` is **0 by
            construction**. That made ``evolve(staleness_policy=)`` a knob that
            could not decide anything: measured over an 8-round run, every one of
            the 15 staleness decisions saw ``eta=0`` and returned ACCEPT, so Full,
            Guarded and Reflective were indistinguishable.

            Above 1, a worker keeps its snapshot for that many rounds and the
            refresh is **staggered** by worker id, so at any moment the workers
            hold a spread of versions and their diffs arrive with a spread of
            ``eta``. That is the same mechanism
            :class:`~agentdescent.orchestrator.AgentDescent` uses to make the
            staleness sweep meaningful, expressed against the round's existing
            snapshot rather than a fresh one -- so it costs no extra ledger read;
            a worker either adopts the snapshot this round already took, or keeps
            the older one it has.
            """
            if refresh_interval <= 1:
                return artifact, base_v
            with snap_lock:
                due = (r % refresh_interval) == (stable_hash(worker) % refresh_interval)
                if worker not in worker_snaps or due:
                    worker_snaps[worker] = (artifact, base_v)
                return worker_snaps[worker]

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
                contract_error.record(e)          # its own lock; first one wins
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
            # This worker's own view of the artifact. Identical to the round's
            # under the default `refresh_interval=1`; older, by design, above it.
            mine, mine_v = _snapshot_for(unit.worker)
            task = by_id[sampler.pick(unit.keys, r)]     # a task from this worker's shard
            # The rollout is the part that can move elsewhere. Everything around
            # it -- which task, what the output implies, who is told about it --
            # reads or writes state that has to stay in this process.
            outcome = eng.executor.rollout(_spec_for(eng, mine, task))
            if not outcome.ok:
                raise _rollout_failure(outcome)
            output = outcome.output
            score = _checked_reward(outcome.reward, task)
            sampler.record(task.id, score)               # learn which tasks carry signal
            if observe_plan is not None:
                # ...and let the parallel strategy learn too, if it wants to.
                # `plan` alone is a pure function of its arguments, which is
                # enough to shard and not enough to schedule: UCB over task
                # clusters had nowhere to receive an outcome, so it lived only in
                # the reference runtime. Optional, so DP and TP are untouched.
                observe_plan(unit, task.id, score)
            with unit_lock:
                ok_units[0] += 1
                health.record_success()
            if score >= solved_threshold:
                return
            proposal = _checked_proposal(
                propose(mine.render(), task, output, score), task)
            if not proposal:
                return
            diff = strategy.to_diff(mine.state, proposal, f"w{unit.worker}", mine_v, artifact_id)
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
                    reward(task, run(mine.apply(diff).render(), task)), task)
                delta = after - score
            else:
                delta = 0.0
            aggregator.ingest(EvidenceCard(
                diff=diff, base_version={artifact_id: mine_v}, touched=[artifact_id],
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
            contract_error.raise_if_set()

            # Decide on the round's tally rather than on the first exception. The
            # same global signal the async path uses: while NO worker has ever
            # completed a rollout, repeated total failure means the backend is
            # misconfigured, so give up quickly and loudly. Once any worker has
            # succeeded the backend demonstrably works, so failures are transient
            # and the run keeps going on the evidence it did gather.
            if failed_units[0] and not ok_units[0]:
                dead_rounds += 1
                if health.should_retire(dead_rounds):
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
            # Same rule as the worker path: scoring held-out runs the agent, so an
            # unmeasurable round is a backend failure like any other.
            if health.should_retire(dead_rounds):
                run_error = first_error[0]
                if verbose:
                    print(f"round {r:>3}  giving up: held-out unmeasurable "
                          f"({run_error})")
                break
            if verbose:
                print(f"round {r:>3}  held-out unmeasurable, carrying last reward: "
                      f"{type(e).__name__}: {str(e)[:100]}")
            continue
        with tp_lock:
            extra = ({"section-violation": section_violations[0]}
                     if section_violations[0] else None)
            section_violations[0] = 0
        # Recording the round, deciding whether to stop, and telling the caller
        # are the same three steps in both loops; only "what is a round" differs.
        info, early_stop = eng.record_round(
            index=r, reward=round_reward, n_items=len(dev.state), reports=reports,
            history=history, early=early, on_round=on_round, extra_reasons=extra)
        if verbose:
            print(f"round {r:>3}  reward={info.held_out_reward:.3f} on "
                  f"{len(held_out)}  size={info.n_items}  "
                  f"+{info.committed}/-{info.rejected}")
        if early_stop is not None:
            stop_reason = early_stop
            if verbose:
                print(f"round {r:>3}  "
                      + (f"target_reward={target_reward} reached, stopping"
                         if early_stop == "target_reward" else
                         f"no improvement for {early.stalled} rounds, stopping"))
            break

    if run_error is None:
        # A clean run publishes the head it produced (see `_publish_stable`);
        # a run that died leaves `stable` where it was, which is the point of
        # having a confirmed branch at all.
        _publish_stable(aggregator)
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
                             ledger_log=_safe_log(ledger), error=run_error,
                             stop_reason="error" if run_error else stop_reason,
                             fusion_trials=_fusion_trials(aggregator),
                             **_cost_fields(eng.meter))
    eng.cleanup()
    return result
