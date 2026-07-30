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
import re
import shutil
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .agents import Completion, claude
from .aggregator import Aggregator, AggregatorConfig, AggregatorFactory
from .evolvable import Contract, Diff, EvidenceCard
from .governance import assert_mutable
from .ledger import Ledger
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
    "It produced:\n{output}\n\n"
    "Propose exactly ONE concise, general rule (a single imperative sentence) to "
    "improve the artifact for this and similar cases. Output only the rule text, "
    "or NONE if no rule would help."
)


@dataclass
class LLMAgent:
    """Adapt a ``Completion`` (from :mod:`agentdescent.agents`) into an :class:`Agent`."""

    complete: Completion
    solve_template: str = _SOLVE_TMPL
    propose_template: str = _PROPOSE_TMPL

    def solve(self, rendered: str, task: Task) -> str:
        return self.complete(
            self.solve_template.format(artifact=rendered, prompt=task.prompt)).strip()

    def propose(self, rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        rule = self.complete(self.propose_template.format(
            artifact=rendered, prompt=task.prompt, output=output, reward=reward)).strip()
        return None if (not rule or rule.upper().startswith("NONE")) else rule


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
class KeyedRules:
    """One entry per *category*: competing proposals contradict and are resolved.

    Proposals look like ``"category: text"``. A new proposal for an existing
    category **overwrites** it, so two workers proposing different text for the
    same category produce a contradiction the aggregator resolves (keeping the
    one that scores better). Unknown categories fall back to append behaviour."""

    categories: Sequence[str]
    title: str = "# Config (by category)"

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
        if not tasks or self._rt is None:
            return 0.0
        return sum(self._rt.eval_one(self, t) for t in tasks) / len(tasks)

    def cheap_eval(self, evidence: EvidenceCard) -> float:
        return self.score([t for t in evidence.trajectory_refs if isinstance(t, Task)])

    def full_eval(self, task_set: Sequence[Task]) -> Dict[str, float]:
        return {"reward": self.score(task_set)}


@dataclass
class _Runtime:
    run: Run
    reward: Reward
    cache: _EvalCache

    def eval_one(self, artifact: EvolvingArtifact, task: Task) -> float:
        key = (artifact._signature(), task.id)
        return self.cache.get_or_eval(
            key, lambda: self.reward(task, self.run(artifact.render(), task)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class RoundInfo:
    round: int
    held_out_reward: float
    n_items: int
    committed: int
    rejected: int


@dataclass
class EvolutionResult:
    state: Dict[str, str]
    rendered: str
    final_reward: float
    history: List[RoundInfo]
    ledger_log: List[str]
    #: ``None`` on a clean run; otherwise ``"<ExcType>: <message>"`` describing the
    #: backend failure that ended the run early. The artifact evolved so far is
    #: still returned -- check this to tell "converged" from "died".
    error: Optional[str] = None

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
                 "n_items": h.n_items, "committed": h.committed, "rejected": h.rejected}
                for h in self.history
            ],
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
                  staleness_policy, aggregator_factory, oracle_budget) -> _Engine:
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
    # round, not truncate: Dataset.val_frac promises "the engine's held-out split
    # is exactly this Dataset's val", and float truncation (13.9999 -> 13) quietly
    # pushed one train item into held-out for many dataset sizes.
    cut = max(1, round(len(tasks) * (1 - held_out_frac)))
    train, held_out = tasks[:cut], tasks[cut:]
    if not held_out:
        train, held_out = tasks[:-1], tasks[-1:]

    runtime = _Runtime(run=run, reward=reward, cache=_EvalCache())

    def serialize(a: EvolvingArtifact) -> dict:
        return {"state": a.state, "blast_radius": a.blast_radius}

    def deserialize(aid: str, version: int, state: dict) -> EvolvingArtifact:
        return EvolvingArtifact(aid, state.get("state", {}), version,
                                state.get("blast_radius", blast_radius), runtime, strategy)

    if repo_path:
        repo = repo_path              # caller-owned (and how a run is resumed): keep it
    else:
        # A scratch ledger per run would otherwise pile up in $TMPDIR forever --
        # one git repo per evolve() call, never reclaimed.
        repo = tempfile.mkdtemp(prefix="agentdescent-evolve-")
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

    # The eval_fn here IS ground truth (deterministic, memoized), so the cheap
    # layers must NOT add noise or sub-sample (see the note in the original).
    verifier = ThreeLayerVerifier(
        eval_fn=lambda a, ts: a.score(ts), held_out=held_out,
        rule_subset=len(held_out), learned_noise=0.0,
        budget=VerifierBudget(oracle_calls_remaining=oracle_budget))

    def _default_aggregator(ledger, verifier, audit, config, policy):
        return Aggregator(ledger, verifier, audit, config, staleness_policy=policy)

    aggregator = (aggregator_factory or _default_aggregator)(
        ledger, verifier, AuditScheduler(),
        agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        staleness_policy)

    return _Engine(ledger, runtime, verifier, aggregator, strategy, run, reward,
                   propose, train, held_out, {t.id: t for t in train},
                   [t.id for t in train], artifact_id, blast_radius)


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
    ``n_workers``. Swap in ``TensorParallel`` / ``PipelineParallel`` or your own
    :class:`~agentdescent.parallel.ParallelStrategy`.

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
        How a round's tasks are partitioned across workers (DP / TP / PP).
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
        Where the git-backed ledger lives. Omit for a scratch repo that is
        cleaned up at exit; **passing the same path again resumes** that ledger.
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
        Hard cap on full held-out oracle evaluations during L1 audits. Once
        spent, the verifier falls back to its cheap layer.
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
        from .async_evolve import async_evolve
        return async_evolve(
            tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
            initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
            n_workers=n_workers, async_ratio=async_ratio,
            max_seconds=20.0 if max_seconds is None else max_seconds,
            max_iters=rounds * max(1, n_workers), held_out_frac=held_out_frac,
            repo_path=repo_path, agg_config=agg_config, staleness_policy=staleness_policy,
            aggregator_factory=aggregator_factory, oracle_budget=oracle_budget,
            self_verify=self_verify, task_sampler=task_sampler,
            on_round=on_round, verbose=verbose)

    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    parallel = parallel or DataParallel()
    sampler = task_sampler or RoundRobin()
    eng = _build_engine(
        tasks, reward, agent=agent, run=run, propose=propose, strategy=strategy,
        initial_state=initial_state, blast_radius=blast_radius, artifact_id=artifact_id,
        held_out_frac=held_out_frac, repo_path=repo_path, agg_config=agg_config,
        staleness_policy=staleness_policy, aggregator_factory=aggregator_factory,
        oracle_budget=oracle_budget)
    ledger, aggregator, strategy = eng.ledger, eng.aggregator, eng.strategy
    run, propose, reward = eng.run, eng.propose, eng.reward
    held_out, by_id, train_ids = eng.held_out, eng.by_id, eng.train_ids

    history: List[RoundInfo] = []
    run_error: Optional[str] = None
    straggler_rounds = 0
    deadline = time.time() + max_seconds if max_seconds else None
    for r in range(rounds):
        if deadline is not None and time.time() >= deadline:
            if verbose:
                print(f"round {r:>3}  stopping: max_seconds={max_seconds} reached")
            break
        snap = ledger.snapshot(Ledger.DEV)
        artifact = snap.get(artifact_id)
        base_v = snap.version.get(artifact_id, 0)
        assert_mutable(artifact)

        def _run_unit(unit) -> None:
            """One worker: rollout -> propose -> ingest evidence (against `snap`)."""
            if not unit.keys:
                return
            task = by_id[sampler.pick(unit.keys, r)]     # a task from this worker's shard
            output = run(artifact.render(), task)
            score = reward(task, output)
            sampler.record(task.id, score)               # learn which tasks carry signal
            if score >= 0.999:
                return
            proposal = propose(artifact.render(), task, output, score)
            if not proposal:
                return
            diff = strategy.to_diff(artifact.state, proposal, f"w{unit.worker}", base_v, artifact_id)
            if diff is None:
                return
            # The self-verify rollout doubles the cost of every proposal, so it is
            # opt-out here exactly as it is on the async path.
            if self_verify:
                after = reward(task, run(artifact.apply(diff).render(), task))
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
                pool = ThreadPoolExecutor(max_workers=min(max_concurrency, len(units)))
                try:
                    futures = [pool.submit(_run_unit, u) for u in units]
                    done, pending = futures_wait(futures, timeout=round_timeout)
                    for f in done:
                        f.result()          # re-raise a genuine backend failure
                    if pending:
                        straggler_rounds += 1
                        if verbose:
                            print(f"round {r:>3}  abandoned {len(pending)} straggler(s) "
                                  f"after round_timeout={round_timeout}s")
                finally:
                    # never block shutdown on an abandoned straggler
                    pool.shutdown(wait=False)
            else:
                for unit in units:
                    _run_unit(unit)

            reports = aggregator.step()
        except Exception as e:  # noqa: BLE001 - a rollout backend failure (e.g. an
            # API/credit error) shouldn't lose the run: stop and return partial results.
            run_error = f"{type(e).__name__}: {str(e)[:200]}"
            if verbose:
                print(f"round {r:>3}  stopped early: {run_error[:140]}")
            break
        committed = sum(1 for x in reports if x.committed_version is not None)
        rejected = sum(1 for x in reports if x.committed_version is None)
        dev = ledger.snapshot(Ledger.DEV).get(artifact_id)
        info = RoundInfo(r, dev.score(held_out), len(dev.state), committed, rejected)
        history.append(info)
        if verbose:
            print(f"round {r:>3}  reward={info.held_out_reward:.3f}  "
                  f"items={info.n_items}  +{committed}/-{rejected}")
        if on_round is not None:
            # A reporting callback must never take the run down with it.
            try:
                on_round(info)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"on_round callback raised: {type(e).__name__}: {e}",
                              RuntimeWarning, stacklevel=2)

    final = ledger.snapshot(Ledger.DEV).get(artifact_id)
    # Scoring runs the agent, so a dead backend must not raise out of the driver
    # and discard everything already committed.
    try:
        final_reward = final.score(held_out)
    except Exception as e:  # noqa: BLE001 - report, keep the partial result
        run_error = run_error or f"{type(e).__name__}: {str(e)[:200]}"
        final_reward = history[-1].held_out_reward if history else 0.0
    if run_error:
        # Never end a run silently: verbose=False is the default, so a partial
        # result is otherwise indistinguishable from a converged one.
        warnings.warn(f"evolve() stopped early after {len(history)} round(s): "
                      f"{run_error}", RuntimeWarning, stacklevel=2)
    return EvolutionResult(state=dict(final.state), rendered=final.render(),
                           final_reward=final_reward, history=history,
                           ledger_log=ledger.log(Ledger.DEV, limit=40), error=run_error)
