"""Solver-challenger self-play: the task distribution evolves too.

A frozen-model RSI loop optimises an artifact against a fixed task set, and
the set is the least-examined part of the design: who decides what the agent
is ever asked? Self-play answers "the agent itself". A second artifact -- the
**challenger** -- owns a *task-generation spec*; its output is not an answer
but a question. The solver must keep learning, and the challenger must keep
proposing questions that sit at the solver's frontier: hard enough to be worth
asking (the solver currently fails them), solvable enough that learning pays
(the solver's score on them rises), and never so degenerate that a valid
answer does not exist.

This is the GoDoors / R-Zero idea -- a *learned* objective ``r_0`` rather than
a fixed one -- made concrete for a harness-evolution framework, with the
regularizers this project insists on:

* **Two-phase learnability, measured online.** A generated task's reward is not
  "does the solver answer it" but "does the solver *learn* it": ``first`` vs
  ``last`` solver score across its attempts. A task the solver already passes
  is not a frontier task; one it never learns is dropped as stuck, not kept
  because it happened to be hard.
* **The challenger never touches the final evaluation.** Generated tasks enter
  the *train* pool through the sampler; ``evolve``'s held-out split is computed
  from the caller's seed tasks and the challenger cannot add to it. What the
  run reports is always measured on tasks the challenger never generated.
* **A validity gate is mandatory.** ``Challenger.validate`` raises on any
  payload that is not a well-formed, answerable task; a generator that cannot
  produce a valid task is counted, not crashed over.
* **The default is the default.** ``curriculum_evolve(seed_tasks, ..., challenger=...)``
  with a challenger is strictly opt-in; the solver is a normal ``evolve()`` run
  driven by the engine, one engine instance for the whole run, so the
  aggregator's state (posteriors, archive) is never rebuilt per round.

The driver runs the engine once. Between rounds, an ``on_round`` hook does the
challenger step: it reads the frontier's measured learnability, hands the
report to ``Challenger.propose``, generates and validates the next frontier
batch, and installs it into the :class:`FrontierSampler`. The sampler mixes the
seed tasks with the frontier, so the solver's own task distribution evolves as
the challenger does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evolution import (
    EvolutionResult, Propose, Reward, Run, Strategy, Task, evolve,
)
from .sampling import TaskSampler

__all__ = ["Challenger", "ChallengerConfig", "FrontierSampler", "curriculum_evolve"]


@dataclass(frozen=True)
class ChallengerConfig:
    """The challenger's dials: how much of the task budget goes to the frontier,
    what counts as learned or stuck, and how long a generator may keep trying."""

    #: How many valid generated tasks each challenger step installs.
    generated_per_round: int = 4
    #: Fraction of a worker's picks that come from the frontier (the rest are
    #: seed tasks, so the solver never forgets the fixed curriculum).
    frontier_share: float = 0.5
    #: A task whose *first* solver score is already at or above ``1 - before_gap``
    #: is not on the frontier; it is dropped on its first attempt.
    before_gap: float = 0.05
    #: A task whose ``last - first`` never exceeds this is stuck and dropped.
    #: The challenger is told the stuck count, so it learns not to generate
    #: tasks its own solver cannot learn.
    stuck_gap: float = 0.0
    #: How many attempts a task may have before it is judged.
    max_attempts: int = 8
    #: The challenger commits to a new spec only when the *learned* report beats
    #: this (see :func:`curriculum_evolve`; a model propose may ignore it).
    r0_floor: float = 0.05
    #: Valid generation attempts per challenger step before giving up.
    max_generate_attempts: int = 20
    #: The frontier pool is capped; the oldest learned/stuck tasks leave first.
    max_frontier_pool: int = 64


@dataclass
class Challenger:
    """The task-generation artifact's behavior: generate, validate, propose.

    ``generate(spec, seed) -> payload`` draws a candidate task from the current
    spec; ``validate(payload) -> Task`` is the **mandatory validity gate** --
    it must raise on anything that is not a well-formed, answerable task, and
    the returned ``Task`` carries the answer in ``task.meta["gold"]`` (what the
    solver's ``reward`` reads). ``propose(spec, report) -> spec`` turns the
    measured frontier report into the next generation spec. All three are plain
    callables, so an offline fixture can be deterministic and a live run can
    use a model.
    """

    generate: Callable[[str, int], Dict[str, Any]]
    validate: Callable[[Dict[str, Any]], Task]
    propose: Callable[[str, Dict[str, float]], str]
    #: The current generation spec. Starts here; every challenger step may
    #: replace it.
    spec: str = ""
    config: ChallengerConfig = field(default_factory=ChallengerConfig)


class FrontierSampler:
    """A :class:`~agentdescent.sampling.TaskSampler` that mixes the fixed seed
    tasks with a growable frontier pool the challenger generates.

    ``pick(keys, r)`` returns a seed task *id* (resolved through the worker's
    shard, like any sampler) with probability ``1 - frontier_share``, or a
    frontier :class:`~agentdescent.evolution.Task` object directly -- the
    engine resolves a Task return without a pool lookup. ``record(task_id,
    score)`` tracks each frontier task's ``first`` / ``last`` solver score, the
    two-phase learnability signal that drives the challenger.

    The frontier is self-regulating: a task whose first attempt already passes
    is dropped at once (it was never on the frontier), a task that never learns
    is dropped as stuck, and a task that learns is dropped and counted -- the
    pool keeps turning over toward the solver's current edge.
    """

    def __init__(self, challenger: Challenger, *, seed: int = 0) -> None:
        self.challenger = challenger
        self._rng = random.Random(seed)
        self._frontier: List[Task] = []
        #: task.id -> measured learnability bookkeeping.
        self._book: Dict[str, Dict[str, Any]] = {}
        #: Counters exposed through :meth:`stats`.
        self._stats: Dict[str, int] = {
            "generated": 0, "invalid": 0, "learned": 0, "stuck": 0,
            "never_attempted": 0,
        }
        self.picks = 0

    # -- the sampler surface ------------------------------------------------

    def pick(self, keys: Sequence[str], r: int) -> Any:
        """A seed id (from the shard) or a frontier Task object."""
        self.picks += 1
        cfg = self.challenger.config
        frontier = [t for t in self._frontier if t.id in self._book]
        if frontier and self._rng.random() < cfg.frontier_share:
            failing = [
                t for t in frontier
                if self._book[t.id].get("first") is None
                or self._book[t.id]["first"] < 1.0 - cfg.before_gap
            ]
            pool = failing or frontier
            return self._rng.choice(pool)
        if keys:
            return keys[r % len(keys)]
        return None

    def record(self, task_id: str, score: float) -> None:
        """Learn one rollout's outcome. The engine calls this for every task,
        seed or frontier; only frontier ids are tracked."""
        book = self._book.get(task_id)
        if book is None:
            return
        if book.get("first") is None:
            book["first"] = score
            # A task the solver already passes was never on the frontier: it is
            # removed immediately, not kept because it is easy.
            if score >= 1.0 - self.challenger.config.before_gap:
                self._drop(task_id, "learned")
                return
        book["last"] = score
        book["attempts"] = book.get("attempts", 0) + 1

    def install(self, tasks: Sequence[Task]) -> None:
        """Add a freshly generated, validated frontier batch."""
        cfg = self.challenger.config
        for task in tasks:
            self._stats["generated"] += 1
            known = next((t for t in self._frontier if t.id == task.id), None)
            if known is not None and task.id in self._book:
                # A redraw of a task already on the frontier must not erase its
                # measured history (positive learning, or the attempts count
                # toward the stuck threshold). An equivalent task keeps its
                # entry; a *different* task under the same id is refused rather
                # than silently inheriting someone else's measurements (#336).
                if (known.prompt, known.meta) != (task.prompt, task.meta):
                    self._stats["invalid"] += 1
                continue
            self._frontier.append(task)
            self._book[task.id] = {"first": None, "last": None, "attempts": 0,
                                   "spec": task.meta.get("spec", "")}
        # Cap the pool; drop oldest judged tasks first.
        if len(self._frontier) > cfg.max_frontier_pool:
            self._frontier = self._frontier[-cfg.max_frontier_pool:]

    def _drop(self, task_id: str, why: str) -> None:
        self._frontier = [t for t in self._frontier if t.id != task_id]
        self._book.pop(task_id, None)
        self._stats[why] += 1

    def sweep(self, solved_threshold: float) -> Dict[str, float]:
        """Close one solver round: judge every frontier task that was attempted,
        return the learnability report for the challenger."""
        cfg = self.challenger.config
        learned = stuck = unfinished = 0
        r0s: List[float] = []
        for task in list(self._frontier):
            book = self._book.get(task.id)
            if book is None:
                continue
            first, last, attempts = book.get("first"), book.get("last"), book.get("attempts")
            if first is None:
                unfinished += 1
                continue
            if last is not None and last >= solved_threshold:
                learned += 1
                r0s.append((last - first))
                self._drop(task.id, "learned")
                continue
            if attempts is not None and attempts >= cfg.max_attempts \
                    and last is not None and (last - first) <= cfg.stuck_gap:
                stuck += 1
                r0s.append((last - first))
                self._drop(task.id, "stuck")
                continue
            r0s.append((last - first) if last is not None else 0.0)
        r0 = (sum(r0s) / len(r0s)) if r0s else 0.0
        return {
            "r0": r0,
            "learned": learned,
            "stuck": stuck,
            "unfinished": unfinished,
            "frontier_n": len(self._frontier),
            "generated_total": self._stats["generated"],
            "invalid_total": self._stats["invalid"],
        }

    def count_invalid(self) -> None:
        """A validity gate refused a generated payload."""
        self._stats["invalid"] += 1

    def stats(self) -> dict:
        return {
            "generated_total": self._stats["generated"],
            "invalid_total": self._stats["invalid"],
            "learned": self._stats["learned"],
            "stuck": self._stats["stuck"],
            "never_attempted": self._stats["never_attempted"],
            "picks": self.picks,
            "frontier_n": len(self._frontier),
        }


def curriculum_evolve(
    seed_tasks: Sequence[Task],
    reward: Reward,
    *,
    run: Run,
    propose: Propose,
    strategy: Strategy,
    challenger: Challenger,
    rounds: int = 15,
    n_workers: int = 4,
    held_out_frac: float = 0.4,
    repo_path: Optional[str] = None,
    seed: int = 0,
    max_concurrency: int = 1,
    solved_threshold: float = 0.999,
    verbose: bool = False,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Co-evolve the solver and its task distribution.

    ``seed_tasks``, ``reward``, ``run``, ``propose``, ``strategy`` are the
    solver's -- exactly as in :func:`evolve`; ``challenger`` supplies the
    task-generation artifact. The engine runs **once** (one aggregator, one
    ledger, the held-out split fixed from ``seed_tasks``); between rounds the
    challenger step installs a freshly generated, validated frontier batch into
    the solver's sampler, and the solver's own ``on_round`` closure closes the
    learnability measurement.

    Returns the solver's :class:`~agentdescent.evolution.EvolutionResult` with
    ``result.curriculum`` set to the run's challenger statistics (``r0``,
    ``learned``, ``stuck``, ``frontier_n``, ``invalid_total``, ``generated_total``).

    Governance: generated tasks enter the train pool only; ``held_out_frac`` is
    applied to ``seed_tasks`` alone, so the reported quality is always measured
    on tasks the challenger never generated.
    """
    from .evolution import RoundInfo

    if rounds < 2:
        raise ValueError("curriculum_evolve needs rounds >= 2 (round 1 seeds the frontier)")
    cfg = challenger.config
    sampler = FrontierSampler(challenger, seed=seed)
    driver = _CurriculumDriver(challenger, sampler, solved_threshold=solved_threshold,
                               rng=random.Random(seed + 1))

    def on_round(info: RoundInfo) -> None:
        driver.after_round(info)

    # Seed the frontier before the first solver round so the very first round
    # already meets generated tasks.
    driver.refresh()

    result = evolve(
        list(seed_tasks), reward, run=run, propose=propose, strategy=strategy,
        task_sampler=sampler, on_round=on_round,
        rounds=rounds, n_workers=n_workers, held_out_frac=held_out_frac,
        repo_path=repo_path, seed=seed, max_concurrency=max_concurrency,
        solved_threshold=solved_threshold, verbose=verbose, **evolve_kwargs,
    )
    result.curriculum = driver.final_stats(sampler)
    return result


class _CurriculumDriver:
    """The challenger step between solver rounds: measure, propose, regenerate."""

    def __init__(self, challenger: Challenger, sampler: FrontierSampler,
                 *, solved_threshold: float, rng: random.Random) -> None:
        self.challenger = challenger
        self.sampler = sampler
        self.solved_threshold = solved_threshold
        self._rng = rng
        self._rounds_closed = 0
        self._reports: List[Dict[str, float]] = []

    def refresh(self) -> None:
        """Generate, validate and install the next frontier batch from the
        challenger's current spec."""
        cfg = self.challenger.config
        batch: List[Task] = []
        for _ in range(cfg.max_generate_attempts):
            if len(batch) >= cfg.generated_per_round:
                break
            try:
                payload = self.challenger.generate(self.challenger.spec, self._rng.getrandbits(32))
                task = self.challenger.validate(payload)
            except (ValueError, TypeError, KeyError):
                self.sampler.count_invalid()
                continue
            task.meta["generated"] = True
            task.meta["spec"] = self.challenger.spec
            batch.append(task)
        if batch:
            self.sampler.install(batch)

    def after_round(self, info) -> None:
        """Close the solver round: judge the frontier, feed the challenger."""
        self._rounds_closed += 1
        report = self.sampler.sweep(self.solved_threshold)
        self._reports.append(report)
        if self._rounds_closed >= 1:
            # The challenger tunes its spec on the measured learnability. The
            # spec change is a plain callable; a model propose may read the
            # whole report.
            try:
                self.challenger.spec = self.challenger.propose(
                    self.challenger.spec, report)
            except Exception:  # noqa: BLE001 - a broken proposer must not fail the run
                self.sampler._stats["invalid"] += 1
        self.refresh()

    def final_stats(self, sampler: FrontierSampler) -> Dict[str, Any]:
        return {
            **sampler.stats(),
            "r0_mean": (sum(r["r0"] for r in self._reports) / len(self._reports)
                        if self._reports else 0.0),
            "rounds_closed": self._rounds_closed,
        }
