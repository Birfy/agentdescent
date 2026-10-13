"""Which task should a worker roll out next? (design doc, section 5.2 -- L-task)

A rollout is the expensive unit of work: one LLM call, or a whole tool-using agent
trajectory that can run for minutes. Spending it on a task the agent *already*
solves teaches the system nothing -- there is no failure to learn from, so no diff
is proposed and the rollout is wasted.

The reference :class:`~agentdescent.scheduler.TaskScheduler` solves this for the
stage-orchestration runtime with UCB over task *clusters* plus a difficulty
filter. This module brings the same idea to the :func:`~agentdescent.evolution.evolve`
engine at *task* granularity, as a plug-in:

* :class:`RoundRobin` -- cycle through the shard (the default; fully deterministic).
* :class:`DifficultyWeighted` -- prefer tasks that carry a learning signal, i.e.
  whose pass rate sits away from the all-pass / all-fail extremes, with a UCB
  exploration bonus so rarely-tried tasks still get sampled.

Both satisfy the :class:`TaskSampler` protocol structurally, so you can pass your
own::

    evolve(tasks, reward, agent=agent, task_sampler=DifficultyWeighted())
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict

from .evolvable import EvidenceCard
from .stats import difficulty_weight
from typing import Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class TaskSampler(Protocol):
    """Chooses the next task id for a worker, and learns from the outcome."""

    def pick(self, keys: Sequence[str], round_index: int) -> str:
        """Return one task id from ``keys`` (never mutate ``keys``)."""
        ...

    def record(self, task_id: str, score: float) -> None:
        """Report the reward a rollout of ``task_id`` achieved (0..1)."""
        ...


class RoundRobin:
    """Cycle through the shard in order -- the deterministic default.

    Reproducible and dependency-free, but it spends rollouts uniformly, including
    on tasks the agent already solves.
    """

    name = "round-robin"

    def pick(self, keys: Sequence[str], round_index: int) -> str:
        if not keys:
            raise ValueError("task_sampler.pick() got an empty key list")
        return keys[round_index % len(keys)]

    def record(self, task_id: str, score: float) -> None:  # nothing to learn
        return None


class DifficultyWeighted:
    """UCB over tasks, weighted by how much learning signal each one carries.

    The weight ``4*p*(1-p)`` (``p`` = observed pass rate) peaks at ``p = 0.5`` and
    vanishes at the extremes: a task that always passes -- or never passes no
    matter the artifact -- yields no usable gradient (the GRPO zero-advantage
    argument), so it is down-weighted in favour of tasks that sometimes fail.
    An untried task keeps the optimistic prior ``p = 0.5`` and a large UCB bonus,
    so exploration still happens.

    ``pass_threshold`` mirrors the engine: a rollout counts as a pass when its
    reward reaches it.

    ``c`` is the UCB exploration constant, and the default is deliberately much
    smaller than the textbook ``1.4``. Most of the exploration here comes from the
    **optimistic prior**: an untried task is assumed to sit at ``p = 0.5``, which
    is exactly where the signal weight is *maximal*, so it is already picked
    eagerly. A large ``c`` on top of that swamps the signal term (which is capped
    at 1.0) and the sampler never stops re-trying tasks it has already shown to be
    uninformative. Measured on a 40-task workload where only 6 tasks carry signal
    (share of rollouts that landed on an informative task, higher is better)::

        c            1.4     0.7     0.4     0.2     0.1     round-robin
        clean       15.5%   19.3%   21.0%   23.4%   27.4%      14.5%
        15% noise      --      --   11.5%   16.3%   18.1%       7.3%

    ``0.2`` is the default: ~1.6-2.2x better than round-robin in both regimes
    while keeping twice the exploration of the empirical optimum, which matters on
    workloads whose reward is noisier than the ones measured here. Lower it to
    focus harder, raise it to explore more.
    """

    name = "difficulty-weighted"

    def __init__(self, c: float = 0.2, pass_threshold: Optional[float] = None,
                 prior: float = 0.5) -> None:
        self.c = c
        # Defaults to the engine's own SOLVED rather than repeating the
        # literal: the docstring says this 'mirrors the engine', which is
        # exactly the coupling a shared constant should carry.
        from .evolution import SOLVED
        self.pass_threshold = SOLVED if pass_threshold is None else pass_threshold
        self.prior = prior
        # task_id -> (passes, trials); untried tasks fall back to the prior.
        self._stats: Dict[str, Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
        self._t = 0
        self._lock = threading.Lock()      # worker threads share one sampler

    def _pass_rate(self, task_id: str) -> float:
        passes, trials = self._stats[task_id]
        if trials <= 0:
            return self.prior
        return passes / trials

    @staticmethod
    def _signal(pass_rate: float) -> float:
        """Learning signal: peaks at pass_rate 0.5, ~0 when always/never solved.

        Shared with :class:`~agentdescent.scheduler.TaskScheduler`, which applies
        the same weight one granularity up (clusters rather than tasks)."""
        return difficulty_weight(pass_rate, floor=1e-3)

    def pick(self, keys: Sequence[str], round_index: int) -> str:
        if not keys:
            raise ValueError("task_sampler.pick() got an empty key list")
        with self._lock:
            self._t += 1
            total = float(self._t)
            best, best_score = keys[0], float("-inf")
            for k in keys:
                _, trials = self._stats[k]
                value = self._signal(self._pass_rate(k))
                # UCB: unexplored tasks (trials == 0) win outright.
                bonus = (self.c * math.sqrt(math.log(total + 1.0) / trials)
                         if trials > 0 else float("inf"))
                score = value + bonus
                if score > best_score:
                    best, best_score = k, score
            return best

    def record(self, task_id: str, score: float) -> None:
        with self._lock:
            passes, trials = self._stats[task_id]
            self._stats[task_id] = (passes + (1.0 if score >= self.pass_threshold else 0.0),
                                    trials + 1.0)

    def stats(self) -> Dict[str, Tuple[float, float]]:
        """Copy of the per-task (passes, trials) counters -- for inspection/tests."""
        with self._lock:
            return dict(self._stats)


class ReplaySampler:
    """A :class:`TaskSampler` that also learns from discarded evidence.

    The engine's aggregator settles the diff it discards -- stale, oversized, or
    lost to a CAS race -- into a bounded pool (``aggregator.settle``). Nobody reads
    it back. This sampler subscribes to that pool and up-weights tasks whose recent
    proposals were thrown away, on the argument that the waste is a signal *of its
    own*: a task that always passes and a task whose proposals keep going stale are
    both "not producing accepted diffs", but the first should be down-weighted
    (nothing to learn) and the second should be up-weighted (the parallel scheduler
    keeps out-competing it, not because it is easy).

    The base score is :class:`DifficultyWeighted`'s -- the pass-rate signal plus a
    UCB exploration bonus. The replay bonus is additive and saturating: a task with
    ``capped_at`` recent settled cards gets the full ``temperature``, so a task
    cannot dominate the score by being out-competed many times.

    ``temperature=0`` (the default) reproduces the base sampler exactly; without a
    wired aggregator the counts never increase, so the bonus stays zero. Wire one:

    .. code-block:: python

        sampler = ReplaySampler(temperature=0.3)
        evolve(tasks, reward, agent=agent, task_sampler=sampler,
               ...)  # the engine calls aggregator.set_settled_consumer(sampler.settle)
    """

    name = "replay"

    def __init__(self, temperature: float = 0.0, capped_at: int = 10) -> None:
        self._base = DifficultyWeighted()
        self.temperature = temperature
        self.capped_at = capped_at
        #: task_id -> number of settled evidence cards the pool handed to this
        #: sampler, counted since construction.
        self._settled_count: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        #: Picks made, and picks where the replay bonus changed the winner --
        #: the observable that tells a run the mechanism fired at all.
        self.picks = 0
        self.replay_affected = 0

    def settle(self, card: EvidenceCard) -> None:
        """Consume one settled evidence card (called by the aggregator's pool).

        ``trajectory_refs`` carries the task objects the diff was rolled out
        against; a card may name several, and every one of them was implicated in
        the discard.
        """
        for ref in card.trajectory_refs:
            task_id = getattr(ref, "id", None)
            if task_id is None:
                continue
            with self._lock:
                self._settled_count[task_id] = self._settled_count[task_id] + 1

    def pick(self, keys: Sequence[str], round_index: int) -> str:
        if not keys:
            raise ValueError("task_sampler.pick() got an empty key list")
        with self._lock:
            self._base._t += 1
            total = float(self._base._t)

            def score_of(k: str, with_replay: bool) -> float:
                passes, trials = self._base._stats[k]
                signal = self._base._signal(
                    passes / trials if trials else self._base.prior)
                bonus = (self._base.c * math.sqrt(
                    math.log(total + 1.0) / trials) if trials > 0 else float("inf"))
                out = signal + bonus
                if with_replay:
                    out += self.temperature * min(
                        self.capped_at, self._settled_count.get(k, 0))
                return out

            base_winner = max(keys, key=lambda k: score_of(k, with_replay=False))
            winner = max(keys, key=lambda k: score_of(k, with_replay=True))
            # replay changed the outcome when the winner differs from the base winner
            if self.temperature > 0 and self._settled_count.get(winner, 0) > 0 \
                    and winner != base_winner:
                self.replay_affected += 1
            self.picks += 1
            return winner

    def record(self, task_id: str, score: float) -> None:
        self._base.record(task_id, score)

    def stats(self) -> Dict[str, Tuple[float, float]]:
        """Copy of the base sampler's per-task (passes, trials) counters."""
        return self._base.stats()

    def settled_counts(self) -> Dict[str, int]:
        """Copy of the per-task settled-card counts -- for inspection/tests."""
        with self._lock:
            return dict(self._settled_count)
