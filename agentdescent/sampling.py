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
