"""Recent-variance task sampling: prefer tasks whose outcome is still moving.

``DifficultyWeighted`` weights a task by its lifetime pass rate, which is the
right signal for a stationary solver and the wrong one when the artifact is
changing: a task that flipped pass/fail in the last few attempts is exactly the
one the next diff can teach, even if its long-run rate looks mediocre. This
sampler keeps a short window per task and picks the highest **recent variance**
plus a UCB exploration bonus.

    evolve(tasks, reward, agent=agent, task_sampler=VarianceSampler(window=5))

It satisfies the same :class:`~agentdescent.sampling.TaskSampler` protocol as the
shipped samplers, so it is a drop-in argument.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Sequence

__all__ = ["VarianceSampler"]


class VarianceSampler:
    """UCB over the recent reward variance of each task.

    ``window`` is how many recent scores define "recent"; ``prior`` is the
    variance assumed for an untried task (0.25 is the maximum for a 0/1 reward);
    ``c`` is the exploration constant. Untried tasks get the prior plus the
    largest bonus, so every task is seen at least once. Ties rotate by
    ``round_index`` so a two-task tie does not starve the second one.
    """

    name = "variance"

    def __init__(self, *, window: int = 5, prior: float = 0.25,
                 c: float = 0.5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        if c < 0:
            raise ValueError("c must be >= 0")
        self.window = int(window)
        self.prior = float(prior)
        self.c = float(c)
        self._scores: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window))
        self._tries: Dict[str, int] = defaultdict(int)
        self._total = 0
        # Workers share one sampler instance; see docs/sampling.md.
        self._lock = threading.Lock()

    def _weight(self, task_id: str) -> float:
        scores = list(self._scores[task_id])
        tries = self._tries[task_id]
        if not scores:
            variance = self.prior
        elif len(scores) == 1:
            variance = self.prior
        else:
            mean = sum(scores) / len(scores)
            variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        bonus = self.c * math.sqrt(math.log(self._total + 2) / (tries + 1))
        return variance + bonus

    def pick(self, keys: Sequence[str], round_index: int) -> str:
        if not keys:
            raise ValueError("task_sampler.pick() got an empty key list")
        with self._lock:
            scores = [self._weight(key) for key in keys]
            best = max(scores)
            tied = [key for key, score in zip(keys, scores)
                    if abs(score - best) <= 1e-12]
            return tied[round_index % len(tied)]

    def record(self, task_id: str, score: float) -> None:
        value = min(1.0, max(0.0, float(score)))
        with self._lock:
            self._scores[task_id].append(value)
            self._tries[task_id] += 1
            self._total += 1
