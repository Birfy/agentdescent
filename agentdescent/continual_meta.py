"""Continually update the meta-policy from logged decisions, with a safety rail.

A meta-policy decides which slot to evolve, which model to call, or whether to
canary. It changes as the archive and the traffic change. A one-shot fit is
wrong for that: the loop needs **online updates with replay, validation and
rollback**.

This vertical slice is a full-feedback contextual bandit:

* :class:`MetaExample` carries context features and the reward of each action;
* :class:`ReplayBuffer` retains a bounded history and samples batches;
* :class:`ContinualMetaLearner` fits a softmax policy by cross-entropy to the
  best action, evaluates the candidate on a validation slice, and only promotes
  it if the expected reward improves by ``min_gain``; otherwise it rolls back;
* a simple feature-mean drift signal warns when the validation context no longer
  looks like the replay context.

    learner = ContinualMetaLearner(["selection", "task_sampler"], dim=4)
    learner.observe(example)
    update = learner.step(validation)
    update.accepted, update.before, update.after

It is deliberately not a deep RL implementation; it is the update/validate/
rollback contract that a PPO/DPO adapter keeps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MetaExample",
    "ReplayBuffer",
    "LinearPolicy",
    "MetaUpdate",
    "ContinualReport",
    "ContinualMetaLearner",
]


def _vec(values: Sequence[float], name: str) -> Tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class MetaExample:
    features: Tuple[float, ...]
    rewards: Mapping[str, float]
    cost: float = 0.0
    at: float = 0.0

    def __post_init__(self) -> None:
        features = _vec(self.features, "features")
        if not self.rewards:
            raise ValueError("rewards must not be empty")
        rewards = {}
        for action, value in self.rewards.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("reward must be finite")
            rewards[str(action)] = number
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and >= 0")
        if not math.isfinite(float(self.at)):
            raise ValueError("at must be finite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "cost", cost)

    @property
    def best_action(self) -> str:
        return max(self.rewards, key=lambda key: (self.rewards[key], key))


class ReplayBuffer:
    """A bounded, deterministic replay buffer."""

    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._rows: List[MetaExample] = []

    def add(self, example: MetaExample) -> None:
        if not isinstance(example, MetaExample):
            raise TypeError("add() expects MetaExample")
        self._rows.append(example)
        if len(self._rows) > self.max_size:
            del self._rows[: len(self._rows) - self.max_size]

    def extend(self, examples: Iterable[MetaExample]) -> None:
        for example in examples:
            self.add(example)

    def sample(self, n: int, *, seed: int = 0) -> Tuple[MetaExample, ...]:
        if n < 1:
            raise ValueError("n must be >= 1")
        if not self._rows:
            raise ValueError("replay buffer is empty")
        rng = random.Random(seed)
        if n >= len(self._rows):
            return tuple(self._rows)
        return tuple(rng.choice(self._rows) for _ in range(n))

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> Tuple[MetaExample, ...]:
        return tuple(self._rows)


@dataclass
class LinearPolicy:
    actions: Tuple[str, ...]
    dim: int
    weights: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if not self.weights:
            self.weights = [[0.0] * self.dim for _ in self.actions]
        if len(self.weights) != len(self.actions) or any(len(row) != self.dim
                                                         for row in self.weights):
            raise ValueError("weights shape does not match actions/dim")

    def probabilities(self, features: Sequence[float]) -> Tuple[float, ...]:
        x = _vec(features, "features")
        if len(x) != self.dim:
            raise ValueError("feature dimension mismatch")
        logits = [sum(a * b for a, b in zip(row, x)) for row in self.weights]
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        total = sum(exps)
        return tuple(value / total for value in exps)

    def act(self, features: Sequence[float], *, epsilon: float = 0.0,
            seed: int = 0) -> str:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        rng = random.Random(seed)
        if epsilon and rng.random() < epsilon:
            return rng.choice(self.actions)
        probs = self.probabilities(features)
        return self.actions[max(range(len(probs)), key=lambda i: (probs[i], -i))]

    def expected_reward(self, examples: Iterable[MetaExample]) -> float:
        rows = list(examples)
        if not rows:
            return 0.0
        total = 0.0
        for example in rows:
            probs = self.probabilities(example.features)
            total += sum(p * example.rewards.get(action, 0.0)
                         for p, action in zip(probs, self.actions))
        return total / len(rows)

    def copy(self) -> "LinearPolicy":
        return LinearPolicy(self.actions, self.dim, [row[:] for row in self.weights])


@dataclass(frozen=True)
class MetaUpdate:
    version: int
    accepted: bool
    before: float
    after: float
    drift: float
    reason: str


@dataclass(frozen=True)
class ContinualReport:
    version: int
    updates: int
    accepted: int
    rejected: int
    replay: int
    detail: str = ""


class ContinualMetaLearner:
    """Update a softmax meta-policy with replay, validation and rollback."""

    def __init__(self, actions: Sequence[str], *,
                 dim: int, learning_rate: float = 0.2, l2: float = 1e-3,
                 replay_size: int = 1000, min_gain: float = 0.0,
                 seed: int = 0) -> None:
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if not math.isfinite(float(l2)) or l2 < 0:
            raise ValueError("l2 must be finite and >= 0")
        if not math.isfinite(float(min_gain)):
            raise ValueError("min_gain must be finite")
        self.actions = tuple(str(action) for action in actions)
        self.policy = LinearPolicy(self.actions, dim)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.min_gain = float(min_gain)
        self.replay = ReplayBuffer(replay_size)
        self.seed = seed
        self.version = 1
        self._updates: List[MetaUpdate] = []
        self._previous_policy: Optional[LinearPolicy] = None

    def observe(self, example: MetaExample) -> None:
        self.replay.add(example)

    def _candidate(self, batch: Sequence[MetaExample]) -> LinearPolicy:
        candidate = self.policy.copy()
        if not batch:
            return candidate
        for example in batch:
            probs = candidate.probabilities(example.features)
            best = example.best_action
            for i, action in enumerate(candidate.actions):
                target = 1.0 if action == best else 0.0
                error = probs[i] - target
                for j in range(candidate.dim):
                    candidate.weights[i][j] -= self.learning_rate * (
                        error * example.features[j] + self.l2 * candidate.weights[i][j])
        return candidate

    def _drift(self, validation: Sequence[MetaExample]) -> float:
        rows = self.replay.rows
        if not rows or not validation:
            return 0.0
        dim = self.policy.dim
        replay_mean = [sum(row.features[j] for row in rows) / len(rows)
                       for j in range(dim)]
        validation_mean = [sum(row.features[j] for row in validation) / len(validation)
                           for j in range(dim)]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(replay_mean, validation_mean)))

    def step(self, validation: Sequence[MetaExample], *,
             batch_size: int = 16, seed: Optional[int] = None) -> MetaUpdate:
        if not validation:
            raise ValueError("validation must not be empty")
        batch = self.replay.sample(batch_size, seed=self.seed if seed is None else seed)
        before = self.policy.expected_reward(validation)
        candidate = self._candidate(batch)
        after = candidate.expected_reward(validation)
        drift = self._drift(validation)
        previous = self.policy.copy()
        if after > before + self.min_gain:
            self._previous_policy = previous
            self.policy = candidate
            self.version += 1
            update = MetaUpdate(self.version, True, before, after, drift, "improved")
        else:
            self._previous_policy = None
            update = MetaUpdate(self.version, False, before, after, drift, "rejected")
        self._updates.append(update)
        return update

    def rollback(self) -> bool:
        """Undo the immediately preceding accepted update.

        Used by a forgetting guard: a policy that improved its validation slice
        but regressed a frozen anchor slice must be reverted. A rejected update
        leaves no previous policy, so ``rollback`` returns ``False``.
        """
        if self._previous_policy is None:
            return False
        if not self._updates or not self._updates[-1].accepted:
            return False
        previous = self._updates[-1]
        self.policy = self._previous_policy
        self._previous_policy = None
        self.version -= 1
        # Keep the attempt in the audit trail as a rejected/rolled-back update
        # instead of pretending it never happened.
        self._updates[-1] = MetaUpdate(
            self.version, False, previous.before, previous.after,
            previous.drift, "rolled_back")
        return True

    def act(self, features: Sequence[float], *, epsilon: float = 0.0) -> str:
        return self.policy.act(features, epsilon=epsilon, seed=self.seed)

    def report(self) -> ContinualReport:
        accepted = sum(1 for update in self._updates if update.accepted)
        return ContinualReport(
            version=self.version, updates=len(self._updates),
            accepted=accepted, rejected=len(self._updates) - accepted,
            replay=len(self.replay),
            detail=f"version={self.version}, accepted={accepted}, "
                   f"replay={len(self.replay)}")
