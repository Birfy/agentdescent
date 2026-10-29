"""Learn a reward model from pairwise human preferences.

Reward shaping by hand encodes the author's taste. Pairwise feedback -- "A is
better than B" -- is cheaper to collect and is the standard interface for RLHF
and DPO. This module is the reward-model half of that loop:

* a Bradley-Terry logistic model over feature vectors;
* ties are target 0.5, not silently dropped;
* deterministic Newton/IRLS with step halving;
* accuracy/log-loss on the training pairs;
* a preference buffer with an uncertainty-based next-query selector.

It does not implement DPO/PPO; it produces the scalar reward those algorithms
consume, and can be plugged into `rewards.scorer` or a verifier.

    model = PreferenceModel().fit(pairs)
    model.score(features)
    model.probability(left, right)
    next_item = buffer.select_next(candidates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PreferencePair",
    "PreferenceReport",
    "PreferenceModel",
    "PreferenceBuffer",
]


def _vec(values: Sequence[float], name: str) -> Tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty")
    for value in out:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite values")
    return out


@dataclass(frozen=True)
class PreferencePair:
    left: Tuple[float, ...]
    right: Tuple[float, ...]
    preferred: str = "left"
    metadata: Any = None

    def __post_init__(self) -> None:
        left = _vec(self.left, "left")
        right = _vec(self.right, "right")
        if len(left) != len(right):
            raise ValueError("left and right must have the same dimension")
        if self.preferred not in ("left", "right", "tie"):
            raise ValueError("preferred must be left, right or tie")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True)
class PreferenceReport:
    n: int
    dim: int
    accuracy: float
    log_loss: float
    weights: Tuple[float, ...]
    verdict: str
    detail: str = ""


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


class PreferenceModel:
    """A Bradley-Terry reward model fitted with deterministic IRLS."""

    def __init__(self, dim: Optional[int] = None) -> None:
        if dim is not None and dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.weights: Tuple[float, ...] = tuple(0.0 for _ in range(dim or 0))
        self.report: Optional[PreferenceReport] = None

    def _require_fitted(self) -> None:
        if self.report is None:
            raise RuntimeError("PreferenceModel requires fit() first")

    def fit(self, pairs: Iterable[PreferencePair], *, steps: int = 100,
            l2: float = 1e-3, tol: float = 1e-10) -> PreferenceReport:
        pairs = list(pairs)
        if not pairs:
            raise ValueError("fit needs at least one pair")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if l2 < 0 or not math.isfinite(l2):
            raise ValueError("l2 must be finite and >= 0")
        dim = len(pairs[0].left)
        if any(len(pair.left) != dim for pair in pairs):
            raise ValueError("all pairs must have the same dimension")
        diffs = []
        for pair in pairs:
            if pair.preferred == "left":
                diffs.append(([a - b for a, b in zip(pair.left, pair.right)], 1.0))
            elif pair.preferred == "right":
                diffs.append(([b - a for a, b in zip(pair.left, pair.right)], 1.0))
            else:
                diffs.append(([a - b for a, b in zip(pair.left, pair.right)], 0.5))
        w = [0.0] * dim
        for _ in range(steps):
            grad = [l2 * value for value in w]
            h = [[l2 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
            for x, y in diffs:
                z = sum(a * b for a, b in zip(w, x))
                p = _sigmoid(z)
                error = p - y
                weight = max(1e-12, p * (1.0 - p))
                for i in range(dim):
                    grad[i] += error * x[i]
                    for j in range(dim):
                        h[i][j] += weight * x[i] * x[j]
            for i in range(dim):
                grad[i] /= len(diffs)
                for j in range(dim):
                    h[i][j] /= len(diffs)
            if sum(abs(value) for value in grad) < tol:
                break
            # Gaussian elimination for H * step = grad.
            matrix = [row[:] + [grad[i]] for i, row in enumerate(h)]
            for col in range(dim):
                pivot = max(range(col, dim), key=lambda row: abs(matrix[row][col]))
                if abs(matrix[pivot][col]) < 1e-12:
                    break
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
                for row in range(col + 1, dim):
                    factor = matrix[row][col] / matrix[col][col]
                    for k in range(col, dim + 1):
                        matrix[row][k] -= factor * matrix[col][k]
            try:
                step = [0.0] * dim
                for row in range(dim - 1, -1, -1):
                    rhs = matrix[row][dim] - sum(matrix[row][k] * step[k]
                                                  for k in range(row + 1, dim))
                    step[row] = rhs / matrix[row][row]
            except ZeroDivisionError:
                break
            scale = 1.0
            for _ in range(30):
                candidate = [a - scale * b for a, b in zip(w, step)]
                if all(math.isfinite(value) for value in candidate):
                    break
                scale *= 0.5
            else:
                break
            w = candidate
        self.weights = tuple(w)
        self.dim = dim
        predictions = []
        losses = []
        correct = total = 0
        for pair in pairs:
            diff = [a - b for a, b in zip(pair.left, pair.right)]
            p = _sigmoid(sum(a * b for a, b in zip(w, diff)))
            predictions.append(p)
            if pair.preferred == "left":
                target = 1.0
            elif pair.preferred == "right":
                target = 0.0
            else:
                target = 0.5
            p_clipped = min(1.0 - 1e-12, max(1e-12, p))
            losses.append(-(target * math.log(p_clipped) +
                            (1 - target) * math.log(1 - p_clipped)))
            if pair.preferred == "tie":
                continue
            total += 1
            if (pair.preferred == "left" and p >= 0.5) or \
                    (pair.preferred == "right" and p < 0.5):
                correct += 1
        accuracy = correct / total if total else 0.0
        log_loss = sum(losses) / len(losses)
        if accuracy >= 0.8:
            verdict = "learned"
        elif accuracy >= 0.6:
            verdict = "weak"
        else:
            verdict = "unlearned"
        self.report = PreferenceReport(
            n=len(pairs), dim=dim, accuracy=accuracy, log_loss=log_loss,
            weights=self.weights, verdict=verdict,
            detail=f"accuracy={accuracy:.3f}, log-loss={log_loss:.4f}")
        return self.report

    def score(self, features: Sequence[float]) -> float:
        self._require_fitted()
        vector = _vec(features, "features")
        if len(vector) != self.dim:
            raise ValueError("feature dimension does not match the fit")
        return sum(a * b for a, b in zip(self.weights, vector))

    def probability(self, left: Sequence[float], right: Sequence[float]) -> float:
        self._require_fitted()
        a = _vec(left, "left")
        b = _vec(right, "right")
        if len(a) != self.dim or len(b) != self.dim:
            raise ValueError("feature dimension does not match the fit")
        return _sigmoid(sum(x * (u - v) for x, u, v in zip(self.weights, a, b)))


class PreferenceBuffer:
    """Collect pairs and pick the most informative next candidate."""

    def __init__(self) -> None:
        self.pairs: List[PreferencePair] = []

    def add(self, pair: PreferencePair) -> None:
        if not isinstance(pair, PreferencePair):
            raise TypeError("add() expects PreferencePair")
        self.pairs.append(pair)

    def extend(self, pairs: Iterable[PreferencePair]) -> None:
        for pair in pairs:
            self.add(pair)

    def select_next(self, candidates: Sequence[Sequence[float]]) -> int:
        """Return the index of the candidate farthest from the labelled set."""
        if not candidates:
            raise ValueError("select_next needs at least one candidate")
        seen = [value for pair in self.pairs for value in (pair.left, pair.right)]
        best_index = 0
        best_distance = -1.0
        for index, candidate in enumerate(candidates):
            vector = _vec(candidate, "candidate")
            if not seen:
                return index
            if any(len(item) != len(vector) for item in seen):
                raise ValueError("candidate dimension does not match the buffer")
            distance = min(math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, item)))
                           for item in seen)
            if distance > best_distance:
                best_index, best_distance = index, distance
        return best_index

    def to_pairs(self) -> Tuple[PreferencePair, ...]:
        return tuple(self.pairs)
