"""Step-level process supervision and calibrated step reranking.

Outcome rewards tell the loop whether the final answer was right; they do not
tell it **which step** deserves credit. A process reward model (PRM) scores each
reasoning/action step, which makes credit assignment, best-of-N and tree search
substantially better than a single end-of-trajectory score.

This module is the reward-model half:

* :class:`StepExample` is one labelled step (features + 0/1 label);
* :class:`ProcessRewardModel` is a logistic step scorer fitted with IRLS;
* :class:`ProcessReport` carries accuracy/log-loss/ECE/Brier and weights;
* ``rank`` / ``select`` rerank candidate steps by the calibrated score.

It can be plugged into `verifier_ensemble`, `credit`, and a planner's value
heuristic. It does not generate trajectories.

    model = ProcessRewardModel()
    report = model.fit(examples)
    model.score(step_features)
    model.select(candidate_features)

``fit`` returns a :class:`ProcessReport`; the fitted model stays on ``model``.
Reports below ``min_n`` steps are marked ``insufficient`` instead of
``calibrated``/``usable`` so downstream gates cannot treat a tiny fit as
trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = [
    "StepExample",
    "ProcessReport",
    "ProcessRewardModel",
]


def _vec(values: Sequence[float], name: str) -> Tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain finite values")
    return out


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class StepExample:
    features: Tuple[float, ...]
    label: float
    trajectory_id: str = ""
    step_index: int = 0
    metadata: Any = None

    def __post_init__(self) -> None:
        features = _vec(self.features, "features")
        label = float(self.label)
        if not math.isfinite(label) or not 0.0 <= label <= 1.0:
            raise ValueError("label must be in [0, 1]")
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True)
class ProcessReport:
    n: int
    dim: int
    accuracy: float
    log_loss: float
    ece: float
    brier: float
    weights: Tuple[float, ...]
    verdict: str
    detail: str = ""


class ProcessRewardModel:
    """A logistic step scorer with an intercept and deterministic IRLS."""

    def __init__(self, dim: Optional[int] = None) -> None:
        if dim is not None and dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.weights: Tuple[float, ...] = tuple(0.0 for _ in range((dim or 0) + 1))
        self.report: Optional[ProcessReport] = None

    def _require_fitted(self) -> None:
        if self.report is None:
            raise RuntimeError("ProcessRewardModel requires fit() first")

    def fit(self, examples: Iterable[StepExample], *, steps: int = 100,
            l2: float = 1e-3, tol: float = 1e-10,
            min_n: int = 5) -> ProcessReport:
        rows = list(examples)
        if not rows:
            raise ValueError("fit needs at least one step")
        if steps < 1 or l2 < 0 or not math.isfinite(l2):
            raise ValueError("steps >= 1 and finite l2 >= 0 are required")
        if min_n < 1:
            raise ValueError("min_n must be >= 1")
        dim = len(rows[0].features)
        if any(len(row.features) != dim for row in rows):
            raise ValueError("all steps must have the same dimension")
        x_rows = [(1.0,) + row.features for row in rows]
        y = [row.label for row in rows]
        w = [0.0] * (dim + 1)
        for _ in range(steps):
            grad = [0.0] * (dim + 1)
            h = [[0.0 for _ in range(dim + 1)] for _ in range(dim + 1)]
            for x, target in zip(x_rows, y):
                p = _sigmoid(sum(a * b for a, b in zip(w, x)))
                error = p - target
                weight = max(1e-12, p * (1 - p))
                for i in range(dim + 1):
                    grad[i] += error * x[i]
                    for j in range(dim + 1):
                        h[i][j] += weight * x[i] * x[j]
            n = len(rows)
            for i in range(dim + 1):
                grad[i] = grad[i] / n + l2 * w[i]
                for j in range(dim + 1):
                    h[i][j] = h[i][j] / n + (l2 if i == j else 0.0)
            if sum(abs(value) for value in grad) < tol:
                break
            matrix = [row[:] + [grad[i]] for i, row in enumerate(h)]
            for col in range(dim + 1):
                pivot = max(range(col, dim + 1), key=lambda r: abs(matrix[r][col]))
                if abs(matrix[pivot][col]) < 1e-12:
                    break
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
                for row in range(col + 1, dim + 1):
                    factor = matrix[row][col] / matrix[col][col]
                    for k in range(col, dim + 2):
                        matrix[row][k] -= factor * matrix[col][k]
            try:
                step = [0.0] * (dim + 1)
                for row in range(dim, -1, -1):
                    rhs = matrix[row][dim + 1] - sum(
                        matrix[row][k] * step[k] for k in range(row + 1, dim + 1))
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
        self.dim = dim
        self.weights = tuple(w)
        preds = [self._sigmoid_with(w, (1.0,) + row.features) for row in rows]
        correct = sum(1 for p, row in zip(preds, rows)
                      if (p >= 0.5) == (row.label >= 0.5))
        accuracy = correct / len(rows)
        losses = []
        for p, row in zip(preds, rows):
            p = min(1 - 1e-12, max(1e-12, p))
            losses.append(-(row.label * math.log(p) +
                            (1 - row.label) * math.log(1 - p)))
        log_loss = sum(losses) / len(losses)
        brier = sum((p - row.label) ** 2 for p, row in zip(preds, rows)) / len(rows)
        ece = _ece(preds, [row.label for row in rows])
        if len(rows) < min_n:
            verdict = "insufficient"
        elif accuracy >= 0.8 and ece <= 0.1:
            verdict = "calibrated"
        elif accuracy >= 0.7:
            verdict = "usable"
        else:
            verdict = "weak"
        self.report = ProcessReport(
            n=len(rows), dim=dim, accuracy=accuracy, log_loss=log_loss,
            ece=ece, brier=brier, weights=self.weights, verdict=verdict,
            detail=f"accuracy={accuracy:.3f}, ECE={ece:.3f}, "
                   f"Brier={brier:.3f}")
        return self.report

    @staticmethod
    def _sigmoid_with(weights: Sequence[float], features: Sequence[float]) -> float:
        return _sigmoid(sum(a * b for a, b in zip(weights, features)))

    def score(self, features: Sequence[float]) -> float:
        self._require_fitted()
        vector = _vec(features, "features")
        if len(vector) != self.dim:
            raise ValueError("feature dimension does not match the fit")
        return self._sigmoid_with(self.weights, (1.0,) + vector)

    def rank(self, candidates: Sequence[Sequence[float]]) -> Tuple[int, ...]:
        self._require_fitted()
        scored = [(self.score(candidate), index)
                  for index, candidate in enumerate(candidates)]
        return tuple(index for _, index in sorted(scored, key=lambda item: (-item[0], item[1])))

    def select(self, candidates: Sequence[Sequence[float]]) -> int:
        ranking = self.rank(candidates)
        if not ranking:
            raise ValueError("select needs at least one candidate")
        return ranking[0]


def _ece(preds: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    buckets = [[] for _ in range(bins)]
    for p, y in zip(preds, labels):
        buckets[min(bins - 1, int(p * bins))].append((p, y))
    total = len(preds)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)
        ece += len(bucket) / total * abs(mean_p - mean_y)
    return ece
