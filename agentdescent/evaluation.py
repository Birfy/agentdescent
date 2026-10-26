"""A budgeted evaluation pass: what the verifier says vs what the oracle paid for.

An acceptance decision often has two signals: a cheap verifier on every task and
an expensive oracle on a sample. This module makes that one explicit operation:
deterministic sampling under a hard oracle-call budget, per-task rows, coverage
and the signed/absolute bias. It is the measurement half of the event model; a
policy or report decides what to do with it.

    report = evaluate(artifact, tasks, verifier=score, oracle=truth,
                      sample_rate=0.2, budget=50)
    report.bias, report.coverage, report.rows
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

__all__ = ["EvaluationReport", "EvaluationRow", "evaluate"]

Scorer = Callable[[Any, Any], float]


def _task_id(task: Any, index: int) -> str:
    if isinstance(task, Mapping):
        value = task.get("id")
        return str(value) if value is not None else str(index)
    value = getattr(task, "id", None)
    return str(value) if value is not None else str(index)


@dataclass(frozen=True)
class EvaluationRow:
    """One audited task's verifier and oracle scores."""

    task_id: str
    verifier: float
    oracle: float

    @property
    def gap(self) -> float:
        return self.verifier - self.oracle


@dataclass(frozen=True)
class EvaluationReport:
    """The verifier/oracle comparison for one artifact."""

    tasks: int
    audited: int
    verifier_mean: float
    #: ``None`` when nothing was audited: no coverage is not "zero bias".
    oracle_mean: Optional[float]
    bias: Optional[float]
    abs_bias: Optional[float]
    audited_verifier_mean: Optional[float]
    coverage: float
    oracle_calls: int
    budget: int
    rows: Tuple[EvaluationRow, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": self.tasks, "audited": self.audited,
            "verifier_mean": self.verifier_mean,
            "oracle_mean": self.oracle_mean,
            "bias": self.bias, "abs_bias": self.abs_bias,
            "audited_verifier_mean": self.audited_verifier_mean,
            "coverage": self.coverage, "oracle_calls": self.oracle_calls,
            "budget": self.budget,
            "rows": [{"task_id": row.task_id, "verifier": row.verifier,
                      "oracle": row.oracle, "gap": row.gap}
                     for row in self.rows],
        }


def evaluate(artifact: Any, tasks: Sequence[Any], *,
             verifier: Scorer, oracle: Scorer,
             sample_rate: float = 0.2, budget: int = 100,
             seed: int = 0) -> EvaluationReport:
    """Compare ``verifier`` with ``oracle`` on a deterministic sample.

    ``verifier`` runs on every task; ``oracle`` runs only on the sample, and the
    sample size is ``min(budget, ceil(len(tasks) * sample_rate))`` (at least one
    when both are positive). ``budget=0`` audits nothing: ``oracle_mean``,
    ``bias`` and ``abs_bias`` are ``None`` and coverage is 0. That is "no truth
    evidence", not zero bias. The verifier runs on every task, but ``bias`` is
    measured on the audited subset only, so ``verifier_mean - oracle_mean`` is
    not generally the bias; ``audited_verifier_mean`` is the matching mean.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be in [0, 1]")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative int")
    task_list = list(tasks)
    verifier_scores = []
    for task in task_list:
        value = float(verifier(artifact, task))
        if not math.isfinite(value):
            raise ValueError(f"verifier returned a non-finite score for task "
                             f"{_task_id(task, len(verifier_scores))!r}")
        verifier_scores.append(value)
    if not task_list:
        return EvaluationReport(
            tasks=0, audited=0, verifier_mean=0.0, oracle_mean=None,
            bias=None, abs_bias=None, audited_verifier_mean=None,
            coverage=0.0, oracle_calls=0, budget=budget)
    wanted = max(1, math.ceil(len(task_list) * sample_rate)) if sample_rate else 0
    n_audited = min(len(task_list), int(budget), wanted)
    chosen = random.Random(seed).sample(range(len(task_list)), n_audited) \
        if n_audited else []
    rows = []
    for index in sorted(chosen):
        task = task_list[index]
        task_id = _task_id(task, index)
        oracle_score = float(oracle(artifact, task))
        if not math.isfinite(oracle_score):
            raise ValueError(f"oracle returned a non-finite score for task "
                             f"{task_id!r}")
        rows.append(EvaluationRow(
            task_id=task_id, verifier=verifier_scores[index],
            oracle=oracle_score))
    verifier_mean = sum(verifier_scores) / len(verifier_scores)
    if rows:
        oracle_mean = sum(row.oracle for row in rows) / len(rows)
        audited_verifier = sum(row.verifier for row in rows) / len(rows)
        bias = audited_verifier - oracle_mean
    else:
        oracle_mean = audited_verifier = bias = None
    return EvaluationReport(
        tasks=len(task_list), audited=len(rows),
        verifier_mean=verifier_mean, oracle_mean=oracle_mean,
        bias=bias, abs_bias=(abs(bias) if bias is not None else None),
        audited_verifier_mean=audited_verifier,
        coverage=len(rows) / len(task_list), oracle_calls=len(rows),
        budget=budget, rows=tuple(rows))
