"""The evaluation pass must respect the oracle budget and expose bias."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentdescent import EvaluationReport, evaluate


@dataclass(frozen=True)
class _Task:
    id: str


TASKS = [_Task(f"t{i}") for i in range(10)]


def _verifier(artifact, task):
    return 0.8


def _oracle(artifact, task):
    return 0.5


def test_it_respects_the_budget_and_reports_bias():
    report = evaluate(None, TASKS, verifier=_verifier, oracle=_oracle,
                      sample_rate=0.5, budget=3, seed=1)
    assert isinstance(report, EvaluationReport)
    assert report.tasks == 10 and report.audited == 3
    assert report.oracle_calls == 3 and report.coverage == pytest.approx(0.3)
    assert report.bias == pytest.approx(0.3) and report.abs_bias == pytest.approx(0.3)
    assert report.verifier_mean == pytest.approx(0.8)   # verifier on all tasks


def test_sampling_is_deterministic():
    a = evaluate(None, TASKS, verifier=_verifier, oracle=_oracle,
                 sample_rate=0.4, budget=10, seed=7)
    b = evaluate(None, TASKS, verifier=_verifier, oracle=_oracle,
                 sample_rate=0.4, budget=10, seed=7)
    assert a == b


def test_zero_budget_means_no_truth_evidence():
    report = evaluate(None, TASKS, verifier=_verifier, oracle=_oracle,
                      sample_rate=0.5, budget=0)
    assert report.audited == 0 and report.coverage == 0.0
    assert report.bias is None and report.abs_bias is None
    assert report.oracle_mean is None


def test_empty_tasks_is_safe():
    report = evaluate(None, [], verifier=_verifier, oracle=_oracle)
    assert report.tasks == 0 and report.rows == ()


def test_parameters_are_validated():
    with pytest.raises(ValueError, match="sample_rate"):
        evaluate(None, TASKS, verifier=_verifier, oracle=_oracle,
                 sample_rate=2)
    with pytest.raises(ValueError, match="budget"):
        evaluate(None, TASKS, verifier=_verifier, oracle=_oracle, budget=-1)
    with pytest.raises(ValueError, match="budget"):
        evaluate(None, TASKS, verifier=_verifier, oracle=_oracle, budget=True)


def test_non_finite_scores_are_refused():
    with pytest.raises(ValueError, match="verifier.*non-finite"):
        evaluate(None, TASKS, verifier=lambda a, t: float("nan"),
                 oracle=_oracle, sample_rate=1.0, budget=10)
    with pytest.raises(ValueError, match="oracle.*non-finite"):
        evaluate(None, TASKS, verifier=_verifier,
                 oracle=lambda a, t: float("inf"), sample_rate=1.0, budget=10)


def test_mapping_tasks_get_their_id():
    report = evaluate(None, [{"id": "a"}, {"id": "b"}], verifier=_verifier,
                      oracle=_oracle, sample_rate=1.0, budget=2)
    assert [row.task_id for row in report.rows] == ["a", "b"]


def test_sampling_uses_ceil():
    report = evaluate(None, TASKS[:5], verifier=_verifier, oracle=_oracle,
                      sample_rate=0.5, budget=10)
    assert report.audited == 3       # ceil(2.5), not round(2.5) == 2
