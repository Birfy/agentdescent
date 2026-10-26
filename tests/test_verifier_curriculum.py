"""The verifier curriculum must target disagreements, not its own comfort zone."""

from __future__ import annotations

import pytest

from agentdescent import (
    CoevolutionReport,
    ProbeRow,
    VerifierAgreement,
    VerifierCurriculum,
)
from agentdescent.aggregator import AggregatorConfig, install_policy
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision, MergeContext


class _Artifact:
    def __init__(self, name):
        self.name = name


ORACLE = {0: 0.2, 1: 0.8, 2: 0.5, 3: 0.9}


def _oracle(artifact, task):
    return ORACLE[task]


def test_probe_reports_the_gap_and_select_ranks_it():
    bias = {0: 0.0, 1: -0.5, 2: 0.1, 3: 0.4}

    def verifier(artifact, task):
        return ORACLE[task] + bias[task]

    cur = VerifierCurriculum([0, 1, 2, 3], verifier, _oracle)
    rows = cur.probe(_Artifact("a"))
    assert all(isinstance(row, ProbeRow) for row in rows)
    assert max(rows, key=lambda row: row.abs_gap).task == 1
    assert cur.select(_Artifact("a"), 2) == [1, 3]
    assert cur.select(_Artifact("a"), 1, mode="overestimate") == [3]
    assert cur.select(_Artifact("a"), 1, mode="underestimate") == [1]


def test_coevolve_feedback_reduces_the_verifier_oracle_gap():
    bias = {0: 0.2, 1: -0.5, 2: 0.1, 3: 0.4}

    def verifier(artifact, task):
        return ORACLE[task] + bias[task]

    cur = VerifierCurriculum([0, 1, 2, 3], verifier, _oracle)

    def update(task, oracle_score):
        bias[task] = 0.0      # the oracle label corrects this task's verifier

    report = cur.coevolve(_Artifact("a"), rounds=3, n=2, update=update)
    assert isinstance(report, CoevolutionReport)
    assert len(report.rounds) == 3
    assert report.improved
    assert report.final_abs_gap < report.initial_abs_gap
    assert report.final_abs_gap == pytest.approx(0.0)


class _Scored:
    def __init__(self, scores):
        self.scores = scores


def _scorer(table):
    def score(artifact, task):
        return table[artifact.name][task]
    return score


def _ctx():
    base = _Scored({})
    base.name = "base"
    cand = _Scored({})
    cand.name = "cand"
    return MergeContext(artifact=base, candidate=cand, cards=[],
                        base_counts=(0.5, 0.5), cand_counts=(0.8, 0.2))


def test_agreement_refuses_a_verifier_only_gain():
    verifier = _scorer({"base": {0: 0.2, 1: 0.2}, "cand": {0: 0.9, 1: 0.9}})
    oracle = _scorer({"base": {0: 0.7, 1: 0.7}, "cand": {0: 0.7, 1: 0.7}})
    gate = VerifierAgreement([0, 1], verifier, oracle, sample_rate=1.0,
                             inner=lambda ctx: AcceptDecision(True))
    decision = gate.accept(_ctx())
    assert decision.accept is False
    assert decision.category == "oracle-disagrees"
    assert "oracle" in decision.detail


def test_agreement_passes_when_the_oracle_confirms_the_gain():
    verifier = _scorer({"base": {0: 0.2}, "cand": {0: 0.9}})
    oracle = _scorer({"base": {0: 0.2}, "cand": {0: 0.8}})
    gate = VerifierAgreement([0], verifier, oracle, sample_rate=1.0,
                             inner=lambda ctx: AcceptDecision(True))
    decision = gate.accept(_ctx())
    assert decision.accept


def test_inner_rejection_is_not_overridden():
    gate = VerifierAgreement(
        [0], _scorer({"base": {0: 1.0}, "cand": {0: 1.0}}),
        _scorer({"base": {0: 1.0}, "cand": {0: 1.0}}), sample_rate=1.0,
        inner=lambda ctx: AcceptDecision(False, "inner-says-no"))
    decision = gate.accept(_ctx())
    assert not decision.accept and decision.category == "inner-says-no"


def test_an_oracle_regression_is_refused_whatever_the_verifier_says():
    verifier = _scorer({"base": {0: 0.5}, "cand": {0: 0.5}})
    oracle = _scorer({"base": {0: 0.9}, "cand": {0: 0.1}})
    gate = VerifierAgreement([0], verifier, oracle, sample_rate=1.0,
                             inner=lambda ctx: AcceptDecision(True))
    decision = gate.accept(_ctx())
    assert not decision.accept and decision.category == "oracle-regression"


def test_install_forwards_the_hooks_to_inner():
    gate = VerifierAgreement([0], lambda a, t: 0.5, lambda a, t: 0.5,
                             inner=DefaultAcceptance())
    install_policy(gate, None, AggregatorConfig())
    assert gate.inner.base_delta is not None
    assert gate.inner.accept_samples is not None


def test_coevolve_reuses_the_oracle_label_it_already_paid_for():
    calls = []
    oracle = {i: (i / 10.0) for i in range(10)}

    def oracle_fn(artifact, task):
        calls.append(task)
        return oracle[task]

    cur = VerifierCurriculum(list(range(10)), lambda a, t: 0.5, oracle_fn)
    cur.coevolve(_Artifact("a"), rounds=3, n=2,
                 update=lambda task, y: None)
    # one probe per round (10 calls), no extra oracle call for update/after
    assert len(calls) <= 30


def test_sample_rate_is_validated():
    with pytest.raises(ValueError, match="sample_rate"):
        VerifierAgreement([], lambda a, t: 0.0, lambda a, t: 0.0,
                          sample_rate=1.5)
