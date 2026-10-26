"""A self-modification must not trade away a protected capability.

The engine's gate judges the target task; these fixtures pin the guard that
adds the RSI-safety question: did this candidate make anything else worse?
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentdescent import CapabilityGuard, CapabilityReport, CapabilitySuite
from agentdescent.aggregator import AggregatorConfig, install_policy
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision, MergeContext


@dataclass(frozen=True)
class _Task:
    id: str


@dataclass
class _Artifact:
    scores: dict

    def score(self, tasks):
        values = [float(self.scores[t.id]) for t in tasks]
        return sum(values) / len(values)


TOOL = [_Task("t1"), _Task("t2")]
JSON = [_Task("j1"), _Task("j2")]


def _ctx(base, cand, *, base_counts=(0.5, 0.5), cand_counts=(0.8, 0.2)):
    return MergeContext(artifact=base, candidate=cand, cards=[],
                        base_counts=base_counts, cand_counts=cand_counts)


def test_it_rejects_a_target_gain_that_breaks_a_protected_suite():
    base = _Artifact({"t1": 1.0, "t2": 1.0, "j1": 1.0, "j2": 1.0})
    cand = _Artifact({"t1": 0.2, "t2": 0.2, "j1": 1.0, "j2": 1.0})
    guard = CapabilityGuard([CapabilitySuite("tool", TOOL, floor=0.85)])
    decision = guard.accept(_ctx(base, cand))
    assert decision.accept is False
    assert decision.category == "capability-regression"
    assert "tool" in decision.detail and "regressed" in decision.detail


def test_a_candidate_that_keeps_every_suite_delegates_to_inner_acceptance():
    base = _Artifact({"t1": 1.0, "t2": 1.0, "j1": 1.0, "j2": 1.0})
    cand = _Artifact({"t1": 1.0, "t2": 1.0, "j1": 1.0, "j2": 1.0})
    guard = CapabilityGuard([CapabilitySuite("tool", TOOL, floor=0.9)],
                            inner=lambda ctx: AcceptDecision(True, detail="inner ok"))
    decision = guard.accept(_ctx(base, cand))
    assert decision.accept and decision.detail == "inner ok"


def test_without_inner_it_uses_the_configured_default_gate():
    base = _Artifact({"t1": 0.5, "t2": 0.5})
    cand = _Artifact({"t1": 0.8, "t2": 0.8})
    guard = CapabilityGuard([CapabilitySuite("tool", TOOL, floor=0.4)])
    guard.configure(AggregatorConfig())
    assert guard.accept(_ctx(base, cand)).accept is True
    worse = _Artifact({"t1": 0.2, "t2": 0.2})
    assert guard.accept(_ctx(base, worse)).accept is False


def test_tolerance_is_relative_to_the_base():
    base = _Artifact({"t1": 0.8, "t2": 0.8})
    cand = _Artifact({"t1": 0.75, "t2": 0.75})
    strict = CapabilityGuard([CapabilitySuite("tool", TOOL, tolerance=0.01)])
    loose = CapabilityGuard([CapabilitySuite("tool", TOOL, tolerance=0.10)])
    strict_report = strict.check(base, cand)
    loose_report = loose.check(base, cand)
    assert not strict_report.passed and loose_report.passed
    assert loose_report.suites["tool"]["bar"] == pytest.approx(0.70)


def test_report_is_data_and_names_every_suite():
    base = _Artifact({"t1": 1.0, "t2": 1.0, "j1": 1.0, "j2": 1.0})
    cand = _Artifact({"t1": 0.0, "t2": 0.0, "j1": 1.0, "j2": 1.0})
    report = CapabilityGuard([CapabilitySuite("tool", TOOL, floor=0.5),
                              CapabilitySuite("json", JSON, floor=0.5)]
                             ).check(base, cand)
    assert isinstance(report, CapabilityReport)
    assert set(report.suites) == {"tool", "json"}
    assert report.regressions == ("tool",)
    assert report.suites["json"]["candidate"] == 1.0


def test_install_forwards_the_hooks_to_inner():
    guard = CapabilityGuard([CapabilitySuite("tool", TOOL)],
                            inner=DefaultAcceptance())
    install_policy(guard, None, AggregatorConfig())
    assert guard.inner.base_delta is not None
    assert guard.inner.accept_samples is not None


def test_a_floor_above_the_base_does_not_block_climbing_toward_it():
    base = _Artifact({"t1": 0.6, "t2": 0.6})
    cand = _Artifact({"t1": 0.8, "t2": 0.8})
    guard = CapabilityGuard([CapabilitySuite("tool", TOOL, floor=0.85,
                                             tolerance=0.02)],
                            inner=lambda ctx: AcceptDecision(True))
    assert guard.accept(_ctx(base, cand)).accept is True


def test_an_empty_or_duplicated_suite_list_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        CapabilityGuard([])
    with pytest.raises(ValueError, match="unique"):
        CapabilityGuard([CapabilitySuite("x", TOOL), CapabilitySuite("x", JSON)])
