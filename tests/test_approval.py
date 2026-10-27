"""High-risk self-modifications stop until a human says yes."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agentdescent import (ApprovalGate, ApprovalStore, Policies, SingleSlot,
                          Task, evolve)
from agentdescent.aggregator import MergeOutcome
from agentdescent.evolvable import Diff
from agentdescent.policies import AcceptDecision, MergeContext


@dataclass
class _Art:
    id: str = "a"

    def render(self):
        return "x"


def _ctx(diff_id="d1", ops=2, blast=0.2, value="v"):
    diff = Diff(diff_id=diff_id, target="a",
                ops={f"k{i}": value for i in range(ops)})
    return MergeContext(artifact=_Art(), candidate=_Art(), cards=[],
                        base_counts=(0.5, 0.5), cand_counts=(0.9, 0.1),
                        diff=diff, trust_radius=blast)


def test_low_risk_changes_delegate_without_a_request(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    assert gate.accept(_ctx(ops=1, blast=0.2)).accept
    assert store.pending() == []


def test_a_high_risk_change_is_queued_then_honours_approval(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    decision = gate.accept(_ctx(ops=8, blast=0.7))
    assert not decision.accept and decision.category == "needs-approval"
    [request] = store.pending()
    store.decide(request.key, approve=True, note="reviewed")
    assert gate.accept(_ctx(ops=8, blast=0.7)).accept
    assert store.pending() == []


def test_a_rejection_is_sticky(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    gate.accept(_ctx(ops=8, blast=0.7))
    key = store.pending()[0].key
    store.decide(key, approve=False)
    decision = gate.accept(_ctx(ops=8, blast=0.7))
    assert decision.category == "approval-rejected"


def test_the_request_is_written_once(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    gate.accept(_ctx(ops=8, blast=0.7))
    gate.accept(_ctx(ops=8, blast=0.7))
    assert len(open(tmp_path / "a.jsonl").read().splitlines()) == 1


def test_thresholds_are_validated():
    with pytest.raises(ValueError, match="risk_threshold"):
        ApprovalGate(ApprovalStore("/tmp/x"), risk_threshold=2)
    with pytest.raises(ValueError, match="ops_threshold"):
        ApprovalGate(ApprovalStore("/tmp/x"), ops_threshold=0)


def test_approval_is_bound_to_content_not_diff_id(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    first = gate.accept(_ctx(diff_id="same", ops=8, value="first"))
    second = gate.accept(_ctx(diff_id="same", ops=8, value="second"))
    assert not first.accept and not second.accept
    pending = store.pending()
    assert len(pending) == 2 and len({r.key for r in pending}) == 2
    store.decide(pending[0].key, approve=True)
    assert gate.accept(_ctx(diff_id="same", ops=8, value="first")).accept
    assert not gate.accept(
        _ctx(diff_id="same", ops=8, value="second")).accept


def test_malformed_store_lines_are_skipped(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('not json\n123\n'
                    '{"event":"request","key":"k","artifact_id":"a",'
                    '"reason":"r","created_at":1}\n')
    store = ApprovalStore(str(path))
    assert [request.key for request in store.pending()] == ["k"]


def test_deciding_an_unknown_key_is_refused(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    with pytest.raises(KeyError, match="no approval request"):
        store.decide("missing", approve=True)


def test_the_request_carries_a_reviewable_summary(tmp_path):
    store = ApprovalStore(str(tmp_path / "a.jsonl"))
    gate = ApprovalGate(store, inner=lambda ctx: AcceptDecision(True))
    gate.accept(_ctx(ops=8))
    assert '"k0"' in store.pending()[0].summary


def test_pending_approval_goes_through_evolve_without_crashing(tmp_path):
    tasks = [Task(id=f"t{i}", prompt="q", meta={"gold": f"g{i}"})
             for i in range(8)]
    result = evolve(
        tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
        run=lambda rendered, t: rendered.strip().split()[-1],
        propose=lambda rendered, t, o, s: "GOLD",
        strategy=SingleSlot(initial_value="seed"), rounds=2, n_workers=2,
        held_out_frac=0.5, seed=0,
        policies=Policies(acceptance=ApprovalGate(
            ApprovalStore(str(tmp_path / "a.jsonl")), ops_threshold=1,
            risk_threshold=1.0)))
    assert result.error is None
    assert result.outcomes().get(MergeOutcome.NEEDS_APPROVAL, 0) > 0
