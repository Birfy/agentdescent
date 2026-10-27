"""The guard pipeline has to be ordered, named and fail-closed."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentdescent import GuardPipeline, Policies, SingleSlot, Task, evolve
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision, MergeContext


@dataclass
class _Art:
    name: str = "a"


def _ctx():
    return MergeContext(artifact=_Art(), candidate=_Art(), cards=[],
                        base_counts=(0.5, 0.5), cand_counts=(0.7, 0.3))


def _guard(name, accept=True, category="", detail=""):
    calls = []

    def check(ctx):
        calls.append(name)
        return AcceptDecision(accept, category, detail or name)

    return check, calls


def test_guards_run_in_order_and_all_must_pass():
    calls = []
    a, ca = _guard("a", detail="a")
    b, cb = _guard("b", detail="b")
    pipeline = GuardPipeline([("a", a), ("b", b)])
    decision = pipeline.accept(_ctx())
    assert decision.accept and "a" in decision.detail and "b" in decision.detail
    assert ca == ["a"] and cb == ["b"]


def test_the_first_refusal_names_the_guard_and_stops():
    calls = []
    a, ca = _guard("a", False, "leak")
    b, cb = _guard("b", True)
    pipeline = GuardPipeline([("a", a), ("b", b)])
    decision = pipeline.accept(_ctx())
    assert not decision.accept
    assert decision.category == "below-threshold"
    assert decision.detail == "a: a"
    assert ca == ["a"] and cb == []


def test_collect_all_records_every_guard_but_reports_the_first_refusal():
    pipeline = GuardPipeline(
        [("a", _guard("a", False, "leak", "A refused")[0]),
         ("b", _guard("b", True, "ok", "B passed")[0])],
        collect_all=True)
    result = pipeline.evaluate(_ctx())
    assert not result.accepted and result.failed_at == "a"
    assert [d.name for d in result.decisions] == ["a", "b"]
    assert result.to_dict()["accepted"] is False
    decision = pipeline.accept(_ctx())
    assert decision.category == "below-threshold"
    assert "A refused" in decision.detail and "B passed" not in decision.detail


def test_an_empty_pipeline_is_the_identity_gate():
    assert GuardPipeline([]).accept(_ctx()).accept


def test_install_hooks_reach_every_guard():
    seen = []

    class Guard:
        def accept(self, ctx):
            return AcceptDecision(True)
        def bind(self, verifier):
            seen.append(("bind", verifier))
        def configure(self, config):
            seen.append(("configure", config))

    pipeline = GuardPipeline([("a", Guard()), ("b", Guard())])
    pipeline.bind("v")
    pipeline.configure("c")
    assert seen == [("bind", "v"), ("bind", "v"), ("configure", "c"), ("configure", "c")]
    assert pipeline.describe() == ("a", "b")


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        GuardPipeline([("a", _guard("a")[0]), ("a", _guard("b")[0])])


def test_a_valid_merge_category_is_preserved_and_the_guard_name_is_in_detail():
    pipeline = GuardPipeline([("stats", _guard("stats", False, "oversized")[0])])
    decision = pipeline.accept(_ctx())
    assert decision.category == "oversized" and decision.detail == "stats: stats"


def test_checkpoint_reject_and_restore_hooks_reach_nested_policies():
    seen = []

    class Stateful:
        def accept(self, ctx):
            return AcceptDecision(True)

        def checkpoint_state(self):
            seen.append("checkpoint")
            return {"n": 1}

        def restore_state(self, state):
            seen.append(("restore", state))

        def on_screen_reject(self):
            seen.append("reject")

    pipeline = GuardPipeline([("stateful", Stateful())])
    state = pipeline.checkpoint_state()
    pipeline.restore_state(state)
    pipeline.on_screen_reject()
    assert state == {"stateful": {"n": 1}}
    assert seen == ["checkpoint", ("restore", {"n": 1}), "reject"]


def test_a_refusing_pipeline_goes_through_evolve_without_crashing():
    tasks = [Task(id=str(i), prompt="q", meta={"gold": f"g{i % 3}"})
             for i in range(12)]

    def refuse(ctx):
        return AcceptDecision(False, "below-threshold", "leaky candidate")

    result = evolve(
        tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
        run=lambda rendered, t: rendered.strip().split()[-1],
        propose=lambda rendered, t, o, s: t.meta["gold"],
        strategy=SingleSlot(initial_value="g0"), rounds=3, n_workers=3,
        max_concurrency=1, held_out_frac=0.5, seed=0,
        policies=Policies(acceptance=GuardPipeline([("leak", refuse)])))
    assert result.error is None
    assert result.outcomes().get("below-threshold", 0) > 0
