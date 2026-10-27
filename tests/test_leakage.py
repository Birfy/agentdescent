"""A self-modification must not learn the answer key instead of the rule."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentdescent import LeakageGuard, Policies, SingleSlot, Task, evolve, find_leakage
from agentdescent.policies import AcceptDecision, MergeContext


@dataclass
class _Art:
    text: str

    def render(self):
        return self.text


def _ctx(cand):
    return MergeContext(artifact=_Art("seed"), candidate=_Art(cand), cards=[],
                        base_counts=(0.5, 0.5), cand_counts=(0.9, 0.1))


def test_it_rejects_a_candidate_that_contains_a_gold_answer():
    guard = LeakageGuard(["321", "the answer is 321"],
                         inner=lambda ctx: AcceptDecision(True))
    decision = guard.accept(_ctx("rule: always answer 321"))
    assert not decision.accept and decision.category == "critic-rejected"
    assert "321" in decision.detail


def test_matching_is_case_and_whitespace_insensitive():
    guard = LeakageGuard(["Queue  Name"], inner=lambda ctx: AcceptDecision(True))
    assert guard.check(_Art("use queue\n  name here")).passed is False


def test_short_protected_strings_are_ignored():
    guard = LeakageGuard(["a", "42"], min_length=3,
                         inner=lambda ctx: AcceptDecision(True))
    assert guard.check(_Art("anything 42")).passed


def test_a_clean_candidate_delegates_to_inner():
    guard = LeakageGuard(["321"], inner=lambda ctx: AcceptDecision(True, detail="ok"))
    decision = guard.accept(_ctx("rule: read the column header"))
    assert decision.accept and decision.detail == "ok"


def test_find_leakage_deduplicates_and_preserves_order():
    assert find_leakage("A and 321 and A", ["a", "321", "A"], min_length=1) == ("a", "321")


def test_min_length_is_validated():
    with pytest.raises(ValueError, match="min_length"):
        LeakageGuard(["x"], min_length=0)


def test_only_newly_introduced_strings_are_refused():
    guard = LeakageGuard(["321"], inner=lambda ctx: AcceptDecision(True))
    ctx = MergeContext(artifact=_Art("seed already says 321"),
                       candidate=_Art("seed already says 321"),
                       cards=[], base_counts=(0.5, 0.5),
                       cand_counts=(0.5, 0.5))
    assert guard.accept(ctx).accept
    ctx = MergeContext(artifact=_Art("seed already says 321"),
                       candidate=_Art("seed says 321 and 321 again"),
                       cards=[], base_counts=(0.5, 0.5),
                       cand_counts=(0.5, 0.5))
    assert guard.accept(ctx).accept


def test_screen_checks_the_diff_text():
    class Card:
        class Diff:
            ops = {"line": "use GOLD-ANSWER"}
        diff = Diff()

    guard = LeakageGuard(["GOLD-ANSWER"])
    assert guard.screen(Card()) == (
        "candidate diff contains protected string(s) 'GOLD-ANSWER'")
    assert guard.screen(object()) == "card carries no diff"


def _leaky_run(**kw):
    tasks = [Task(id=f"t{i}", prompt="q", meta={"gold": f"g{i}"})
             for i in range(8)]
    return evolve(
        tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
        run=lambda rendered, t: rendered.strip().split()[-1],
        propose=lambda rendered, t, o, s: "GOLD-ANSWER",
        strategy=SingleSlot(initial_value="seed"), rounds=3, n_workers=2,
        held_out_frac=0.5, seed=0, **kw)


def test_a_leaking_screen_rejects_the_card_without_crashing_evolve():
    result = _leaky_run(policies=Policies(
        screen=LeakageGuard(["GOLD-ANSWER"])))
    assert result.error is None
    assert result.outcomes().get("critic-rejected", 0) > 0


def test_the_acceptance_backstop_uses_a_real_outcome_category():
    result = _leaky_run(policies=Policies(
        acceptance=LeakageGuard(["GOLD-ANSWER"])))
    assert result.error is None
    assert result.outcomes().get("critic-rejected", 0) > 0
