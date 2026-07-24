"""Tests for user-customizable evolution strategies."""

from difflib import SequenceMatcher
from typing import Optional

from concordia.aggregator import diffs_contradict
from concordia.evolution import (
    AppendRules,
    KeyedRules,
    Strategy,
    Task,
    evolve,
)


# -- built-in strategies -----------------------------------------------------


def test_append_rules_dedupes():
    s = AppendRules()
    d1 = s.to_diff({}, "Lowercase the text.", "w0", 1, "sk")
    assert d1 is not None
    state = dict(d1.ops)
    # the same proposal from another worker produces no new diff.
    assert s.to_diff(state, "  lowercase the text.  ", "w1", 1, "sk") is None


def test_keyed_rules_make_competing_proposals_contradict():
    s = KeyedRules(categories=["method"])
    a = s.to_diff({}, "method: sort case-insensitively", "w0", 1, "sk")
    b = s.to_diff({}, "method: sort by ASCII value", "w1", 1, "sk")
    # same category key, different value -> a contradiction the aggregator resolves.
    assert a is not None and b is not None
    assert diffs_contradict(a, b)


def test_keyed_rules_falls_back_to_append_for_unknown_category():
    s = KeyedRules(categories=["method"])
    d = s.to_diff({}, "just a freeform tip", "w0", 1, "sk")
    assert d is not None
    # keyed under a content hash, not a category
    assert list(d.ops)[0].startswith("r")


# -- a fully custom strategy plugs in ----------------------------------------


class SingleValueStrategy:
    """A minimal custom strategy: the skill is one slot; each proposal replaces
    it (so proposals always contradict and the best-scoring one wins)."""

    def initial(self):
        return {}

    def render(self, state):
        return state.get("v", "(none)")

    def to_diff(self, state, proposal, author, base_version, target):
        from concordia.evolvable import Diff
        if state.get("v") == proposal:
            return None
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": proposal}, author=author)


def test_custom_strategy_is_structural_and_runs():
    assert isinstance(SingleValueStrategy(), Strategy)

    class Agent:
        # learns to answer "lower" by proposing the literal target text
        def solve(self, skill_text, task):
            return skill_text if skill_text != "(none)" else task.prompt
        def propose(self, skill_text, task, output, reward) -> Optional[str]:
            return task.meta["expected"]

    tasks = [Task(id=f"t{i}", prompt=f"X{i}",
                  meta={"expected": "GOAL"}) for i in range(12)]
    reward = lambda t, o: SequenceMatcher(None, o, t.meta["expected"]).ratio()

    result = evolve(tasks, reward, agent=Agent(), strategy=SingleValueStrategy(),
                    rounds=6, n_workers=2)
    assert result.final_reward >= result.history[0].held_out_reward
    assert result.state.get("v") == "GOAL"
