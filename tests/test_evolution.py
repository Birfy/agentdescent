"""Tests for agent-driven skill evolution.

Uses tiny deterministic in-process agents (no LLM) to verify the convenient
``evolve`` API: any agent implementing the two-method protocol drives it,
good rules are learned, and harmful rules are rejected on held-out reward.
"""

from difflib import SequenceMatcher
from typing import Optional

from concordia.evolution import (
    Agent,
    LLMAgent,
    Task,
    evolve,
    rule_id,
)


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt=f"  MiXeD Case {i}!  ",
                 meta={"expected": f"mixed case {i}!".strip()}) for i in range(n)]


def _reward(task, output):
    return SequenceMatcher(None, output, task.meta["expected"]).ratio()


class GoodAgent:
    """Learns 'lowercase' and 'strip' rules; proposes the one that's missing."""

    def solve(self, skill_text: str, task: Task) -> str:
        out = task.prompt
        if "lowercase" in skill_text:
            out = out.lower()
        if "Strip" in skill_text:
            out = out.strip()
        return out

    def propose(self, skill_text, task, output, reward) -> Optional[str]:
        if "lowercase" not in skill_text:
            return "Convert the text to lowercase."
        if "Strip" not in skill_text:
            return "Strip surrounding whitespace."
        return None


class HarmfulAgent:
    """Fails the tasks, and only ever proposes a rule that makes output worse."""

    def solve(self, skill_text: str, task: Task) -> str:
        out = task.prompt  # unchanged -> imperfect, so it keeps proposing
        if "UPPERCASE" in skill_text:
            out = out.upper()  # the harmful rule makes it strictly worse
        return out

    def propose(self, skill_text, task, output, reward) -> Optional[str]:
        return "Convert everything to UPPERCASE."


def test_any_agent_drives_evolution_and_learns_rules():
    result = evolve(_tasks(), _reward, agent=GoodAgent(), rounds=10, n_workers=3)
    assert result.history[0].held_out_reward < result.final_reward
    assert result.final_reward > 0.95
    assert len(result.state) >= 2  # both useful rules were learned


def test_harmful_rules_are_rejected():
    result = evolve(_tasks(), _reward, agent=HarmfulAgent(), rounds=8, n_workers=3)
    # the only proposals hurt held-out reward, so none should be committed.
    assert len(result.state) == 0
    assert sum(r.rejected for r in result.history) > 0


def test_protocol_is_structural():
    # a bare object with the two methods satisfies the Agent protocol.
    assert isinstance(GoodAgent(), Agent)


def test_llm_agent_wraps_a_completion_fn():
    calls = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return "NONE" if "Propose" in prompt else "the result"

    agent = LLMAgent(complete)
    task = Task(id="t", prompt="hi")
    assert agent.solve("# Skill Playbook", task) == "the result"
    assert agent.propose("# Skill Playbook", task, "out", 0.2) is None
    assert len(calls) == 2


def test_rule_id_dedupes_identical_text():
    assert rule_id("Lowercase the text.") == rule_id("  lowercase the text.  ")
    assert rule_id("A") != rule_id("B")


# -- pluggable aggregator ----------------------------------------------------

def test_custom_aggregator_factory_plugs_in():
    from concordia.aggregator import Aggregator, AggregatorProtocol

    calls = {"steps": 0}

    class CountingAggregator(Aggregator):
        def step(self):
            calls["steps"] += 1
            return super().step()

    def factory(ledger, verifier, audit, config, policy):
        agg = CountingAggregator(ledger, verifier, audit, config, staleness_policy=policy)
        assert isinstance(agg, AggregatorProtocol)      # satisfies the contract
        return agg

    result = evolve(_tasks(), _reward, agent=GoodAgent(), rounds=5, n_workers=2,
                    aggregator_factory=factory)
    assert calls["steps"] == 5                          # one step per round, via our aggregator
    assert result.final_reward >= result.history[0].held_out_reward
