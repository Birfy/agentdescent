"""Evolving an L1 artifact (harness / verifier), not just L2 skills.

'What evolves' is chosen by registration + blast radius, so the same machinery
that evolves a skill evolves a harness -- just under stricter L1 governance."""

from difflib import SequenceMatcher
from typing import Optional

from concordia.governance import Layer, classify
from concordia.scheduler import AuditScheduler
from concordia.evolution import EvolvingArtifact, Task, evolve


def test_blast_radius_selects_governance_layer():
    assert classify(EvolvingArtifact("skill", blast_radius=0.2)) is Layer.L2_FAST      # local skill
    assert classify(EvolvingArtifact("harness", blast_radius=0.6)) is Layer.L1_SLOW    # harness/verifier


def test_high_blast_radius_forces_oracle_audit():
    audit = AuditScheduler()
    assert audit.force_oracle(0.6, "harness") is True   # L1 change -> oracle-gated
    assert audit.force_oracle(0.2, "skill") is False    # L2 skill -> cheap layers ok


class _HarnessAgent:
    """Learns two harness rules (a routing rule + a context rule)."""

    def solve(self, skill_text: str, task: Task) -> str:
        out = task.prompt
        if "route" in skill_text:
            out = out.replace("RAW", "routed")
        if "trim" in skill_text:
            out = out.strip()
        return out

    def propose(self, skill_text, task, output, reward) -> Optional[str]:
        if "route" not in skill_text:
            return "route: rewrite RAW to routed"
        if "trim" not in skill_text:
            return "trim: strip whitespace"
        return None


def test_evolve_harness_end_to_end():
    tasks = [Task(id=f"t{i}", prompt=f"  RAW payload {i}  ",
                  meta={"expected": f"routed payload {i}"}) for i in range(12)]
    reward = lambda t, o: SequenceMatcher(None, o, t.meta["expected"]).ratio()

    result = evolve(tasks, reward, agent=_HarnessAgent(), blast_radius=0.6,
                    rounds=8, n_workers=2)
    # the L1 harness improved on held-out despite oracle-gated, conservative merges
    assert result.final_reward > result.history[0].held_out_reward
    assert result.final_reward > 0.9
