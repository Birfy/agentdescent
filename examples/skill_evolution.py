"""Agent-driven skill self-evolution -- runnable example.

Evolves a text-normalization skill from an empty playbook to a correct one,
driven by an agent. The default agent is a deterministic mock (no API key, no
network) so this runs anywhere; pass ``--claude`` to drive it with a real Claude
agent instead.

    python -m examples.skill_evolution
    python -m examples.skill_evolution --claude          # needs ANTHROPIC_API_KEY

The skill is a playbook of text-transform rules. Each task is a messy string
whose expected output requires applying several transforms. Workers run tasks,
observe failures, and propose the missing rule; the aggregator merges the good
rules from parallel workers and rejects unhelpful ones on held-out reward.
"""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from concordia.skillevo import Agent, Task, evolve_skill, claude_agent


# -- the transforms the skill can learn --------------------------------------

TRANSFORMS: Dict[str, "callable"] = {
    "Strip leading and trailing whitespace from the text.": lambda s: s.strip(),
    "Convert the text to lowercase.": lambda s: s.lower(),
    "Remove all punctuation characters.": lambda s: re.sub(r"[^\w\s]", "", s),
    "Collapse runs of whitespace into a single space.": lambda s: re.sub(r"\s+", " ", s),
    "Replace tab characters with a single space.": lambda s: s.replace("\t", " "),
}
RULES = list(TRANSFORMS)
# a canonical order so the composed target is well-defined.
ORDER = [RULES[4], RULES[1], RULES[2], RULES[3], RULES[0]]  # tabs,lower,punct,collapse,strip


def normalize(text: str, rules: List[str]) -> str:
    """Apply the given transforms (in canonical order) to text."""
    out = text
    for rule in ORDER:
        if rule in rules:
            out = TRANSFORMS[rule](out)
    return out


def make_tasks(n: int = 16, seed: int = 3) -> List[Task]:
    import random
    rng = random.Random(seed)
    raws = [
        "  Hello,   World!!  ", "\tFoo\tBAR baz. ", "MiXeD   Case?!  ",
        " multiple    spaces here ", "Tabs\tand  \tPUNCT... ",
        "  UPPER lower  Mix!  ", "trailing space   ", "  a,b,c;d:e  ",
    ]
    tasks = []
    for i in range(n):
        raw = rng.choice(raws) + f" [{i}]"
        expected = normalize(raw, RULES)  # target = all transforms applied
        tasks.append(Task(id=f"t{i}", prompt=raw, meta={"expected": expected}))
    return tasks


def reward(task: Task, output: str) -> float:
    """Graded similarity between the agent's output and the fully-normalized target."""
    return SequenceMatcher(None, output, task.meta["expected"]).ratio()


# -- a deterministic mock agent (stands in for an LLM) -----------------------


class MockNormalizerAgent:
    """Executes whatever transform-rules are in the playbook; on failure,
    proposes a plausible missing rule (occasionally a wrong one, to exercise
    the aggregator's rejection of unhelpful rules)."""

    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        import random
        self.noise = noise
        self._rng = random.Random(seed)

    def solve(self, skill_text: str, task: Task) -> str:
        active = [r for r in RULES if r in skill_text]
        return normalize(task.prompt, active)

    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> Optional[str]:
        missing = [r for r in RULES if r not in skill_text]
        if not missing:
            return None
        if self._rng.random() < self.noise:
            return "Convert the text to UPPERCASE."  # harmful; must be rejected
        # pick a missing transform deterministically by task, so different
        # workers propose different (complementary) rules -> fusion.
        return missing[hash(task.id) % len(missing)]


def main() -> None:
    use_claude = "--claude" in sys.argv
    tasks = make_tasks()

    if use_claude:
        agent: Agent = claude_agent(model="claude-opus-4-8")
        rounds, n_workers = 8, 3
        print("Using Claude agent (this makes real API calls)...\n")
    else:
        agent = MockNormalizerAgent(noise=0.34, seed=1)
        rounds, n_workers = 12, 4
        print("Using deterministic mock agent (no API calls)...\n")

    result = evolve_skill(agent, tasks, reward, rounds=rounds,
                          n_workers=n_workers, held_out_frac=0.4, verbose=True)

    print("\n=== evolved playbook ===")
    print(result.playbook)
    print(f"\nheld-out reward: {result.history[0].held_out_reward:.3f} "
          f"-> {result.final_reward:.3f}")
    print(f"rules learned: {len(result.rules)}")


if __name__ == "__main__":
    main()
