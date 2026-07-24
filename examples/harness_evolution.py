"""Evolving a HARNESS with the general engine (no LLM required).

Same engine as the skill example, different artifact. Here the evolving thing is
a **request-processing harness** -- a pipeline of steps (route / normalize /
trim) rather than a skill playbook -- and it is registered at a higher
``blast_radius`` so it lives in the **L1 governance layer** (every merge forced
through the oracle, wider staleness tolerance).

This example also shows you don't need an LLM ``Agent``: you can drive
:func:`concordia.evolution.evolve` with plain ``run`` / ``reward`` / ``propose``
functions -- i.e. just *write the rules of evolution* and it runs.

    python -m examples.harness_evolution
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from concordia.evolution import KeyedRules, Task, evolve
from concordia.governance import Layer, classify
from concordia.evolution import EvolvingArtifact

# the pipeline steps the harness can learn, applied in this fixed order.
STEPS = {
    "route": lambda s: s.replace("RAW:", "routed:"),
    "normalize": lambda s: re.sub(r"\s+", " ", s.lower()),
    "trim": lambda s: s.strip(),
}
ORDER = ["route", "normalize", "trim"]


def process(text: str, active: list) -> str:
    for step in ORDER:
        if step in active:
            text = STEPS[step](text)
    return text


def make_tasks(n=16, seed=5):
    import random
    rng = random.Random(seed)
    raws = ["  RAW: Hello   World  ", "\tRAW:  Foo BAR  ", " RAW:  a   b  c ",
            "RAW: MiXeD   Case  "]
    tasks = []
    for i in range(n):
        raw = rng.choice(raws) + f" [{i}]"
        tasks.append(Task(id=f"t{i}", prompt=raw,
                          meta={"expected": process(raw, ORDER)}))
    return tasks


def reward(task: Task, output: str) -> float:
    return SequenceMatcher(None, output, task.meta["expected"]).ratio()


# -- the harness "actor": run the pipeline, and propose a missing step -------


def run(rendered_harness: str, task: Task) -> str:
    active = [s for s in ORDER if f"## {s}" in rendered_harness]
    return process(task.prompt, active)


def propose(rendered_harness: str, task: Task, output: str, r: float) -> Optional[str]:
    missing = [s for s in ORDER if f"## {s}" not in rendered_harness]
    if not missing:
        return None
    # different tasks surface different missing steps -> parallel workers propose
    # complementary steps that the aggregator fuses into one harness.
    step = missing[hash(task.id) % len(missing)]
    return f"{step}: apply the {step} step"


def main() -> None:
    tasks = make_tasks()
    print("Artifact : request-processing harness (steps: route / normalize / trim)")
    print(f"Governance: L1 (blast_radius=0.6) -> "
          f"{classify(EvolvingArtifact('harness', blast_radius=0.6)).name}, "
          f"every merge oracle-gated\n")

    result = evolve(
        tasks, reward,
        run=run, propose=propose,                    # plain functions, no LLM
        strategy=KeyedRules(categories=ORDER),       # one entry per step
        blast_radius=0.6,                            # -> L1 harness governance
        artifact_id="harness",
        rounds=10, n_workers=3, verbose=True,
    )

    print("\n=== evolved harness ===")
    print(result.rendered)
    print(f"\nheld-out score: {result.history[0].held_out_reward:.3f} "
          f"-> {result.final_reward:.3f}")
    print(f"steps learned : {len(result.state)}")


if __name__ == "__main__":
    main()
