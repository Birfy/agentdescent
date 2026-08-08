"""Environment analogue: **Voyager** (MineDojo/Voyager@55e45a88).

Preserved: an **add-only skill library** (one entry per goal key; entries
accumulate and union-merge, matching upstream's versioned, never-overwritten
library); repair driven by **environment feedback** -- the deterministic world
executes the attempt and reports the first failed step, which is what the
repair prompt sees (never a gold trace); the **critic** as the engine's worker
self-check (`self_verify`): every proposal is re-rolled and judged by the
environment reward before it reaches the gate, upstream's "judge success from
the environment, not the agent's claim"; and frontier task focus via the
:class:`~agentdescent.sampling.DifficultyWeighted` sampler.

Boundaries: a deterministic crafting world replaces Minecraft; key-match
retrieval replaces embedding retrieval; the task pool plus difficulty sampling
is an analogue of the generative curriculum, which proposes novel tasks.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import canonical_json, parse_json_object
from examples._method_policy import MethodPolicy, SkillLibrary
from examples._method_runner import standard_main


FIDELITY = "environment_analogue"

INGREDIENTS = (
    "mint", "berry", "lemon", "ginger", "apple", "peach",
    "pear", "plum", "orange", "basil", "cherry", "melon",
)

PRIMITIVES = "sanitize, collect, heat, combine, serve (as 'primitive:argument')"


def _required(ingredient: str) -> List[str]:
    return [
        "sanitize:vessel",
        "collect:water",
        f"collect:{ingredient}",
        "heat:water",
        f"combine:water+{ingredient}",
        "serve:drink",
    ]


def _action_list(text: str, key: str) -> List[str]:
    payload = parse_json_object(text)
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"{key} must be a non-empty bounded list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must contain strings")
    return [item.strip().lower() for item in value]


def simulate(actions: Sequence[str], ingredient: str) -> Tuple[bool, str]:
    """Execute an action sequence; report the first unmet step, Voyager-style.

    The message names what failed and what the world observed -- not the
    required program.
    """
    required = _required(ingredient)
    cursor = 0
    for action in actions:
        if cursor < len(required) and action == required[cursor]:
            cursor += 1
    if cursor == len(required):
        return True, "goal reached: drink served"
    missing = required[cursor]
    verb = missing.split(":", 1)[0]
    return False, (
        f"execution stopped before '{verb}' succeeded: the environment "
        f"expected a '{verb}' step that never happened in order (progress "
        f"{cursor}/{len(required)}). Available primitives: {PRIMITIVES}."
    )


def _skill_value(text: str) -> str:
    steps = _action_list(text, "steps")
    return canonical_json({"steps": steps})


def _tasks() -> List[Task]:
    return [
        Task(
            id=f"recipe:{ingredient}",
            prompt=f"brew and serve {ingredient} tea",
            meta={"ingredient": ingredient},
        )
        for ingredient in INGREDIENTS
    ]


def _split(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    rows = _tasks()
    random.Random(seed).shuffle(rows)
    return rows[:4], rows[4:8], rows[8:12]


def _retrieve(rendered: str, ingredient: str) -> str:
    """Key-match retrieval: the goal's own entry, else the generic skill."""
    sections = {}
    current = None
    for line in rendered.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    exact = sections.get(f"skill {ingredient}")
    generic = sections.get("skill generic")
    chosen = exact if exact is not None else generic
    return "\n".join(chosen) if chosen else "{}"


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        ingredient = str(task.meta["ingredient"])
        return llm(
            (
                "You are Voyager's action agent in a deterministic crafting "
                "world. Use the retrieved executable skill and adapt "
                "placeholders to the visible ingredient. Available primitives: "
                f"{PRIMITIVES}.\n\n"
                f"Goal: {task.prompt}\nVisible ingredient: {ingredient}\n"
                f"Retrieved skill: {_retrieve(rendered, ingredient)}\n\n"
                'Return JSON only as {"actions": ["primitive:argument", ...]}.'
            ),
            unit=task.id,
        )

    def reward(task: Task, output: str) -> float:
        try:
            actions = _action_list(output, "actions")
        except ValueError:
            return 0.0
        ok, _ = simulate(actions, str(task.meta["ingredient"]))
        return float(ok)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        ingredient = str(task.meta["ingredient"])
        try:
            actions = _action_list(output, "actions")
            _, env_feedback = simulate(actions, ingredient)
        except ValueError:
            env_feedback = (
                "the environment rejected the attempt: it was not a JSON "
                f"action list. Available primitives: {PRIMITIVES}."
            )
        raw = llm(
            (
                "Voyager repairs executable programs from environment errors. "
                "Create a reusable skill with {ingredient} as a placeholder.\n\n"
                f"Goal: {task.prompt}\nAttempt: {output[:500]}\nReward: {score}\n"
                f"Environment feedback: {env_feedback}\n\n"
                'Return JSON only as {"steps": ["primitive:argument", ...]}.'
            ),
            unit=task.id,
        )
        # The reusable placeholder skill evolves under the generic key --
        # held-out goals have disjoint ingredients, so only a placeholder
        # skill can pass the gate. Per-goal keys remain for specialization.
        return f"skill generic: {raw}"

    train, held_out, test = _split(seed)
    categories = ["skill generic"] + [f"skill {i}" for i in INGREDIENTS]
    return MethodPolicy(
        name="voyager",
        fidelity=FIDELITY,
        notes=(
            "Skills accumulate under goal keys and are never overwritten, matching the add-only library.",
            "Repair sees the environment's first-failure feedback, never a gold trace.",
            "The critic is the engine's self-verify rollout: proposals are re-run and judged by the environment reward.",
            "A deterministic crafting world replaces Minecraft; the task pool plus difficulty sampling stands in for the generative curriculum.",
        ),
        strategy=SkillLibrary(
            categories=categories,
            value_validator=_skill_value,
            initial_entries={
                "skill generic": canonical_json(
                    {"steps": ["collect:water", "collect:{ingredient}",
                               "serve:drink"]}),
            },
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=reward,
        proposal_calls_per_candidate=1,
        engine=Policies(task_sampler=DifficultyWeighted()),
        reflective=False,
        self_verify=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
