"""Environment analogue: **Voyager** (MineDojo/Voyager@55e45a88).

Preserved: a skill is added **only when the critic says the task succeeded**
(`voyager.py`: ``if info["success"]: self.skill_manager.add_new_skill(info)``);
repair driven by **environment feedback** -- the deterministic world executes
the attempt and reports the first failed step, which is what the repair prompt
sees (never a gold trace); the **critic** as the engine's worker self-check
(`self_verify`): every proposal is re-rolled and judged by the environment
reward before it reaches the gate, upstream's "judge success from the
environment, not the agent's claim"; and frontier task focus via the
:class:`~agentdescent.sampling.DifficultyWeighted` sampler.

**The library is not add-only, and this port used to say it was.**
`SkillManager.add_new_skill` prints *"Skill {name} already exists. Rewriting!"*,
deletes the vector-store entry and reassigns ``self.skills[program_name]``. The
older code is dumped to disk as ``{name}V2.js`` and **retrieval never reads
it**: ``retrieve_skills`` returns ``self.skills[...]["code"]``, so the live
library holds exactly one version per name -- the newest. Versioned-on-disk and
overwritten-in-memory are different libraries, and only the second one is the
algorithm. Each goal key here therefore holds the latest accepted skill rather
than an accumulation.

Boundaries: a deterministic crafting world replaces Minecraft; key-match
retrieval replaces embedding retrieval over descriptions (``retrieval_top_k=5``,
so upstream shows the action agent several skills and this shows one); the task
pool plus difficulty sampling is an analogue of the generative curriculum, which
proposes novel tasks.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Sequence, Tuple

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import canonical_json, parse_json_object
from examples._method_policy import MethodPolicy, SkillLibrary
from examples._method_runner import standard_main


FIDELITY = "environment_analogue"

#: 48, not 12, and the size is a constraint rather than a garnish: `run_port`
#: refuses a run whose train split is smaller than the worker count, so twelve
#: goals split 4/4/4 capped this port at four workers with a four-goal gate.
INGREDIENTS = (
    "mint", "berry", "lemon", "ginger", "apple", "peach",
    "pear", "plum", "orange", "basil", "cherry", "melon",
    "thyme", "sage", "clove", "fennel", "anise", "cocoa",
    "papaya", "guava", "lychee", "quince", "damson", "medlar",
    "nutmeg", "cassia", "sorrel", "borage", "hyssop", "lovage",
    "banana", "mango", "grape", "fig", "date", "apricot",
    "rosehip", "elder", "juniper", "lavender", "chamomile", "verbena",
    "tamarind", "persimmon", "kumquat", "pomelo", "cranberry", "blackcurrant",
)

PRIMITIVES = "sanitize, collect, heat, combine, serve (as 'primitive:argument')"


#: What each required step needs, beyond its verb. A step whose argument the
#: world never names -- the *vessel* that gets sanitized, the *drink* that gets
#: served -- is matched on the verb alone; a step whose argument is derivable
#: from the goal must contain it.
#:
#: Measured against string equality, both unnamed arguments were lotteries. The
#: solver discovered all three missing steps and still scored zero, writing
#: `sanitize:water` / `sanitize:berry` (never `vessel`, a word that appears
#: nowhere it can see) and `combine:berry:water` / `combine:water_clove` /
#: `combine:thyme_water` (never `water+berry`). Three seeds, 0.000 each,
#: `accepted=0/80`, and **no invalid proposals** -- well-formed skills against an
#: unreachable target.
_STEPS = (
    ("sanitize", ()),               # of what, the world does not say
    ("collect", ("water",)),
    ("collect", ("{ingredient}",)),
    ("heat", ("water",)),
    ("combine", ("water", "{ingredient}")),
    ("serve", ()),                  # what is served, the world does not say
)


def _required(ingredient: str) -> List[str]:
    """The canonical rendering of the required sequence, for display and tests."""
    out = []
    for verb, terms in _STEPS:
        filled = [t.replace("{ingredient}", ingredient) for t in terms]
        out.append(f"{verb}:{'+'.join(filled)}" if filled else f"{verb}:vessel"
                   if verb == "sanitize" else f"{verb}:drink" if not filled
                   else f"{verb}:{'+'.join(filled)}")
    return [item.lower() for item in out]


def _action_list(text: str, key: str) -> List[str]:
    payload = parse_json_object(text)
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"{key} must be a non-empty bounded list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must contain strings")
    return [item.strip().lower() for item in value]


def _matches(action: str, index: int, ingredient: str) -> bool:
    """Does this action perform the required step at `index`?

    Verb plus *content*, not string equality -- see `_STEPS` for why. What still
    has to be discovered is the **sequence**, which is the skill the paper is
    about; the spelling of an argument the world never names is not.
    """
    verb, _, argument = action.partition(":")
    want_verb, terms = _STEPS[index]
    if verb.strip() != want_verb:
        return False
    return all(term.replace("{ingredient}", ingredient) in argument
               for term in terms)


def simulate(actions: Sequence[str], ingredient: str) -> Tuple[bool, str]:
    """Execute an action sequence; report the first unmet step, Voyager-style.

    The message names what failed and what the world observed -- not the
    required program.
    """
    required = _required(ingredient)
    cursor = 0
    for action in actions:
        if cursor < len(required) and _matches(action, cursor, ingredient):
            cursor += 1
    if cursor == len(required):
        return True, "goal reached: drink served"
    verb = _STEPS[cursor][0]
    # Whether the verb is *absent* or merely *late* is the difference between
    # "add a step" and "reorder". Saying only "a 'collect' step never happened"
    # to an agent that wrote two of them is not a terse error, it is a wrong
    # one -- and the agent does the only thing it says, which is add a third.
    #
    # Measured: the solver had already discovered all six actions and was
    # writing `[collect, collect, sanitize, heat, combine, serve]` against a
    # world that wants sanitize first. Three seeds, 0.000, `accepted=0/80`.
    # Naming the ordering keeps the rule the docstring claims -- no gold trace,
    # nothing about the steps still to come -- while describing what happened.
    written = [a.partition(":")[0].strip() for a in actions]
    late = written.count(verb) > sum(
        1 for i in range(cursor) if _STEPS[i][0] == verb)
    if late:
        detail = (f"a '{verb}' step is present but out of order: the "
                  f"environment reached it after steps it has to come before")
    else:
        detail = (f"the environment expected a '{verb}' step that never "
                  f"happened")
    return False, (
        f"execution stopped before '{verb}' succeeded: {detail} (progress "
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
    third = len(rows) // 3
    return rows[:third], rows[third:2 * third], rows[2 * third:3 * third]


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
            "A skill is added only when the environment judges the attempt successful, as add_new_skill is called only under info['success'].",
            "A goal key holds the latest accepted skill, not an accumulation: add_new_skill rewrites self.skills in place and retrieve_skills reads that, so upstream's on-disk V2 dumps are never retrieved.",
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
