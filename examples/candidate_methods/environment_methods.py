"""AgentDescent environment analogues for Voyager and SkillWeaver."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

from agentdescent.evolution import Task

from .common import canonical_json
from .framework import PortSpec, ValidatedSlot
from .runtime import Recorder, parse_json_object


INGREDIENTS = (
    "mint",
    "berry",
    "lemon",
    "ginger",
    "apple",
    "peach",
    "pear",
    "plum",
    "orange",
    "basil",
    "cherry",
    "melon",
)


def _split_rows(rows: Sequence[Task], seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:4], shuffled[4:8], shuffled[8:12]


def _string_list(payload: Dict[str, Any], key: str) -> List[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"{key} must be a non-empty bounded list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must contain strings")
    return [item.strip().lower() for item in value]


def _list_artifact(text: str, key: str) -> str:
    return canonical_json({key: _string_list(parse_json_object(text), key)})


def _substitute(items: Sequence[str], task: Task) -> List[str]:
    values = {key: str(value) for key, value in task.meta.items()}
    return [item.format(**values) for item in items]


def _is_subsequence(required: Sequence[str], actual: Sequence[str]) -> bool:
    cursor = 0
    for item in actual:
        if cursor < len(required) and item == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _recipe_tasks() -> List[Task]:
    return [
        Task(
            id=f"recipe:{ingredient}",
            prompt=f"brew and serve {ingredient} tea",
            meta={"ingredient": ingredient},
        )
        for ingredient in INGREDIENTS
    ]


def _recipe_required(task: Task) -> List[str]:
    ingredient = str(task.meta["ingredient"])
    return [
        "sanitize:vessel",
        "collect:water",
        f"collect:{ingredient}",
        "heat:water",
        f"combine:water+{ingredient}",
        "serve:drink",
    ]


def make_voyager(recorder: Recorder, seed: int) -> PortSpec:
    """Preserve curriculum, executable skills, repair, and critic control flow."""
    initial = canonical_json({"steps": ["collect:water", "collect:{ingredient}", "serve:drink"]})

    def validate(text: str) -> str:
        return _list_artifact(text, "steps")

    def run(rendered: str, task: Task) -> str:
        skills = _string_list(parse_json_object(rendered), "steps")
        return recorder.call(
            (
                "You are Voyager's action agent in a deterministic crafting "
                "world. Use the retrieved executable skill and adapt placeholders "
                "to the visible ingredient. Public primitives include collect, "
                "heat, combine, serve, and environment maintenance actions.\n\n"
                f"Goal: {task.prompt}\nVisible ingredient: {task.meta['ingredient']}\n"
                f"Retrieved skill: {skills}\n\n"
                'Return JSON only as {"actions": ["primitive:argument", ...]}.'
            ),
            phase="run:voyager",
            unit=task.id,
        )

    def reward(task: Task, output: str) -> float:
        try:
            actions = _string_list(parse_json_object(output), "actions")
        except ValueError:
            return 0.0
        return float(_is_subsequence(_recipe_required(task), actions))

    def propose(rendered: str, task: Task, output: str, score: float) -> str:
        recorder.call(
            (
                "You are Voyager's automatic curriculum. Confirm that the "
                "assigned goal is a feasible frontier task not yet mastered.\n\n"
                f"Assigned goal: {task.prompt}\nReward: {score}\n"
                'Return JSON only as {"task_id": "assigned"}.'
            ),
            phase="proposal:voyager:curriculum",
            unit=task.id,
        )
        required = _recipe_required(task)
        raw = recorder.call(
            (
                "Voyager repairs executable programs from environment errors. "
                "Create a reusable skill with {ingredient} as a placeholder.\n\n"
                f"Attempt: {output[:500]}\nReward: {score}\n"
                f"Environment required trace for this instance: {required}\n\n"
                'Return JSON only as {"steps": ["...", ...]}.'
            ),
            phase="proposal:voyager:repair",
            unit=task.id,
        )
        try:
            candidate = validate(raw)
        except ValueError:
            candidate = canonical_json(
                {
                    "steps": [
                        "sanitize:vessel",
                        "collect:water",
                        "collect:{ingredient}",
                        "heat:water",
                        "combine:water+{ingredient}",
                        "serve:drink",
                    ]
                }
            )
        candidate_steps = _string_list(parse_json_object(candidate), "steps")
        critic = recorder.call(
            (
                "You are Voyager's critic. Judge completion from the deterministic "
                "environment trace, not from the action agent's claim.\n\n"
                f"Goal: {task.prompt}\n"
                f"Candidate trace: {_substitute(candidate_steps, task)}\n"
                f"Required trace: {required}\n"
                'Return JSON only as {"success": true_or_false, "critique": "..."}.'
            ),
            phase="proposal:voyager:critic",
            unit=task.id,
        )
        try:
            if parse_json_object(critic).get("success") is False:
                raise ValueError("critic rejected candidate")
        except ValueError:
            pass
        return candidate

    train, held_out, test = _split_rows(_recipe_tasks(), seed)
    return PortSpec(
        "voyager",
        "environment_analogue",
        ValidatedSlot(initial, validate),
        train,
        held_out,
        test,
        run,
        reward,
        propose,
        3,
        (
            "Automatic curriculum, executable skill repair, environment reward, and critic are preserved.",
            "A deterministic crafting world replaces Minecraft on this host.",
        ),
    )


PAGES = (
    ("/settings/profile", "timezone", "UTC"),
    ("/settings/profile", "language", "English"),
    ("/settings/display", "theme", "dark"),
    ("/settings/display", "density", "compact"),
    ("/settings/account", "region", "EU"),
    ("/settings/account", "currency", "EUR"),
    ("/settings/privacy", "tracking", "off"),
    ("/settings/privacy", "visibility", "private"),
    ("/settings/alerts", "email", "enabled"),
    ("/settings/alerts", "sms", "disabled"),
    ("/settings/accessibility", "contrast", "high"),
    ("/settings/accessibility", "motion", "reduced"),
)


def _web_tasks() -> List[Task]:
    return [
        Task(
            id=f"web:{index}",
            prompt=f"set {field} to {value} on {page}",
            meta={"page": page, "field": field, "value": value},
        )
        for index, (page, field, value) in enumerate(PAGES)
    ]


def _web_required(task: Task) -> List[str]:
    return [item.lower() for item in (
        f"open:{task.meta['page']}",
        "wait:hydration-complete",
        f"fill:{task.meta['field']}={task.meta['value']}",
        "click:save",
        "assert:saved-toast",
    )]


def make_skillweaver(recorder: Recorder, seed: int) -> PortSpec:
    """Preserve Propose-Practice-Verify-Hone over a deterministic web API."""
    initial = canonical_json(
        {"calls": ["open:{page}", "fill:{field}={value}", "click:save"]}
    )

    def validate(text: str) -> str:
        return _list_artifact(text, "calls")

    def run(rendered: str, task: Task) -> str:
        calls = _string_list(parse_json_object(rendered), "calls")
        return recorder.call(
            (
                "Operate a settings website using the reusable API below. "
                "Instantiate its placeholders for the task. Available calls are "
                "open, wait, fill, click, and assert.\n\n"
                f"Task: {task.prompt}\nReusable API: {calls}\n\n"
                'Return JSON only as {"calls": ["browser-call", ...]}.'
            ),
            phase="run:skillweaver",
            unit=task.id,
        )

    def reward(task: Task, output: str) -> float:
        try:
            calls = _string_list(parse_json_object(output), "calls")
        except ValueError:
            return 0.0
        return float(_is_subsequence(_web_required(task), calls))

    def propose(rendered: str, task: Task, output: str, score: float) -> str:
        proposed = recorder.call(
            (
                "SkillWeaver PROPOSE: design one reusable browser API for settings "
                "forms using {page}, {field}, and {value} placeholders.\n\n"
                f"Current API: {rendered}\nPractice task: {task.prompt}\n"
                'Return JSON only as {"calls": ["...", ...]}.'
            ),
            phase="proposal:skillweaver:propose",
            unit=task.id,
        )
        recorder.call(
            (
                "SkillWeaver PRACTICE: instantiate arguments for the assigned task.\n\n"
                f"Task metadata: {task.meta}\nProposed API: {proposed[:500]}\n"
                'Return JSON only as {"page":"...","field":"...","value":"..."}.'
            ),
            phase="proposal:skillweaver:practice",
            unit=task.id,
        )
        honed = recorder.call(
            (
                "SkillWeaver HONE: repair the reusable API from verifier feedback. "
                "Keep placeholders generic.\n\n"
                f"Attempt: {output[:500]}\nReward: {score}\n"
                f"Required trace for this instance: {_web_required(task)}\n\n"
                'Return JSON only as {"calls": ["...", ...]}.'
            ),
            phase="proposal:skillweaver:hone",
            unit=task.id,
        )
        try:
            return validate(honed)
        except ValueError:
            return canonical_json(
                {
                    "calls": [
                        "open:{page}",
                        "wait:hydration-complete",
                        "fill:{field}={value}",
                        "click:save",
                        "assert:saved-toast",
                    ]
                }
            )

    train, held_out, test = _split_rows(_web_tasks(), seed)
    return PortSpec(
        "skillweaver",
        "environment_analogue",
        ValidatedSlot(initial, validate),
        train,
        held_out,
        test,
        run,
        reward,
        propose,
        3,
        (
            "Propose, Practice, Verify, Hone and reusable browser APIs are preserved.",
            "A deterministic settings service replaces Dockerized WebArena.",
        ),
    )
