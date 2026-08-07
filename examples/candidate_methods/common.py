"""Shared artifact and task helpers for the AgentDescent candidate ports."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

from agentdescent.evolution import Task

from .domain import MoneyTask, split_tasks
from .runtime import Recorder, parse_json_object


STARTING_INSTRUCTION = (
    "Solve the user's arithmetic problem carefully. Return only the final answer."
)


def artifact_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def clip_instruction(value: Any, *, fallback: str = STARTING_INSTRUCTION) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip()
    if not cleaned or len(cleaned) > 900:
        return fallback
    return cleaned


def canonical_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def instruction_validator(text: str) -> str:
    return clip_instruction(text, fallback="")


def money_task(task: MoneyTask, *, split: str) -> Task:
    return Task(
        id=f"{split}:{task.id}",
        prompt=task.question,
        meta={"answer_cents": task.answer_cents, "split": split},
    )


def money_splits(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    train, held_out, test = split_tasks(seed)
    return (
        [money_task(task, split="train") for task in train],
        [money_task(task, split="held-out") for task in held_out],
        [money_task(task, split="test") for task in test],
    )


def money_reward(task: Task, output: str) -> float:
    return float(output.strip().rstrip(".!") == str(task.meta["answer_cents"]))


def solve_money(
    recorder: Recorder,
    instruction: str,
    task: Task,
    *,
    phase: str,
) -> str:
    return recorder.call(
        f"System instruction:\n{instruction}\n\nUser problem:\n{task.prompt}",
        phase=phase,
        unit=task.id,
    )


def feedback(task: Task, output: str) -> str:
    return (
        f"Question: {task.prompt}\n"
        f"Candidate answer: {output.strip()[:120]!r}\n"
        f"Strict evaluator answer: {str(task.meta['answer_cents'])!r}\n"
        "The evaluator uses one reusable output convention across the domain. "
        "Infer that convention without memorizing this item."
    )


def parse_json_fields(text: str, fields: Sequence[str]) -> Dict[str, str]:
    payload = parse_json_object(text)
    result = {field: clip_instruction(payload.get(field), fallback="") for field in fields}
    if not all(result.values()):
        raise ValueError("model response omitted required string fields")
    return result
