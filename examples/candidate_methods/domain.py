"""Deterministic held-out domain shared by the candidate-method experiments."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MoneyTask:
    id: str
    question: str
    answer_cents: str


# The questions deliberately ask for an amount without revealing the evaluator's
# integer-cents convention. Evolution sees that convention only through training
# feedback; validation and test answers are never inserted into candidate prompts.
TASKS: Tuple[MoneyTask, ...] = (
    MoneyTask("m00", "A notebook costs $2.75 and a pen costs $1.40. What is the total amount?", "415"),
    MoneyTask("m01", "A ticket costs $12.50 and a coupon removes $3.25. What amount remains?", "925"),
    MoneyTask("m02", "Three labels cost $1.20 each and tape costs $0.55. What is the total amount?", "415"),
    MoneyTask("m03", "A meal is $8.99 and a drink is $2.01. What is the total amount?", "1100"),
    MoneyTask("m04", "Two people split a $5.40 charge equally. What amount does each person pay?", "270"),
    MoneyTask("m05", "Add a 15 percent tip to a $20.00 bill. What is the final amount?", "2300"),
    MoneyTask("m06", "A sandwich is $4.75 and two juices are $1.10 each. What is the total amount?", "695"),
    MoneyTask("m07", "You pay $10.00 for an item costing $2.35. What amount of change is due?", "765"),
    MoneyTask("m08", "Six postcards cost $0.85 each. What is the total amount?", "510"),
    MoneyTask("m09", "An $18.75 item has 8 percent tax. What is the final amount?", "2025"),
    MoneyTask("m10", "A $7.50 item is discounted by 20 percent. What is the sale amount?", "600"),
    MoneyTask("m11", "Four friends split a $13.20 bill equally. What amount does each pay?", "330"),
)


def split_tasks(seed: int) -> Tuple[List[MoneyTask], List[MoneyTask], List[MoneyTask]]:
    """Return disjoint feedback, validation, and test splits for one seed."""
    rows = list(TASKS)
    random.Random(seed).shuffle(rows)
    return rows[:4], rows[4:8], rows[8:12]


_FINAL_INTEGER = re.compile(
    r"\A\s*(?:(?:final\s+answer|answer)\s*:\s*)?([+-]?\d+)\s*[.!]?\s*\Z",
    re.IGNORECASE,
)


def parse_integer_answer(response: str) -> Optional[str]:
    """Accept a bare integer, while rejecting dollars, decimals, and prose."""
    match = _FINAL_INTEGER.fullmatch(response)
    return match.group(1) if match else None


def score_answer(task: MoneyTask, response: str) -> float:
    return float(parse_integer_answer(response) == task.answer_cents)


def training_feedback(task: MoneyTask, response: str) -> str:
    parsed = parse_integer_answer(response)
    shown = parsed if parsed is not None else response.strip()[:120]
    return (
        f"Question: {task.question}\n"
        f"Candidate answer: {shown!r}\n"
        f"Strict evaluator answer: {task.answer_cents!r}\n"
        "The evaluator uses one reusable output convention across the domain. "
        "Infer that convention and improve the agent without memorizing this item."
    )


def accuracy(tasks: Sequence[MoneyTask], responses: Sequence[str]) -> float:
    if len(tasks) != len(responses):
        raise ValueError("tasks and responses must have equal length")
    if not tasks:
        return 0.0
    return sum(score_answer(task, response) for task, response in zip(tasks, responses)) / len(tasks)
