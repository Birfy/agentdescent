"""Deterministic money domain shared by the candidate-method experiments.

One grader. ``parse_integer_answer`` accepts a bare integer or an
``Answer: <int>`` line and rejects dollars, decimals, and prose; every port that
scores against this domain scores through :func:`score_answer`, and the offline
tests exercise the same function the live runs use.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from agentdescent.evolution import Task


STARTING_INSTRUCTION = (
    "Solve the user's arithmetic problem carefully. Return only the final answer."
)


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


_FINAL_INTEGER = re.compile(
    r"\A\s*(?:(?:final\s+answer|answer)\s*:\s*)?([+-]?\d+)\s*[.!]?\s*\Z",
    re.IGNORECASE,
)


def parse_integer_answer(response: str) -> Optional[str]:
    """Accept a bare integer, while rejecting dollars, decimals, and prose."""
    match = _FINAL_INTEGER.fullmatch(response)
    return match.group(1) if match else None


def money_task(task: MoneyTask, *, split: str) -> Task:
    return Task(
        id=f"{split}:{task.id}",
        prompt=task.question,
        meta={"answer_cents": task.answer_cents, "split": split},
    )


def money_splits(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    """Disjoint train / held-out / test splits for one seed."""
    rows = list(TASKS)
    random.Random(seed).shuffle(rows)
    return (
        [money_task(task, split="train") for task in rows[:4]],
        [money_task(task, split="held-out") for task in rows[4:8]],
        [money_task(task, split="test") for task in rows[8:12]],
    )


def score_answer(expected_cents: str, response: str) -> float:
    return float(parse_integer_answer(response) == expected_cents)


def money_reward(task: Task, output: str) -> float:
    return score_answer(str(task.meta["answer_cents"]), output)


def feedback(task: Task, output: str) -> str:
    parsed = parse_integer_answer(output.strip())
    shown = parsed if parsed is not None else output.strip()[:120]
    return (
        f"Question: {task.prompt}\n"
        f"Candidate answer: {shown!r}\n"
        f"Strict evaluator answer: {str(task.meta['answer_cents'])!r}\n"
        "The evaluator uses one reusable output convention across the domain. "
        "Infer that convention without memorizing this item."
    )


def solve_money(llm, instruction: str, task: Task, *, subphase: str = "") -> str:
    return llm(
        f"System instruction:\n{instruction}\n\nUser problem:\n{task.prompt}",
        subphase=subphase,
        unit=task.id,
    )


def accuracy(expected: Sequence[str], responses: Sequence[str]) -> float:
    if len(expected) != len(responses):
        raise ValueError("expected and responses must have equal length")
    if not expected:
        return 0.0
    return sum(
        score_answer(answer, response)
        for answer, response in zip(expected, responses)
    ) / len(expected)
