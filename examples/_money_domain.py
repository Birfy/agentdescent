"""Deterministic money domain shared by the candidate-method experiments.

One grader. ``parse_integer_answer`` reads the **final line** of a reply and
accepts a bare integer or an ``Answer: <int>`` line there, rejecting dollars,
decimals, and prose; every port that scores against this domain scores through
:func:`score_answer`, and the offline tests exercise the same function the live
runs use.

*Final line*, not the whole reply, and the difference is the domain's ceiling.
The grader used to ``fullmatch`` the entire response, which made the output
convention and the reasoning **mutually exclusive**: a prompt that satisfied the
format had to forbid the working, and without the working the model's arithmetic
collapsed. Measured on `deepseek-v4-flash` over all 48 items, same model, same
convention:

===================================  =====
prompt / grader                      score
===================================  =====
no working, whole-reply grader       0.562
with working, whole-reply grader     0.000   <- the working itself failed
with working, final-line grader      0.979
===================================  =====

The middle row is the one that matters: a *correct* answer scored zero because it
was shown its arithmetic. That is not difficulty, it is a grader that forbids a
chain of thought -- and the benchmarks these ports come from do the opposite,
GSM8K extracting the final number after its ``####`` marker. What the port still
has to discover is unchanged: that the amount is an **integer number of cents**,
stated nowhere in the problems.
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
#
# 48 items in ten arithmetic families -- sum, difference, n-of-one-plus-one,
# equal split, percent tip, mixed multi-item, change due, n-times-price, percent
# tax, percent discount. The size is not decoration: `run_port` refuses a run
# whose train split is smaller than the worker count, so the original twelve
# capped every port in this family at four workers and put the acceptance gate on
# four held-out items, where one arithmetic slip moves the score by 0.25.
#
# Nor is the domain a single on/off convention. Measured on `deepseek-v4-flash`:
# the starting instruction scores **0.000** -- every answer comes back as
# `$4.15` -- but simply stating the convention outright only reaches **0.417**,
# because forcing a bare integer surfaces real arithmetic errors (`27` for a
# $2.70 half-split, `1320` for a bill that still needed dividing). A prompt has
# to teach the convention *and* the working to clear the domain.
#
# The 36 items after `m11` were generated and each answer was re-derived from the
# dollar figures its own question prints, using Decimal rather than the integer
# cents the table was built in. A wrong answer here does not raise -- it silently
# becomes an unsolvable task and reads as a method that failed to learn.
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
    MoneyTask("m12", "A stapler costs $3.40 and a ruler costs $1.25. What is the total amount?", "465"),
    MoneyTask("m13", "A mug costs $6.15 and a coaster costs $2.85. What is the total amount?", "900"),
    MoneyTask("m14", "A folder costs $2.60 and a clip pack costs $1.75. What is the total amount?", "435"),
    MoneyTask("m15", "A lamp costs $18.45 and a bulb costs $6.55. What is the total amount?", "2500"),
    MoneyTask("m16", "A jacket costs $45.20 and a coupon removes $12.75. What amount remains?", "3245"),
    MoneyTask("m17", "A book costs $18.90 and a coupon removes $4.45. What amount remains?", "1445"),
    MoneyTask("m18", "A pass costs $30.50 and a coupon removes $9.95. What amount remains?", "2055"),
    MoneyTask("m19", "4 brackets cost $1.45 each and a charger costs $8.90. What is the total amount?", "1470"),
    MoneyTask("m20", "5 envelopes cost $0.60 each and a stamp roll costs $7.25. What is the total amount?", "1025"),
    MoneyTask("m21", "3 cloths cost $2.35 each and a bucket costs $4.15. What is the total amount?", "1120"),
    MoneyTask("m22", "3 people split a $12.45 charge equally. What amount does each person pay?", "415"),
    MoneyTask("m23", "5 people split a $13.20 charge equally. What amount does each person pay?", "264"),
    MoneyTask("m24", "6 people split a $9.30 charge equally. What amount does each person pay?", "155"),
    MoneyTask("m25", "4 people split a $49.00 charge equally. What amount does each person pay?", "1225"),
    MoneyTask("m26", "Add a 20 percent tip to a $24.00 bill. What is the final amount?", "2880"),
    MoneyTask("m27", "Add a 10 percent tip to a $16.50 bill. What is the final amount?", "1815"),
    MoneyTask("m28", "Add a 25 percent tip to a $36.00 bill. What is the final amount?", "4500"),
    MoneyTask("m29", "Add a 40 percent tip to an $8.50 bill. What is the final amount?", "1190"),
    MoneyTask("m30", "A salad is $7.25 and 3 rolls are $1.40 each. What is the total amount?", "1145"),
    MoneyTask("m31", "A kit is $12.99 and 2 filters are $3.50 each. What is the total amount?", "1999"),
    MoneyTask("m32", "A frame is $8.99 and 4 hooks are $0.75 each. What is the total amount?", "1199"),
    MoneyTask("m33", "You pay $50.00 for an item costing $17.85. What amount of change is due?", "3215"),
    MoneyTask("m34", "You pay $20.00 for an item costing $6.45. What amount of change is due?", "1355"),
    MoneyTask("m35", "You pay $100.00 for an item costing $37.99. What amount of change is due?", "6201"),
    MoneyTask("m36", "7 markers cost $1.35 each. What is the total amount?", "945"),
    MoneyTask("m37", "9 tags cost $0.40 each. What is the total amount?", "360"),
    MoneyTask("m38", "12 screws cost $0.25 each. What is the total amount?", "300"),
    MoneyTask("m39", "8 ribbons cost $1.75 each. What is the total amount?", "1400"),
    MoneyTask("m40", "A $25.00 item has 8 percent tax. What is the final amount?", "2700"),
    MoneyTask("m41", "A $40.00 item has 6 percent tax. What is the final amount?", "4240"),
    MoneyTask("m42", "A $12.50 item has 4 percent tax. What is the final amount?", "1300"),
    MoneyTask("m43", "A $75.00 item has 12 percent tax. What is the final amount?", "8400"),
    MoneyTask("m44", "A $32.00 item is discounted by 25 percent. What is the sale amount?", "2400"),
    MoneyTask("m45", "A $9.00 item is discounted by 30 percent. What is the sale amount?", "630"),
    MoneyTask("m46", "A $45.00 item is discounted by 15 percent. What is the sale amount?", "3825"),
    MoneyTask("m47", "A $16.00 item is discounted by 75 percent. What is the sale amount?", "400"),
)


_FINAL_INTEGER = re.compile(
    r"\A\s*(?:(?:final\s+answer|answer)\s*:\s*)?([+-]?\d+)\s*[.!]?\s*\Z",
    re.IGNORECASE,
)


def parse_integer_answer(response: str) -> Optional[str]:
    """The reply's **final line**, as a bare integer -- or None.

    Dollars, decimals and prose are still rejected, on that line: `$4.15`,
    `4.15` and `about 415 cents` are all None, so the integer-cents convention is
    still something a method has to discover. What is no longer rejected is a
    reply that *shows its arithmetic* before answering.

    Only the last non-empty line is read. A reply that ends mid-working -- the
    common failure, `18.90 - 4.45 = 14.45` -- has no bare integer there and
    scores zero, as it should.
    """
    lines = [line for line in (response or "").strip().splitlines() if line.strip()]
    if not lines:
        return None
    match = _FINAL_INTEGER.fullmatch(lines[-1])
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
    third = len(rows) // 3
    return (
        [money_task(task, split="train") for task in rows[:third]],
        [money_task(task, split="held-out") for task in rows[third:2 * third]],
        [money_task(task, split="test") for task in rows[2 * third:3 * third]],
    )


def score_answer(expected_cents: str, response: str) -> float:
    return float(parse_integer_answer(response) == expected_cents)


def money_reward(task: Task, output: str) -> float:
    return score_answer(str(task.meta["answer_cents"]), output)


def feedback(task: Task, output: str) -> str:
    """What the grader read, and what it wanted -- not what the answer is.

    The middle line says *the final line* on purpose. The grader used to match
    the whole reply and now matches only the last one, and feedback that still
    implies the whole reply is read is feedback that is false: every method
    evolving against it wrote prompts forbidding explanation, which is the one
    thing that costs this domain its arithmetic. Same measurement, same model,
    all 48 items -- a prompt that only fixes the format reaches 0.479, one that
    also works the arithmetic out reaches 1.000.

    What still has to be discovered is untouched: that the amount is an integer
    number of *cents*. Only the shape of the channel is described.
    """
    text = output.strip()
    read = [line for line in text.splitlines() if line.strip()]
    read = read[-1].strip() if read else ""
    return (
        f"Question: {task.prompt}\n"
        f"The evaluator read the final line of your reply: {read[:120]!r}\n"
        f"It wanted: {str(task.meta['answer_cents'])!r}\n"
        "Only that last line is graded, so anything before it is free. The "
        "evaluator uses one reusable output convention across the domain; infer "
        "that convention without memorizing this item."
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
