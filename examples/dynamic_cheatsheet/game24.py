"""A vendored, fully offline Game-of-24 slice for the Dynamic Cheatsheet port.

Game of 24 is the task on which the Dynamic Cheatsheet paper shows its largest
effect: the model discovers that brute-forcing the puzzle in Python beats mental
arithmetic, writes that down in its cheatsheet, and reuses it on later puzzles.
This module vendors 36 puzzles (four integers in 1..13, each solvable) so CI
needs no dataset and no network, and grades the way upstream's
``eval_for_GameOf24`` does -- the expression must use each given number exactly
once and evaluate to 24 -- except that evaluation is a small exact-``Fraction``
AST walk instead of ``eval``.

**No gold answer is needed to grade**, and none is shown to a method. Each task
also carries ``meta["reference"]``, one solution found by exhaustive search; it
exists only so a test can assert that no prompt ever contains it. Scoring reads
the reply, never the reference.
"""

from __future__ import annotations

import ast
import itertools
import random
import re
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from agentdescent.evolution import Task

TARGET = 24

#: 36 four-number puzzles, drawn once (seed 292) from quadruples in 1..13 that
#: have between 1 and 12 distinct exact solutions -- hard enough that a bare
#: answer is often wrong, all solvable. `test_every_puzzle_is_solvable` re-derives
#: that independently.
PUZZLES: Tuple[Tuple[int, int, int, int], ...] = (
    (5, 6, 9, 10), (4, 7, 11, 12), (2, 11, 12, 13), (1, 4, 6, 10),
    (3, 4, 5, 13), (5, 9, 9, 12), (1, 7, 13, 13), (2, 8, 12, 13),
    (3, 5, 5, 7), (3, 12, 12, 13), (5, 6, 7, 9), (1, 3, 4, 6),
    (6, 11, 12, 13), (2, 7, 8, 8), (3, 6, 6, 7), (1, 3, 3, 6),
    (4, 5, 10, 10), (4, 4, 5, 6), (1, 12, 12, 12), (6, 10, 10, 13),
    (1, 4, 4, 9), (6, 10, 12, 13), (4, 5, 8, 10), (5, 5, 6, 11),
    (1, 6, 8, 13), (3, 5, 12, 13), (3, 3, 8, 10), (3, 6, 8, 12),
    (2, 2, 12, 13), (5, 6, 7, 12), (1, 2, 2, 7), (3, 8, 11, 12),
    (2, 4, 5, 10), (3, 6, 9, 11), (1, 4, 5, 10), (1, 2, 6, 13),
)

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def solve_exact(numbers: Sequence[int]) -> Optional[str]:
    """One expression over every number, exactly once, equal to 24 -- or None."""
    def go(items):
        if len(items) == 1:
            return items[0][1] if items[0][0] == TARGET else None
        for i, j in itertools.combinations(range(len(items)), 2):
            (a, ea), (b, eb) = items[i], items[j]
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            options = [(a + b, f"({ea} + {eb})"), (a * b, f"({ea} * {eb})"),
                       (a - b, f"({ea} - {eb})"), (b - a, f"({eb} - {ea})")]
            if b != 0:
                options.append((a / b, f"({ea} / {eb})"))
            if a != 0:
                options.append((b / a, f"({eb} / {ea})"))
            for value, text in options:
                found = go(rest + [(value, text)])
                if found is not None:
                    return found
        return None

    return go([(Fraction(n), str(n)) for n in numbers])


def evaluate(expression: str) -> Tuple[Fraction, List[int]]:
    """Exact value and the integer literals used. Raises ``ValueError``.

    Only ``+ - * /``, parentheses and non-negative integer literals are allowed;
    anything else (names, calls, powers, unary minus on a group) is rejected, so
    a model reply can never reach ``eval``.
    """
    text = (expression.replace("\u00d7", "*").replace("\u00f7", "/")
            .replace("x", "*").strip())
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("not an expression") from exc
    literals: List[int] = []

    def walk(node) -> Fraction:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if (isinstance(node, ast.Constant) and type(node.value) is int
                and node.value >= 0):
            literals.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    return walk(tree), literals


def parse_expression(response: str) -> Optional[str]:
    """The last ``<answer>`` block, left of any ``=``; None when absent."""
    blocks = _ANSWER.findall(response or "")
    if not blocks:
        return None
    body = blocks[-1].strip()
    if "=" in body:
        body = body.split("=")[0]
    return body.strip() or None


def score_answer(numbers: Sequence[int], response: str) -> float:
    expression = parse_expression(response)
    if expression is None:
        return 0.0
    try:
        value, used = evaluate(expression)
    except (ValueError, RecursionError):
        return 0.0
    if Counter(used) != Counter(numbers) or value != TARGET:
        return 0.0
    return 1.0


def numbers_of(task: Task) -> List[int]:
    return [int(part) for part in task.prompt.split()]


def game24_reward(task: Task, output: str) -> float:
    return score_answer(numbers_of(task), output)


def _task(numbers: Tuple[int, ...], split: str, index: int) -> Task:
    reference = solve_exact(numbers)
    assert reference is not None, numbers
    return Task(
        id=f"{split}:g{index:02d}",
        prompt=" ".join(str(n) for n in numbers),
        meta={"split": split, "reference": reference},
    )


def game24_splits(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    """Disjoint 12 / 12 / 12 train, held-out and test splits for one seed."""
    rows = list(PUZZLES)
    random.Random(seed).shuffle(rows)
    third = len(rows) // 3
    names = ("train", "held-out", "test")
    return tuple(  # type: ignore[return-value]
        [_task(numbers, name, index)
         for index, numbers in enumerate(rows[k * third:(k + 1) * third])]
        for k, name in enumerate(names)
    )


def reference_answers() -> Dict[str, str]:
    """Every reference solution, for the leak tests only."""
    return {" ".join(map(str, p)): solve_exact(p) or "" for p in PUZZLES}
