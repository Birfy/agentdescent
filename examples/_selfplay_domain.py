"""Shared zero-data cart domain for the self-play analogues.

Training rollouts keep the upstream loop: the proposer generates a task and the
solver answers it, verified by a trusted local renderer. **Evaluation does
not**: the held-out and test splits are frozen at build time from the seed, so
the evolved artifact cannot influence its own test set and baseline/final are
measured on the same problems -- solver-only, one deterministic answer each.

Two verifiable task kinds stand in for Absolute Zero's three (induction is the
documented omission): ``deduction`` (compute the cart total) and ``abduction``
(infer the one hidden unit price from the total).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from agentdescent.evolution import Task

from ._measure import canonical_json, parse_json_object
from ._money_domain import parse_integer_answer, score_answer


@dataclass(frozen=True)
class CartTask:
    item_cents: Tuple[int, ...]
    quantities: Tuple[int, ...]
    kind: str = "deduction"          # or "abduction": infer item_cents[0]

    @property
    def total(self) -> int:
        return sum(p * q for p, q in zip(self.item_cents, self.quantities))

    @property
    def question(self) -> str:
        terms = [
            f"{q} type-{i} item{'s' if q != 1 else ''} at ${p / 100:.2f} each"
            for i, (p, q) in enumerate(zip(self.item_cents, self.quantities), 1)
        ]
        if self.kind == "abduction":
            shown = ["? (hidden)"] + [f"${p / 100:.2f}" for p in self.item_cents[1:]]
            return (
                "A cart contains " + ", ".join(
                    f"{q} type-{i} items at {price} each"
                    for i, (q, price) in enumerate(zip(self.quantities, shown), 1))
                + f". The total amount is {self.total} cents. "
                "What is the type-1 unit price in cents?"
            )
        return "A cart contains " + ", ".join(terms) + ". What is the total amount?"

    @property
    def answer(self) -> str:
        return str(self.item_cents[0] if self.kind == "abduction" else self.total)


def validate_generated(payload: Dict[str, Any]) -> CartTask:
    item_cents = payload.get("item_cents")
    quantities = payload.get("quantities")
    if not isinstance(item_cents, list) or not isinstance(quantities, list):
        raise ValueError("item_cents and quantities must be lists")
    if not 2 <= len(item_cents) == len(quantities) <= 8:
        raise ValueError("generated task must have two to eight terms")
    if not all(isinstance(v, int) and 1 <= v <= 20_000 for v in item_cents):
        raise ValueError("item_cents values are invalid")
    if not all(isinstance(v, int) and 1 <= v <= 100 for v in quantities):
        raise ValueError("quantity values are invalid")
    return CartTask(tuple(item_cents), tuple(quantities))


def frozen_cart(rng: random.Random, kind: str) -> CartTask:
    terms = rng.randint(2, 4)
    return CartTask(
        tuple(rng.randint(50, 5000) for _ in range(terms)),
        tuple(rng.randint(1, 12) for _ in range(terms)),
        kind,
    )


#: Self-play training slots, and frozen evaluation carts per split. 16, not 4:
#: `run_port` refuses a run whose train split is smaller than the worker count,
#: so four slots capped all three inference-analogue ports at four workers with
#: a four-cart gate, where one item moves the score by 0.25.
SELFPLAY_SLOTS = 16
FROZEN_PER_SPLIT = 16


def selfplay_splits(seed: int, name: str) -> Tuple[List[Task], List[Task], List[Task]]:
    """`SELFPLAY_SLOTS` self-play training slots; two frozen evaluation splits.

    The evaluation carts are drawn from the seed at build time, so the evolved
    memory cannot shape its own test set -- the property the three ports that
    share this domain each claim, and the reason the carts are generated here
    rather than by the proposer.
    """
    rng = random.Random((seed, name).__repr__())
    train = [
        Task(
            id=f"{name}:slot:{seed + index}",
            prompt="Generate and solve one verifier-grounded latent curriculum item.",
            meta={"kind": "selfplay", "slot": seed + index},
        )
        for index in range(SELFPLAY_SLOTS)
    ]

    def frozen(split: str, count: int) -> List[Task]:
        rows = []
        for index in range(count):
            cart = frozen_cart(rng, "abduction" if index % 2 else "deduction")
            rows.append(Task(
                id=f"{name}:{split}:{index}",
                prompt=cart.question,
                meta={"kind": "frozen", "answer": cart.answer, "split": split},
            ))
        return rows

    return (train, frozen("held-out", FROZEN_PER_SPLIT),
            frozen("test", FROZEN_PER_SPLIT))


def proposer_prompt(role: str, slot: int, memory: str) -> str:
    return (
        f"You are the {role} in a zero-data self-play loop. Propose a novel "
        "money calculation near the solver's capability frontier. A trusted local "
        "renderer creates the word problem and verifies the arithmetic. Use 2 to "
        "8 terms, item_cents 1..20000, quantities 1..100. No benchmark label is "
        "available.\n\n"
        f"Curriculum memory: {memory}\nSlot: {slot}\n"
        'Return JSON only with integer-list "item_cents" and "quantities".'
    )


def solver_prompt(question: str, memory: str) -> str:
    return (
        "Solve the problem. Follow policy memory exactly and return only the "
        f"final answer.\n\nPolicy memory:\n{memory}\n\nProblem:\n{question}"
    )


def trajectory(cart: CartTask, final: str, *, valid: bool) -> str:
    return canonical_json(
        {
            "item_cents": list(cart.item_cents),
            "quantities": list(cart.quantities),
            "kind": cart.kind,
            "final": final.strip()[:120],
            "valid": valid,
        }
    )


def majority_share(finals: Sequence[str]) -> float:
    """`score = max_count / len(results)`: the share agreeing with the majority.

    R-Zero's `question_evaluate/evaluate.py` and Agent0's `generate_results` are
    the same computation, and neither uses ground truth -- both reward a
    generated question the solver is *self-inconsistent* on, which is the whole
    point of a method with no labels for what it just wrote.

    Unparseable replies count as their own distinct answers rather than being
    dropped: a solver that cannot state an answer is not one that agrees with
    itself, and dropping them makes an incoherent batch read as certain.
    """
    counts: Dict[object, int] = {}
    for index, final in enumerate(finals):
        parsed = parse_integer_answer(final.strip())
        key = parsed if parsed is not None else f"__unparsed_{index}"
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / float(len(finals)) if finals else 1.0


def trajectory_reward(_task: Task, output: str) -> float:
    """Verifier-grounded reward with the domain's tolerant integer parse."""
    try:
        payload = parse_json_object(output)
        if payload.get("valid") is not True:
            return 0.0
        cart = validate_generated(payload)
        if payload.get("kind") == "abduction":
            cart = CartTask(cart.item_cents, cart.quantities, "abduction")
        final = str(payload.get("final", ""))
        parsed = parse_integer_answer(final)
        return float(parsed is not None and parsed == cart.answer)
    except ValueError:
        return 0.0


def frozen_reward(task: Task, output: str) -> float:
    return score_answer(str(task.meta["answer"]), output)


def mixed_reward(task: Task, output: str) -> float:
    if task.meta.get("kind") == "frozen":
        return frozen_reward(task, output)
    return trajectory_reward(task, output)
