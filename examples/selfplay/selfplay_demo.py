#!/usr/bin/env python3
"""Solver-challenger self-play on a cart-arithmetic domain.

A frozen model (or a deterministic rule, ``--dry-run``) is the solver: it holds
a skill that answers cart totals only when the item prices stay within its
capability cap. A second artifact -- the challenger -- owns a task-generation
spec (the price and quantity ranges) and proposes ever-wider frontiers. The
solver's score on the generated tasks rises as its skill learns them; that
first-vs-last rise is ``r0``, the learned objective that drives the challenger.

Run offline, no key, no network:

    python -m examples.selfplay.selfplay_demo --dry-run

With a model for the challenger's propose (the solver skill still deterministic):

    python -m examples.selfplay.selfplay_demo --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass

from agentdescent import (
    Challenger,
    ChallengerConfig,
    SingleSlot,
    Task,
    curriculum_evolve,
    openai_compatible,
)


@dataclass(frozen=True)
class Cart:
    """A deterministic arithmetic task: N items at prices p_i, quantity q_i."""

    prices: tuple
    quantities: tuple

    @property
    def total(self) -> int:
        return sum(p * q for p, q in zip(self.prices, self.quantities))

    @property
    def question(self) -> str:
        terms = ", ".join(
            f"{q} item(s) at ${p / 100:.2f}"
            for p, q in zip(self.prices, self.quantities))
        return f"A cart contains {terms}. What is the total amount in cents?"

    @property
    def answer(self) -> str:
        return str(self.total)


# -- the solver: a capability-capped skill ---------------------------------------

CAP_RE = re.compile(r"cap:(\d+)")


def solver_run(rendered: str, task: Task) -> str:
    cap = int(CAP_RE.search(rendered).group(1)) if CAP_RE.search(rendered) else 0
    prices = task.meta.get("prices", ())
    if prices and max(prices) <= cap:
        return task.meta["answer"]
    return "maybe " + task.meta["answer"]


def solver_reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == str(task.meta["answer"]).strip() else 0.0


def solver_propose(rendered: str, task: Task, output: str, reward: float) -> str:
    if reward >= 0.999:
        return "cap:0"
    prices = task.meta.get("prices", ())
    need = max(prices) if prices else 0
    cap = int(CAP_RE.search(rendered).group(1)) if CAP_RE.search(rendered) else 0
    if need > cap:
        return f"cap:{need + 10}"
    return "cap:1"


# -- the challenger: task generation at an evolving frontier ----------------------

def make_challenger(model=None, spec: str = "200") -> Challenger:
    if model is None:
        def propose(spec_text: str, report: dict) -> str:
            # deterministic rule: follow the measured learnability
            limit = int(spec_text) if str(spec_text).isdigit() else 200
            if report.get("r0", 0.0) > 0.05 and limit < 5000:
                return str(limit * 2)
            return spec_text
    else:
        def propose(spec_text: str, report: dict) -> str:
            try:
                return (model(
                    f"You set the curriculum difficulty; current spec {spec_text!r}, "
                    f"measured learnability r0={report.get('r0', 0.0):.2f} "
                    f"({report.get('learned', 0)} learned, "
                    f"{report.get('stuck', 0)} stuck). Return a single integer "
                    "difficulty limit (a larger one is harder).").strip()
                    or spec_text)
            except Exception:
                return spec_text

    def generate(spec_text: str, seed: int) -> dict:
        rng = random.Random(seed)
        limit = int(spec_text) if str(spec_text).isdigit() else 200
        n = rng.randint(2, 5)
        prices = tuple(rng.randint(10, limit) for _ in range(n))
        quantities = tuple(rng.randint(1, 20) for _ in range(n))
        return {"prices": prices, "quantities": quantities}

    def validate(payload: dict) -> Task:
        prices = payload["prices"]
        quantities = payload["quantities"]
        if not (isinstance(prices, tuple) and isinstance(quantities, tuple)):
            raise ValueError("generated cart must have tuple prices/quantities")
        if not 2 <= len(prices) == len(quantities) <= 5:
            raise ValueError("generated cart must have two to five items")
        if not all(isinstance(p, int) and 1 <= p <= 10000 for p in prices):
            raise ValueError("generated prices are invalid")
        if not all(isinstance(q, int) and 1 <= q <= 100 for q in quantities):
            raise ValueError("generated quantities are invalid")
        cart = Cart(prices, quantities)
        return Task(id=f"g{'-'.join(map(str, prices))}",
                    prompt=cart.question,
                    meta={"answer": cart.answer, "prices": prices})

    return Challenger(generate=generate, validate=validate, propose=propose,
                      spec=spec,
                      config=ChallengerConfig(generated_per_round=4,
                                              frontier_share=0.5,
                                              r0_floor=0.05))


def seed_tasks(n: int = 12) -> list:
    out = []
    for i in range(n):
        cart = Cart((i * 25 + 10,), (2,))
        out.append(Task(id=f"s{i}", prompt=cart.question,
                        meta={"answer": cart.answer, "prices": cart.prices}))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="no model, deterministic propose, offline")
    parser.add_argument("--model", default=None, help="model name for the challenger")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model and not args.dry_run:
        model = openai_compatible(model=args.model)
    else:
        model = None

    result = curriculum_evolve(
        seed_tasks(),
        solver_reward,
        run=solver_run,
        propose=solver_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        challenger=make_challenger(model=model),
        rounds=args.rounds,
        n_workers=args.workers,
        seed=args.seed,
        verbose=True,
    )

    print(f"final skill: {result.rendered}")
    print(f"held-out reward: {result.final_reward:.3f} "
          f"outcomes: {result.outcomes()}")
    if result.curriculum:
        c = result.curriculum
        print(f"self-play: r0={c['r0_mean']:.3f} "
              f"learned={c['learned']} stuck={c['stuck']} "
              f"generated={c['generated_total']} invalid={c['invalid_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
