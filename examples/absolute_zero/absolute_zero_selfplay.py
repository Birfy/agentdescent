"""Inference analogue: **Absolute Zero** (LeapLabTHU/Absolute-Zero-Reasoner@484afa48).

Preserved: one model plays proposer and solver; a trusted local renderer is the
grounded verifier; training rollouts generate their own curriculum. Evaluation
splits are **frozen at build time** from the seed — the evolved memory cannot
influence its own test set. The proposer's learnability signal is surfaced
verbatim: upstream's propose reward is ``1−r̄`` when solvable and 0 at the
extremes (monotone in solve rate, *not* peaked at 0.5, which is why no
difficulty-weighted sampler is attached here), and the update prompt carries
the solver's observed success rate.

Boundaries: verbal policy memory replaces TRR++ weight updates; deduction and
abduction stand in for the paper's three task types (induction is omitted).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from agentdescent.evolution import Task

from examples._measure import parse_json_object
from examples._method_policy import MethodPolicy, ValidatedSlot, clip_text
from examples._method_runner import standard_main
from examples._money_domain import STARTING_INSTRUCTION
from examples._selfplay_domain import (CartTask, mixed_reward, proposer_prompt,
                                       selfplay_splits, solver_prompt,
                                       trajectory, validate_generated)


FIDELITY = "inference_analogue"

_FALLBACK = CartTask((125, 240), (1, 2))


def _memory(text: str) -> str:
    value = clip_text(text)
    if not value:
        raise ValueError("empty policy memory")
    return value


def build(seed: int) -> MethodPolicy:
    solve_rate: Dict[str, float] = {"attempts": 0.0, "solved": 0.0}
    lock = threading.Lock()

    def solve(llm, rendered: str, task: Task) -> str:
        if task.meta.get("kind") == "frozen":
            return llm(solver_prompt(task.prompt, rendered),
                       subphase="solver", unit=task.id)
        slot = int(task.meta["slot"])
        raw = llm(proposer_prompt("Absolute Zero proposer", slot, rendered),
                  subphase="proposer", unit=task.id)
        valid = True
        try:
            cart = validate_generated(parse_json_object(raw))
        except ValueError:
            cart, valid = _FALLBACK, False
        final = llm(solver_prompt(cart.question, rendered),
                    subphase="solver", unit=task.id)
        return trajectory(cart, final, valid=valid)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        with lock:
            solve_rate["attempts"] += 1
            solve_rate["solved"] += score
            rate = solve_rate["solved"] / solve_rate["attempts"]
        learnability = 0.0 if rate in (0.0, 1.0) else 1.0 - rate
        return llm(
            (
                "Absolute Zero updates the same model from grounded verifier "
                "reward. Distill one reusable policy lesson, including the "
                "output representation; omit item-specific numbers.\n\n"
                f"Latest trajectory: {output[:400]}\nLatest reward: {score}\n"
                f"Observed solver success rate r={rate:.2f} "
                f"(propose reward 1-r = {learnability:.2f}; tasks at the "
                "extremes teach nothing).\n"
                f"Current policy memory:\n{rendered}"
            ),
            subphase="update",
            unit=task.id,
        )

    train, held_out, test = selfplay_splits(seed, "absolute_zero")
    return MethodPolicy(
        name="absolute_zero",
        fidelity=FIDELITY,
        notes=(
            "Proposer, solver, grounded verifier, and zero-data curriculum are preserved in training rollouts.",
            "Held-out and test carts are frozen from the seed; the evolved memory cannot shape its own test set.",
            "The propose-reward shape 1-r (zero at both extremes, monotone) is surfaced in the update prompt.",
            "Verbal policy memory replaces TRR++ weight updates; induction tasks are omitted.",
        ),
        strategy=ValidatedSlot(initial_value=STARTING_INSTRUCTION,
                               validator=_memory),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=mixed_reward,
        proposal_calls_per_candidate=1,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
