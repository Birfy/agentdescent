"""Mechanism microport: **Self-Refine** (madaan/self-refine@9a206d41).

Preserved: FEEDBACK and REFINE are **separate calls** on the same model, and
the loop stops early when the feedback contains the upstream stop signal (the
gsm runner checks for the literal "it is correct"). On a stop no refinement is
proposed -- the candidate budget entry is spent and produces no diff, which is
why the runner treats the reserved proposal-call count as an upper bound.

Boundary (fundamental, not a detail): upstream refines the *answer* to one
instance; this port refines the *instruction artifact* and the held-out rerun
plays the role of the refined attempt.
"""

from __future__ import annotations

from typing import Optional

from agentdescent.evolution import Task

from examples._method_policy import MethodPolicy, ValidatedSlot, clip_text
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, feedback,
                                    money_reward, money_splits, solve_money)


FIDELITY = "mechanism_microport"

STOP_SIGNAL = "it is correct"


def _instruction(text: str) -> str:
    value = clip_text(text)
    if not value:
        raise ValueError("empty instruction")
    return value


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        return solve_money(llm, rendered, task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        critique = llm(
            (
                "You are Self-Refine's FEEDBACK module. Give specific actionable "
                f"feedback on this attempt. If the answer is already correct, say "
                f'"{STOP_SIGNAL}".\n\n'
                f"Instruction used:\n{rendered}\n\nReward: {reward}\n"
                f"{feedback(task, output)}"
            ),
            subphase="feedback",
            unit=task.id,
        )
        if STOP_SIGNAL in critique.lower():
            return None
        return llm(
            (
                "You are Self-Refine's REFINE module. Apply the feedback to "
                "produce one improved reusable instruction; omit item-specific "
                "numbers.\n\n"
                f"Current instruction:\n{rendered}\n\nFeedback:\n{critique}"
            ),
            subphase="refine",
            unit=task.id,
        )

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="self_refine",
        fidelity=FIDELITY,
        notes=(
            "FEEDBACK and REFINE are separate model calls, and the upstream stop signal ends a refinement early.",
            "GENERATE, FEEDBACK, and REFINE map to rollout, proposal, and held-out rerun.",
            "Upstream refines the answer to one instance; this port refines the instruction artifact.",
        ),
        strategy=ValidatedSlot(initial_value=STARTING_INSTRUCTION,
                               validator=_instruction),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=money_reward,
        proposal_calls_per_candidate=2,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
