"""Mechanism microport: **Reflexion** (noahshinn/reflexion@218cf0ef).

Preserved: the memory is **append-only and bounded** -- upstream appends each
verbal reflection and prompts with the last Ω (`memory[-3:]` in the alfworld
runs; the paper bounds Ω to 1-3). :class:`~examples._method_policy.WindowedMemory`
is exactly that shape: every accepted reflection is a new commit-ordered ledger
key, the rendered prompt shows the last three, and parallel workers'
reflections **union-merge** with no ranking evaluation, because appends never
contradict.

Boundaries: upstream retries the same failed instance; the framework's
held-out rerun is the analogue. The equal-budget design requests a reflection
after every rollout, where upstream reflects only on failure.
"""

from __future__ import annotations

from typing import Optional

from agentdescent.evolution import Task

from examples._method_policy import MethodPolicy, WindowedMemory
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, feedback,
                                    money_reward, money_splits, solve_money)


FIDELITY = "mechanism_microport"


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        return solve_money(llm, rendered, task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        return llm(
            (
                "You are Reflexion's verbal reflection module. Convert the failed "
                "trajectory and external evaluator signal into one reusable "
                "episodic memory entry for retry.\n\n"
                f"Current memory:\n{rendered}\n\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return only one reusable lesson; omit item-specific numbers."
            ),
            unit=task.id,
        )

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="reflexion",
        fidelity=FIDELITY,
        notes=(
            "Memory is append-only and rendered as the last three entries, matching upstream's bounded window.",
            "Attempt, external feedback, verbal reflection, and retry map to rollout, proposal, and gate rerun.",
            "A deterministic arithmetic evaluator replaces HotpotQA/ALFWorld.",
        ),
        strategy=WindowedMemory(seed_text=STARTING_INSTRUCTION, window=3),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=money_reward,
        proposal_calls_per_candidate=1,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
