"""Mechanism microport: **PromptBreeder** (arXiv:2309.16797, no released code).

Preserved (paper, Sections 3-4): a population of units each holding a
task-prompt and a mutation-prompt; **binary tournament** replication (sample
two, mutate the winner); mutation operators drawn per replication event,
including **hyper-mutation** -- the self-referential rewrite of the
mutation-prompt by itself; fitness measured on a training batch.

The population lives in the engine's selection seam: :class:`BinaryTournament`
is PromptBreeder's Algorithm-1 parent rule as a ~20-line
:class:`~agentdescent.selection.SelectionPolicy`. The genome is a
:class:`~examples._method_policy.FieldSlots` artifact, so the two prompts are
separate plain-text ledger keys -- edits to different fields union-merge, and
contested fields are model-merged, with no ranking evaluation either way.

Boundaries: population and batch are budget-sized (paper: 50 units, batch 100);
three of the nine mutation operators (zero-order, first-order, hyper-mutation);
no few-shot workings-out in the unit.
"""

from __future__ import annotations

import itertools
import random
import threading
from typing import Optional, Sequence

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.selection import SelectionContext, SingleHead

from examples._measure import parse_json_object
from examples._method_policy import FieldSlots, MethodPolicy, read_fields
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, feedback,
                                    money_reward, money_splits, solve_money)


FIDELITY = "mechanism_microport"

MUTATION_PROMPT = (
    "Infer a reusable instruction from evaluator feedback and improve "
    "this mutation instruction too."
)


class BinaryTournament(SingleHead):
    """PromptBreeder's replication rule: sample two units, breed the winner.

    An unscored candidate wins its tournament (same optimistic rationale as
    :class:`~agentdescent.selection.Beam`): a unit that has never been measured
    is exploration the paper's random pairing would also have reached.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, ctx: SelectionContext, n: int) -> Sequence:
        candidates = list(ctx.candidates)
        if len(candidates) <= 1:
            return super().select(ctx, n)
        winners = []
        for _ in range(n):
            pair = self.rng.sample(candidates, 2)
            winners.append(max(
                pair, key=lambda c: (c.score is None, c.score or 0.0)))
        return winners


def _operator_prompt(operator: str, genome: dict, task: Task, output: str,
                     reward: float) -> str:
    if operator == "zero_order":
        return (
            "PromptBreeder zero-order mutation: write a fresh task prompt for "
            "this problem domain from a hint of the problem, without seeing the "
            "current prompt. Generalize; do not include the benchmark item's "
            "numeric answer.\n\n"
            f"Problem hint: {task.prompt}\nReward of current unit: {reward}\n"
            f"{feedback(task, output)}\n\n"
            'Return JSON only with a "task_prompt" string.'
        )
    if operator == "hyper":
        return (
            "PromptBreeder hyper-mutation: apply the mutation prompt to ITSELF "
            "to produce a better mutation prompt. This is the self-referential "
            "operator; do not touch the task prompt.\n\n"
            f"Mutation prompt: {genome['mutation_prompt']}\n"
            f"Recent reward: {reward}\n\n"
            'Return JSON only with a "mutation_prompt" string.'
        )
    return (
        "PromptBreeder first-order mutation: apply the mutation prompt to the "
        "task prompt using execution feedback. Generalize; do not include the "
        "benchmark item's numeric answer.\n\n"
        f"Task prompt: {genome['task_prompt']}\n"
        f"Mutation prompt: {genome['mutation_prompt']}\n"
        f"Reward: {reward}\n{feedback(task, output)}\n\n"
        'Return JSON only with "task_prompt" and "mutation_prompt" strings.'
    )


def build(seed: int) -> MethodPolicy:
    operators = itertools.cycle(("first_order", "zero_order", "hyper"))
    lock = threading.Lock()

    def next_operator() -> str:
        with lock:
            return next(operators)

    def solve(llm, rendered: str, task: Task) -> str:
        genome = read_fields(rendered)
        return solve_money(llm, genome.get("task_prompt", STARTING_INSTRUCTION),
                           task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        genome = read_fields(rendered)
        genome.setdefault("task_prompt", STARTING_INSTRUCTION)
        genome.setdefault("mutation_prompt", MUTATION_PROMPT)
        return llm(
            _operator_prompt(next_operator(), genome, task, output, reward),
            unit=task.id,
        )

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="promptbreeder",
        fidelity=FIDELITY,
        notes=(
            "Binary tournament replication and task/mutation-prompt co-evolution follow Algorithm 1.",
            "Zero-order, first-order, and hyper-mutation operators rotate per replication event (3 of the paper's 9).",
            "Population and fitness batch are budget-sized; PromptBreeder has no official released implementation.",
        ),
        strategy=FieldSlots(
            fields={"task_prompt": STARTING_INSTRUCTION,
                    "mutation_prompt": MUTATION_PROMPT},
            parse=parse_json_object,
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=money_reward,
        proposal_calls_per_candidate=1,
        engine=Policies(selection=BinaryTournament(seed)),
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
