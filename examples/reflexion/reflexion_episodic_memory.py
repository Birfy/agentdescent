"""Mechanism microport: **Reflexion** (noahshinn/reflexion@218cf0ef).

Preserved: the reflection query carries **two worked examples** of a failed
attempt followed by its plan (`FEW_SHOT_EXAMPLES`, prepended by
`_generate_reflection_query` under "Here are two examples:"); reflection happens
**only after a failure**
(`generate_reflections.update_memory`: ``if not env['is_success'] and not
env['skip']``; `agents.py`: ``if self.step_n > 0 and not self.is_correct()``);
the reflection prompt asks for a **plan that accounts for the mistake**, shown
the previous plans numbered by trial, which is `_generate_reflection_query`; and
memory is **append-only and bounded**, prompted with the last Ω
(``memory[-3:]`` in the alfworld runs).
:class:`~examples._method_policy.WindowedMemory` is that: every accepted
reflection is a new commit-ordered ledger key, the rendered prompt shows the
last three, and parallel workers' reflections **union-merge** with no ranking
evaluation, because appends never contradict.

Declared generalisation -- **memory is global here, and per-instance upstream.**
`env_configs[i]['memory']` is one list per task instance, and the HotpotQA header
says so outright: *"You have attempted to answer following question before and
failed."* Reflexion is an inference-time method that retries **the same
instance**, and it does not claim transfer to unseen ones. A faithful
per-instance memory would be empty for every held-out task, so the row would
measure nothing at all; this port carries one shared memory instead and is
therefore asking a question Reflexion does not ask -- whether verbal reflection
*transfers*. That is worth measuring and is not what the paper measured, so it
is stated here rather than implied by a window that happens to match.

Boundary: upstream retries the same failed instance; the framework's held-out
rerun is the analogue.
"""

from __future__ import annotations

from typing import Optional

from agentdescent.evolution import Task

from examples._method_policy import MethodPolicy, WindowedMemory, read_fields
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, feedback,
                                    money_reward, money_splits, solve_money)


FIDELITY = "mechanism_microport"

#: `prompts.REFLECTION_HEADER`, with "question" generalised to "problem" because
#: this memory is shared across the domain rather than bound to one instance.
MEMORY_HEADER = (
    "# Plans from past attempts. You have attempted problems like this before "
    "and failed; these plans say how to avoid failing the same way. Use them to "
    "improve your strategy (most recent last)."
)

#: `reflexion_few_shot_examples.txt`, in this domain. Without them the reflector
#: does not write a plan -- it answers the arithmetic. Measured: 40 reflections
#: against no examples produced 40 bare numbers (`925`, `264`, `$32.15`), one of
#: which mentioned the output convention, and the memory finished empty because
#: nothing that useless ever cleared the acceptance gate. The prompt ends in
#: "New plan:" and contains a question and its answer; with nothing to imitate,
#: answering the question is the likelier continuation.
#:
#: The two items are **invented**, not drawn from `TASKS`. The first draft used
#: `m03` and `m11`, which land in held-out or test depending on the seed, so
#: every reflection prompt in the run handed over two graded answers --
#: `test_the_examples_do_not_hand_over_a_held_out_answer` is what caught it.
FEW_SHOT_EXAMPLES = """A ticket is $6.30 and a badge is $1.20. What is the total amount?
Attempt: The total is $7.50.
The evaluator read the final line of the reply: '$7.50'
It wanted: '750'
STATUS: FAIL
New plan: I computed the sum correctly and then reported it the way a person \
writes money, with a dollar sign and a decimal point. The evaluator did not \
accept that string. I should have converted the dollar amount to whole cents \
and written that integer on its own line with nothing else on it. Next time I \
will finish the arithmetic in dollars, multiply by 100, and put only that \
integer on the final line.

Five friends split a $21.50 bill equally. What amount does each pay?
Attempt: 2150
The evaluator read the final line of the reply: '2150'
It wanted: '430'
STATUS: FAIL
New plan: I wrote the total in cents instead of each person's share -- I had \
the output convention right and skipped an operation the question asked for. I \
should have divided by the number of people before converting. Next time I will \
restate which quantity is being asked for before computing, and check that every \
operation named in the question appears in my working."""


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        return solve_money(llm, rendered, task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        # `if not env['is_success'] and not env['skip']`. Reflecting on a success
        # too is not a cheaper version of Reflexion -- it fills the window with
        # entries the paper's memory never contains, and the window is bounded,
        # so each one evicts a lesson drawn from an actual failure. The runner's
        # budget check is `observed <= expected`, so declining costs nothing.
        if reward >= 1.0:
            return None
        return llm(
            (
                "You will be given the history of a past attempt at a problem "
                "you were unsuccessful at. Do not summarize the problem; think "
                "about the strategy you took. Devise a concise, new plan of "
                "action that accounts for your mistake, with reference to the "
                "specific steps you should have taken. Write the plan, not an "
                "answer. Here are two examples:\n\n"
                f"{FEW_SHOT_EXAMPLES}\n\n"
                f"{rendered}\n\n"
                f"Here is the attempt:\n{feedback(task, output)}\n"
                f"External evaluator reward: {reward}\n"
                "STATUS: FAIL\n\n"
                "New plan:"
            ),
            unit=task.id,
        )

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="reflexion",
        fidelity=FIDELITY,
        notes=(
            "The reflection query carries two worked examples of a failed attempt and its plan, as _generate_reflection_query prepends FEW_SHOT_EXAMPLES; without them the reflector answers the arithmetic instead of planning.",
            "Reflection is requested only after a failed rollout, as update_memory and Agent.run both gate on success.",
            "The reflection prompt asks for a plan that accounts for the mistake and shows the previous plans, matching _generate_reflection_query.",
            "Memory is append-only and rendered as the last three entries, matching upstream's bounded window.",
            "Memory is global where upstream's is per task instance: Reflexion retries the same instance and claims no transfer, so this port asks whether reflection transfers at all.",
            "A deterministic arithmetic evaluator replaces HotpotQA/ALFWorld.",
        ),
        strategy=WindowedMemory(seed_text=STARTING_INSTRUCTION, window=3,
                                title=MEMORY_HEADER),
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
