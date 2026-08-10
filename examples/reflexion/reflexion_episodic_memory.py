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
from examples._gsm8k_domain import (STARTING_INSTRUCTION, feedback,
                                    gsm8k_reward, gsm8k_splits, solve_gsm8k)


FIDELITY = "mechanism_microport"

#: `prompts.REFLECTION_HEADER`, with "question" generalised to "problem" because
#: this memory is shared across the domain rather than bound to one instance.
MEMORY_HEADER = (
    "# Plans from past attempts. You have attempted problems like this before "
    "and failed; these plans say how to avoid failing the same way. Use them to "
    "improve your strategy (most recent last)."
)

#: `reflexion_few_shot_examples.txt`, in this domain. Without them the reflector
#: does not write a plan -- it answers the arithmetic. Measured on the domain
#: this port used to run: 40 reflections against no examples produced 40 bare
#: numbers, one of which mentioned anything reusable, and the memory finished
#: empty because nothing that useless cleared the acceptance gate. The prompt
#: ends in "New plan:" and contains a question and its answer; with nothing to
#: imitate, answering the question is the likelier continuation.
#:
#: The two items are **invented**, not drawn from GSM8K -- a worked example built
#: from a real row hands its graded answer to every proposal in the run, which is
#: what `test_the_examples_do_not_hand_over_a_held_out_answer` checks.
FEW_SHOT_EXAMPLES = """A baker sells 3 trays of 12 buns for 2 dollars a bun. What does he earn?
Attempt: 3 trays is 36 buns, so 36 times 2. The answer is 62.
The evaluator read the last number in the reply: '62'
It wanted: '72'
STATUS: FAIL
New plan: I set the problem up correctly and then did 36 x 2 in my head and got \
it wrong. I should have written that multiplication out on its own line where I \
could see it. Next time I will put every calculation on its own line rather than \
carrying it mentally, and re-read each one before using its result.

A shop had 20 crates, sold 8, then received 5 more. How many now?
Attempt: 20 - 8 = 12, then 12 + 5 = 17. So the shop started with 20 crates.
The evaluator read the last number in the reply: '20'
It wanted: '17'
STATUS: FAIL
New plan: My arithmetic was right and reached 17, and then I added a closing \
sentence that mentioned an earlier number. The grader reads the **last** number \
in the reply, so that sentence replaced my answer with the wrong one. Next time \
I will finish with the final number and write nothing after it."""


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        return solve_gsm8k(llm, rendered, task)

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

    train, held_out, test = gsm8k_splits(seed)
    return MethodPolicy(
        name="reflexion",
        fidelity=FIDELITY,
        notes=(
            "The reflection query carries two worked examples of a failed attempt and its plan, as _generate_reflection_query prepends FEW_SHOT_EXAMPLES; without them the reflector answers the arithmetic instead of planning.",
            "Reflection is requested only after a failed rollout, as update_memory and Agent.run both gate on success.",
            "The reflection prompt asks for a plan that accounts for the mistake and shows the previous plans, matching _generate_reflection_query.",
            "Memory is append-only and rendered as the last three entries, matching upstream's bounded window.",
            "Memory is global where upstream's is per task instance: Reflexion retries the same instance and claims no transfer, so this port asks whether reflection transfers at all.",
            "GSM8K replaces HotpotQA/ALFWorld: real questions, a real answer key, and the standard grader.",
        ),
        strategy=WindowedMemory(seed_text=STARTING_INSTRUCTION, window=3,
                                title=MEMORY_HEADER),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=gsm8k_reward,
        proposal_calls_per_candidate=1,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
