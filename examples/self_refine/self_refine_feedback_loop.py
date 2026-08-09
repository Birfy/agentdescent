"""Mechanism microport: **Self-Refine** (madaan/self-refine@9a206d41).

Preserved, from the task whose domain this is -- **gsm**, arithmetic word
problems. `GSMFeedback.__call__` is **one model call** that emits the critique
and the improved artifact together, split on a marker
(``entire_output.split("def solution():")``), and `iterative_gsm` then checks
the critique half for the stop signal. Six of the repository's seven tasks do
have a separate ``task_iterate.py`` REFINE module; gsm does not, and gsm is the
one this port's domain matches and the one it takes its stop signal from.

The critique prompt carries **worked examples of the fused shape**
(``data/prompt/gsm/feedback.txt``, four of them, each a wrong solution followed
by a block-by-block critique and then the corrected code). Without examples the
model does not emit the marker at all, and the refinement half comes back empty
-- the same failure [Reflexion](algo-reflexion.md) had, where a reflector with
nothing to imitate answered the arithmetic instead.

Stated rather than claimed: **the stop signal is stronger here than upstream.**
``iterative_gsm`` breaks on ``"it is correct" in feedback.lower()``, and the
phrase appears nowhere in ``data/prompt/gsm/`` -- nothing teaches the model to
emit it, so upstream's loop runs its full ``max_attempts``. This port instructs
the critique to say it when the attempt is already right, which makes the check
fire. That is the paper's stopping criterion working rather than upstream's
vestigial check reproduced, and the difference belongs in the notes.

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

#: `entire_output.split("def solution():")` -- the marker that separates the
#: critique from the refinement inside one completion.
REFINE_MARKER = "# Improved instruction:"

#: `data/prompt/gsm/feedback.txt`, in this domain: a wrong attempt, a critique
#: that walks the parts, then the corrected artifact after the marker. Two rather
#: than upstream's four, and invented rather than drawn from `TASKS` -- a worked
#: example built from a real item hands its graded answer to every proposal in
#: the run.
FEW_SHOT = f"""Instruction: Solve the problem and give the answer.
Attempt on "A ticket is $6.30 and a badge is $1.20. What is the total amount?": \
The total is $7.50.
The evaluator read the final line of the reply: '$7.50'
It wanted: '750'

# There is an error above because of a lack of understanding of what the \
evaluator accepts. What is the error? Go through the instruction part by part.

# Let us check the instruction step by step.
# "Solve the problem" -- fine, the arithmetic was right.
# "give the answer" -- wrong! It does not say in what form. The attempt wrote \
dollars and the evaluator wanted whole cents as a bare integer, so a correct \
computation scored zero.
# It also does not say where the answer goes, and only the final line is read.

{REFINE_MARKER}
Solve the problem. Work the arithmetic out in dollars, then convert to whole \
cents. Put nothing but that integer on the final line.

### END ###

Instruction: Solve the problem and put the amount in cents on the last line.
Attempt on "Five friends split a $21.50 bill equally. What amount does each \
pay?": 2150
The evaluator read the final line of the reply: '2150'
It wanted: '430'

# There is an error above because of a lack of understanding of what the \
evaluator accepts. What is the error? Go through the instruction part by part.

# Let us check the instruction step by step.
# "put the amount in cents on the last line" -- fine, the form was right.
# "Solve the problem" -- wrong! It does not say to check that every operation \
the question asks for actually happens. The attempt converted the whole bill \
instead of one share, so the form was right and the quantity was not.

{REFINE_MARKER}
Solve the problem. First restate which quantity is being asked for, then work \
the arithmetic out in dollars, then convert to whole cents. Put nothing but \
that integer on the final line.

### END ###"""


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
        raw = llm(
            (
                f"{FEW_SHOT}\n\n"
                f"Instruction: {rendered}\n"
                f"Attempt on \"{task.prompt}\":\n{feedback(task, output)}\n"
                f"External evaluator reward: {reward}\n\n"
                "# There is an error above because of a lack of understanding of "
                "what the evaluator accepts. What is the error? Go through the "
                f"instruction part by part. If the attempt is already right, say "
                f'"{STOP_SIGNAL}" and stop.\n'
                f"Then write the improved reusable instruction after a line "
                f"reading exactly {REFINE_MARKER!r}. Omit item-specific numbers.\n"
            ),
            subphase="feedback",
            unit=task.id,
        )
        critique, _, refinement = raw.partition(REFINE_MARKER)
        # `iterative_gsm` checks the *critique* half, not the whole completion:
        # the refined artifact routinely restates the goal, so testing the whole
        # reply would stop on the refinement's own words.
        if STOP_SIGNAL in critique.lower():
            return None
        return refinement.strip() or None

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="self_refine",
        fidelity=FIDELITY,
        notes=(
            "FEEDBACK and REFINE are one model call split on a marker, as GSMFeedback.__call__ is; six of the seven upstream tasks separate them, and gsm -- this port's domain -- does not.",
            "The critique prompt carries worked examples of the fused shape, as data/prompt/gsm/feedback.txt does.",
            "The stop signal is checked on the critique half only, as iterative_gsm does.",
            "The stop signal is stronger than upstream's: 'it is correct' appears nowhere in data/prompt/gsm, so nothing teaches the model to emit it and upstream's loop runs its full max_attempts. This port asks for it, so the check fires.",
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
        proposal_calls_per_candidate=1,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
