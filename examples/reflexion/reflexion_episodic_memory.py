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

**Domain: GSM-Hard, not GSM8K**, for a reason specific to this mechanism.
Reflexion is fed by failures, and on GSM8K this port's held-out score was
0.75-0.80 -- four rollouts in five succeeded and wrote nothing, so a 32-candidate
probe produced 7 reflections. Worse, the failures that did occur had no shared
cause: each was an idiosyncratic misreading, and a memory read against *other*
questions has nothing to carry. GSM-Hard is the same 1319 questions with large
numbers substituted, where the wrapper scores 0.391 and failures concentrate on
one cause -- arithmetic done in the model's head. That is a failure mode a
transferable rule can actually address, which is what this port is asking about.
"""

from __future__ import annotations

import re
from typing import Optional

from agentdescent.evolution import Task

from examples._method_policy import MethodPolicy, WindowedMemory, read_fields
from examples._method_runner import standard_main
from examples._gsmhard_domain import (STARTING_INSTRUCTION, feedback,
                                      gsmhard_reward, gsmhard_splits,
                                      solve_gsmhard)


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
#:
#: They also carry **no arithmetic of their own**. An earlier version showed the
#: failed working (`36 x 2`, `20 - 8 = 12`) before generalising, and the model
#: imitated the shape rather than the conclusion: measured on GSM8K, all four
#: reflections in a run were re-derivations of the item just failed -- "I
#: correctly identified 2 large pizzas and 2 small, but I multiplied wrong" --
#: which is useless in a memory read against *other* questions. Every proposal
#: was well-formed, every one was refused by the gate, and the memory finished
#: empty across three seeds.
#: The block shape mirrors the live one exactly -- feedback, reward, ``STATUS:
#: FAIL``, ``New plan:`` -- because that is what makes them *examples* of the
#: completion being asked for rather than two paragraphs of advice. Upstream's
#: file has the same property and `test_the_reflection_query_carries_worked_
#: examples` pins it by counting the markers.
FEW_SHOT_EXAMPLES = """Here is the attempt:
Question: a problem about a baker's trays of buns
The evaluator read the last number in your reply: '62'
It wanted: '72'
External evaluator reward: 0.0
STATUS: FAIL

New plan: I did one of the multiplications in my head instead of writing it \
down, and got it wrong. From now on every calculation goes on its own line \
before I use its result, however small it looks.

Here is the attempt:
Question: a problem about a shop's crates
The evaluator read the last number in your reply: '20'
It wanted: '17'
External evaluator reward: 0.0
STATUS: FAIL

New plan: My arithmetic reached the right value and then I wrote a closing \
sentence that mentioned an earlier number. The grader reads the last number in \
the reply, so anything after the answer replaces it. From now on I finish with \
the answer and write nothing after it."""


#: A reflection that states no rule is not a short reflection, it is a different
#: kind of output. The prompt ends in "New plan:" and shows the failed question,
#: so answering the question is a live continuation however firmly the
#: instructions forbid it: measured on GSM8K, 2 of 7 reflections in a probe were
#: the bare string `624` and `48`. Those merged into the window as entries and
#: displaced real plans, because `WindowedMemory` is bounded and append-only.
#:
#: Four words is a **shape** floor, not a quality bar, and it is deliberately
#: nowhere near the boundary: every degenerate output ever observed here scored
#: **zero** -- `624`, `48`, `925`, `$32.15` contain no word at all -- while the
#: tersest real plan in the tests, "check the units before answering", scores
#: five. A threshold sitting between 0 and 5 does not have to be argued for.
#: Whether a plan that clears it is any *good* is the held-out gate's question,
#: and answering it here would be the strategy scoring its own proposals.
_WORD = re.compile(r"[A-Za-z]{2,}")


def is_a_plan(entry: str) -> bool:
    """Does this reflection state a rule rather than answer the question?"""
    return len(_WORD.findall(entry or "")) >= 4


def _for_instance(rendered: str, task: Task) -> str:
    """The rendered memory, narrowed to entries written for this task.

    Upstream's memory is a list per environment; here it is one artifact whose
    entries carry a `[task-id]` tag, and this drops the rest. On a held-out task
    that leaves nothing, which is exactly upstream's position and exactly why
    the default does not do it.
    """
    kept, tag = [], f"[{task.id}]"
    for line in rendered.splitlines():
        if not line.startswith("- ") or tag in line:
            kept.append(line)
    return "\n".join(kept)


def build(seed: int, *, per_instance: bool = False) -> MethodPolicy:
    """``per_instance=True`` is the faithful variant.

    Upstream keys memory on the task (`env_configs[i]['memory']`) and retries
    **that** instance. Under the framework's held-out gate such a memory is
    empty for every reported task, so the row measures nothing -- which is why
    the default generalises it to one shared memory and says so.

    The flag exists because a substitution nobody can run is a claim nobody can
    check, the same reason Godel Agent has ``--gateless``. It is expected to
    score its baseline and accept nothing; that outcome is the control, not a
    failure.
    """

    def solve(llm, rendered: str, task: Task) -> str:
        if per_instance:
            # Only this task's own entries, as upstream's per-env memory is.
            rendered = _for_instance(rendered, task)
        return solve_gsmhard(llm, rendered, task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        # `if not env['is_success'] and not env['skip']`. Reflecting on a success
        # too is not a cheaper version of Reflexion -- it fills the window with
        # entries the paper's memory never contains, and the window is bounded,
        # so each one evicts a lesson drawn from an actual failure. The runner's
        # budget check is `observed <= expected`, so declining costs nothing.
        if reward >= 1.0:
            return None
        prefix = f"[{task.id}] " if per_instance else ""
        raw = llm(
            (
                "You will be given a past attempt at a problem you got wrong. "
                "Write one plan that stops the same *kind* of mistake on other "
                "problems.\n\n"
                "This memory is read while solving questions you have not seen, "
                "so a plan naming this problem's numbers, quantities or "
                "characters is worth nothing when it is read. Do not restate the "
                "arithmetic and do not solve it again -- say what you should "
                "have done differently, as a rule. Here are two examples:\n\n"
                f"{FEW_SHOT_EXAMPLES}\n\n"
                f"{rendered}\n\n"
                f"Here is the attempt:\n{feedback(task, output)}\n"
                f"External evaluator reward: {reward}\n"
                "STATUS: FAIL\n\n"
                "New plan:"
            ),
            unit=task.id,
        )
        # Tagged so `_for_instance` can hand the solver only its own entries.
        # The tag is the whole of what per-instance costs here: the memory is
        # still one ledger artifact, and only the *rendering* narrows.
        return f"{prefix}{raw.strip()}" if raw and raw.strip() else raw

    train, held_out, test = gsmhard_splits(seed)
    return MethodPolicy(
        name="reflexion",
        fidelity=FIDELITY,
        notes=(
            "The reflection query carries two worked examples of a failed attempt and its plan, as _generate_reflection_query prepends FEW_SHOT_EXAMPLES; without them the reflector answers the arithmetic instead of planning.",
            "Reflection is requested only after a failed rollout, as update_memory and Agent.run both gate on success.",
            "The reflection prompt asks for a plan that accounts for the mistake and shows the previous plans, matching _generate_reflection_query.",
            "Memory is append-only and rendered as the last three entries, matching upstream's bounded window.",
            "Memory is global where upstream's is per task instance: Reflexion retries the same instance and claims no transfer, so this port asks whether reflection transfers at all. --per-instance runs the faithful variant.",
            "The reflection prompt asks for a rule that survives being read against other questions, because a shared memory needs one -- upstream asks for the specific steps that should have been taken, which is right when the next thing you do is retry that same instance.",
            "An entry stating no rule is refused as malformed rather than appended: the prompt shows the failed question and ends in 'New plan:', so a bare answer is a live continuation, and the window is bounded so one displaces a real plan.",
            "GSM-Hard replaces HotpotQA/ALFWorld: real questions, a real answer key, and the standard grader. GSM8K left this port at 0.75-0.80, where four rollouts in five wrote no reflection at all.",
        ),
        strategy=WindowedMemory(seed_text=STARTING_INSTRUCTION, window=3,
                                title=MEMORY_HEADER, validator=is_a_plan),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=gsmhard_reward,
        proposal_calls_per_candidate=1,
        reflective=True,
    )


def _per_instance_flag(parser) -> None:
    parser.add_argument(
        "--per-instance",
        action="store_true",
        help=("run the faithful variant: a reflection is read back only while "
              "solving the task that produced it, as upstream's per-environment "
              "memory is. Expect it to accept nothing and score its baseline -- "
              "that is the control that makes the shared-memory default a "
              "checkable substitution rather than an unverifiable claim"))


def main(argv=None) -> int:
    return standard_main(build, argv, extra_args=_per_instance_flag,
                         build_kwargs=lambda a: {"per_instance": a.per_instance})


if __name__ == "__main__":
    raise SystemExit(main())
