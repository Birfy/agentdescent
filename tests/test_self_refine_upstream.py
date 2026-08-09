"""Self-Refine against `madaan/self-refine@9a206d41`, task `gsm`.

The port's domain is arithmetic word problems, so gsm is the matching task and
the one the stop signal was taken from -- and gsm is the single task in that
repository whose FEEDBACK and REFINE are **one call**, not two.
"""
from __future__ import annotations

from agentdescent.evolution import Task

from examples.self_refine import self_refine_feedback_loop as sr


def _task():
    return Task(id="train:m00", prompt="A pen costs $1.40. What is the total?",
                meta={"answer_cents": "140", "split": "train"})


def _fused(critique: str, refinement: str) -> str:
    return f"{critique}\n{sr.REFINE_MARKER}\n{refinement}"


def test_feedback_and_refine_are_one_call():
    """`GSMFeedback.__call__` makes a single request and splits the completion on
    `"def solution():"` -- prose before is the critique, code after is the
    improved solution. Two calls is the shape of the other six tasks'
    `task_iterate.py`, and gsm has no such module."""
    policy = sr.build(0)
    calls = []

    def llm(prompt, **kw):
        calls.append(prompt)
        return _fused("# wrong! it says nothing about units.", "Answer in cents.")

    out = policy.propose(llm, "Solve it.", _task(), "$1.40", 0.0)
    assert len(calls) == 1, f"{len(calls)} calls; upstream's gsm makes one"
    assert out == "Answer in cents."
    assert policy.proposal_calls_per_candidate == 1


def test_the_stop_signal_is_read_from_the_critique_half_only():
    """`iterative_gsm` tests `fb_and_maybe_soln["feedback"]`, not the whole
    completion. A refinement routinely restates the goal, so testing the whole
    reply stops the loop on the refinement's own words."""
    policy = sr.build(0)
    llm = lambda prompt, **kw: _fused(
        "# the instruction is missing the output form.",
        f"Answer in cents. Check the result until {sr.STOP_SIGNAL}.")
    assert policy.propose(llm, "Solve it.", _task(), "$1.40", 0.0) is not None


def test_a_correct_attempt_stops_without_proposing():
    policy = sr.build(0)
    llm = lambda prompt, **kw: _fused(
        f"# {sr.STOP_SIGNAL}, nothing to change.", "unused")
    assert policy.propose(llm, "Solve it.", _task(), "140", 1.0) is None


def test_a_reply_with_no_marker_proposes_nothing_rather_than_the_critique():
    """Without the marker the refinement half is empty, and returning the
    critique instead would commit a *criticism* as the next instruction."""
    policy = sr.build(0)
    llm = lambda prompt, **kw: "# there is an error here and I will not say more"
    assert policy.propose(llm, "Solve it.", _task(), "$1.40", 0.0) is None


def test_the_prompt_carries_worked_examples_of_the_fused_shape():
    """`data/prompt/gsm/feedback.txt` holds four. Without them the model does not
    emit the marker and the refinement half comes back empty -- the same failure
    Reflexion had, where a reflector with nothing to imitate answered the
    arithmetic instead of planning."""
    policy = sr.build(0)
    seen = {}

    def llm(prompt, **kw):
        seen["prompt"] = prompt
        return _fused("# fine", "Answer in cents.")

    policy.propose(llm, "Solve it.", _task(), "$1.40", 0.0)
    prompt = seen["prompt"]
    assert prompt.count(sr.REFINE_MARKER) >= 3, (
        "the examples must demonstrate the marker, not merely ask for it")
    assert sr.FEW_SHOT in prompt


def test_the_examples_do_not_hand_over_a_graded_answer():
    """A worked example built from a real item hands its answer to every proposal
    in the run; this is the leak `test_the_examples_do_not_hand_over_a_held_out_
    answer` caught in Reflexion."""
    from examples._money_domain import TASKS

    for task in TASKS:
        assert f"It wanted: '{task.answer_cents}'" not in sr.FEW_SHOT, (
            f"the examples hand over {task.id}'s answer")
        assert task.question not in sr.FEW_SHOT


def test_the_stronger_stop_signal_is_declared():
    """`"it is correct"` appears nowhere in `data/prompt/gsm/`, so nothing teaches
    the model to emit it and upstream's check never fires -- its loop runs the
    full `max_attempts`. This port asks for the phrase, which makes the check
    real. That is the paper's stopping criterion working rather than upstream's
    behaviour reproduced, and the notes have to say which."""
    notes = " ".join(sr.build(0).notes).lower()
    assert "stronger than upstream" in notes
    assert "max_attempts" in notes
