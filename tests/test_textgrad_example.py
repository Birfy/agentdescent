"""TextGrad against ``zou-group/textgrad@75e912e2``: the prompt ``Variable`` and
one textual-gradient step, ported as a ``mechanism_microport``.

Offline throughout: the model is a function in this file, the GSM8K domain is the
committed sample (``conftest.py``), and the engine, strategy, reflective merge
and held-out gate are the real ones.
"""
from __future__ import annotations

import socket

import pytest

from agentdescent.agents import Usage, metered
from agentdescent.evolution import Task
from examples._gsm8k_domain import STARTING_INSTRUCTION, load_split, gold_answer
from examples._measure import Recorder
from examples._method_runner import run_port
from examples.textgrad import textgrad_prompt_optimization as tg


def _task() -> Task:
    return Task(id="train:0", prompt="A pen costs 3 dollars. Two pens cost?",
                meta={"answer": "6", "split": "train"})


def _tagged(text: str) -> str:
    return f"Here you go.\n{tg.TAGS[0]}\n{text}\n{tg.TAGS[1]}\nDone."


class Scripted:
    """A model that answers the gradient call, then the step call, and records."""

    def __init__(self, gradient="Ask for the working, line by line.",
                 step=None):
        self.gradient = gradient
        self.step = step if step is not None else _tagged(
            "Solve the problem. Write each calculation on its own line, then "
            "finish with the final number.")
        self.calls = []

    def __call__(self, prompt, *, subphase="", unit=""):
        self.calls.append((subphase, prompt))
        return self.gradient if subphase == "backward" else self.step


# -- the critique -> rewrite step ------------------------------------------------


def test_a_critique_becomes_a_diff_on_the_instruction_slot():
    policy = tg.build(0)
    llm = Scripted()
    proposal = policy.propose(llm, STARTING_INSTRUCTION, _task(),
                              "The answer is 5", 0.0)
    strategy = policy.strategy
    diff = strategy.to_diff(strategy.initial(), proposal, "w0", 1, "artifact")
    assert diff is not None
    assert list(diff.ops) == [strategy.key]
    assert diff.ops[strategy.key].startswith("Solve the problem. Write each")
    assert "<IMPROVED_VARIABLE>" not in diff.ops[strategy.key]


def test_the_proposal_is_a_gradient_call_then_a_step_call():
    """`loss.backward()` then `optimizer.step()`: two calls, in that order, and
    the step is shown the gradient it is meant to follow."""
    policy = tg.build(0)
    llm = Scripted(gradient="UNIQUE-CRITIQUE-TEXT")
    policy.propose(llm, STARTING_INSTRUCTION, _task(), "The answer is 5", 0.0)
    assert [phase for phase, _ in llm.calls] == ["backward", "step"]
    assert policy.proposal_calls_per_candidate == 2
    backward, step = llm.calls[0][1], llm.calls[1][1]
    # upstream's gradient engine is told not to propose a rewrite itself
    assert "DO NOT propose a new version of the variable" in backward
    assert STARTING_INSTRUCTION in backward and "<LM_OUTPUT>" in backward
    assert "<FEEDBACK>UNIQUE-CRITIQUE-TEXT</FEEDBACK>" in step
    assert "<IMPROVED_VARIABLE>" in step and "Constraint 1:" in step


def test_a_correct_attempt_proposes_nothing_and_calls_nothing():
    policy = tg.build(0)
    llm = Scripted()
    assert policy.propose(llm, STARTING_INSTRUCTION, _task(), "6", 1.0) is None
    assert llm.calls == []


def test_a_reply_without_the_tags_is_invalid_and_never_substituted():
    """Upstream raises IndexError. Here the candidate is spent and counted, and
    the raw reply -- which would be a whole conversation -- is not committed."""
    policy = tg.build(0)
    strategy = policy.strategy
    for reply in ("I would rewrite it as follows: be careful.",
                  f"{tg.TAGS[0]} opened but never closed"):
        proposal = policy.propose(Scripted(step=reply), STARTING_INSTRUCTION,
                                  _task(), "The answer is 5", 0.0)
        assert proposal, "a falsy proposal never reaches to_diff, so is uncounted"
        assert strategy.to_diff(strategy.initial(), proposal, "w", 1, "a") is None
    assert strategy.invalid_proposals == 2


def test_an_empty_gradient_costs_the_candidate_without_a_step_call():
    policy = tg.build(0)
    llm = Scripted(gradient="   ")
    proposal = policy.propose(llm, STARTING_INSTRUCTION, _task(), "5", 0.0)
    assert proposal and tg.TAGS[0] not in proposal
    strategy = policy.strategy
    assert strategy.to_diff(strategy.initial(), proposal, "w", 1, "a") is None
    assert strategy.invalid_proposals == 1
    assert [phase for phase, _ in llm.calls] == ["backward"]


def test_the_gold_answer_appears_only_as_the_evaluators_line():
    """The prompts' fixed text is upstream's; the answer key reaches the model
    once, as the evaluator's reading of the attempt -- no worked example here is
    drawn from a graded item."""
    prompt = tg.backward_prompt(STARTING_INSTRUCTION, _task(), "The answer is 5")
    assert prompt.count("It wanted: '6'") == 1
    fixed = tg.BACKWARD_SYSTEM_PROMPT + tg.OPTIMIZER_SYSTEM_PROMPT
    for split in ("train", "test"):
        for row in load_split(split):
            assert row["question"].strip() not in fixed


def test_malformed_replies_are_counted_through_the_real_proposal_adapter(capsys):
    """Through `evolve`, not a hand call to `to_diff`: a falsy proposal is dropped
    by the adapter before the strategy sees it, which would leave a spent
    candidate out of the invalid count."""
    def complete(prompt):
        lower = prompt.lower()
        if "you are the gradient (feedback) engine" in lower:
            return "The instruction should change."
        if "you are part of an optimization system that improves text" in lower:
            return "I would rewrite it, but I will not use the tags."
        return "The answer is 0"

    usage = Usage()
    recorder = Recorder(metered(complete, usage), usage)
    policy = tg.build(0)
    payload = run_port(policy, recorder, mode="sync_parallel", seed=0,
                       workers=2, candidate_budget=4, max_seconds=60.0,
                       shutdown_grace=5.0).compact()
    assert payload["invalid_candidates"] >= 1
    assert payload["accepted"] == 0


# -- content addressing and no-ops ------------------------------------------------


def test_a_rewrite_equal_to_the_current_text_is_a_noop():
    strategy = tg.build(0).strategy
    state = strategy.initial()
    same = strategy.to_diff(state, _tagged(f"  {STARTING_INSTRUCTION}\n"), "w0", 1, "a")
    assert same is None
    assert strategy.noop_steps == 1
    assert strategy.invalid_proposals == 0


def test_identical_critiques_give_the_same_text_hashed_diff_per_author():
    """The id hashes the text, per author and base version. Two *workers* still
    get distinct ids for the same text, like every other slot's diffs."""
    policy = tg.build(0)
    strategy = policy.strategy
    state = strategy.initial()
    first = policy.propose(Scripted(), STARTING_INSTRUCTION, _task(), "5", 0.0)
    second = policy.propose(Scripted(), STARTING_INSTRUCTION, _task(), "7", 0.0)
    a = strategy.to_diff(state, first, "w0", 3, "a")
    b = strategy.to_diff(state, second, "w0", 3, "a")
    assert a.diff_id == b.diff_id and a.ops == b.ops
    other = strategy.to_diff(state, _tagged("Show every step, then the number."), "w0", 3, "a")
    assert other.diff_id != a.diff_id
    assert strategy.to_diff(state, first, "w1", 3, "a").diff_id != a.diff_id


# -- the held-out gate -------------------------------------------------------------

_BETTER = "Work step by step and end with the final number."
_WORSE = "Answer from memory without any working."


def _model(rewrite, seed_solves_even=False):
    """Solves a task iff the instruction in force says 'step by step'; the
    gradient and step calls are the scripted ones.

    ``seed_solves_even`` lets the *seed* instruction solve the items whose gold
    answer is even, so a rewrite can do strictly worse than the head."""
    gold = {}
    for split in ("train", "test"):
        for row in load_split(split):
            gold[row["question"].strip()] = gold_answer(row["answer"])

    def complete(prompt):
        lower = prompt.lower()
        if "you are the gradient (feedback) engine" in lower:
            return "The instruction should change."
        if "you are part of an optimization system that improves text" in lower:
            return f"{tg.TAGS[0]}{rewrite}{tg.TAGS[1]}"
        for question, answer in gold.items():
            if question in prompt:
                if "step by step" in lower:
                    return f"Working it through, the answer is {answer}"
                if seed_solves_even and int(answer) % 2 == 0 \
                        and _WORSE not in prompt:
                    return f"The answer is {answer}"
                return f"The answer is {int(answer) + 1}"
        return "I do not know."

    return complete


def _run(rewrite, capsys, seed_solves_even=False):
    usage = Usage()
    recorder = Recorder(metered(_model(rewrite, seed_solves_even), usage), usage)
    policy = tg.build(0)
    payload = run_port(policy, recorder, mode="sync_parallel", seed=0,
                       workers=2, candidate_budget=4, max_seconds=60.0,
                       shutdown_grace=5.0).compact()
    artifact = capsys.readouterr().out.split("[final artifact]\n", 1)[1]
    return policy, payload, artifact


def test_a_rewrite_that_regresses_heldout_quality_is_rejected(capsys):
    """A true regression: the seed solves the even-answer items and the rewrite
    solves none, so validation drops. Upstream's `run_validation_revert` would
    revert this too."""
    policy, payload, artifact = _run(_WORSE, capsys, seed_solves_even=True)
    assert payload["baseline_validation_quality"] > 0.0
    assert payload["budget"]["observed_proposal_calls"] >= 2
    assert payload["accepted"] == 0
    assert payload["validation_quality"] == payload["baseline_validation_quality"]
    assert artifact.strip().startswith(STARTING_INSTRUCTION)
    assert _WORSE not in artifact


def test_a_rewrite_that_only_ties_heldout_quality_is_also_rejected(capsys):
    """A declared departure, pinned: upstream's revert rule keeps a rewrite whose
    validation score *ties* the previous one (it reverts only on `val <
    previous`); the engine's Beta-posterior gate resolves a tie by rejecting.
    Here the seed and the rewrite both score zero on every item."""
    policy, payload, artifact = _run(_WORSE, capsys)
    assert payload["budget"]["observed_proposal_calls"] >= 2   # a real proposal
    assert payload["validation_quality"] == payload["baseline_validation_quality"]
    assert payload["accepted"] == 0
    assert _WORSE not in artifact
    assert payload["final_quality"] == payload["baseline_quality"]


def test_a_rewrite_that_improves_heldout_quality_is_accepted(capsys):
    """The control for the test above: the gate is a comparison, not a wall."""
    policy, payload, artifact = _run(_BETTER, capsys)
    assert payload["accepted"] >= 1
    assert payload["validation_quality"] > payload["baseline_validation_quality"]
    assert _BETTER in artifact
    assert payload["final_quality"] > payload["baseline_quality"]
    assert payload["budget"]["matched"] is True


# -- the entry point ---------------------------------------------------------------


def test_dry_run_needs_no_network_and_no_api_key(monkeypatch, capsys):
    """Neither the GSM8K fetch nor a model call is reachable: the suite's sample
    switch is removed and every socket is made to fail."""
    monkeypatch.delenv("AGENTDESCENT_GSM8K_SAMPLE", raising=False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("--dry-run opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert tg.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "textgrad [mechanism_microport]" in out
    assert "proposal calls=4" in out
    assert "committed GSM8K sample" in out and "correction" not in out
    # the sample is scoped to the build, not left switched on
    import os
    assert "AGENTDESCENT_GSM8K_SAMPLE" not in os.environ


def test_the_textgrad_package_is_never_imported():
    import sys
    assert "textgrad" not in sys.modules
    assert not hasattr(tg, "textgrad")


def test_it_is_registered_in_the_candidate_matrix():
    from bench.candidate_methods import ALGORITHMS
    fidelity, builder = ALGORITHMS["textgrad"]
    assert fidelity == "mechanism_microport"
    assert builder(0).name == "textgrad"
