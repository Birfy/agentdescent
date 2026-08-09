"""Reflexion against `noahshinn/reflexion@218cf0ef`.

The trigger is the mechanism. Reflexion reflects on *failure*; a port that
reflects on every rollout still runs, still fills a memory, and fills it with
entries the paper's memory never contains -- while the window is bounded, so each
one evicts a lesson drawn from an actual failure.
"""
from __future__ import annotations

from agentdescent.evolution import Task

from examples.reflexion import reflexion_episodic_memory as rx


def _task():
    return Task(id="train:m00", prompt="A pen costs $1.40. What is the total?",
                meta={"answer_cents": "140", "split": "train"})


def _policy():
    return rx.build(0)


def test_a_successful_rollout_produces_no_reflection():
    """`update_memory`: `if not env['is_success'] and not env['skip']`.
    `agents.Agent.run`: `if self.step_n > 0 and not self.is_correct()`."""
    policy = _policy()
    calls = []
    llm = lambda prompt, **kw: calls.append(prompt) or "a plan"

    rendered = policy.strategy.render(policy.strategy.initial())
    assert policy.propose(llm, rendered, _task(), "140", 1.0) is None
    assert calls == [], "a reflection was requested after a success"


def test_a_failed_rollout_does_produce_one():
    policy = _policy()
    calls = []
    llm = lambda prompt, **kw: (calls.append(prompt), "a plan")[1]

    rendered = policy.strategy.render(policy.strategy.initial())
    assert policy.propose(llm, rendered, _task(), "$1.40", 0.0) == "a plan"
    assert len(calls) == 1


def test_the_reflection_prompt_asks_for_a_plan_and_shows_the_past_ones():
    """`_generate_reflection_query` asks for "a concise, new plan of action that
    accounts for your mistake with reference to specific actions you should have
    taken", and appends the previous plans. A prompt asking instead for "one
    reusable lesson, omit item-specific numbers" is a different operator."""
    policy = _policy()
    seen = {}

    def llm(prompt, **kw):
        seen["prompt"] = prompt
        return "a plan"

    state = policy.strategy.initial()
    diff = policy.strategy.to_diff(state, "check the units before answering",
                                   "w", 1, "candidate-reflexion")
    state = {**state, **diff.ops}
    policy.propose(llm, policy.strategy.render(state), _task(), "$1.40", 0.0)

    prompt = seen["prompt"]
    assert "new plan of action that accounts for your mistake" in prompt
    assert "specific steps you should have taken" in prompt
    assert "check the units before answering" in prompt, (
        "the previous plans were not shown to the reflector")
    assert prompt.rstrip().endswith("New plan:")


def test_the_window_is_the_last_three_entries():
    """`memory[-3:]` in the alfworld runs; the paper bounds omega to 1-3."""
    policy = _policy()
    state = policy.strategy.initial()
    for i in range(5):
        diff = policy.strategy.to_diff(state, f"plan {i}", "w", i,
                                       "candidate-reflexion")
        state = {**state, **diff.ops}
    rendered = policy.strategy.render(state)
    assert "plan 4" in rendered and "plan 3" in rendered and "plan 2" in rendered
    assert "plan 0" not in rendered and "plan 1" not in rendered


def test_the_rendered_memory_frames_the_entries_as_plans_after_failure():
    """`REFLECTION_HEADER` tells the solver these are plans from *failed*
    attempts. Rendering them as an undifferentiated list drops the framing the
    solver is meant to act on."""
    policy = _policy()
    rendered = policy.strategy.render(policy.strategy.initial())
    assert "failed" in rendered.lower() and "plan" in rendered.lower()


def test_the_global_memory_generalisation_is_declared():
    """Upstream keys memory on the task instance (`env_configs[i]['memory']`) and
    its header says "You have attempted to answer following question before".
    This port shares one memory across the domain, which is a different question
    -- whether reflection *transfers* -- and the notes have to say so, because a
    reader comparing the window alone would find them identical."""
    notes = " ".join(_policy().notes).lower()
    assert "per task instance" in notes and "transfer" in notes


def test_the_reflection_query_carries_worked_examples():
    """`_generate_reflection_query` prepends `FEW_SHOT_EXAMPLES` under "Here are
    two examples:", and dropping them is not a shortened prompt -- it changes
    what the module does.

    Measured without them: 40 reflections produced 40 bare numbers -- `925`,
    `264`, `$32.15` -- one of which mentioned the output convention, and the
    memory finished empty because nothing that useless cleared the gate. The
    query ends in "New plan:" and contains a question and its answer; with
    nothing to imitate, answering the question is the likelier continuation.
    """
    policy = _policy()
    seen = {}

    def llm(prompt, **kw):
        seen["prompt"] = prompt
        return "a plan"

    rendered = policy.strategy.render(policy.strategy.initial())
    policy.propose(llm, rendered, _task(), "$1.40", 0.0)

    prompt = seen["prompt"]
    assert "Here are two examples:" in prompt
    assert prompt.count("New plan:") >= 3, (
        "the examples must show the answer shape, not just ask for it")
    assert prompt.count("STATUS: FAIL") >= 3
    assert rx.FEW_SHOT_EXAMPLES in prompt


def test_the_examples_are_plans_rather_than_answers():
    """An example that ends in a bare number teaches the model to end in a bare
    number, which is exactly the failure the examples exist to prevent."""
    for block in rx.FEW_SHOT_EXAMPLES.split("New plan: ")[1:]:
        plan = block.split("\n")[0]
        assert len(plan) > 80, f"too terse to be a plan: {plan!r}"
        assert "next time" in plan.lower() or "should have" in plan.lower()


def test_the_examples_do_not_hand_over_a_held_out_answer():
    """They are worked examples, so they contain answers -- which makes it worth
    checking that the items they use are the domain's own, and that a method
    memorising them learns nothing a held-out item would reward."""
    from examples._money_domain import money_splits

    for seed in (0, 1, 2):
        _, held_out, test = money_splits(seed)
        for task in list(held_out) + list(test):
            answer = str(task.meta["answer_cents"])
            shown = f"It wanted: '{answer}'"
            assert shown not in rx.FEW_SHOT_EXAMPLES, (
                f"seed {seed}: the examples hand over {task.id}'s answer")
