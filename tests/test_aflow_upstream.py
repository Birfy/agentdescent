"""AFlow against `FoundationAgents/AFlow@3f457218`, not against its own docstring.

The port ran, reported "soft mixed probability", and picked uniformly. Every test
here is a line of upstream that the port had paraphrased rather than implemented.
"""
from __future__ import annotations

import json

import pytest

from examples.aflow import aflow_workflow_search as af


def test_the_probability_formula_matches_compute_probabilities():
    """`mixed = lambda_ * (1/n) + (1 - lambda_) * exp(a*(s-s_max))/sum(...)`."""
    import math

    scores = [0.9, 0.55, 0.4]
    policy = af.SoftMixed()
    scaled = [s * af.SCORE_SCALE for s in scores]
    s_max = max(scaled)
    weights = [math.exp(af.ALPHA * (s - s_max)) for s in scaled]
    total = sum(weights)
    expected = [af.LAMBDA / 3 + (1 - af.LAMBDA) * w / total for w in weights]

    got = policy.probabilities(scores)
    assert [round(p, 12) for p in got] == [round(p, 12) for p in expected]
    assert abs(sum(got) - 1.0) < 1e-9


def test_a_flat_pool_is_uniform_and_a_spread_pool_is_not():
    policy = af.SoftMixed()
    flat = policy.probabilities([0.5, 0.5, 0.5, 0.5])
    assert max(flat) - min(flat) < 1e-9
    # λ/n + (1-λ) is the ceiling the mixture allows, and an 0.8 gap saturates it:
    # exp(0.2*100*-0.8) is 1e-7, so the softmax term is entirely on the leader.
    spread = policy.probabilities([0.9, 0.1])
    ceiling = af.LAMBDA / 2 + (1 - af.LAMBDA)
    assert spread[0] == pytest.approx(ceiling, rel=1e-6), (
        f"exploitation is missing: {spread}")


def test_lambda_is_the_floor_a_hopeless_workflow_keeps():
    """The uniform term is why a workflow that scores zero is still reachable --
    upstream's exploration, and the reason λ is not folded into α."""
    policy = af.SoftMixed()
    probs = policy.probabilities([1.0, 0.0, 0.0, 0.0])
    assert min(probs) == pytest.approx(af.LAMBDA / 4, rel=1e-6)


# -- experience --------------------------------------------------------------


def test_experience_reports_the_parents_score_and_prohibits_every_past_edit():
    """`format_experience` prohibits successes *and* failures. What it prevents
    is repetition, not failure -- a port that only listed failures would keep
    proposing the modifications that already worked, which upstream's
    `check_modification` rejects outright."""
    exp = af.Experience()
    exp.record("workflow-A", "add a units check", 0.40)
    exp.resolve("workflow-A", "add a units check", 0.55)      # succeeded
    exp.record("workflow-A", "shorten the prompt", 0.40)
    exp.resolve("workflow-A", "shorten the prompt", 0.20)     # failed

    text = exp.render("workflow-A", 0.40)
    assert "Original Score: 0.400" in text
    assert "-Absolutely prohibit add a units check" in text
    assert "-Absolutely prohibit shorten the prompt (Score: 0.400)" in text
    assert exp.tried("workflow-A") == ["add a units check", "shorten the prompt"]


def test_experience_is_per_parent_not_global():
    """Upstream keys `experience.json` on the *father node*: a modification tried
    from one workflow says nothing about trying it from another."""
    exp = af.Experience()
    exp.record("workflow-A", "add a units check", 0.4)
    assert exp.tried("workflow-B") == []
    assert "No experience data found" in exp.render("workflow-B", 0.0)


def test_a_repeated_modification_is_regenerated_once():
    """`check_modification` loops until the modification is new. Accepting a
    repeat instead spends a candidate on an edit already known not to help."""
    policy = af.build(0)
    rendered = policy.strategy.render(policy.strategy.initial())
    task = _task()

    replies = [
        json.dumps({"modification": "same edit", "solve_instruction": "a",
                    "review_instruction": "b"}),
        json.dumps({"modification": "a different edit", "solve_instruction": "c",
                    "review_instruction": "d"}),
    ]
    units = []

    def llm(prompt, **kw):
        units.append(kw.get("unit", ""))
        return replies[min(len(units) - 1, len(replies) - 1)]

    policy.propose(llm, rendered, task, "140", 0.0)   # records "same edit"
    units.clear()
    replies[:] = [replies[0], replies[1]]
    out = policy.propose(llm, rendered, task, "140", 0.0)

    assert len(units) == 2, "the repeat was accepted instead of regenerated"
    assert units[1].endswith(":regenerate")
    assert json.loads(out)["modification"] == "a different edit"


def test_the_regeneration_is_inside_the_declared_budget():
    assert af.build(0).proposal_calls_per_candidate == 2


def _task():
    from agentdescent.evolution import Task
    return Task(id="train:0", prompt="A pen costs 3 dollars. Two pens cost?",
                meta={"answer": "6", "split": "train"})
