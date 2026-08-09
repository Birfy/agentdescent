"""The three inference analogues against their upstreams.

Absolute Zero: `LeapLabTHU/Absolute-Zero-Reasoner@484afa48`
R-Zero:        `Chengsong-Huang/R-Zero@5699329d`
Agent0:        `aiming-lab/Agent0@f775b510`
"""
from __future__ import annotations

import json

from examples._selfplay_domain import selfplay_splits
from examples.absolute_zero import absolute_zero_selfplay as az


def test_the_splits_are_wide_enough_for_the_workers_the_matrix_runs():
    """`run_port` refuses a run whose train split is under the worker count, so
    four self-play slots capped all three ports sharing this domain at four
    workers, with a four-cart gate where one item moves the score by 0.25."""
    for name in ("absolute_zero", "r_zero", "agent0"):
        train, held_out, test = selfplay_splits(0, name)
        assert len(train) == len(held_out) == len(test) == 16, name
        ids = [{t.id for t in g} for g in (train, held_out, test)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), name


def test_the_evaluation_carts_are_frozen_from_the_seed():
    """The property all three ports claim: the evolved memory cannot shape its
    own test set, which is why the carts are drawn here and not by the
    proposer."""
    for name in ("absolute_zero", "r_zero"):
        first = selfplay_splits(3, name)
        again = selfplay_splits(3, name)
        for a, b in zip(first[1] + first[2], again[1] + again[2]):
            assert a.prompt == b.prompt and a.meta["answer"] == b.meta["answer"]
    # ...and different seeds really do give different carts.
    assert ({t.prompt for t in selfplay_splits(0, "absolute_zero")[2]}
            != {t.prompt for t in selfplay_splits(1, "absolute_zero")[2]})


def test_the_learnability_signal_is_this_items_rate_not_the_runs_average():
    """`accuracies[uid]` is the solve rate of *that* problem over its rollout
    group. A running mean across the run tells the proposer a number about
    problems it did not write, and this port surfaced one while its notes said
    the reward was carried verbatim.

    One solver sample also makes the rate 0 or 1, so `1 - r` is zero either way
    and the signal the paper's proposer is trained on does not exist at all.
    """
    assert az.SOLVER_SAMPLES >= 2

    policy = az.build(0)
    train, _, _ = selfplay_splits(0, "absolute_zero")
    task = train[0]

    cart = {"item_cents": [100, 200], "quantities": [1, 1]}
    replies = iter([json.dumps(cart), "300", "wrong"])   # one right, one wrong
    seen = {}

    def llm(prompt, **kw):
        if "Return JSON only" in prompt:
            return next(replies)
        return next(replies)

    policy.solve(llm, "memory", task)

    def capture(prompt, **kw):
        seen["prompt"] = prompt
        return "a lesson"

    policy.propose(capture, "memory", task, "{}", 0.0)
    assert "r=0.50" in seen["prompt"], seen["prompt"]
    assert "1-r = 0.50" in seen["prompt"]


def test_the_reward_is_zero_at_both_extremes():
    """`(1 - accuracy) if accuracy > 0 else 0.0` -- upstream's `one_minus`. It is
    monotone in the solve rate and *not* peaked at 0.5, which is why no
    difficulty-weighted sampler is attached to this port."""
    policy = az.build(0)
    train, _, _ = selfplay_splits(0, "absolute_zero")

    for outcomes, want in ((["300", "300"], "0.00"),      # solved every time
                           (["nope", "nope"], "0.00"),    # solved never
                           (["300", "nope"], "0.50")):
        task = train[0]
        cart = {"item_cents": [100, 200], "quantities": [1, 1]}
        replies = iter([json.dumps(cart)] + outcomes)
        policy.solve(lambda p, **k: next(replies), "memory", task)
        seen = {}
        policy.propose(lambda p, **k: (seen.setdefault("p", p), "x")[1],
                       "memory", task, "{}", 0.0)
        assert f"1-r = {want}" in seen["p"], (outcomes, seen["p"])
