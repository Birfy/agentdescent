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


# -- R-Zero -------------------------------------------------------------------


def test_the_challengers_signal_carries_no_ground_truth():
    """`question_evaluate/evaluate.py` takes `score = max_count / len(results)`
    over `--num_samples` solver samples: the share agreeing with the **majority
    answer**. R-Zero has no ground truth for a question its Challenger just
    wrote -- that is the premise -- and rewards questions the Solver is
    *self-inconsistent* on.

    This port computed `p_hat = (score + agreement * score) / 2`, mixing in the
    grounded verifier's reward, which measures something the paper cannot.
    """
    from examples.r_zero.r_zero_challenger_solver import majority_share

    assert majority_share(["300", "300", "300", "300"]) == 1.0
    assert majority_share(["300", "300", "300", "999"]) == 0.75
    assert majority_share(["300", "300", "999", "999"]) == 0.5
    assert majority_share(["1", "2", "3", "4"]) == 0.25
    # A correct majority and a wrong one are the same number: no truth involved.
    assert (majority_share(["300", "300", "999"])
            == majority_share(["999", "999", "300"]))


def test_unparseable_replies_do_not_look_like_agreement():
    """Dropping them would make an incoherent batch read as certain, which is
    the opposite of the Challenger's target."""
    from examples.r_zero.r_zero_challenger_solver import majority_share

    assert majority_share(["$3.00", "three", "?", ""]) == 0.25


def test_the_uncertainty_reward_peaks_at_a_half():
    """`min(score, 1 - score)`, maximal when the Solver agrees with itself half
    the time -- and this is why `DifficultyWeighted`'s 4p(1-p) is attached here
    and not to Absolute Zero, whose 1-r is monotone instead."""
    from examples.r_zero.r_zero_challenger_solver import majority_share

    rates = [majority_share(f) for f in (["1", "1", "1", "1"],
                                         ["1", "1", "1", "2"],
                                         ["1", "1", "2", "2"])]
    rewards = [min(p, 1 - p) for p in rates]
    assert rewards == [0.0, 0.25, 0.5]
    assert rewards[2] == max(rewards)


def test_r_zero_samples_the_solver_more_than_twice():
    """Two samples give the majority share only two values, 0.5 and 1.0, so the
    Challenger sees a coin flip rather than a frontier."""
    from examples.r_zero import r_zero_challenger_solver as rz

    assert rz.SOLVER_SAMPLES >= 4


# -- Agent0 -------------------------------------------------------------------


def test_the_tool_reward_is_the_capped_weighted_call_count():
    """`calculate_tool_reward(predict, weight=0.05, cap=4)`:
    `min(tool_call_count, cap) * weight`. The port's prompt said "with a
    tool-use bonus" and carried no number, while its notes claimed the component
    was surfaced."""
    from examples.agent0.agent0_tool_curriculum import tool_reward, TOOL_CAP

    assert tool_reward(0) == 0.0
    assert tool_reward(1) == 0.05
    assert tool_reward(4) == 0.2
    assert tool_reward(9) == tool_reward(TOOL_CAP), "the cap does not bind"
    assert tool_reward(-1) == 0.0


def test_agent0_uncertainty_is_self_consistency_not_the_grounded_reward():
    """`min(score, 1 - score)` over `generate_results`' majority share -- the
    same computation R-Zero uses. The port fed it the single rollout's grounded
    reward, which is 0 or 1, and `1 - 2|p - 0.5|` is zero at both: the
    curriculum signal did not exist on any item of any run.

    Upstream's term is also `min(p, 1-p)`, where the port used `1 - 2|p - 0.5|`
    -- exactly twice it.
    """
    from examples._selfplay_domain import majority_share
    from examples.agent0 import agent0_tool_curriculum as ag

    assert ag.EXECUTOR_SAMPLES >= 4

    for finals, want in ((["300"] * 4, 0.0),
                         (["300", "300", "300", "9"], 0.25),
                         (["300", "300", "9", "9"], 0.5)):
        p_hat = majority_share(finals)
        assert min(p_hat, 1 - p_hat) == want, finals

    # The grounded reward, which the port used, is 0 or 1 -- and both give zero.
    for score in (0.0, 1.0):
        assert 1.0 - 2.0 * abs(score - 0.5) == 0.0


def test_the_two_ports_share_one_majority_share():
    """R-Zero's `question_evaluate/evaluate.py` and Agent0's `generate_results`
    are the same computation, so a fix to one should not leave the other."""
    from examples._selfplay_domain import majority_share as shared
    from examples.r_zero import r_zero_challenger_solver as rz

    assert rz.majority_share is shared
