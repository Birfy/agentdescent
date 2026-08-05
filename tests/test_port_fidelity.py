"""Constants and control flow the ports claim to take from the original repos.

`docs/self-evolution-examples.md` states the rule these follow: *"Faithful to the
repo, not just the paper. Where the released code diverges from the paper's
claims, the example follows the code and says so."* That only means anything if
the traceable parts are pinned, so this file asserts the ones with an exact
upstream source -- the numbers a reader would otherwise have to diff by hand.

Offline: every value here is a literal, checked against the line it was taken
from. No network, no clone.
"""

import inspect

import pytest

from examples.adas import adas_meta_agent_search as adas
from examples.dgm import dgm_self_improve as dgm
from examples.evoskill import evoskill_skill_discovery as evoskill


# -- DGM (jennyzzt/dgm) ---------------------------------------------------------


def test_the_parent_selection_rule_matches_dgm_outer():
    """`DGM_outer.py:91-100`, method `score_child_prop`::

        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(...)]
    """
    weights = dgm.dgm_parent_weights(scores=[0.5, 0.5], children=[0, 1])
    # equal scores, so the ratio is decided by the novelty term alone: 1 vs 1/2
    assert weights[0] == pytest.approx(2 * weights[1])
    assert sum(weights) == pytest.approx(1.0)
    # the sigmoid is steep (x10) and centred on 0.5
    high, low = dgm.dgm_parent_weights([0.9, 0.1], [0, 0])
    assert high > 40 * low, "the sigmoid should be sharply peaked at 10x"


def test_the_staged_eval_subset_sizes_and_threshold():
    """`swe_bench/subsets/{small,medium,big}.json` really are 10 / 50 / 140, and
    `DGM_outer.py:268` sets `test_more_threshold = 0.4`."""
    assert (dgm.STAGE_SMALL, dgm.STAGE_MEDIUM, dgm.STAGE_BIG) == (10, 50, 140)
    assert dgm.TEST_MORE_THRESHOLD == 0.4


def test_the_ladder_stops_at_medium_as_upstreams_loop_does():
    """Upstream's self-improve loop passes exactly two subsets:

        self_improve(test_task_list=swe_issues_sm,
                     test_more_threshold=test_more_threshold,
                     test_task_list_more=swe_issues_med, ...)

    `big.json` belongs to the separate full-evaluation path, gated by the
    *archive-relative* `full_eval_threshold` -- not by the fixed 0.4. Running it
    as a third rung on the same threshold changed what `agent.score` means: a high
    scorer's became a 140-instance number while a low scorer's stayed a
    10-instance one, and both fed the same selection sigmoid.
    """
    seen = []

    def evaluate_fn(agent, tasks):
        seen.append(len(tasks))
        return 1.0                       # always clears the threshold

    score = dgm.staged_evaluate(
        dgm.initial_agent(), evaluate_fn,
        small=[{}] * 10, medium=[{}] * 50, big=[{}] * 140)
    assert seen == [10, 50], f"escalated to {seen}, upstream stops after medium"
    assert score == 1.0


def test_a_low_scorer_never_escalates():
    seen = []
    dgm.staged_evaluate(dgm.initial_agent(),
                        lambda a, ts: (seen.append(len(ts)), 0.1)[1],
                        small=[{}] * 10, medium=[{}] * 50)
    assert seen == [10], "escalation must require score > test_more_threshold"


# -- ADAS (ShengranHu/ADAS) -----------------------------------------------------


def test_the_seven_mgsm_seeds_match_get_init_archive():
    """`_mgsm/mgsm_prompt.py:get_init_archive()` -- name for name."""
    names = {a["name"] for a in adas.seed_archive()}
    assert names == {
        "Chain-of-Thought",
        "Self-Consistency with Chain-of-Thought",
        "Self-Refine (Reflexion)",
        "LLM Debate",
        "Step-back Abstraction",
        "Quality-Diversity",
        "Dynamic Assignment of Roles",
    }


def test_the_bootstrap_resample_deviation_is_documented():
    """Upstream `_mgsm/utils.py:88` uses `num_bootstrap_samples=100000`.

    2000 is a deliberate speed trade for an example that runs in seconds. A
    deviation in a file whose page says "faithful" has to be stated, not inferred.
    """
    assert inspect.signature(adas.bootstrap_ci).parameters["n_resamples"].default == 2000
    doc = inspect.getdoc(adas.bootstrap_ci) or ""
    assert "100000" in doc, "the upstream value must appear where the deviation is"


def test_the_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = adas.bootstrap_ci([1.0] * 7 + [0.0] * 3)
    assert mean == pytest.approx(0.7, abs=0.05)
    assert lo <= mean <= hi


# -- EvoSkill (sentient-agi/EvoSkill) -------------------------------------------


def test_the_tolerance_ladder_and_pass_threshold():
    """`src/loop/runner.py:79` and `:319` -- both verbatim."""
    assert evoskill.TOLERANCE_LEVELS == [0.05, 0.01, 0.1, 0.0, 0.025]
    assert evoskill.PASS_THRESHOLD == 0.8


def test_the_tolerance_weights_favour_strictness():
    """`weight = 1 / (1 + 20 * tolerance)`, per the upstream docstring."""
    for tol in evoskill.TOLERANCE_LEVELS:
        assert 1.0 / (1.0 + 20.0 * tol) > 0.0
    # exact match: 0.0 -> 1.00, 0.01 -> 0.83, 0.025 -> 0.67, 0.05 -> 0.50, 0.1 -> 0.33
    assert 1.0 / (1.0 + 20.0 * 0.0) == pytest.approx(1.00, abs=0.005)
    assert 1.0 / (1.0 + 20.0 * 0.01) == pytest.approx(0.83, abs=0.005)
    assert 1.0 / (1.0 + 20.0 * 0.1) == pytest.approx(0.33, abs=0.005)


def test_the_frontier_size_matches_the_registry_default():
    """`src/registry/manager.py:379` -- `def update_frontier(..., max_size: int = 5)`."""
    assert evoskill.Frontier().max_size == 5


def test_the_frontier_is_a_bounded_top_k_leaderboard_not_a_pareto_front():
    """The paper's abstract says "a Pareto frontier of agent programs governs
    selection"; `update_frontier` is a leaderboard on one scalar. The stated rule
    is to follow the code, so this pins the code's behaviour."""
    f = evoskill.Frontier(max_size=2)
    assert f.update({"a": "x"}, 0.5), "admit while there is room"
    assert f.update({"b": "x"}, 0.9)
    assert not f.update({"c": "x"}, 0.1), \
        "a worse candidate must not displace a better one"
    assert f.update({"d": "x"}, 0.7), "a better candidate replaces the worst member"
    assert len(f.members) == 2
    assert sorted(score for _, score in f.members) == [0.7, 0.9]
    assert f.select_parent()[1] == 0.9, "the parent is the best member (strategy=best)"
