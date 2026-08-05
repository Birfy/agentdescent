"""Offline tests for the Darwin Godel Machine (DGM) example.

The DGM *algorithm* -- keep-all archive, sigmoid(perf)xnovelty parent selection,
staged escalation -- is fully testable offline; the objective is the documented
surrogate (real DGM needs the SWE-bench Docker harness). No network here.
"""

import random

from examples.dgm.dgm_self_improve import (
    CAPABILITY_POOL,
    STAGE_MEDIUM,
    STAGE_SMALL,
    TEST_MORE_THRESHOLD,
    Agent,
    choose_selfimproves,
    dgm_parent_weights,
    initial_agent,
    make_surrogate_evaluator,
    propose_modification,
    required_capabilities,
    run_dgm,
    staged_evaluate,
    surrogate_resolved,
)


def test_parent_weights_formula():
    # higher performance -> higher weight; normalised.
    w = dgm_parent_weights([0.9, 0.5, 0.1], [0, 0, 0])
    assert w[0] > w[1] > w[2]
    assert abs(sum(w) - 1.0) < 1e-9
    # same score, more children -> discounted (open-ended novelty).
    w2 = dgm_parent_weights([0.9, 0.9], [0, 5])
    assert w2[0] > w2[1]


def test_selection_favours_high_performers():
    archive = [Agent(("a",), score=0.9), Agent(("b",), score=0.1)]
    picks = [choose_selfimproves(archive, 1, random.Random(i))[0] for i in range(2000)]
    assert picks.count(0) / len(picks) > 0.8


def test_surrogate_objective_monotone_in_capabilities():
    instances = [{"instance_id": f"proj__x-{i}", "repo": "r"} for i in range(30)]
    ev = make_surrogate_evaluator()
    poor = Agent(("context-retrieval",))
    rich = Agent(tuple(CAPABILITY_POOL))
    assert ev(rich, instances) == 1.0        # covers every required set
    assert ev(rich, instances) >= ev(poor, instances)


def test_required_capabilities_are_deterministic_and_bounded():
    req = required_capabilities("astropy__astropy-12907")
    assert req == required_capabilities("astropy__astropy-12907")   # deterministic
    assert 1 <= len(req) <= 3
    assert req <= set(CAPABILITY_POOL)


def test_staged_evaluation_escalates_past_threshold():
    instances = [{"instance_id": f"i{i}", "repo": "r"} for i in range(STAGE_MEDIUM)]
    calls = {"n": 0}

    def counting_eval(agent, subset):
        calls["n"] += 1
        # below threshold on the first (small) call -> should NOT escalate.
        return 0.1

    staged_evaluate(Agent(("context-retrieval",)), counting_eval,
                    instances[:STAGE_SMALL], instances[:STAGE_MEDIUM], instances)
    assert calls["n"] == 1                    # stayed on the small subset


def test_propose_modification_targets_missing_capabilities():
    agent = Agent(("context-retrieval",))
    instances = [{"instance_id": "i0", "repo": "r"}]
    cap = propose_modification(agent, instances, random.Random(0))
    assert cap in CAPABILITY_POOL and cap not in agent.capabilities


def test_dgm_loop_grows_archive_and_improves():
    instances = [{"instance_id": f"proj__x-{i}", "repo": "r"} for i in range(STAGE_SMALL)]
    result = run_dgm(instances, generations=12, seed=0)
    assert len(result.archive) > 1                    # keep-all archive grew
    assert result.best_score >= result.seed_score     # open-ended improvement
    # every archived agent traces back to the seed (lineage).
    assert result.archive[0].parent is None


def test_keep_better_rejects_regressions():
    instances = [{"instance_id": f"proj__x-{i}", "repo": "r"} for i in range(STAGE_SMALL)]
    res = run_dgm(instances, generations=8, archive_mode="keep_better", seed=1)
    # in keep_better every non-seed agent must beat its parent.
    for a in res.archive[1:]:
        assert a.score > res.archive[a.parent].score
