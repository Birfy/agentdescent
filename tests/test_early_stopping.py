"""Stop buying rollouts once the artifact has converged.

Without early stopping a run always spent all `rounds`. Measured on a workload
that converges after two: 141 model calls for a result reached at call 69 — 51% of
the budget spent re-running tasks against an artifact that had stopped changing.
That is real money on an LLM backend.
"""

import pytest

from agentdescent.evolution import AppendRules, Task, evolve


def _run(**kw):
    calls = {"n": 0}

    def run(rendered, task):
        calls["n"] += 1
        return "good" if task.meta["h"] in rendered else "bad"

    tasks = [Task(id=str(i), prompt="q", meta={"h": f"H{i % 5}"}) for i in range(20)]
    res = evolve(tasks, lambda t, o: 1.0 if o == "good" else 0.0, run=run,
                 propose=lambda rendered, t, o, s: t.meta["h"],
                 strategy=AppendRules(), rounds=20, n_workers=4,
                 max_concurrency=4, held_out_frac=0.4, **kw)
    return res, calls["n"]


def test_target_reward_stops_early_and_saves_calls():
    baseline, baseline_calls = _run()
    early, early_calls = _run(target_reward=1.0)

    assert early.final_reward >= 1.0, "must still reach the target"
    assert len(early.history) < len(baseline.history)
    assert early_calls < baseline_calls * 0.75, \
        f"expected a large saving, got {early_calls} vs {baseline_calls}"


def test_patience_stops_a_plateau():
    res, _ = _run(patience=3)
    assert len(res.history) < 20
    assert res.final_reward > 0.0


def test_patience_does_not_fire_while_improving():
    """It must count *consecutive* non-improving rounds, not any of them."""
    res, _ = _run(patience=10)
    assert res.final_reward == 1.0


def test_no_early_stopping_by_default():
    res, _ = _run()
    assert len(res.history) == 20, "default behaviour must be unchanged"


def test_target_reward_is_reported_through_history():
    res, _ = _run(target_reward=1.0)
    assert res.history[-1].held_out_reward >= 1.0
    assert res.error is None


def test_both_knobs_are_documented():
    import inspect

    doc = evolve.__doc__ or ""
    for name in ("target_reward", "patience"):
        assert name in doc
        assert name in inspect.signature(evolve).parameters
