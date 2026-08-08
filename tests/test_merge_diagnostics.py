"""A run that commits nothing must say why.

`committed`/`rejected` counts cannot distinguish "proposals reached the gate and
failed to beat the baseline" from "they never reached it" -- and those need
opposite fixes.
"""
from agentdescent.evolution import SingleSlot, Task, evolve

TASKS = [Task(id=str(i), prompt="p", meta={"gold": "g"}) for i in range(6)]


def _run(**kw):
    # The proposal must be *decisively* unhelpful: with candidate and baseline
    # held-out rewards exactly equal, P(delta>0) sits on the annealed threshold
    # and the accept draw is a near coin flip -- ~3% of seeded runs committed
    # the no-op diff in round 1, after which SingleSlot deduped every later
    # proposal and `below-threshold` never appeared. A strictly worse candidate
    # keeps the test's point (proposals reach the gate and fail) without the
    # knife edge; defaults.py labels the measured drop `below-threshold` too.
    return evolve(TASKS, lambda t, o: 0.5 if o == "x" else 0.2,
                  run=lambda a, t: "x" if a == "v" else "y",
                  propose=lambda *a: "does not help",
                  strategy=SingleSlot(initial_value="v"),
                  rounds=4, held_out_frac=0.5, **kw)


def test_rejections_name_a_category():
    res = _run(n_workers=2, max_concurrency=2)
    assert res.outcomes(), "a run that rejected everything reported no reason"
    assert "below-threshold" in res.outcomes()


def test_categories_are_countable_across_rounds():
    """`reason` interpolates measured numbers, so it cannot be a tally key."""
    res = _run(n_workers=2, max_concurrency=2)
    assert sum(res.outcomes().values()) == sum(
        h.committed + h.rejected for h in res.history)


def test_outcomes_survive_save_and_load():
    import os, tempfile
    from agentdescent.evolution import EvolutionResult
    res = _run(n_workers=2, max_concurrency=2)
    path = tempfile.mktemp(suffix=".json")
    try:
        res.save(path)
        assert EvolutionResult.load(path).outcomes() == res.outcomes()
    finally:
        os.path.exists(path) and os.unlink(path)


def test_the_async_path_reports_them_too():
    res = _run(n_workers=2, asynchronous=True, max_seconds=4.0)
    assert res.outcomes()


def test_a_committing_run_is_labelled_committed():
    """The category is not a rejection-only field."""
    tasks = [Task(id=str(i), prompt=f"{i}", meta={"gold": str(i)}) for i in range(8)]
    res = evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                 run=lambda rendered, t: t.prompt if "echo" in rendered else "?",
                 propose=lambda *a: "echo",
                 strategy=SingleSlot(initial_value="guess"),
                 rounds=3, n_workers=2, max_concurrency=2, held_out_frac=0.5)
    assert res.outcomes().get("committed", 0) >= 1
