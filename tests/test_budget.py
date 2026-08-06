"""The budget an equal-budget comparison is actually allowed to fix.

`rounds` is not one. Configurations differ in how much model a round buys --
`n_workers=8` buys eight times what `n_workers=1` does -- so a budget fixed in
rounds hands the wider configuration more model and then reports the extra model
as a win for parallelism. `bench/harness.py` says this in prose; these tests are
what makes it enforceable, on both runtimes and in both units.

The overshoot is tested, not hidden. The synchronous path checks at the round
barrier because a half-merged round is not a state worth comparing, so a run ends
*past* its budget by up to one round's worth. A caller who compares on the number
they asked for rather than the number the run reported gets a wrong answer
quietly; the test below pins how far apart those two can be.
"""

import pytest

from agentdescent.evolution import AppendRules, Task, evolve


def _tasks(n=20):
    return [Task(id=str(i), prompt="q", meta={"h": f"H{i % 5}"}) for i in range(n)]


def _run(rendered, task):
    return "good" if task.meta["h"] in rendered else "bad"


def _reward(task, output):
    return 1.0 if output == "good" else 0.0


def _propose(rendered, task, output, score):
    return task.meta["h"]


def _evolve(**kw):
    kw.setdefault("rounds", 20)
    kw.setdefault("n_workers", 4)
    kw.setdefault("max_concurrency", 4)
    return evolve(_tasks(), _reward, run=_run, propose=_propose,
                  strategy=AppendRules(), held_out_frac=0.4, **kw)


# -- synchronous ------------------------------------------------------------


def test_rollout_budget_stops_the_run_and_says_so():
    res = _evolve(max_rollouts=20)
    assert res.stop_reason == "max_rollouts"
    assert res.error is None
    assert len(res.history) < 20, "it must stop before spending every round"


def test_call_budget_stops_the_run_and_says_so():
    res = _evolve(max_calls=20)
    assert res.stop_reason == "max_calls"
    assert res.error is None


def test_the_two_budgets_are_not_derivable_from_each_other():
    """A rollout that solves its task never proposes, so calls != k x rollouts.

    This is why both bounds exist. If the ratio were fixed, `max_calls` would be
    decoration on `max_rollouts`.
    """
    res = _evolve(max_rollouts=40)
    ratio = res.usage.calls / max(1, res.rollouts)
    assert ratio != pytest.approx(2.0, abs=1e-9), \
        "a fixed 2x would mean every rollout proposed; then one budget suffices"


def test_the_overshoot_is_bounded_by_one_round():
    """The barrier check means a run ends past its budget -- but not unboundedly.

    Compare on what the run reported, never on what it was asked for; the gap
    below is exactly why.
    """
    res = _evolve(max_rollouts=20)
    per_round = max(h.rollouts - prev.rollouts
                    for prev, h in zip(res.history, res.history[1:])) \
        if len(res.history) > 1 else res.rollouts
    assert res.rollouts >= 20
    assert res.rollouts <= 20 + per_round


def test_a_budget_nothing_reaches_changes_nothing():
    """An unreached bound must leave the run's own stopping point intact."""
    unbounded = _evolve()
    generous = _evolve(max_rollouts=10 ** 6, max_calls=10 ** 6)
    assert generous.stop_reason == unbounded.stop_reason
    assert len(generous.history) == len(unbounded.history)


def test_default_is_unbounded():
    res = _evolve(rounds=3)
    assert res.stop_reason == "rounds"


def test_a_budget_in_rounds_hands_the_wider_configuration_more_model():
    """Guard the premise. If this ever stopped being true the rest is pointless."""
    narrow = _evolve(rounds=3, n_workers=1, max_concurrency=1)
    wide = _evolve(rounds=3, n_workers=8, max_concurrency=8)
    assert wide.rollouts > narrow.rollouts * 2, \
        f"rounds bought {narrow.rollouts} vs {wide.rollouts}; the whole file " \
        "exists because that gap gets reported as a win for parallelism"


def test_a_budget_in_rollouts_does_not():
    """The fix: the same two configurations, now spending comparably.

    Not *identically*: the barrier check means each overshoots by up to its own
    last round, and the wide configuration's round is wider. Within a factor of
    two is what a barrier-checked budget can promise, against the eight above.
    """
    narrow = _evolve(rounds=10_000, n_workers=1, max_concurrency=1,
                     max_rollouts=40)
    wide = _evolve(rounds=10_000, n_workers=8, max_concurrency=8, max_rollouts=40)
    assert narrow.stop_reason == wide.stop_reason == "max_rollouts"
    assert wide.rollouts <= narrow.rollouts * 2, \
        f"narrow spent {narrow.rollouts}, wide spent {wide.rollouts}"


# -- barrier-free -----------------------------------------------------------


def test_async_takes_the_same_two_budgets():
    res = _evolve(asynchronous=True, max_seconds=20.0, max_rollouts=12,
                  max_concurrency=1)
    assert res.stop_reason in ("max_iters", "max_calls", "target_reward",
                               "max_seconds")
    assert res.error is None


def test_async_max_rollouts_silences_the_rounds_reinterpretation_warning():
    """`rounds` on the async path is a rollout budget in disguise; saying the
    budget outright is the way to stop guessing at it."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _evolve(asynchronous=True, max_seconds=5.0, rounds=3, max_concurrency=1)
    assert any("reinterpreted as a budget" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _evolve(asynchronous=True, max_seconds=5.0, rounds=3, max_rollouts=12,
                max_concurrency=1)
    assert not any("reinterpreted as a budget" in str(w.message) for w in caught), \
        "max_rollouts= makes the budget explicit, so there is nothing to warn about"
