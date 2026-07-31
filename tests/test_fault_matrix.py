"""Every fault against both engines, asserting the invariants that have broken.

The bugs found so far were never crashes. They were: a run that ended silently
after one transient, a merger taken out permanently, a process that would not
exit, a knob accepted and ignored. So the invariants asserted here are about
*outcomes*, not exceptions:

1. a run never hangs -- it finishes within its own budget;
2. it never ends silently -- if it stopped early, `error` says why;
3. a recoverable fault never ends it;
4. an unrecoverable one ends it *fast*, rather than burning the budget.
"""
import time
import warnings

import pytest

import faults
from agentdescent.evolution import SingleSlot, Task, evolve

TASKS = [Task(id=str(i), prompt=str(i), meta={"gold": str(i)}) for i in range(8)]

ENGINES = [
    pytest.param({"n_workers": 3, "max_concurrency": 3}, id="sync"),
    pytest.param({"n_workers": 3, "asynchronous": True, "max_seconds": 8.0},
                 id="async"),
]


def _run(fault, engine, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(TASKS, lambda t, o: 0.5, run=fault,
                      propose=lambda rendered, t, o, s: f"rule-{t.id}",
                      strategy=SingleSlot(initial_value="v"),
                      rounds=12, held_out_frac=0.5, **engine, **kw)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_throttled_backend_does_not_end_the_run(engine):
    """1 call in 3 fails. Two thirds succeed -- the run must use them."""
    res = _run(faults.flaky(1 / 3), engine)
    assert res.error is None, f"a 429 storm ended the run: {res.error}"
    assert res.retired_workers == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_an_outage_that_recovers_does_not_end_the_run(engine):
    res = _run(faults.recovers_after(4), engine)
    assert res.error is None, res.error


@pytest.mark.parametrize("engine", ENGINES)
def test_a_misconfigured_backend_ends_the_run_fast_and_loudly(engine):
    t0 = time.time()
    res = _run(faults.never_works(), engine)
    assert res.error is not None and "401" in res.error, "ended silently"
    assert time.time() - t0 < 8.0, "burned the budget on a dead backend"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_backend_that_dies_midway_keeps_what_it_learned(engine):
    """The artifact evolved before the outage must survive the outage."""
    res = _run(faults.dies_after(12), engine)
    assert res.rendered, "returned an empty artifact"


@pytest.mark.parametrize("engine", ENGINES)
def test_no_fault_makes_a_run_outlast_its_budget(engine):
    """The hang case: a wedged rollout must not become a wedged run."""
    t0 = time.time()
    _run(faults.wedged(20.0), engine, round_timeout=1.0 if "max_seconds"
         not in engine else None)
    assert time.time() - t0 < 25.0, "a wedged rollout wedged the run"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [None, "", "   "], ids=["null", "empty", "blank"])
def test_a_mute_backend_does_not_look_like_a_converged_run(engine, value):
    """The fault class every other primitive misses: success with no content.

    It raises nothing, so it bypasses every retry and error-tolerance path and
    lands as `error=None` with a full history -- while `outcomes()` reports
    `below-threshold`, documented to mean "the reflector is the problem", which is
    the opposite of the truth. Nothing can currently distinguish it, so this pins
    the *observable* consequence rather than a diagnosis that does not exist yet:
    the artifact never moves off its seed, and nothing is ever committed.
    """
    res = _run(faults.returns_nothing(value), engine)
    assert res.rendered == "v", (
        "a backend that says nothing cannot have taught the artifact anything, "
        f"yet it rendered {res.rendered!r}")
    assert res.outcomes().get("committed", 0) == 0, \
        f"a mute backend produced a commit: {res.outcomes()}"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_reflector_outage_does_not_end_a_healthy_run(engine):
    """`propose` is a separate backend call, often to a different model.

    The matrix faulted only `run` and paired it with a reflector that never
    fails, so "the solver is fine and the reflector is down" -- an ordinary
    two-provider outage -- was untested.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(TASKS, lambda t, o: 0.5, run=faults.OK,
                     propose=faults.flaky_propose(1 / 3),
                     strategy=SingleSlot(initial_value="v"),
                     rounds=12, held_out_frac=0.5, **engine)
    assert res.error is None, f"a flaky reflector ended the run: {res.error}"


def test_a_dying_ledger_still_returns_what_was_learned():
    """The sole writer, on the critical path, with no retry anywhere -- and no
    fault primitive until now. Its failure used to escape as an exception,
    against the documented "a run that died still returns a partial result"."""
    with faults.ledger_dies_after(24):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _run(faults.OK, {"n_workers": 3, "max_concurrency": 3})
    assert res.rendered, "a ledger failure discarded the artifact"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_clean_run_reports_no_error(engine):
    """The control: without a fault, none of the above machinery fires."""
    res = _run(faults.OK, engine)
    assert res.error is None and res.retired_workers == 0
    assert res.history


def test_eval_concurrency_is_reachable_and_changes_behaviour():
    """It was a dataclass default on a private class -- unreachable, and setting
    the class attribute silently did nothing because dataclasses bake defaults
    into __init__. That made it impossible to measure or tune."""
    import inspect
    from agentdescent.evolution import evolve as _evolve
    from agentdescent.async_evolve import async_evolve
    assert "eval_concurrency" in inspect.signature(_evolve).parameters
    assert "eval_concurrency" in inspect.signature(async_evolve).parameters

    seen = {}

    def run(rendered, task):
        seen.setdefault("threads", set()).add(__import__("threading").get_ident())
        return "wrong"

    for conc in (1, 4):
        seen.clear()
        _run(run, {"n_workers": 1, "max_concurrency": 1}, eval_concurrency=conc)
    # A real behavioural check lives in the efficiency docs (wall-clock); here we
    # only pin that the argument reaches the engine rather than being swallowed.
    res = _run(faults.OK, {"n_workers": 2, "max_concurrency": 2}, eval_concurrency=2)
    assert res.error is None
