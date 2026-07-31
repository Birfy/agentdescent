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
def test_a_clean_run_reports_no_error(engine):
    """The control: without a fault, none of the above machinery fires."""
    res = _run(faults.OK, engine)
    assert res.error is None and res.retired_workers == 0
    assert res.history
