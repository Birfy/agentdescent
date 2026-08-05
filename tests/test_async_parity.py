"""Capabilities the general engine was missing, and one seam the adapter keeps.

This file used to guard **two independent barrier-free implementations** against
drifting apart: `AsyncAgentDescent` had its own worker threads, merger thread,
published head and backpressure, and shared no code with `async_evolve`. The cost
was that fixes and capabilities lived in whichever one they were written for --
the worker-retirement heuristic was measured wrong and fixed in one, backpressure
and duration-aware straggler detection existed only in the other.

`AsyncAgentDescent` is an adapter now (#93), so most of what was here tested
`async_evolve` through a wrapper -- which `test_fault_matrix.py` asserts directly
against both engines, and `test_worker_resilience.py` asserts in more detail.
Those went. What is left is what only this file says:

* the capabilities that were reference-only are reachable from the general API
  and actually fire -- backpressure, straggler detection, and the diagnostics
  that report them;
* one test of the **adapter's own seam** (`rollout=`), because a broken seam
  would otherwise make every reference example silently stop exercising failures;
* one that the retirement rule has a single implementation, by name -- the two
  loops are `evolve` and `async_evolve` now, and that is still two.
"""

import tempfile
import time
import warnings

import pytest

from agentdescent import AppendRules, Task, async_evolve
from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.aggregator import Aggregator
from agentdescent.domains.router import make_task_universe, router_run
from agentdescent.scheduler import DurationEstimator


def _tasks(n=12):
    return [Task(id=str(i), prompt=f"q{i}", meta={"gold": "y"}) for i in range(n)]


def _async_evolve(**kw):
    kw.setdefault("n_workers", 3)
    kw.setdefault("max_seconds", 3.0)
    kw.setdefault("self_verify", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return async_evolve(_tasks(), kw.pop("reward", lambda t, o: 0.0),
                            strategy=AppendRules(), **kw)


# -- the retirement heuristic, on both sides ------------------------------------


def test_a_misconfigured_backend_still_retires_every_worker_fast():
    """While nothing has ever succeeded, give up loudly rather than spin.

    Kept out of a larger group that all asserted this through the adapter,
    because it is also the only test of the adapter's `rollout=` seam: if that
    stopped reaching the engine, every reference example would quietly stop
    exercising failures and nothing else would notice.
    """
    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=3, max_seconds=20.0, seed=1)
    with tempfile.TemporaryDirectory() as repo:
        def dead(rendered, task):
            raise RuntimeError("401 unauthorized")

        system = AsyncAgentDescent(repo, universe, config=cfg, rollout=dead)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = system.run()
    assert stats.error is not None and "401" in stats.error, "ended silently"
    assert time.time() - t0 < 15.0, "burned the whole budget on a dead backend"


# -- capabilities the general path was missing ----------------------------------


def test_backpressure_is_reachable_from_the_general_api():
    """`concepts.md` says this guard is what keeps `async_ratio > alpha` from
    livelocking under Guarded. It existed only in the reference runtime."""
    res = _async_evolve(async_ratio=50, stall_patience=1,
                        reward=lambda t, o: 0.0,
                        run=lambda r, t: "no",
                        propose=lambda r, t, o, s: f"rule {time.time()}")
    assert res.forced_refreshes >= 0        # the field exists and is reported
    assert hasattr(res, "forced_refreshes")


def test_a_stalled_pipeline_forces_a_resync():
    """Cards arriving, nothing committing: that is a livelock, not slow progress."""
    n = [0]

    def propose(rendered, task, output, score):
        n[0] += 1
        return f"rule number {n[0]}"

    res = _async_evolve(async_ratio=1000, stall_patience=1, max_seconds=4.0,
                        reward=lambda t, o: 0.5, run=lambda r, t: "no",
                        propose=propose)
    assert res.forced_refreshes > 0, (
        "the pipeline stalled and no worker was ever told to resync; "
        "with a lag budget this large nothing would move head on its own")


def test_straggler_detection_is_reachable_from_the_general_api():
    """L-traj lived only in a runtime that accepts nothing but TaskUniverse."""
    slow_id = "3"
    tasks = [Task(id=str(i), prompt="q" * (200 if str(i) == slow_id else 20),
                  meta={"gold": "y"}) for i in range(12)]

    def run(rendered, task):
        time.sleep(0.35 if task.id == slow_id else 0.01)
        return "no"

    est = DurationEstimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = async_evolve(tasks, lambda t, o: 0.0, run=run,
                           propose=lambda r, t, o, s: f"rule {time.time()}",
                           strategy=AppendRules(), n_workers=3, max_seconds=4.0,
                           duration_estimator=est, self_verify=False)
    assert res.stragglers > 0, "the one deliberately slow rollout was not detected"
    intercept, slope = est.params
    assert slope != 0.0, "the estimator never fitted a cost law"


def test_no_estimator_means_no_straggler_accounting():
    """Opt-in: timing every rollout against a fitted law is not free."""
    res = _async_evolve(run=lambda r, t: "no", propose=lambda *a: None)
    assert res.stragglers == 0


def test_the_new_diagnostics_survive_save_and_load(tmp_path):
    from agentdescent import EvolutionResult

    res = _async_evolve(run=lambda r, t: "no", propose=lambda *a: None)
    path = tmp_path / "r.json"
    res.save(str(path))
    loaded = EvolutionResult.load(str(path))
    assert loaded.forced_refreshes == res.forced_refreshes
    assert loaded.stragglers == res.stragglers


def test_both_loops_retire_workers_through_the_same_object():
    """The rule that was hand-ported once must not be re-implemented again.

    `pipeline.py`'s module docstring records that a measured fix had to be
    carried across by hand between two implementations of the same shape. The
    reference runtimes are adapters now, but `evolve` and `async_evolve` are
    still two loops, and the synchronous one had re-grown its own copy once
    already (`any_success` + `dead_rounds >= max_worker_errors`). This asserts
    there is one implementation, by name, in both.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "agentdescent"
    for module in ("evolution.py", "async_evolve.py"):
        text = (src / module).read_text(encoding="utf-8")
        assert "WorkerHealth" in text, f"{module} does not use the shared rule"
        assert "should_retire" in text, f"{module} does not ask it the question"
        # the shape of the re-implementation, so it cannot come back quietly
        assert not re.search(r"any_success\[0\]\s*and", text), (
            f"{module} has re-implemented the retirement rule inline")
