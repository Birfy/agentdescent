"""The two async pipelines must not diverge in behaviour.

`AsyncAgentDescent` (reference stack, synthetic router domain) and `async_evolve`
(general, and the only one a real workload reaches) implement the same
barrier-free shape independently and share no code. The cost was that fixes and
capabilities lived in whichever one they were written for:

* the worker-retirement heuristic was **measured wrong** and fixed in
  `async_evolve` -- "at a 1-in-3 call failure rate the old blanket rule retired
  all three workers in 22s with nothing learned" -- while `AsyncAgentDescent` kept
  the rule that paragraph describes, along with a merger that ended the run on its
  first exception (the other pattern that measurement removed);
* backpressure and duration-aware straggler detection existed only in
  `AsyncAgentDescent`, which accepts nothing but `TaskUniverse` -- so the guard
  `concepts.md` says prevents a livelock, and the whole L-traj mechanism, were
  unreachable from the API every real workload uses.

These pin the behaviour on *both* sides rather than the implementation, so they
keep holding if the two are ever merged into one.
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


def test_the_reference_runtime_no_longer_sheds_workers_over_a_transient():
    """Every worker shares one backend, so shedding workers cannot relieve
    throttling -- it only guarantees the run dies."""
    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=4, max_seconds=3.0, seed=1)
    with tempfile.TemporaryDirectory() as repo:
        calls = {"n": 0}

        def flaky(rendered, task):
            calls["n"] += 1
            if calls["n"] > 2 and calls["n"] % 3:   # ~2 in 3, once it has worked
                raise RuntimeError("429 rate limited")
            return router_run(rendered, task)

        # This used to patch `system.workers[i].run`. The runtime is an adapter
        # over `async_evolve` now, so the seam is the rollout itself -- and the
        # merger's held-out scoring goes through it too, which the old seam
        # bypassed. Hence the two guaranteed successes: the invariant under test
        # is "once the backend has demonstrably worked, do not shed workers", so
        # the run has to reach that state before the failures start.
        system = AsyncAgentDescent(repo, universe, config=cfg, rollout=flaky)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = system.run()
    assert stats.retired_workers == 0, (
        f"{stats.retired_workers} worker(s) retired over a transient the backend "
        "recovers from; shedding them cannot relieve a shared throttle")


def test_a_misconfigured_backend_still_retires_every_worker_fast():
    """The other half: while nothing has ever succeeded, give up loudly."""
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


def test_the_reference_merger_survives_a_transient():
    """It scores held-out every sweep, so it is a backend caller like any other.

    Ending the run on its first exception made the merger a single point of
    failure -- measured in `async_evolve` as 0 sweeps while the workers were fine.
    """
    universe = make_task_universe(seed=7)
    # Generous budget on purpose: the retry backs off 2s after one failure, which
    # would be half of a short run and would test the clock, not the tolerance.
    cfg = AsyncConfig(n_workers=2, max_seconds=12.0, target_accuracy=2.0, seed=1)
    with tempfile.TemporaryDirectory() as repo:
        calls = {"n": 0}

        def factory(ledger, verifier, audit, config, policy):
            # The aggregator is built by the run now, so the injection point is
            # the factory rather than an attribute patched before it starts.
            agg = Aggregator(ledger, verifier, audit, config,
                             staleness_policy=policy)
            original = agg.step

            def flaky_step():
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("503 transient")
                return original()

            agg.step = flaky_step
            return agg

        system = AsyncAgentDescent(repo, universe, config=cfg,
                                   aggregator_factory=factory)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = system.run()
    assert stats.sweeps > 2, \
        f"the merger stopped at the first transient ({stats.sweeps} sweeps)"


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

    `pipeline.py`'s module docstring records that these two runtimes implemented
    the same shape independently, and that a measured fix had to be carried
    across by hand. The synchronous loop had re-grown its own copy
    (`any_success` + `dead_rounds >= max_worker_errors`); this asserts there is
    one implementation, by name, in both.
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
