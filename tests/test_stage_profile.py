"""Is the gate on the critical path? The counters that can answer it.

`eval_seconds` cannot. It sums across the evaluation pool, so eight threads
scoring for a second each reads 8.0 whether the run spent eight seconds in the
gate or one -- and the question this file exists for is whether the gate **stops
anyone from working**, which is a wall-clock question about one thread.

So three counters, and the invariants that make them readable:

* `merge_seconds` -- the merger's busy time, on the single thread that merges.
  Declared since the first version of `metrics.py` and written by nobody, so
  every published run reported 0.0, which reads as "merging was free";
* `merge_gate_seconds` -- the part of it blocked on evaluation. A **subset**;
* `worker_starved_seconds` -- what a busy merger actually cost, in rollouts not
  started. A merger at 90% occupancy that starved nobody has hidden itself
  behind the rollouts, which is the entire point of the barrier-free path.

The last test is the one that would fail if the mechanism were not real: a run
whose *held-out* scoring is slow and whose rollouts are fast must starve its
workers, because the gate runs on the merger and the merger is one thread.
"""

import threading
import time

import pytest

from agentdescent import AppendRules, Task, evolve
from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import EvolutionResult
from agentdescent.metrics import Meter


# ---------------------------------------------------------------------------
# The offline domain, plus a switch that makes only the gate slow
# ---------------------------------------------------------------------------

#: Tasks are split by position, so with `held_out_frac=0.5` over eight tasks the
#: back half is the held-out set -- which is what the gate scores and the workers
#: never roll out. Sleeping only on those separates the two workloads.
N_TASKS = 8
HELD_OUT_IDS = {f"t{i}" for i in range(N_TASKS // 2, N_TASKS)}
#: Shared by `_async` and `_starved_share`, which have to agree: the second
#: divides by it, and a helper that guessed would silently rescale the ratio.
_WORKERS = 4


def _tasks(n=N_TASKS):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(n)]


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _run(rendered, task):
    return task.meta["gold"] if task.id in rendered else "?"


def _slow_gate_run(rendered, task):
    """Fast rollouts, slow evaluations. Sleeping releases the GIL, so this is
    the same shape as a real backend where the gate is the expensive stage."""
    if task.id in HELD_OUT_IDS:
        time.sleep(0.02)
    return _run(rendered, task)


def _slow_rollout_run(rendered, task):
    """The mirror image: slow rollouts, a gate that costs nothing.

    This is the configuration where the merger keeps up, and it is the only way
    to show that `worker_starved_seconds` can read low -- on this domain a
    *fast* rollout is a dictionary lookup, so workers outrun the merger whatever
    the gate is doing (`docs/efficiency.md` says the same thing about
    `async_ratio`)."""
    if task.id not in HELD_OUT_IDS:
        time.sleep(0.05)
    return _run(rendered, task)


def _propose(rendered, task, output, reward):
    return task.id


def _sync(**kw):
    kw.setdefault("rounds", 4)
    kw.setdefault("n_workers", 2)
    kw.setdefault("held_out_frac", 0.5)
    kw.setdefault("run", _run)
    return evolve(_tasks(), _reward, propose=_propose,
                  strategy=AppendRules(), **kw)


def _async(**kw):
    """A barrier-free run with the staleness warning silenced.

    That warning is expected on this domain and is not what is under test:
    `docs/efficiency.md` records that a dictionary-lookup rollout drifts past any
    lag budget immediately, which is why the async rows there run at lag 0-1."""
    import warnings

    kw.setdefault("n_workers", _WORKERS)
    kw.setdefault("max_seconds", 2.0)
    kw.setdefault("held_out_frac", 0.5)
    kw.setdefault("run", _run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return async_evolve(_tasks(), _reward, propose=_propose,
                            strategy=AppendRules(), **kw)


# ---------------------------------------------------------------------------
# Meter.timed
# ---------------------------------------------------------------------------


def test_timed_rejects_an_unknown_counter_before_running_the_block():
    """Fail on the name, not after the block has already paid for itself."""
    m = Meter()
    ran = []
    with pytest.raises(KeyError, match="unknown meter counter"):
        with m.timed("merge_second"):        # missing 's'
            ran.append(1)
    assert ran == []


def test_timed_records_a_block_that_raised():
    """A sweep that failed still spent its time.

    Without this the phase counters read lowest on exactly the runs worth
    looking at -- a merger failing every sweep would report as idle."""
    m = Meter()
    with pytest.raises(RuntimeError):
        with m.timed("merge_seconds"):
            time.sleep(0.01)
            raise RuntimeError("nope")
    assert m.snapshot().merge_seconds >= 0.01


def test_timed_is_thread_safe():
    """Same reason `add` is: `+=` is three bytecodes."""
    m = Meter()

    def bump():
        for _ in range(20):
            with m.timed("merge_gate_seconds"):
                pass

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.snapshot().merge_gate_seconds >= 0.0    # exactness is `add`'s test


# ---------------------------------------------------------------------------
# The counters, end to end
# ---------------------------------------------------------------------------


def test_merge_seconds_is_written_at_all():
    """The regression this whole file starts from.

    `merge_seconds` was declared in `MeterSnapshot`, in `Meter`, and in
    `Meter.snapshot()`, and written by nothing -- so every run ever published
    reported 0.0, which is indistinguishable from "merging was free"."""
    r = _sync()
    assert r.merge_seconds > 0.0


def test_the_gate_share_is_a_subset_not_a_second_total():
    """`merge_gate_seconds <= merge_seconds`, always.

    They are nested timers, so a reader who adds them double-counts the gate.
    The invariant is what makes `gate_share()` a ratio rather than a number that
    can exceed 1."""
    for r in (_sync(), _async()):
        assert r.merge_gate_seconds <= r.merge_seconds + 1e-9
        assert 0.0 <= r.gate_share() <= 1.0


def test_merge_seconds_is_wallclock_on_one_thread():
    """Unlike `rollout_seconds` and `eval_seconds`, this one cannot exceed the
    run: there is exactly one merger, so its busy time is bounded by the clock.

    That bound is what makes `merger_occupancy()` meaningful; a summed counter
    in its place would routinely report occupancy above 100%."""
    for r in (_sync(), _async()):
        assert r.merge_seconds <= r.wallclock + 0.5      # slack for the final scoring


def test_the_synchronous_path_starves_nobody_by_this_measure():
    """0 by construction, and deliberately not a hidden restatement of the barrier.

    The barrier idles every worker for exactly `merge_seconds`, so a counter
    here would carry no information `merge_seconds` does not. Non-zero would
    mean the async gate had been wired into the synchronous loop."""
    assert _sync().worker_starved_seconds == 0.0


def test_a_slow_gate_starves_the_workers_on_the_async_path():
    """The mechanism, demonstrated rather than asserted from the design.

    Held-out scoring sleeps, rollouts do not. The gate runs on the merger, the
    merger is one thread, and `async_ratio` bounds how far ahead the workers may
    get -- so a slow gate has to show up as workers with nowhere to put a
    finished card. If this ever reads 0, evaluation is no longer on the merger's
    critical path and issue #151 is done.
    """
    r = _async(run=_slow_gate_run, async_ratio=1, max_seconds=1.5)
    assert r.worker_starved_seconds > 0.0
    # And the gate is where the merger's time went, not the merging.
    assert r.gate_share() > 0.5


def _starved_share(r) -> float:
    """Starvation as a fraction of the worker-time the run had to spend.

    The raw counter is a sum across workers, so it is only comparable between
    runs once it is divided by `n_workers x wallclock`. Reading it against
    `wallclock` alone makes a four-worker run look four times worse than a
    one-worker run that idled just as much."""
    available = _WORKERS * r.wallclock
    return r.worker_starved_seconds / available if available else 0.0


def test_starvation_is_exactly_zero_when_the_gate_cannot_fire():
    """The other half: the counter has to be able to read low.

    A counter non-zero on every run cannot distinguish the case it was added
    for -- timing the backpressure loop unconditionally would do exactly that,
    since the two `intake_lock` acquisitions would put a floor under it.

    Asserted against a lag budget nothing can exceed rather than against a
    threshold, so it is **exact and load-independent**. A wall-clock comparison
    would read differently on a busy machine, and a counter's floor should not.

    The corresponding *measurement* (n=3) -- 12-17% starved with slow rollouts
    and a cheap gate, against 36-41% the other way round -- lives in
    `examples/efficiency.py --only stages`, which is where the numbers in
    `docs/efficiency.md` come from.
    """
    r = _async(run=_slow_rollout_run, async_ratio=1_000_000, max_seconds=1.0)
    assert r.rollouts > 0                       # it did work, it just was not blocked
    assert r.worker_starved_seconds == 0.0


def test_narrowing_the_gates_own_pool_still_starves_workers():
    """Serialising the gate's own pool is the controlled way to implicate it.

    Same workload, same rollouts, same evaluations -- only `eval_concurrency`
    moves, so the gate's wall-clock is the single variable. Asserted one-sided:
    a loaded machine can only make starvation *worse*, so `> 0` cannot flake,
    while the ordering against a wider pool can and did.

    Measured across three trials: 36-41% at the default width, 52-75% at 1.
    Quoted as observed ranges, not as an effect size -- three runs of one
    configuration of this engine can span 1.5x (see docs/efficiency.md).
    """
    r = _async(run=_slow_gate_run, async_ratio=1, max_seconds=1.5,
               eval_concurrency=1)
    assert _starved_share(r) > 0.0
    assert r.gate_share() > 0.5


def test_the_two_ratios_are_the_arithmetic_they_claim():
    """`gate_share` is gate/merge and `merger_occupancy` is merge/wall.

    Asserted on a hand-built result rather than a run: the ratios are what a
    reader divides by, and getting the denominators the wrong way round is both
    easy and invisible in a plausible-looking percentage."""
    r = EvolutionResult(state={}, rendered="", final_reward=0.0,
                        history=[], ledger_log=[],
                        wallclock=10.0, merge_seconds=4.0,
                        merge_gate_seconds=3.0)
    assert r.gate_share() == pytest.approx(0.75)
    assert r.merger_occupancy() == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_ratios_are_defined_when_nothing_merged():
    """A run whose budget expired before its first sweep has no gate share, as
    distinct from having a low one. Neither may raise."""
    empty = EvolutionResult(state={}, rendered="", final_reward=0.0,
                            history=[], ledger_log=[])
    assert empty.gate_share() == 0.0
    assert empty.merger_occupancy() == 0.0


def test_the_counters_survive_a_save_and_load(tmp_path):
    """A cost field that does not round-trip is a cost field that cannot be
    compared across runs, which is the only way any of these get used."""
    r = _sync()
    path = str(tmp_path / "run.json")
    r.save(path)
    back = EvolutionResult.load(path)
    assert back.merge_seconds == pytest.approx(r.merge_seconds)
    assert back.merge_gate_seconds == pytest.approx(r.merge_gate_seconds)
    assert back.worker_starved_seconds == pytest.approx(r.worker_starved_seconds)


def test_a_result_written_before_these_existed_still_loads(tmp_path):
    """`load` reads every cost field with a default, so an older file loads as a
    run that simply did not measure them."""
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps(
        {"state": {}, "rendered": "", "final_reward": 0.5}), encoding="utf-8")
    back = EvolutionResult.load(str(path))
    assert back.merge_seconds == 0.0
    assert back.worker_starved_seconds == 0.0


def test_cost_summary_mentions_the_merger():
    """It is the line a reader sees first; a counter absent from it is a counter
    nobody reads."""
    line = _sync().cost_summary()
    assert "merger" in line
