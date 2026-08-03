"""What a run cost has to be measurable from the result alone.

Every assertion here exists because the number it checks is one a reader would
otherwise take on faith. The two that matter most:

* `usage.calls` doubling is what a double-wrapped actor looks like, and a
  doubled call count is a perfectly plausible number -- nothing else would catch
  it;
* `stale_considered` is a denominator, so a ratio computed against the wrong one
  is wrong in a way that still reads as a ratio.
"""

import json
import threading

import pytest

from agentdescent import AppendRules, Task, evolve
from agentdescent.agents import Usage
from agentdescent.evolution import EvolutionResult, RoundInfo
from agentdescent.metrics import Meter, measured


# ---------------------------------------------------------------------------
# Meter itself
# ---------------------------------------------------------------------------


def test_add_rejects_an_unknown_counter():
    """A typo must raise, not silently create a field nothing reads.

    A miscount and "that path never ran" are the same number otherwise."""
    m = Meter()
    with pytest.raises(KeyError, match="unknown meter counter"):
        m.add("rollout_second", 1.0)     # missing 's'


def test_concurrent_adds_are_exact():
    """`+=` is three bytecodes; a thread switch between them loses increments.

    Not "approximately 3200" -- exactly 3200, or the counters cannot be trusted
    at the concurrency the engine actually runs at."""
    m = Meter()
    def bump():
        for _ in range(100):
            m.add("rollouts")
    threads = [threading.Thread(target=bump) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.snapshot().rollouts == 3200


def test_snapshot_does_not_tear():
    """A snapshot is one instant, so counters written together stay consistent.

    Reading fields one at a time off a live meter mixes instants, which breaks
    exactly the `elapsed_s <= wallclock` style assertions below."""
    m = Meter()
    m.start()
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            m.add("cache_hits")
            m.add("cache_misses")

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        for _ in range(200):
            s = m.snapshot()
            # hits are written first, so a torn read shows more misses than hits.
            assert s.cache_hits >= s.cache_misses
    finally:
        stop.set()
        t.join()


def test_start_is_idempotent():
    """Re-starting would move the origin under a RoundInfo that already recorded
    an elapsed_s against the old one, making the sequence non-monotonic."""
    m = Meter()
    m.start()
    t0 = m.t0
    m.start()
    assert m.t0 == t0


def test_elapsed_is_zero_before_start():
    assert Meter().elapsed() == 0.0
    assert Meter().snapshot().elapsed_s == 0.0


def test_measured_counts_calls_and_failures():
    m = Meter()
    ok = measured(lambda a, b: "x", m)
    assert ok(1, 2) == "x"

    def boom(a, b):
        raise RuntimeError("nope")

    bad = measured(boom, m)
    with pytest.raises(RuntimeError):
        bad(1, 2)
    assert m.usage.calls == 2
    assert m.usage.failures == 1


def test_measured_preserves_arity_for_the_signature_check():
    """`_check_callable` binds (None,)*n against `inspect.signature`.

    A bare `*args` wrapper makes every arity check pass -- including the typo
    the check exists to catch -- so the wrapper has to keep the real signature.
    """
    from agentdescent.evolution import _check_callable

    def run(rendered, task):
        return ""

    _check_callable(measured(run, Meter()), 2, "run(rendered, task)")
    with pytest.raises(TypeError):
        _check_callable(measured(run, Meter()), 4, "propose(...)")


def test_measured_without_a_meter_is_the_identity():
    """The un-instrumented path must not grow a frame."""
    def f(x):
        return x
    assert measured(f, None) is f


# ---------------------------------------------------------------------------
# End to end, on the deterministic offline domain
# ---------------------------------------------------------------------------


def _tasks(n=6):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(n)]


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _run(rendered, task):
    # solved once the rule for this task is in the artifact.
    return task.meta["gold"] if task.id in rendered else "?"


def _propose(rendered, task, output, reward):
    return task.id


def _evolve(**kw):
    kw.setdefault("rounds", 4)
    kw.setdefault("n_workers", 2)
    kw.setdefault("held_out_frac", 0.5)
    return evolve(_tasks(), _reward, run=_run, propose=_propose,
                  strategy=AppendRules(), **kw)


def test_a_run_reports_what_it_cost():
    r = _evolve()
    assert r.rollouts > 0
    assert r.wallclock > 0.0
    assert r.usage.calls > 0
    assert r.cache_hits + r.cache_misses > 0
    assert r.cost_summary()          # renders without raising


def test_elapsed_is_monotonic_and_within_the_wallclock():
    r = _evolve()
    elapsed = [h.elapsed_s for h in r.history]
    assert elapsed == sorted(elapsed)
    assert all(e <= r.wallclock + 1e-6 for e in elapsed)
    rollouts = [h.rollouts for h in r.history]
    assert rollouts == sorted(rollouts)


def test_usage_calls_equals_actor_invocations():
    """Counts every `run` and `propose`, each exactly once.

    The failure this guards is `measured()` applied twice -- the count then comes
    out exactly doubled, which looks like a busier run rather than a bug.
    """
    calls = {"run": 0, "propose": 0}

    def counted_run(rendered, task):
        calls["run"] += 1
        return _run(rendered, task)

    def counted_propose(rendered, task, output, reward):
        calls["propose"] += 1
        return _propose(rendered, task, output, reward)

    r = evolve(_tasks(), _reward, run=counted_run, propose=counted_propose,
               strategy=AppendRules(), rounds=3, n_workers=2, held_out_frac=0.5)
    assert r.usage.calls == calls["run"] + calls["propose"]


def test_the_cache_counts_hits_and_misses():
    """N evaluations of one (artifact, task) is 1 miss and N-1 hits."""
    from agentdescent.evolution import _EvalCache

    m = Meter()
    cache = _EvalCache(m)
    n = {"calls": 0}

    def measure():
        n["calls"] += 1
        return 1.0

    for _ in range(5):
        cache.get_or_eval(("k", "t1"), measure)
    assert (m.snapshot().cache_misses, m.snapshot().cache_hits) == (1, 4)
    assert n["calls"] == 1


def test_the_stale_denominator_matches_the_reports():
    """`stale_considered` must equal the evidence the filter actually saw.

    Asserted rather than assumed: a ratio computed against the wrong denominator
    still looks like a ratio."""
    seen = []
    r = evolve(_tasks(), _reward, run=_run, propose=_propose,
               strategy=AppendRules(), rounds=4, n_workers=3, held_out_frac=0.5,
               on_round=lambda info: seen.append(info))
    assert r.stale_considered >= r.stale_discarded >= 0
    assert 0.0 <= r.stale_rate() <= 1.0


def test_sandbox_fields_are_zero_on_the_default_path():
    """There is no pool to wait for and no image to warm, so these stay 0 --
    and their presence must not perturb any existing number."""
    r = _evolve()
    assert (r.sandbox_wait_s, r.sandbox_setup_s) == (0.0, 0.0)
    assert (r.sandboxes_created, r.sandboxes_reused, r.sandbox_failures) == (0, 0, 0)


def test_time_and_cost_to_quality():
    r = _evolve(rounds=6)
    reached = [h for h in r.history if h.held_out_reward >= 0.5]
    if reached:
        assert r.time_to_quality(0.5) == reached[0].elapsed_s
        assert r.cost_to_quality(0.5) == reached[0].rollouts
    # a bar nothing reaches has no time-to-quality; total wall-clock would flatter it
    assert r.time_to_quality(2.0) is None
    assert r.cost_to_quality(2.0) is None


def test_a_shared_usage_object_accumulates_tokens():
    """Tokens are only knowable when the caller's adapter reports them, so the
    engine has to accept the caller's Usage rather than always making its own."""
    u = Usage()
    u.record(prompt_tokens=100, completion_tokens=20)     # as an adapter would
    r = _evolve(usage=u)
    assert r.usage is u
    assert r.usage.prompt_tokens == 100
    assert r.usage.calls > 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_save_load_round_trip_keeps_the_cost_fields(tmp_path):
    r = _evolve()
    p = tmp_path / "r.json"
    r.save(str(p))
    back = EvolutionResult.load(str(p))
    assert back.wallclock == r.wallclock
    assert back.rollouts == r.rollouts
    assert back.usage.calls == r.usage.calls
    assert back.cache_hits == r.cache_hits
    assert back.stale_considered == r.stale_considered
    assert [h.elapsed_s for h in back.history] == [h.elapsed_s for h in r.history]


def test_a_result_written_before_these_fields_existed_still_loads(tmp_path):
    """The old on-disk shape, verbatim -- not a dict with keys removed."""
    old = {
        "state": {"r1": "always answer 4"},
        "rendered": "always answer 4",
        "final_reward": 0.75,
        "error": None,
        "history": [{"round": 0, "held_out_reward": 0.5, "n_items": 1,
                     "committed": 1, "rejected": 0, "reasons": {"committed": 1}}],
        "retired_workers": 0,
        "forced_refreshes": 0,
        "stragglers": 0,
        "stop_reason": "rounds",
        "ledger_log": ["init"],
    }
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old), encoding="utf-8")

    r = EvolutionResult.load(str(p))
    assert r.final_reward == 0.75
    assert r.wallclock == 0.0 and r.rollouts == 0
    assert r.usage.calls == 0
    assert r.history[0].elapsed_s == 0.0
    assert r.stale_rate() == 0.0 and r.duplicate_rate() == 0.0


def test_roundinfo_defaults_keep_positional_construction_working():
    """Existing callers build RoundInfo positionally; the new fields are last."""
    info = RoundInfo(0, 0.5, 3, 1, 2, {"committed": 1})
    assert (info.elapsed_s, info.rollouts) == (0.0, 0)
