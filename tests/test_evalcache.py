"""The cache has to stay right when more than one thing is asking.

Evaluation is the expensive half of a run -- the gate is 193.6s against 90.0s for
the same work at `eval_concurrency` 1 and 8, and every one of those seconds is a
real rollout. So the two ways the plain dictionary stopped being right under
concurrency are worth pinning:

* N callers for one uncomputed key produced N computations;
* one key meant one answer, even when the answers came from different
  environments and only one of them was the answer to the question being asked.
"""

import json
import threading
import time

import pytest

from agentdescent.evalcache import CacheProtocol, FileCache, MemoryCache, cache_key
from agentdescent.metrics import Meter


def test_both_backends_satisfy_the_protocol(tmp_path):
    assert isinstance(MemoryCache(), CacheProtocol)
    assert isinstance(FileCache(str(tmp_path)), CacheProtocol)


def test_a_repeated_key_is_computed_once():
    calls = {"n": 0}

    def measure():
        calls["n"] += 1
        return 0.5

    meter = Meter()
    cache = MemoryCache(meter)
    for _ in range(5):
        assert cache.get_or_eval(("k", "t1", ""), measure) == 0.5
    assert calls["n"] == 1
    s = meter.snapshot()
    assert (s.cache_misses, s.cache_hits) == (1, 4)


def test_concurrent_callers_for_one_key_compute_once():
    """Single-flight. The old dictionary checked, released its lock, then
    computed -- so every thread that arrived before the first finished missed and
    computed too. Correct, and wasteful in exactly the case caching is for."""
    calls, started = {"n": 0}, threading.Event()

    def slow():
        calls["n"] += 1
        started.set()
        time.sleep(0.15)              # long enough for the others to pile up
        return 1.0

    meter = Meter()
    cache = MemoryCache(meter)
    out = []
    threads = [threading.Thread(target=lambda: out.append(
        cache.get_or_eval(("k", "t", ""), slow))) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert calls["n"] == 1, f"{calls['n']} threads computed the same value"
    assert out == [1.0] * 32
    assert meter.snapshot().cache_inflight_joins == 31


def test_a_failing_computation_reaches_everyone_waiting():
    """The waiters must not hang, and must not silently get a wrong value."""
    def boom():
        time.sleep(0.05)
        raise RuntimeError("the gate failed")

    cache = MemoryCache()
    errors = []

    def call():
        try:
            cache.get_or_eval(("k", "t", ""), boom)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(errors) == 8 and all("the gate failed" in e for e in errors)
    # and the key is free again, so a later attempt can succeed
    assert cache.get_or_eval(("k", "t", ""), lambda: 0.25) == 0.25


def test_two_environments_do_not_answer_each_others_questions():
    """The poisoning case, and the reason the fingerprint is in the key.

    `code_runner`'s score depends on what `setup_cmd` installed, the python minor
    version, whether there was a network. Sharing a cache across images means one
    environment's measurement answers another's question -- and that number is
    what the commit gates read."""
    meter = Meter()
    cache = MemoryCache(meter)
    calls = []

    def measure(value):
        def fn():
            calls.append(value)
            return value
        return fn

    a = cache.get_or_eval(cache_key("rendered", "t1", "image-a"), measure(1.0))
    b = cache.get_or_eval(cache_key("rendered", "t1", "image-b"), measure(0.0))
    assert (a, b) == (1.0, 0.0)
    assert calls == [1.0, 0.0], "one environment's score was reused for the other"
    assert meter.snapshot().cache_misses == 2


def test_the_key_is_what_the_artifact_renders_to_not_its_state():
    """A historical fix that must not regress.

    `eval_one` passes only `render()` to `run`, so two states that render
    identically cannot score differently. Keying on state made a strategy that
    carries bookkeeping beside the artifact re-evaluate the whole held-out set
    because a label changed."""
    from agentdescent import AppendRules
    from agentdescent.evolution import EvolvingArtifact

    strategy = AppendRules()
    a = EvolvingArtifact("x", {"r1": "always answer 4"}, 1, 0.2, None, strategy)
    b = EvolvingArtifact("x", {"r1": "always answer 4"}, 7, 0.2, None, strategy)
    assert a._signature() == b._signature(), (
        "version is not part of what an evaluation depends on")


# ---------------------------------------------------------------------------
# Across processes
# ---------------------------------------------------------------------------


def test_a_file_cache_is_visible_to_a_second_reader(tmp_path):
    """One process pays; the next one does not. That is the whole point of it."""
    calls = {"n": 0}

    def measure():
        calls["n"] += 1
        return 0.75

    first = FileCache(str(tmp_path))
    assert first.get_or_eval(cache_key("r", "t", "fp"), measure) == 0.75

    second = FileCache(str(tmp_path))            # a fresh process would look like this
    assert second.get_or_eval(cache_key("r", "t", "fp"), measure) == 0.75
    assert calls["n"] == 1, "the second reader recomputed a value already on disk"


def test_a_file_cache_separates_environments_too(tmp_path):
    cache = FileCache(str(tmp_path))
    cache.get_or_eval(cache_key("r", "t", "image-a"), lambda: 1.0)
    assert cache.get_or_eval(cache_key("r", "t", "image-b"), lambda: 0.0) == 0.0


def test_an_unwritable_cache_does_not_fail_the_run(tmp_path):
    """The answer has already been paid for. Losing the chance to reuse it is a
    smaller problem than losing the run."""
    cache = FileCache(str(tmp_path / "gone"))
    import shutil
    shutil.rmtree(str(tmp_path / "gone"))
    assert cache.get_or_eval(cache_key("r", "t", ""), lambda: 0.5) == 0.5


def test_a_half_written_file_is_never_read(tmp_path):
    """Written to a sibling and renamed, so a reader sees all of it or none."""
    cache = FileCache(str(tmp_path))
    key = cache_key("r", "t", "")
    cache.get_or_eval(key, lambda: 0.5)
    path = cache._path(key)
    with open(path, encoding="utf-8") as fh:
        json.load(fh)                            # parses: it was never partial
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# Through the engine
# ---------------------------------------------------------------------------


def test_a_run_can_be_given_a_shared_cache(tmp_path):
    from agentdescent import AppendRules, Policies, Task, evolve

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]
    shared = FileCache(str(tmp_path))

    def run_it():
        return evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                      run=lambda r, t: t.meta["gold"] if t.id in r else "?",
                      propose=lambda r, t, o, s: t.id, strategy=AppendRules(),
                      rounds=3, n_workers=2, held_out_frac=0.5, seed=0,
                      policies=Policies(eval_cache=shared))

    first, second = run_it(), run_it()
    assert first.final_reward == second.final_reward
    # the second run found the first one's evaluations already on disk
    assert second.cache_hits > 0
