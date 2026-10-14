"""The trajectory store has to stay right under the exact conditions a run has.

Rollout records are written from every worker thread and the evaluation pool at
once, they dedupe on an identity the cache already defines, they must never fail
a rollout when the disk does, and their text is private by default. Each of
those is a property worth pinning:

* record/get round-trip, and identity-based dedupe across processes and threads;
* concurrency -- N writers of one evaluation land one file, never a torn one;
* failure -- a hostile filesystem degrades the recording, not the run;
* privacy and caps -- event detail is metadata unless text recording is on;
* wiring -- `evolve()` and `async_evolve()` record into the ledger's own
  directory when asked, keep it out of the git index, and leave the run
  unchanged when not asked.
"""

import json
import os
import subprocess
import threading
import time

import pytest

from agentdescent.evalcache import MemoryCache
from agentdescent.evolution import (
    AppendRules,
    EvolvingArtifact,
    Task,
    _Runtime,
    evolve,
)
from agentdescent.metrics import Meter
from agentdescent.trajectory import (
    PURPOSE_EVAL,
    PURPOSE_ROLLOUT,
    PURPOSE_SELF_VERIFY,
    RECORD_TEXT_ENV,
    Trajectory,
    TrajectoryEvent,
    TrajectoryStore,
    trajectory_key,
)


# -- the store in isolation ---------------------------------------------------


def _record(store, *, artifact_id="artifact", rendered="render me", task_id="t1",
            env="", purpose=PURPOSE_ROLLOUT, output="out", reward=1.0,
            **kw):
    return store.record(Trajectory(
        artifact_id=artifact_id, task_id=task_id, env_fingerprint=env,
        rendered=rendered, purpose=purpose, output=output, reward=reward, **kw))


def test_round_trip_preserves_every_field(tmp_path):
    store = TrajectoryStore(str(tmp_path), record_text=True)
    trace = _record(
        store, artifact_id="skill", rendered="prompt text", task_id="task-9",
        env="img:v1", purpose=PURPOSE_SELF_VERIFY, base_version='{"skill": 3}',
        output="the answer", reward=0.5, cost_tokens=12, cost_wallclock=1.5,
        lease_id="lease-1", worker_id="w2",
        events=(TrajectoryEvent("step", "solve", seconds=0.1, cost_tokens=4,
                                detail="thinking..."),))
    assert trace is not None
    back = store.get(trace)
    assert back is not None
    assert back.artifact_id == "skill"
    assert back.rendered == "prompt text"
    assert back.task_id == "task-9"
    assert back.env_fingerprint == "img:v1"
    assert back.purpose == PURPOSE_SELF_VERIFY
    assert back.base_version == '{"skill": 3}'
    assert back.output == "the answer"
    assert back.reward == 0.5
    assert back.cost_tokens == 12
    assert back.cost_wallclock == 1.5
    assert back.lease_id == "lease-1"
    assert back.worker_id == "w2"
    assert back.ts > 0
    assert back.trace_id == trace
    assert len(back.events) == 1
    assert back.events[0].kind == "step"
    assert back.events[0].detail == "thinking..."
    assert back.trace_id.startswith("skill:")


def test_identity_is_the_cache_triple_and_dedupes(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    first = _record(store, rendered="r", task_id="t", env="e")
    assert first == f"artifact:{trajectory_key('r', 't', 'e')}"
    second = _record(store, rendered="r", task_id="t", env="e")
    assert second is None            # dedupe: nothing new to write
    assert store.count() == 1
    # a different task, a different environment, a different render: all new
    assert _record(store, rendered="r", task_id="t2", env="e") is not None
    assert _record(store, rendered="r", task_id="t", env="e2") is not None
    assert _record(store, rendered="r2", task_id="t", env="e") is not None
    assert store.count() == 4


def test_identity_is_unambiguous_for_embedded_nuls(tmp_path):
    # The identity encodes three parts; a byte that can appear *inside* a part
    # must never be able to double as a part boundary. These two triples differ
    # only in which field holds the NUL, and must hash differently -- with a
    # shared-separator encoding they would collide and the second trajectory
    # would be silently treated as a duplicate of the first.
    store = TrajectoryStore(str(tmp_path))
    t1 = _record(store, rendered="a\x00b", task_id="c", env="")
    t2 = _record(store, rendered="a", task_id="b\x00c", env="")
    assert t1 is not None and t2 is not None
    assert t1 != t2
    assert store.count() == 2
    # ordinary identical-input dedup still holds alongside
    assert _record(store, rendered="a\x00b", task_id="c", env="") is None
    assert store.count() == 2


def test_dedupe_and_identity_are_across_artifacts_by_render(tmp_path):
    # Two artifact ids rendering identically are one evaluation -- the same
    # property the eval cache leans on, and what makes a re-run idempotent.
    store = TrajectoryStore(str(tmp_path))
    a = _record(store, artifact_id="a", rendered="same", task_id="t")
    b = _record(store, artifact_id="b", rendered="same", task_id="t")
    assert b is None
    assert store.count() == 1
    assert store.get(a).artifact_id == "a"   # first writer's id wins


def test_concurrent_writers_of_one_identity_land_one_file(tmp_path):
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter)
    traj = Trajectory(artifact_id="a", task_id="t", env_fingerprint="",
                      rendered="shared", purpose=PURPOSE_ROLLOUT,
                      output="o", reward=1.0)
    n = 16
    results = []
    barrier = threading.Barrier(n)

    def work():
        barrier.wait()
        results.append(store.record(traj))

    threads = [threading.Thread(target=work) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 1
    # The exists-check + write pair is locked, so exactly one writer records
    # and the rest dedupe -- the counters say precisely that, not "n writes
    # happened and one file survived".
    written = {r for r in results if r is not None}
    assert len(written) == 1
    s = meter.snapshot()
    assert s.trajectories_recorded == 1
    assert s.trajectories_dedup == n - 1
    assert s.trajectory_write_failures == 0
    back = store.get(next(iter(written)))
    assert back is not None and back.rendered == "shared"


def test_concurrent_distinct_identities_all_land(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    n = 24
    barrier = threading.Barrier(n)

    def work(i):
        barrier.wait()
        store.record(Trajectory(
            artifact_id="a", task_id=f"t{i}", env_fingerprint="",
            rendered=f"r{i}", purpose=PURPOSE_ROLLOUT))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == n
    assert {r.task_id for r in store.iter_records(artifact_id="a")} == \
        {f"t{i}" for i in range(n)}


def test_a_hostile_artifact_id_cannot_escape_the_root(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    trace = _record(store, artifact_id="../../etc/passwd")
    assert trace is not None
    path = store.path_for(trace)
    assert path is not None
    assert os.path.normpath(path).startswith(os.path.normpath(str(tmp_path)))
    assert store.get(trace) is not None
    # the traversal component never survives into a directory name
    assert not os.path.exists(os.path.join(str(tmp_path), "..", "..", "etc"))


def test_a_colon_in_artifact_id_still_round_trips_through_get(tmp_path):
    """trace_id is ``"<artifact_id>:<hex>"``; a colon inside the id must not
    break ``get`` / ``path_for``. ``evolve()`` itself forbids colons, but the
    store is a public API and namespaced ids are otherwise legal."""
    store = TrajectoryStore(str(tmp_path))
    trace = _record(store, artifact_id="ns:skills")
    assert trace is not None
    assert trace.startswith("ns:skills:")
    assert trace.count(":") >= 2
    back = store.get(trace)
    assert back is not None
    assert back.artifact_id == "ns:skills"
    assert back.trace_id == trace
    path = store.path_for(trace)
    assert path is not None
    assert os.path.normpath(path).startswith(os.path.normpath(str(tmp_path)))
    assert os.path.isfile(path)
    # multi-colon still resolves to the trailing hex key
    deeper = _record(store, artifact_id="org:team:bot", task_id="t2",
                     rendered="other render")
    assert store.get(deeper) is not None
    assert store.get(deeper).artifact_id == "org:team:bot"


def test_get_and_path_for_reject_unknown_ids(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    assert store.get("not-a-trace") is None
    assert store.get("") is None
    assert store.get("a:") is None
    assert store.get("a:../..") is None
    assert store.path_for("nope") is None


def test_write_failure_is_counted_not_raised(tmp_path):
    # A path whose parent is a regular file: makedirs must fail, and the run's
    # record() must degrade to None while counting the failure.
    blocker = tmp_path / "a_file"
    blocker.write_text("occupied")
    meter = Meter()
    store = TrajectoryStore(str(blocker / "trajectories"), meter=meter)
    trace = _record(store)
    assert trace is None
    assert meter.snapshot().trajectory_write_failures == 1
    assert meter.snapshot().trajectories_recorded == 0


def test_event_text_is_private_by_default(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    _record(store, events=(TrajectoryEvent("model_call", "solve",
                                           detail="secret reasoning"),))
    back = next(store.iter_records())
    assert back.events[0].detail is None
    assert back.events[0].kind == "model_call"       # metadata survives


def test_event_text_can_be_turned_on_explicitly(tmp_path):
    store = TrajectoryStore(str(tmp_path), record_text=True)
    _record(store, events=(TrajectoryEvent("model_call", "solve",
                                           detail="secret reasoning"),))
    back = next(store.iter_records())
    assert back.events[0].detail == "secret reasoning"


def test_event_text_follows_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(RECORD_TEXT_ENV, "1")
    store = TrajectoryStore(str(tmp_path / "on"))
    _record(store, events=(TrajectoryEvent("step", "run", detail="kept"),))
    assert next(store.iter_records()).events[0].detail == "kept"
    monkeypatch.delenv(RECORD_TEXT_ENV)
    store2 = TrajectoryStore(str(tmp_path / "off"))
    _record(store2, events=(TrajectoryEvent("step", "run", detail="dropped"),))
    assert next(store2.iter_records()).events[0].detail is None


def test_event_count_and_detail_caps_are_enforced(tmp_path):
    store = TrajectoryStore(str(tmp_path), max_events=2, max_detail_chars=5,
                            record_text=True)
    events = tuple(
        TrajectoryEvent("step", "s", detail=f"detail-{i}") for i in range(4))
    _record(store, events=events)
    back = next(store.iter_records())
    assert len(back.events) == 2                       # count capped
    assert back.events[0].detail == "detai"            # detail capped


def test_rendered_can_be_capped_and_is_flagged(tmp_path):
    store = TrajectoryStore(str(tmp_path), max_rendered_chars=8)
    trace = _record(store, rendered="0123456789abcdef")
    back = store.get(trace)
    assert back.rendered == "01234567"
    assert back.rendered_truncated is True


def test_iter_records_filters_and_skips_corrupt_files(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    _record(store, artifact_id="a", task_id="t1", purpose=PURPOSE_ROLLOUT)
    _record(store, artifact_id="a", task_id="t2", purpose=PURPOSE_EVAL)
    # distinct identity so it does not dedupe into a's t1 record
    _record(store, artifact_id="b", task_id="t3", purpose=PURPOSE_ROLLOUT)
    # a corrupt record must be skipped, not raised
    with open(os.path.join(str(tmp_path), "0000.json"), "w") as fh:
        fh.write("not json")

    got = list(store.iter_records(artifact_id="a"))
    assert len(got) == 2
    assert {r.task_id for r in got} == {"t1", "t2"}
    assert {r.artifact_id for r in store.iter_records(purpose=PURPOSE_ROLLOUT)} == \
        {"a", "b"}
    assert len(list(store.iter_records(artifact_id="b"))) == 1
    assert len(list(store.iter_records(task_id="t1"))) == 1


def test_records_round_trip_through_json(tmp_path):
    store = TrajectoryStore(str(tmp_path))
    trace = _record(store, artifact_id="a", task_id="t", rendered="r",
                    purpose=PURPOSE_EVAL,
                    events=(TrajectoryEvent("tool_call", "calculator",
                                            seconds=0.2, cost_tokens=3),))
    raw = store.get(trace)
    assert raw is not None
    blob = json.dumps(raw.to_dict(), default=str, allow_nan=False)
    again = Trajectory.from_dict(json.loads(blob))
    assert again == raw


# -- the engine wiring ---------------------------------------------------------


def _offline_actors():
    def run(rendered, task):
        return "answer" + (" 2026" if "year" in rendered else "")

    def reward(task, output):
        return 1.0 if "2026" in output else 0.0

    def propose(rendered, task, output, reward):
        return None if reward >= 0.999 else "always state the year"

    return run, reward, propose


def _tasks(n=8):
    return [Task(id=f"t{i}", prompt=f"item {i}") for i in range(n)]


def test_evolve_records_rollouts_into_the_ledger_dir(tmp_path):
    run, reward, propose = _offline_actors()
    repo = str(tmp_path / "repo")
    result = evolve(_tasks(), reward, run=run, propose=propose, rounds=2,
                    n_workers=2, record_trajectories=True, repo_path=repo,
                    max_concurrency=2)
    assert result.trajectories_path is not None
    store = TrajectoryStore(result.trajectories_path)
    assert store.count() >= 1
    assert result.trajectories_recorded >= 1
    purposes = {r.purpose for r in store.iter_records()}
    assert PURPOSE_ROLLOUT in purposes
    # the ledger's own index must not contain the records
    out = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain"],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert out.stdout.strip() == ""
    assert os.path.exists(os.path.join(repo, "trajectories"))


def test_evolve_records_gate_evals_and_self_verify(tmp_path):
    run, reward, propose = _offline_actors()
    repo = str(tmp_path / "repo")
    result = evolve(_tasks(12), reward, run=run, propose=propose, rounds=2,
                    n_workers=2, repo_path=repo, record_trajectories=True,
                    max_concurrency=2)
    store = TrajectoryStore(result.trajectories_path)
    purposes = {r.purpose for r in store.iter_records()}
    assert purposes >= {PURPOSE_ROLLOUT, PURPOSE_EVAL, PURPOSE_SELF_VERIFY}
    # The self-verify boundary record carries the provider's output -- the same
    # contract as the rollout path -- so offline consumers see the candidate's
    # answer, not just the score it earned.
    for rec in store.iter_records(purpose=PURPOSE_SELF_VERIFY):
        assert rec.output is not None
    # Gate records carry the artifact version they measured -- a downstream
    # gate/audit consumer must be able to attribute the measurement.
    eval_recs = list(store.iter_records(purpose=PURPOSE_EVAL))
    assert eval_recs, "expected gate records"
    for rec in eval_recs:
        assert rec.base_version, "gate record missing version attribution"


def test_trajectories_are_off_by_default(tmp_path):
    run, reward, propose = _offline_actors()
    result = evolve(_tasks(), reward, run=run, propose=propose, rounds=1,
                    n_workers=2, repo_path=str(tmp_path / "repo"))
    assert result.trajectories_recorded == 0
    assert result.trajectories_dedup == 0
    assert result.trajectories_path is None
    assert not os.path.exists(str(tmp_path / "repo" / "trajectories"))


def test_explicit_store_wins_over_the_boolean(tmp_path):
    run, reward, propose = _offline_actors()
    custom = TrajectoryStore(str(tmp_path / "custom"))
    result = evolve(_tasks(), reward, run=run, propose=propose, rounds=1,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    record_trajectories=True, trajectory_store=custom)
    assert result.trajectories_path == str(tmp_path / "custom")
    assert not os.path.exists(str(tmp_path / "repo" / "trajectories"))
    assert custom.count() >= 1
    # The engine attached its meter to the caller's store (which had none), so
    # the run's result accounts for the writes exactly as it does for a store
    # it built itself -- a run that recorded trajectories and reported zero
    # would be indistinguishable from one that recorded nothing.
    assert result.trajectories_recorded >= 1


def test_explicit_store_with_its_own_meter_keeps_its_accounting(tmp_path):
    # A shared store that outlives the run reports into its own totals; the
    # engine must not clobber a caller-supplied meter. The run's result stays
    # zero -- the caller chose where the accounting goes.
    from agentdescent.metrics import Meter

    own = Meter()
    custom = TrajectoryStore(str(tmp_path / "custom"), meter=own)
    run, reward, propose = _offline_actors()
    result = evolve(_tasks(), reward, run=run, propose=propose, rounds=1,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    trajectory_store=custom)
    assert custom.count() >= 1
    assert own.snapshot().trajectories_recorded >= 1
    assert result.trajectories_recorded == 0
    assert custom.meter is own          # the caller's meter was left in place


def _run_store_reuse_probe(tmp_path, *, asynchronous):
    """The reviewer's meter-lifetime repro, parametrised over both drivers.

    Reuse the same originally-unmetered store across two runs with *distinct*
    task ids (so run 2's evaluations are not dedups of run 1's): each run must
    report its own writes in its own result, and run 2's writes must not land
    in run 1's meter -- run 1's meter is a private per-run object, never a
    caller-owned one the engine must preserve."""
    from agentdescent.async_evolve import async_evolve

    run, reward, propose = _offline_actors()
    custom = TrajectoryStore(str(tmp_path / "custom"))
    assert custom._caller_meter is False

    def once(label, repo):
        tasks = [Task(id=f"{label}-{i}", prompt=f"{label} item {i}")
                 for i in range(8)]
        kw = dict(reward=reward, run=run, propose=propose, n_workers=2,
                  repo_path=str(repo), trajectory_store=custom)
        if asynchronous:
            kw.update(max_seconds=1.5, self_verify=False)
            return async_evolve(tasks, **kw)
        return evolve(tasks, **kw, rounds=1)

    repo1 = str(tmp_path / "repo1")
    r1 = once("first", repo1)
    assert r1.trajectories_recorded >= 1, "run 1 must report its own writes"
    meter1 = custom.meter
    assert meter1 is not None, "run 1 must attach its meter to the store"
    recorded_in_meter1 = meter1.snapshot().trajectories_recorded

    repo2 = str(tmp_path / "repo2")
    r2 = once("second", repo2)
    assert r2.trajectories_recorded >= 1, (
        "run 2 must report its own writes, not zero")
    meter2 = custom.meter
    assert meter2 is not None and meter2 is not meter1, (
        "run 2 must swap in its own meter, not reuse run 1's")
    # Run 1's meter absorbed only run 1's writes -- run 2's are not in it.
    assert meter1.snapshot().trajectories_recorded == recorded_in_meter1, (
        "run 2's writes must not land in run 1's meter")
    # The store's caller-owned flag stays False: the engine owns the accounting
    # and swaps the meter per run.
    assert custom._caller_meter is False
    return r1, r2


def test_reusing_an_unmetered_store_reports_each_run_s_accounting(tmp_path):
    _run_store_reuse_probe(tmp_path, asynchronous=False)


def test_reusing_an_unmetered_store_reports_each_async_run_s_accounting(tmp_path):
    _run_store_reuse_probe(tmp_path, asynchronous=True)


def test_async_evolve_records_rollouts(tmp_path):
    from agentdescent.async_evolve import async_evolve

    run, reward, propose = _offline_actors()
    repo = str(tmp_path / "repo")
    result = async_evolve(_tasks(), reward, run=run, propose=propose,
                          n_workers=2, max_seconds=1.5,
                          repo_path=repo, record_trajectories=True,
                          self_verify=False)
    assert result.trajectories_path is not None
    store = TrajectoryStore(result.trajectories_path)
    assert store.count() >= 1
    assert result.trajectories_recorded >= 1
    out = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                         capture_output=True, text=True)
    assert out.stdout.strip() == ""


def test_a_resumed_run_shares_and_dedupes_records(tmp_path):
    # Recording into the same durable repo across two runs: the second run's
    # identical evaluations are duplicates, not new records.
    run, reward, propose = _offline_actors()
    repo = str(tmp_path / "repo")
    first = evolve(_tasks(), reward, run=run, propose=propose, rounds=1,
                   n_workers=2, repo_path=repo, record_trajectories=True)
    store1 = TrajectoryStore(first.trajectories_path)
    before = store1.count()
    second = evolve(_tasks(), reward, run=run, propose=propose, rounds=1,
                    n_workers=2, repo_path=repo, record_trajectories=True)
    store2 = TrajectoryStore(second.trajectories_path)
    assert store2.count() == before
    assert second.trajectories_dedup >= 1


# -- the runtime choke point ----------------------------------------------------


def test_eval_one_records_once_even_when_a_retry_happened(tmp_path):
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter)
    calls = {"n": 0}

    def run(rendered, task):
        return "out"

    def reward(task, output):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient backend failure")
        return 1.0

    runtime = _Runtime(run=run, reward=reward, cache=MemoryCache(meter),
                       meter=meter, trajectory_store=store)
    artifact = EvolvingArtifact("a", {"k": "v"}, strategy=AppendRules(),
                                runtime=runtime)
    task = Task(id="t1", prompt="p")
    assert runtime.eval_one(artifact, task) == 1.0
    assert calls["n"] == 2                       # the transient was retried
    assert store.count() == 1                    # ...but recorded once
    assert next(store.iter_records()).purpose == PURPOSE_EVAL
    assert meter.snapshot().trajectory_write_failures == 0


def test_eval_one_records_nothing_when_no_store(tmp_path):
    runtime = _Runtime(run=lambda r, t: "out",
                       reward=lambda t, o: 1.0,
                       cache=MemoryCache())
    artifact = EvolvingArtifact("a", {"k": "v"}, strategy=AppendRules(),
                                runtime=runtime)
    assert runtime.eval_one(artifact, Task(id="t1", prompt="p")) == 1.0


# -- an unencodable record degrades the recording, never the rollout ----------


class _Unprintable:
    def __str__(self):
        raise RuntimeError("no str for you")


@pytest.mark.parametrize("output", [
    {"logprob": float("nan")},          # allow_nan=False -> ValueError
    {("a", "b"): 1},                    # non-string dict key -> TypeError
    _Unprintable(),                     # default=str itself raises
], ids=["nan-inside-output", "tuple-key", "str-raises"])
def test_unencodable_record_is_counted_not_raised(tmp_path, output):
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter)
    assert _record(store, output=output) is None
    s = meter.snapshot()
    assert s.trajectory_write_failures == 1
    assert s.trajectories_recorded == 0
    assert os.listdir(str(tmp_path)) == []          # no record, no orphan tmp
    # The identity was never written, so a later encodable record still lands.
    assert _record(store, output="fine") is not None
    assert store.count() == 1


def test_circular_output_is_counted_not_raised(tmp_path):
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter)
    loop = []
    loop.append(loop)
    assert _record(store, output=loop) is None
    assert meter.snapshot().trajectory_write_failures == 1
    assert os.listdir(str(tmp_path)) == []


def test_non_string_event_detail_is_counted_not_raised(tmp_path):
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter, record_text=True)
    event = TrajectoryEvent("step", "run", detail=12345)   # not subscriptable
    assert _record(store, events=(event,)) is None
    assert meter.snapshot().trajectory_write_failures == 1


def test_eval_one_does_not_rerun_a_rollout_whose_record_cannot_be_encoded(tmp_path):
    """The gate already has its answer; the store failing to serialise the
    output must not be mistaken for a backend transient and re-run the
    rollout (or fail the gate)."""
    meter = Meter()
    store = TrajectoryStore(str(tmp_path), meter=meter)
    calls = {"run": 0}

    def run(rendered, task):
        calls["run"] += 1
        return {"answer": "x", "logprob": float("nan")}

    runtime = _Runtime(run=run, reward=lambda t, o: 1.0,
                       cache=MemoryCache(meter), meter=meter,
                       trajectory_store=store)
    artifact = EvolvingArtifact("a", {"k": "v"}, strategy=AppendRules(),
                                runtime=runtime)
    assert runtime.eval_one(artifact, Task(id="t1", prompt="p")) == 1.0
    assert calls["run"] == 1                     # paid once, not ATTEMPTS times
    assert store.count() == 0
    assert meter.snapshot().trajectory_write_failures == 1
