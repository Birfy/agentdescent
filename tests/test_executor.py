"""Rollouts run somewhere; that somewhere is replaceable, and it can die.

The point of processes here is not speed -- `docs/efficiency.md` measures 7.1x on
I/O against 1.0x on CPU, and a rollout is almost all waiting on a model. It is
that the code being evolved is model-authored, so a worker that segfaults or gets
OOM-killed is ordinary, and in a thread it takes the run with it.

Which makes the interesting assertions here failure ones, not throughput ones.
"""

import multiprocessing as mp
import os
import sys
import time

import pytest

from agentdescent import Task
from agentdescent.executor import Executor, Result, ThreadExecutor
from agentdescent.filetree import canonical
from agentdescent.supervisor import ProcessExecutor, _refuse_to_recurse
from agentdescent.workspec import Ref, RolloutSpec

TREE = canonical({"main.py": "import os;print(f'42@{os.getpid()}')"})


def spec(i=0, entrypoint=None):
    return RolloutSpec(
        rendered=TREE,
        task=Task(f"t{i}", f"q{i}", meta={"gold": "42"}),
        run=Ref("agentdescent.runners:code_runner",
                {"entrypoint": entrypoint or [sys.executable, "main.py"]}),
        reward=Ref("agentdescent.rewards:last_number"))


# ---------------------------------------------------------------------------
# Both executors satisfy the same contract
# ---------------------------------------------------------------------------


def test_both_executors_satisfy_the_protocol():
    assert isinstance(ThreadExecutor(1), Executor)
    assert isinstance(ProcessExecutor(1), Executor)


def test_the_thread_executor_takes_the_callables_directly():
    """In-process there is no boundary, so a closure is fine -- and staying that
    way is what keeps every existing caller working untouched."""
    ex = ThreadExecutor(2, run=lambda rendered, task: task.meta["gold"],
                        reward=lambda task, out: 1.0 if out == "42" else 0.0)
    try:
        results = list(ex.map_rollouts([spec(i) for i in range(4)]))
    finally:
        ex.shutdown()
    assert len(results) == 4
    assert all(r.ok and r.reward == 1.0 for r in results)


def test_results_arrive_as_they_finish_not_in_order():
    """A caller that must wait for the first spec before seeing the second has
    serialised itself, and rollout durations have a long tail."""
    order = []

    def slow(rendered, task):
        delay = 0.30 if task.id == "t0" else 0.01
        time.sleep(delay)
        order.append(task.id)
        return "42"

    ex = ThreadExecutor(4, run=slow, reward=lambda t, o: 1.0)
    try:
        seen = [r.task_id for r in ex.map_rollouts([spec(i) for i in range(4)])]
    finally:
        ex.shutdown()
    assert set(seen) == {"t0", "t1", "t2", "t3"}
    assert seen[0] != "t0", f"the slow one blocked the rest: {seen}"


def test_a_failing_rollout_is_reported_not_raised():
    """One bad rollout is evidence, not the end of the run."""
    def boom(rendered, task):
        raise RuntimeError("the model said no")

    ex = ThreadExecutor(2, run=boom, reward=lambda t, o: 1.0)
    try:
        results = list(ex.map_rollouts([spec(i) for i in range(3)]))
    finally:
        ex.shutdown()
    assert len(results) == 3
    assert all(not r.ok and "the model said no" in r.error for r in results)
    assert all(r.kind == "model" for r in results)


# ---------------------------------------------------------------------------
# Crossing a process boundary
# ---------------------------------------------------------------------------


def test_the_start_method_is_spawn():
    """`fork` from a threaded parent produces children holding locks nothing will
    release. The engine is threaded, so this is not a preference."""
    ex = ProcessExecutor(1)
    try:
        assert ex._ctx.get_start_method() == "spawn"
    finally:
        ex.shutdown()


def test_work_really_runs_in_another_process():
    ex = ProcessExecutor(2)
    try:
        results = list(ex.map_rollouts([spec(i) for i in range(4)]))
    finally:
        ex.shutdown()
    assert len(results) == 4, [r.error for r in results]
    assert all(r.ok for r in results), [r.error for r in results]
    pids = {r.output.rsplit("@", 1)[1] for r in results}
    assert str(os.getpid()) not in pids, "the rollout ran in the parent"


def test_a_closure_is_refused_before_any_rollout_runs():
    """The alternative is a pickling error from a queue's feeder thread, minutes
    in, naming neither the argument nor the fix."""
    bad = RolloutSpec(rendered="r", task=Task("t", "q"),
                      run=Ref("agentdescent.rewards:last_number"),
                      reward=Ref("agentdescent.rewards:last_number"))
    object.__setattr__(bad, "run", lambda rendered, task: "x")   # a closure

    ex = ProcessExecutor(1)
    try:
        with pytest.raises(TypeError, match="cannot cross a process boundary"):
            list(ex.map_rollouts([bad]))
    finally:
        ex.shutdown()


def test_building_an_executor_inside_a_worker_is_refused():
    """`spawn` re-imports `__main__` in every child, so an executor built at
    module level builds one in each child, which builds one in each grandchild.
    The machine fills with processes and nothing reports -- it reads as slow
    rather than as a fault, which is why this refuses rather than warns."""
    assert _refuse_to_recurse() is None            # in the parent: fine

    original = mp.parent_process

    def pretend_child():
        return object()

    mp.parent_process = pretend_child
    try:
        with pytest.raises(RuntimeError, match="__main__"):
            _refuse_to_recurse()
    finally:
        mp.parent_process = original


def test_killing_one_worker_does_not_take_the_others_with_it():
    """The assertion `ProcessPoolExecutor` cannot pass.

    There, an abruptly-dead worker raises `BrokenProcessPool` and every in-flight
    task dies with it -- so a single OOM from one model-authored candidate ends
    the run. Here the survivors keep working and the dead one's task is sent
    somewhere else."""
    ex = ProcessExecutor(3)
    ex.start()
    victim = ex._slots[0].process
    others = [ex._slots[i].process for i in (1, 2)]
    try:
        os.kill(victim.pid, 9)
        deadline = time.time() + 10
        while time.time() < deadline and victim.is_alive():
            time.sleep(0.05)
        assert not victim.is_alive(), "the victim survived SIGKILL"
        assert all(p.is_alive() for p in others), (
            "killing one worker killed the others -- this is the BrokenProcessPool "
            "behaviour the supervisor exists to avoid")

        results = list(ex.map_rollouts([spec(i) for i in range(4)]))
        assert len(results) == 4, [r.error for r in results]
        assert all(r.ok for r in results), [r.error for r in results]
    finally:
        ex.shutdown()


def test_a_task_whose_worker_dies_is_re_dispatched_under_the_same_lease():
    """Same lease id on purpose: a worker only *presumed* dead may still report,
    and the id is what lets the caller drop the duplicate. Without it one
    misjudged timeout puts two cards for the same task into the evidence pool."""
    ex = ProcessExecutor(2, hang_timeout=0.5)
    ex.start()
    try:
        pending = {"L1": spec(1)}
        inflight = {"L1": time.time() - 60}       # long overdue
        to_send = []
        out = ex._reclaim(pending, inflight, {"L1": 0}, {}, to_send)
        assert not out, "an overdue task should be retried, not failed"
        assert [s.lease_id for s in to_send] == [pending["L1"].lease_id]
    finally:
        ex.shutdown()


def test_a_task_that_keeps_killing_workers_is_given_up_on():
    """Otherwise one poisonous candidate re-dispatches forever."""
    ex = ProcessExecutor(2, hang_timeout=0.5)
    try:
        s = spec(9)
        pending, to_send = {s.lease_id: s}, []
        attempts = {s.lease_id: 2}
        out = ex._reclaim(pending, {s.lease_id: time.time() - 60},
                          {s.lease_id: 0}, attempts, to_send)
        assert len(out) == 1 and out[0].kind == "infrastructure"
        assert not to_send and not pending
    finally:
        ex.shutdown()


def test_shutdown_leaves_no_workers_behind():
    ex = ProcessExecutor(2)
    ex.start()
    processes = [slot.process for slot in ex._slots.values()]
    assert processes and all(p.is_alive() for p in processes)
    ex.shutdown(grace=3.0)
    deadline = time.time() + 10
    while time.time() < deadline and any(p.is_alive() for p in processes):
        time.sleep(0.1)
    assert not any(p.is_alive() for p in processes), "workers outlived shutdown"


def test_shutdown_is_idempotent():
    ex = ProcessExecutor(1)
    ex.start()
    ex.shutdown()
    ex.shutdown()


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_infrastructure_failures_are_counted_apart_from_model_failures():
    """`WorkerHealth` retires only while nothing has ever succeeded -- exactly
    the window in which sandboxes fail to start. Counting the two together
    retires every worker at startup and finishes the run clean at zero
    concurrency."""
    from agentdescent.supervisor import _classify
    from agentdescent.workspec import RefError

    assert _classify(RefError("nope")) == "caller"
    assert _classify(OSError("no space left on device")) == "infrastructure"
    assert _classify(MemoryError()) == "infrastructure"
    assert _classify(RuntimeError("the model refused")) == "model"


# ---------------------------------------------------------------------------
# Task-level recovery
# ---------------------------------------------------------------------------


def test_a_late_result_from_a_presumed_dead_worker_is_dropped_and_counted():
    """The idempotency that makes re-dispatch safe.

    A worker declared lost may simply have been slow. When it finishes, its
    result arrives for a task somebody else has already answered -- and accepting
    both would put two cards for one task into the evidence pool, which is a
    quiet doubling of one worker's vote.

    Dropping it is correct and invisible, so it is counted: an over-eager
    re-dispatch policy otherwise looks exactly like a well-tuned one."""
    from agentdescent.metrics import Meter

    meter = Meter()
    ex = ProcessExecutor(2, meter=meter)
    try:
        specs = [spec(i) for i in range(2)]
        results = list(ex.map_rollouts(specs))
        assert len(results) == 2

        # the straggler reports, after its lease has already been answered
        ex._results.put(("DONE", 0, specs[0].lease_id, specs[0].task.id,
                         "late", 1.0, 0.1))
        again = list(ex.map_rollouts([spec(9)]))
        assert len(again) == 1 and again[0].task_id == "t9", (
            "a duplicate for an answered task was reported as a result")
    finally:
        ex.shutdown()
    assert meter.snapshot().duplicates_dropped >= 1


def test_re_dispatch_is_counted():
    ex = ProcessExecutor(2, hang_timeout=0.5)
    from agentdescent.metrics import Meter
    ex.meter = Meter()
    ex.start()
    try:
        s = spec(3)
        to_send = []
        ex._reclaim({s.lease_id: s}, {s.lease_id: time.time() - 60},
                    {s.lease_id: 0}, {}, to_send)
        assert to_send and ex.meter.snapshot().redispatched == 1
    finally:
        ex.shutdown()


def test_the_loop_never_blocks_forever_on_a_result_that_cannot_come():
    """If a worker dies holding the only copy of a task, the run must end -- with
    a failure for that task -- rather than wait for a message nobody will send."""
    ex = ProcessExecutor(1, hang_timeout=0.2)
    try:
        s = spec(5)
        pending, to_send = {s.lease_id: s}, []
        out = ex._reclaim(pending, {s.lease_id: time.time() - 60},
                          {s.lease_id: 0}, {s.lease_id: 5}, to_send)
        assert len(out) == 1 and not out[0].ok
        assert out[0].kind == "infrastructure"
        assert not pending, "the task stayed pending and the loop would not end"
    finally:
        ex.shutdown()


# ---------------------------------------------------------------------------
# The seam, in the round body
# ---------------------------------------------------------------------------


def test_the_round_body_routes_its_rollout_through_the_executor():
    """The seam is load-bearing, not decorative: a counting executor sees every
    rollout the round performs."""
    from agentdescent import AppendRules, Policies, evolve
    from agentdescent.executor import ThreadExecutor

    seen = []

    class Counting(ThreadExecutor):
        def rollout(self, spec):
            seen.append(spec.task.id)
            return super().rollout(spec)

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]
    ex = Counting(2, run=lambda r, t: t.meta["gold"] if t.id in r else "?",
                  reward=lambda t, o: 1.0 if o == t.meta["gold"] else 0.0)
    try:
        result = evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                        run=lambda r, t: "unused", propose=lambda r, t, o, s: t.id,
                        strategy=AppendRules(), rounds=3, n_workers=2,
                        held_out_frac=0.5, seed=0, policies=Policies(executor=ex))
    finally:
        ex.shutdown()
    assert seen, "the round body never asked the executor for a rollout"
    assert result.rollouts == len(seen), (
        f"{result.rollouts} rollouts counted but {len(seen)} went through the seam")


def test_a_caller_contract_failure_still_stops_the_run():
    """The executor reports rather than raises, because one bad rollout is
    evidence. But a broken contract is not evidence -- it makes the whole run
    meaningless -- and that distinction has to survive the trip through
    `Result.kind`."""
    from agentdescent import AppendRules, Policies, evolve
    from agentdescent.evolution import ProposalContractError
    from agentdescent.executor import ThreadExecutor

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]
    ex = ThreadExecutor(2, run=lambda r, t: "x",
                        reward=lambda t, o: 5.0)      # outside [0, 1]: a caller bug
    try:
        with pytest.raises(Exception) as excinfo:
            evolve(tasks, lambda t, o: 5.0, run=lambda r, t: "x",
                   propose=lambda r, t, o, s: None, strategy=AppendRules(),
                   rounds=2, n_workers=1, held_out_frac=0.5, seed=0,
                   policies=Policies(executor=ex))
        assert "range" in str(excinfo.value).lower() or "0, 1" in str(excinfo.value)
    finally:
        ex.shutdown()


def test_a_supplied_executor_runs_the_callers_actors():
    """`evolve()` hands its actors to whatever executor it was given.

    Without that the executor falls back to resolving the spec's `Ref`s -- and
    `evolve()` is handed `run` and `reward` as closures, which have no name to
    resolve. The failure was silent in the worst way: every rollout failed, the
    gate still scored the artifact, and the run returned `rollouts=0` with a
    plausible `final_reward` and no exception.
    """
    from agentdescent import AppendRules, Policies, evolve
    from agentdescent.executor import ThreadExecutor

    calls = []
    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]

    def run(rendered, task):
        calls.append(task.id)
        return task.meta["gold"]

    ex = ThreadExecutor(2)                    # no actors: only evolve() has them
    try:
        result = evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                        run=run, propose=lambda r, t, o, s: t.id,
                        strategy=AppendRules(), rounds=3, n_workers=2,
                        held_out_frac=0.5, seed=0, policies=Policies(executor=ex))
    finally:
        ex.shutdown()
    assert calls, "the caller's `run` was never invoked through the executor"
    assert result.rollouts > 0, "a supplied executor produced no rollouts at all"
    assert result.error is None, result.error


def test_an_executor_evolve_cannot_drive_is_refused_not_run():
    """A cross-process executor cannot take closures, and `evolve()` has nothing
    else to give it. Refusing at build time is the only honest answer: the run it
    would otherwise produce measures nothing and says so nowhere."""
    from agentdescent import AppendRules, Policies, evolve
    from agentdescent.supervisor import ProcessExecutor

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]
    with pytest.raises(TypeError) as excinfo:
        evolve(tasks, lambda t, o: 1.0, run=lambda r, t: "x",
               propose=lambda r, t, o, s: None, strategy=AppendRules(),
               rounds=1, n_workers=1, held_out_frac=0.5, seed=0,
               policies=Policies(executor=ProcessExecutor(1)))
    assert "attach_actors" in str(excinfo.value)


def test_a_spec_built_by_evolve_refuses_to_resolve_its_actors():
    """The spec carries a reference that raises, not a plausible default.

    It used to name `agents:echo` and `rewards:contains` -- real callables with
    the wrong signature -- so the failure arrived as a `TypeError` about argument
    counts, naming neither the cause nor the fix.
    """
    from agentdescent.evolution import undescribable_actor
    from agentdescent.workspec import Ref, RefError

    ref = Ref("agentdescent.evolution:undescribable_actor", {"which": "reward"})
    with pytest.raises(RefError) as excinfo:
        ref.resolve()
    assert "reward" in str(excinfo.value)
    assert "closure" in str(excinfo.value)
    with pytest.raises(RefError):
        undescribable_actor("run")
