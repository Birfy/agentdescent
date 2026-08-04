"""Rollouts in other processes, supervised by this one.

The code being evolved is written by a model, so the interesting failures are not
exceptions. They are a worker that segfaults, gets OOM-killed, or stops answering
while holding a sandbox. A pool that treats those as "the pool is broken" -- which
`ProcessPoolExecutor` does -- turns one bad candidate into a dead run, which is
the opposite of why anyone reaches for processes.

So this is not a pool. It is a set of long-lived workers with a supervisor that
decides, on its own, when one of them is gone.

Four decisions worth stating, because each replaced something that looked
reasonable:

**`spawn`, never `fork`.** The engine is threaded. A `fork` from a process whose
other threads hold locks produces a child holding locks that nothing will ever
release, and the symptom is an occasional hang rather than an error. The cost is
that a worker re-imports on start, which is why workers are persistent rather
than per-task.

**Liveness is `is_alive()`, not a heartbeat.** A worker is single-threaded: once
it is inside a rollout it cannot send anything, and a rollout can legitimately
take ten minutes. A heartbeat TTL short enough to detect death quickly would kill
workers that are merely working. `multiprocessing` already knows whether a
process is alive, exactly and for free. The heartbeat that remains is for the
other failure -- alive but wedged -- and its timeout is therefore *longer* than
any legitimate rollout, not shorter.

**The results queue is unbounded, the task queue is not.** Back-pressure belongs
on the work going in. Bounding results deadlocks: a full results queue blocks the
worker in `put`, so it never takes another task, so the supervisor waits for a
result that will never come.

**Re-dispatch reuses the lease id.** A worker wrongly presumed dead has its task
sent to somebody else; if the first one then finishes, both report. The lease id
is what lets the caller drop the second, and without it one misjudged timeout
puts two cards for the same task into the evidence pool.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .executor import Result
from .metrics import Meter
from .workspec import DEFAULT_ALLOWED_PREFIXES, RolloutSpec

__all__ = ["ProcessExecutor", "worker_main"]

_STOP = "__agentdescent_stop__"

#: How long a worker may be inside one rollout before it is presumed wedged.
#: Longer than any legitimate rollout on purpose -- see the module docstring. It
#: is a deadlock detector, not a liveness check.
DEFAULT_HANG_TIMEOUT = 900.0


def _install_rlimits(memory_mb: Optional[int]) -> None:
    """Cap what a worker can take, where the platform allows it.

    POSIX only: Windows has no `resource` module. Rather than pretending
    otherwise, the container provider is where a cross-platform memory ceiling
    actually exists."""
    if memory_mb is None:
        return
    try:
        import resource
    except ImportError:                       # pragma: no cover -- Windows
        return
    limit = int(memory_mb) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):             # pragma: no cover
        pass


def worker_main(wid: int, tasks, results, allowed: Sequence[str],
                memory_mb: Optional[int] = None) -> None:
    """A worker's whole life. Module-level because `spawn` imports it by name.

    A closure could not be used here even if it were tidier -- which is the same
    constraint that made `Ref` necessary, seen from the other side.
    """
    if os.name != "nt":
        os.setsid()               # our own process group, so a kill takes the tree
    _install_rlimits(memory_mb)

    while True:
        spec = tasks.get()
        if spec == _STOP:
            break
        results.put(("START", wid, spec.lease_id, time.time()))
        started = time.time()
        try:
            run, reward = spec.resolve(allowed)
            output = run(spec.rendered, spec.task)
            value = reward(spec.task, output)
            results.put(("DONE", wid, spec.lease_id, spec.task.id, output, value,
                         time.time() - started))
        except Exception as e:  # noqa: BLE001
            results.put(("FAILED", wid, spec.lease_id, spec.task.id,
                         f"{type(e).__name__}: {str(e)[:200]}",
                         _classify(e), time.time() - started))
    results.put(("BYE", wid, None, 0.0))


def _classify(exc: BaseException) -> str:
    """Which counter a failure belongs to.

    Infrastructure failures must not feed the worker-retirement rule: it only
    fires while no worker has ever succeeded, which is precisely the window in
    which a sandbox fails to start. Counting them together retires every worker
    at startup and finishes the run clean at zero concurrency."""
    from .workspec import RefError

    if isinstance(exc, RefError):
        return "caller"
    name = type(exc).__name__
    if name in ("EngineError", "SandboxLeak", "OSError", "MemoryError"):
        return "infrastructure"
    return "model"


@dataclass
class _Slot:
    wid: int
    process: Any
    restarts: int = 0


class ProcessExecutor:
    """Persistent worker processes, with re-dispatch when one dies.

    Requires specs whose `Ref`s resolve on the far side; a closure cannot cross,
    and the failure says so before any rollout runs rather than as a pickling
    error mid-flight.
    """

    def __init__(self, max_workers: int = 4, *, meter: Optional[Meter] = None,
                 allowed_prefixes: Sequence[str] = DEFAULT_ALLOWED_PREFIXES,
                 hang_timeout: float = DEFAULT_HANG_TIMEOUT,
                 max_restarts: int = 3, memory_mb: Optional[int] = None) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.meter = meter
        self.allowed_prefixes = tuple(allowed_prefixes)
        self.hang_timeout = hang_timeout
        self.max_restarts = max_restarts
        self.memory_mb = memory_mb
        self._ctx = mp.get_context("spawn")
        self._tasks = self._ctx.Queue(maxsize=max_workers * 2)
        self._results = self._ctx.Queue()          # unbounded on purpose
        self._slots: Dict[int, _Slot] = {}
        self._lock = threading.Lock()
        self._closed = False

    # -- workers --------------------------------------------------------------

    def _spawn(self, wid: int) -> _Slot:
        p = self._ctx.Process(
            target=worker_main,
            args=(wid, self._tasks, self._results, self.allowed_prefixes,
                  self.memory_mb),
            daemon=True)          # a stuck worker must not keep the process alive
        p.start()
        return _Slot(wid, p)

    def start(self) -> None:
        _refuse_to_recurse()
        with self._lock:
            for wid in range(self.max_workers):
                if wid not in self._slots:
                    self._slots[wid] = self._spawn(wid)

    # -- running --------------------------------------------------------------

    def map_rollouts(self, specs: Sequence[RolloutSpec]) -> Iterator[Result]:
        if self._closed:
            raise RuntimeError("executor is shut down")
        specs = list(specs)
        if not specs:
            return
        _refuse_closures(specs)
        self.start()

        pending: Dict[str, RolloutSpec] = {s.lease_id: s for s in specs}
        inflight: Dict[str, float] = {}            # lease -> started at
        owner: Dict[str, int] = {}                 # lease -> worker id
        attempts: Dict[str, int] = {}
        to_send = list(specs)

        while pending:
            while to_send and len(inflight) < self.max_workers * 2:
                spec = to_send.pop(0)
                self._tasks.put(spec)
                inflight[spec.lease_id] = time.time()

            try:
                message = self._results.get(timeout=1.0)
            except queue.Empty:
                message = None

            if message is not None:
                kind = message[0]
                if kind == "START":
                    _, wid, lease, at = message
                    owner[lease] = wid
                    inflight[lease] = at
                    continue
                if kind == "BYE":
                    continue
                if kind == "DONE":
                    _, wid, lease, task_id, output, value, seconds = message
                    inflight.pop(lease, None)
                    if pending.pop(lease, None) is None:
                        # A worker that was only *presumed* dead, finishing after
                        # its task had been given to somebody else. Dropping it is
                        # correct and invisible, so it is counted: an over-eager
                        # re-dispatch policy otherwise looks exactly like a
                        # well-tuned one.
                        if self.meter is not None:
                            self.meter.add("duplicates_dropped")
                        continue
                    if self.meter is not None:
                        self.meter.add("rollouts")
                        self.meter.add("rollout_seconds", seconds)
                    yield Result(lease, task_id, output=output, reward=value,
                                 seconds=seconds)
                    continue
                if kind == "FAILED":
                    _, wid, lease, task_id, error, failure_kind, seconds = message
                    inflight.pop(lease, None)
                    if pending.pop(lease, None) is None:
                        if self.meter is not None:
                            self.meter.add("duplicates_dropped")
                        continue
                    if self.meter is not None and failure_kind == "infrastructure":
                        self.meter.add("sandbox_failures")
                    yield Result(lease, task_id, error=error, kind=failure_kind,
                                 seconds=seconds)
                    continue

            for result in self._reclaim(pending, inflight, owner, attempts, to_send):
                yield result

        return

    def _reclaim(self, pending, inflight, owner, attempts, to_send) -> List[Result]:
        """Notice dead or wedged workers and put their work back.

        Death is checked first and cheaply: `is_alive()` is exact, and it covers
        the OOM/segfault case that motivates processes at all. The hang timeout
        is the fallback for a worker that is alive and stuck, and it is long
        enough that no working rollout trips it."""
        out: List[Result] = []
        now = time.time()
        dead = [wid for wid, slot in list(self._slots.items())
                if not slot.process.is_alive()]
        wedged = [lease for lease, started in list(inflight.items())
                  if now - started > self.hang_timeout]

        for lease in list(inflight):
            wid = owner.get(lease)
            if (wid is not None and wid in dead) or lease in wedged:
                inflight.pop(lease, None)
                spec = pending.get(lease)
                if spec is None:
                    continue
                attempts[lease] = attempts.get(lease, 0) + 1
                if attempts[lease] > 2:
                    pending.pop(lease, None)
                    out.append(Result(
                        lease, spec.task.id,
                        error=("worker died repeatedly while running this task"),
                        kind="infrastructure"))
                    continue
                # Same lease id on purpose: if the original worker was only
                # presumed dead and reports later, the caller can tell the two
                # apart and drop one.
                if self.meter is not None:
                    self.meter.add("redispatched")
                to_send.append(spec)

        for wid in dead:
            slot = self._slots.get(wid)
            if slot is None:
                continue
            if slot.restarts >= self.max_restarts:
                # A worker that keeps dying is a crash loop; leave the slot empty
                # rather than burning CPU restarting it. At least one worker must
                # survive, or the run silently becomes serial and then stalls.
                if len(self._slots) > 1:
                    self._slots.pop(wid, None)
                    continue
            new = self._spawn(wid)
            new.restarts = slot.restarts + 1
            self._slots[wid] = new
        return out

    # -- shutdown -------------------------------------------------------------

    def attach_meter(self, meter: Meter) -> None:
        """Report into this run's counters. See `ThreadExecutor.attach_meter`."""
        self.meter = meter

    def rollout(self, spec: RolloutSpec) -> Result:
        """One rollout in a worker process, with the same recovery as a batch."""
        for result in self.map_rollouts([spec]):
            return result
        raise RuntimeError("the executor returned no result for a rollout")

    def shutdown(self, grace: float = 5.0) -> None:
        """Stop the workers, in the one order that does not hang.

        A child that has written to a queue cannot exit until the data is drained,
        so joining before draining deadlocks. Drain, then stop, then join, then
        kill whatever is left."""
        if self._closed:
            return
        self._closed = True

        deadline = time.time() + grace
        while time.time() < deadline:                     # 1. drain
            try:
                self._results.get(timeout=0.05)
            except queue.Empty:
                break

        for _ in self._slots:                             # 2. ask them to stop
            try:
                self._tasks.put(_STOP, timeout=1.0)
            except Exception:
                break

        for slot in list(self._slots.values()):           # 3. join, then insist
            slot.process.join(timeout=max(0.1, grace / 2))
            if slot.process.is_alive():
                _kill_tree(slot.process)
        self._slots.clear()


def _refuse_to_recurse() -> None:
    """Refuse to spawn workers from inside a worker.

    `spawn` starts a child by **re-importing `__main__`**. A script that builds a
    `ProcessExecutor` at module level therefore builds one again in every child,
    which builds one in every grandchild: the machine fills with processes and
    nothing ever reports, so it reads as "slow" rather than as a fault.

    Python's own guard for this only fires if the child gets as far as spawning,
    which is too late to be useful. `parent_process()` is not `None` in any
    spawned child, so this fires on the first one and names the fix.
    """
    if mp.parent_process() is not None:
        raise RuntimeError(
            "ProcessExecutor was constructed inside a worker process. With the "
            "'spawn' start method each worker re-imports __main__, so anything "
            "at module level runs again in every child -- including this. Put "
            "the run behind `if __name__ == \"__main__\":`, or move it into a "
            "function the module does not call on import.")


def _kill_tree(process) -> None:
    """Kill a worker and whatever it started."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:                                             # pragma: no cover
            process.kill()
    except (OSError, AttributeError):
        try:
            process.kill()
        except Exception:
            pass


def _refuse_closures(specs: Sequence[RolloutSpec]) -> None:
    """Fail before the first rollout if the work cannot cross a process.

    The alternative is a pickling error from inside a queue's feeder thread,
    minutes in, naming neither the argument nor what to do about it."""
    import pickle

    for spec in specs[:1]:            # they share a shape; one is enough
        try:
            pickle.dumps(spec)
        except Exception as e:
            raise TypeError(
                f"this rollout cannot cross a process boundary: {e}. Every "
                "factory in this package returns a closure, so `run=` and "
                "`reward=` have to be given as `Ref(\"module:factory\", {...})` "
                "for a process executor. In-process, ThreadExecutor takes them "
                "as they are.") from None
