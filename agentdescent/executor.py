"""Where rollouts run, as a replaceable thing.

Today's answer is threads, and for the workload that is measured to be the right
one: a rollout is almost entirely waiting on a model, and `docs/efficiency.md`
records **7.1x** on I/O against **1.0x** on CPU for eight of them. Nothing here
is an attempt to make rollouts faster by using processes.

What processes buy is different and not available any other way:

* **fault isolation** -- the code being evolved is model-authored, so a segfault,
  an OOM or a runaway allocation is an ordinary Tuesday, and in a thread each of
  those takes the run with it;
* **capacity beyond one machine**, later;
* **heterogeneous workers**, later still.

**Not `ProcessPoolExecutor`.** Four reasons, and the first is disqualifying:

1. one worker dying abruptly breaks the whole pool (`BrokenProcessPool`) and
   every in-flight task with it. Fault isolation implemented on top of something
   that fails as a unit is not fault isolation;
2. `max_tasks_per_child` arrived in 3.11 and this package supports 3.9, so a
   worker that leaks cannot be recycled;
3. it has no notion of a sandbox, and no place to say "this task needs an
   environment with fingerprint X, wait for one";
4. its default start method is `fork` on Linux, and this engine is threaded. A
   `fork` from a process holding locks in other threads produces a child holding
   locks that will never be released.

So: persistent workers, a bounded task queue, and a supervisor that decides on
its own when a worker is gone. About three hundred lines, all standard library.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Protocol, Sequence, runtime_checkable

from .metrics import Meter
from .workspec import RolloutSpec

__all__ = ["Executor", "Result", "ThreadExecutor"]


@dataclass
class Result:
    """What one rollout produced, or why it did not.

    ``error`` and ``output`` are both present because a rollout that failed still
    identifies itself: the caller has to know *which* task to re-dispatch, and a
    bare exception does not say."""

    lease_id: str
    task_id: str
    output: Optional[str] = None
    reward: Optional[float] = None
    error: Optional[str] = None
    #: How the failure should be counted. Infrastructure failures must not feed
    #: the worker-retirement rule, which fires hardest before any worker has
    #: succeeded -- exactly when sandboxes fail to start.
    kind: str = "ok"          # "ok" | "model" | "infrastructure" | "caller"
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class Executor(Protocol):
    """Runs rollouts somewhere. Threads here, processes and hosts later."""

    def map_rollouts(self, specs: Sequence[RolloutSpec]) -> Iterator[Result]: ...

    def shutdown(self, grace: float = 5.0) -> None: ...

    def rollout(self, spec: RolloutSpec) -> Result:
        """One rollout, wherever this executor runs them.

        The engine's round body already has its own concurrency -- a thread per
        worker unit under a semaphore -- and the work either side of the rollout
        (choosing the task, turning an output into a diff, handing it to the
        aggregator) touches state that has to stay in this process. So the seam
        is the rollout itself, not the batch: routing one at a time is what lets
        `run` move to another process without moving the control plane with it.
        """


class ThreadExecutor:
    """The default: a bounded pool of threads in this process.

    Deliberately the same shape as the process one, so that swapping them is a
    change of executor rather than a change of engine. It accepts specs whose
    `Ref`s resolve *or* closures already bound into them, because in-process
    there is no boundary for a closure to fail to cross.
    """

    def __init__(self, max_workers: int = 4, *, meter: Optional[Meter] = None,
                 sandbox_pool: Optional[Any] = None,
                 run: Optional[Callable[[str, Any], str]] = None,
                 reward: Optional[Callable[[Any, str], float]] = None) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.meter = meter
        self.sandbox_pool = sandbox_pool
        # In-process, the actors can be handed over directly. A spec still
        # carries `Ref`s so the same spec works across a boundary, but resolving
        # them here would rebuild a model client per rollout.
        self._run = run
        self._reward = reward
        self._closed = False

    # -- the work -------------------------------------------------------------

    def _one(self, spec: RolloutSpec) -> Result:
        t0 = time.time()
        try:
            run = self._run or spec.run.resolve()
            reward = self._reward or spec.reward.resolve()
        except Exception as e:                        # a Ref that will not resolve
            return Result(spec.lease_id, spec.task.id, error=f"{type(e).__name__}: {e}",
                          kind="caller", seconds=time.time() - t0)
        try:
            if self.sandbox_pool is not None:
                with self.sandbox_pool.lease(spec.sandbox):
                    output = run(spec.rendered, spec.task)
            else:
                output = run(spec.rendered, spec.task)
            value = reward(spec.task, output)
        except Exception as e:  # noqa: BLE001 -- a backend or candidate failure
            return Result(spec.lease_id, spec.task.id,
                          error=f"{type(e).__name__}: {str(e)[:200]}",
                          kind="model", seconds=time.time() - t0)
        seconds = time.time() - t0
        if self.meter is not None:
            self.meter.add("rollouts")
            self.meter.add("rollout_seconds", seconds)
        return Result(spec.lease_id, spec.task.id, output=output, reward=value,
                      seconds=seconds)

    def map_rollouts(self, specs: Sequence[RolloutSpec]) -> Iterator[Result]:
        """Run every spec, yielding results **as they finish**.

        Not in order: a caller that waits for the first spec before looking at the
        second has serialised itself, and the long tail of rollout durations is
        exactly what makes that expensive."""
        if self._closed:
            raise RuntimeError("executor is shut down")
        specs = list(specs)
        if not specs:
            return
        done: "queue.Queue[Result]" = queue.Queue()
        gate = threading.Semaphore(min(self.max_workers, len(specs)))

        def work(spec: RolloutSpec) -> None:
            with gate:
                done.put(self._one(spec))

        threads = [threading.Thread(target=work, args=(s,), daemon=True)
                   for s in specs]
        for t in threads:
            t.start()
        for _ in specs:
            yield done.get()

    def attach_meter(self, meter: Meter) -> None:
        """Report into this run's counters.

        A supplied executor is built before the engine exists, so it has no
        meter -- and then `rollouts` reads zero for the configuration whose whole
        point was to move rollouts somewhere else. The same shape of mistake the
        evaluation cache made, in the next seam along."""
        self.meter = meter

    def rollout(self, spec: RolloutSpec) -> Result:
        """One rollout in this process. Same path `map_rollouts` takes."""
        return self._one(spec)

    def shutdown(self, grace: float = 5.0) -> None:
        """Release anything still held. Idempotent.

        The threads are daemons and cannot be cancelled, so this does not wait for
        them: what it must do is make sure no sandbox is still leased, because
        those are bounded and an abandoned one is a slot nobody can use."""
        self._closed = True
        if self.sandbox_pool is not None:
            self.sandbox_pool.close()
