"""The evaluation group: the gate's own concurrency, separate from the rollouts'.

Evaluation and exploration are different workloads that happen to call the same
function. A rollout is long-tailed, frequently fails, and its result is one
opinion among many -- losing one costs a little evidence. An evaluation is
batched, cacheable, and decides whether a change is committed; losing one costs
the decision.

They were already separate in the sense that `EvolvingArtifact.score` opened its
own pool. What they were not is *managed*: that pool was built and thrown away on
every call -- 83 of them in a six-round run over twelve tasks -- so it had a size
and nothing else. No identity means nothing to bound, nothing to observe, and no
way to hand evaluation a different substrate from rollouts.

This is that pool, kept. It is deliberately small: one bounded group with its own
lifetime, and a `map` that preserves order.

`map` **propagates** the first exception, deliberately. A rollout that fails is
one opinion missing from a pool of them; a failed *evaluation* is a missing term
in the mean a commit gate reads, and averaging over the ones that happened to
succeed answers the gate's question with a number that is not the answer. The
caller decides what an unmeasurable gate means -- `_Runtime.eval_one` retries,
and `evolve()` counts the round as unmeasurable -- and it can only decide that if
it is told.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, TypeVar

__all__ = ["EvaluatorGroup"]

T = TypeVar("T")
R = TypeVar("R")


class EvaluatorGroup:
    """A bounded, reusable pool for gate work.

    `max_workers` is the ceiling on evaluations in flight, not a target: a gate
    over four tasks uses four threads whatever the ceiling is.
    """

    def __init__(self, max_workers: int = 8, *, sandbox_pool: Optional[Any] = None,
                 meter: Optional[Any] = None) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        #: Shared with the rollout side when there is one, so that the two
        #: workloads sit under a single ceiling rather than two that add up.
        self.sandbox_pool = sandbox_pool
        self.meter = meter
        self._pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._closed = False
        self._peak = 0
        self._inflight = 0

    def _executor(self) -> ThreadPoolExecutor:
        """The pool, built on demand -- including after a shutdown.

        `shutdown` releases threads; it does not end the group's life. A finished
        run releases its evaluator, and a caller inspecting the result afterwards
        (`verifier.eval_counts` on the returned aggregator is a documented way to
        do that) would otherwise meet an exception about a pool it never knew
        existed. A lazily-created pool that refuses to be created again was a
        model of a resource that does not behave that way.
        """
        with self._lock:
            self._closed = False
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    self.max_workers, thread_name_prefix="agentdescent-eval")
            return self._pool

    def map(self, fn: Callable[[T], R], items: Sequence[T]) -> List[R]:
        """Apply ``fn`` to every item, in parallel, preserving order.

        Order is preserved because a caller averaging scores over a fixed task
        list will eventually want to know which score belongs to which task, and
        a completion-ordered result quietly makes that impossible.
        """
        items = list(items)
        if not items:
            return []
        if len(items) == 1 or self.max_workers == 1:
            return [fn(items[0])] if len(items) == 1 else [fn(i) for i in items]

        pool = self._executor()
        with self._lock:
            self._inflight += len(items)
            self._peak = max(self._peak, self._inflight)
        try:
            return list(pool.map(fn, items))
        finally:
            with self._lock:
                self._inflight -= len(items)

    def stats(self) -> dict:
        """What this group has been asked to do.

        `in_flight` and `peak_inflight` count evaluations **submitted and not yet
        returned**, which is what a caller sizing `eval_concurrency` wants to
        know: how much work arrived at once. They are not a thread count and
        exceed `max_workers` whenever a batch is bigger than the pool -- the
        pool is the ceiling on how many of them are *running*.
        """
        with self._lock:
            return {"max_workers": self.max_workers, "peak_inflight": self._peak,
                    "in_flight": self._inflight, "closed": self._closed}

    def shutdown(self, wait: bool = True) -> None:
        """Release the threads. Idempotent.

        `wait=False` exists for the path where a run is already being abandoned:
        an evaluation still in flight is holding a rollout that cannot be
        cancelled, and blocking on it would keep the interpreter alive."""
        with self._lock:
            pool, self._pool, self._closed = self._pool, None, True
        if pool is not None:
            pool.shutdown(wait=wait)
