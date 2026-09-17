"""Memoised evaluations, and what changes when more than one process wants them.

Evaluation is the expensive half of a run: `docs/efficiency.md` measures the gate
at 193.6s against 90.0s for the same work at `eval_concurrency` 1 and 8, and each
of those seconds is a real rollout. So the cache is not a nicety, and it stops
working in two specific ways as soon as there is more than one worker.

**Two callers, one key, two computations.** The in-process dictionary checked for
a key, released its lock, computed, and stored -- so N threads asking for the same
uncomputed value all missed and all computed. Wasteful in one process; across
processes it is N copies of an expensive gate. `single-flight` makes the first
caller compute while the others wait for that result.

**Two environments, one key, a wrong answer.** The key was `(rendered, task.id)`.
That is complete only while every measurement comes from the same toolchain. Once
one rollout runs under `python:3.11-slim` and another under a different image,
sharing a cache between them means one environment's score answers the other's
question -- and that score is what the commit gates read. The environment's
fingerprint is part of the key, so those are simply different entries.

**What the key is not** is the artifact's state. It is what the artifact
*renders to*, because `eval_one` passes only `render()` to `run`: two states that
render identically cannot score differently, and keying on state made a strategy
that carries bookkeeping beside the artifact (ADAS keeps a design's name and
rationale) re-evaluate the whole held-out set because a label changed.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

__all__ = ["CacheProtocol", "FileCache", "MemoryCache", "cache_key"]


def cache_key(rendered: str, task_id: str, fingerprint: str = "") -> Tuple[str, str, str]:
    """The identity of one evaluation.

    All three parts are needed, and the third is the one that is easy to leave
    out: without it a shared cache answers a question asked in one environment
    with a measurement taken in another."""
    return (rendered, task_id, fingerprint)


@runtime_checkable
class CacheProtocol(Protocol):
    """Somewhere to keep evaluations. In one process, across many, or on disk."""

    def get_or_eval(self, key: Any, fn: Callable[[], float]) -> float: ...


class _Inflight:
    """One key being computed, and everyone waiting for it."""

    __slots__ = ("event", "value", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Optional[float] = None
        self.error: Optional[BaseException] = None


class MemoryCache:
    """In-process, single-flight, counted.

    Single-flight is the difference from a plain dictionary: concurrent callers
    for the same uncomputed key produce **one** computation, not one each. The
    old code checked, released the lock, then computed -- correct, and wasteful
    exactly when the value is expensive enough to be worth caching.
    """

    def __init__(self, meter: Optional[Any] = None) -> None:
        self._values: Dict[Any, float] = {}
        self._inflight: Dict[Any, _Inflight] = {}
        self._lock = threading.Lock()
        self._meter = meter

    def get_or_eval(self, key: Any, fn: Callable[[], float]) -> float:
        with self._lock:
            if key in self._values:
                self._count("cache_hits")
                return self._values[key]
            waiting = self._inflight.get(key)
            if waiting is None:
                waiting = self._inflight[key] = _Inflight()
                mine = True
            else:
                mine = False

        if not mine:
            # Somebody is already computing this. Waiting counts as a hit for the
            # duplicate-computation figure -- the work was not done twice -- but
            # separately, because "waited for" and "found ready" have different
            # costs and only one of them is free.
            self._count("cache_hits")
            self._count("cache_inflight_joins")
            waiting.event.wait()
            if waiting.error is not None:
                raise waiting.error
            return waiting.value           # type: ignore[return-value]

        self._count("cache_misses")
        try:
            value = fn()
        except BaseException as exc:       # noqa: BLE001 -- re-raised to everyone
            waiting.error = exc
            with self._lock:
                self._inflight.pop(key, None)
            waiting.event.set()
            raise
        with self._lock:
            self._values[key] = value
            self._inflight.pop(key, None)
        waiting.value = value
        waiting.event.set()
        return value

    def attach_meter(self, meter: Any) -> None:
        """Report into *this* run's counters, replacing any earlier run's.

        A caller-supplied cache is built before the engine exists, so it has no
        meter -- and without this the duplicate-computation figure silently reads
        zero for exactly the configuration the cache was added to measure.

        It replaces rather than keeps the first, because a shared cache outlives
        the run that created it: that is the point of sharing one. Keeping the
        first meter sends the second run's hits into the first run's totals,
        which is worse than not counting at all -- the number looks like a
        measurement of the wrong run."""
        self._meter = meter

    def _count(self, counter: str) -> None:
        if self._meter is not None:
            self._meter.add(counter)


class FileCache:
    """A directory of evaluations, so separate processes can share them.

    Deliberately files rather than a database: the point is that two processes on
    one machine stop paying twice for the same gate, and that needs no server.
    A network cache is the same protocol with a different backend, and it should
    arrive with the cross-machine work that would justify running one.

    Wraps a `MemoryCache`, so single-flight within a process still holds and the
    file is only consulted once per process per key.
    """

    def __init__(self, directory: str, meter: Optional[Any] = None) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._memory = MemoryCache(meter)

    def attach_meter(self, meter: Any) -> None:
        self._memory.attach_meter(meter)

    def _path(self, key: Any) -> str:
        import hashlib

        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return os.path.join(self.directory, f"{digest}.json")

    def get_or_eval(self, key: Any, fn: Callable[[], float]) -> float:
        def from_disk_or_compute() -> float:
            path = self._path(key)
            try:
                with open(path, encoding="utf-8") as fh:
                    return float(json.load(fh)["value"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
            value = fn()
            # Write to a sibling and rename: a reader must never see half a file,
            # and on POSIX rename is atomic. A cache that cannot be written is
            # still a correct cache, so failures here are dropped rather than
            # failing a run that has already paid for the answer.
            try:
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"key": list(key) if isinstance(key, tuple) else key,
                               "value": value}, fh, default=str)
                os.replace(tmp, path)
            except OSError:
                pass
            return value

        return self._memory.get_or_eval(key, from_disk_or_compute)
