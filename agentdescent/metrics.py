"""What a run cost, measured at the places the engine already funnels through.

A run used to report *what* it produced and nothing about what it took to get
there. `EvolutionResult` carried the artifact, the per-round reward and the merge
categories; a refactor that doubled the wall-clock and tripled the model calls
left every one of those numbers untouched, which makes "did this change make
things worse?" unanswerable from the result object alone.

The five questions this module exists to answer -- time to a quality bar, cost to
that bar, how much of the pipeline was stale, how much was recomputed, and how
much was spent waiting on a sandbox rather than on a model -- all decompose into
counters. So this is counters, and deliberately nothing else: no formatting, no
export, no sampling. `Meter` is written from every worker thread and the merger
thread at once, which is the only reason it is more than a dataclass.

**Sum, not wall-clock.** ``rollout_seconds`` adds up per-rollout durations across
concurrent workers, so with N workers it routinely exceeds ``wallclock``. It is
the answer to "how much rollout was there", not "how long did rollouts take"; a
reader who takes it as the latter computes occupancy above 100%.

**Model time and sandbox time are separate on purpose.** Both look identical in a
total wall-clock, and telling them apart is the whole point of the sandbox
counters: "8 workers only bought 2x" reads as a staleness problem, a slow model
and a queue of containers installing dependencies, and those need opposite fixes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Optional, TypeVar

from .agents import Usage

__all__ = ["Meter", "MeterSnapshot", "measured"]

T = TypeVar("T")


@dataclass(frozen=True)
class MeterSnapshot:
    """An immutable read of a :class:`Meter`, taken under its lock.

    Reading the fields one at a time from a live meter gives a mix of instants --
    fine for a log line, not fine for `elapsed_s <= wallclock` style assertions,
    which are exactly what the tests check."""

    elapsed_s: float = 0.0
    rollouts: int = 0
    rollout_seconds: float = 0.0
    eval_seconds: float = 0.0
    merge_seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_inflight_joins: int = 0
    stale_considered: int = 0
    stale_discarded: int = 0
    sandbox_wait_s: float = 0.0
    sandbox_setup_s: float = 0.0
    sandboxes_created: int = 0
    sandboxes_reused: int = 0
    sandbox_failures: int = 0
    env_mismatch: int = 0


#: Every counter `add` accepts. Spelled out rather than derived from `__dict__`
#: so a typo raises instead of quietly creating a field nothing ever reads --
#: a miscount is indistinguishable from "that path never ran".
_COUNTERS = frozenset({
    "rollouts", "rollout_seconds", "eval_seconds", "merge_seconds",
    "cache_hits", "cache_misses", "cache_inflight_joins",
    "stale_considered", "stale_discarded",
    "sandbox_wait_s", "sandbox_setup_s", "sandboxes_created",
    "sandboxes_reused", "sandbox_failures", "env_mismatch",
})


@dataclass
class Meter:
    """Every counter a run accumulates. Safe to share across threads.

    ``+=`` is not atomic in Python -- read, add and store are three bytecodes and
    a thread switch between them loses an increment. With eight workers and a
    merger thread all recording into one meter, that is not a theoretical race:
    it is a slow, silent undercount that makes the numbers look merely
    disappointing rather than wrong. Hence one lock and one write path.
    """

    #: Wall-clock origin, set by the engine before the first unit of work.
    #: 0.0 until then, which is what makes `elapsed()` return 0 for a meter that
    #: was never started rather than the seconds since 1970.
    t0: float = 0.0

    #: Rollouts completed (successful or not) -- the denominator for per-rollout
    #: costs and the sample size behind `rollout_seconds`.
    rollouts: int = 0
    #: Sum over rollouts, not wall-clock. See the module docstring.
    rollout_seconds: float = 0.0
    eval_seconds: float = 0.0
    merge_seconds: float = 0.0

    #: Model spend. Reuses `agents.Usage` rather than duplicating its fields, so
    #: `evolve(usage=...)` and `claude(usage=...)` can share one object and the
    #: caller sees a single set of totals.
    #:
    #: `calls` counts **actor invocations** (`run` and `propose`), not provider
    #: requests: a `cli_agent` rollout is one call here and many requests to the
    #: model. Token counts only appear when the same `Usage` also reaches an
    #: adapter that reports them (`claude`, `openai_compatible`) -- an opaque
    #: `run` has no way to surface them, and inventing a number would be worse
    #: than reporting zero.
    usage: Usage = field(default_factory=Usage)

    #: Evaluation cache. `misses` counts evaluations actually performed;
    #: `inflight_joins` counts callers that waited on an in-flight computation
    #: instead of starting a duplicate one (0 until single-flight lands).
    cache_hits: int = 0
    cache_misses: int = 0
    cache_inflight_joins: int = 0

    #: Staleness needs a denominator: `discarded` alone cannot distinguish "the
    #: lag budget is too tight" from "barely anything was proposed".
    stale_considered: int = 0
    stale_discarded: int = 0

    #: Sandbox accounting. Zero on the default single-workspace path; filled once
    #: the sandbox pool exists. Kept here from the start so the result schema
    #: does not change again when it does.
    sandbox_wait_s: float = 0.0
    sandbox_setup_s: float = 0.0
    sandboxes_created: int = 0
    sandboxes_reused: int = 0
    sandbox_failures: int = 0
    #: Candidates whose base and candidate measurements came from different
    #: environments, so the comparison was refused. Non-zero means the pool is
    #: heterogeneous and quality comparisons from that run need a caveat.
    env_mismatch: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- writing ---------------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of the run. Idempotent: the first call wins.

        Re-starting would move the origin under a `RoundInfo` that already
        recorded an `elapsed_s` against the old one, making the sequence
        non-monotonic."""
        with self._lock:
            if self.t0 == 0.0:
                self.t0 = time.time()

    def add(self, counter: str, value: float = 1) -> None:
        """Add to one counter. The only write path, so the only place to lock."""
        if counter not in _COUNTERS:
            raise KeyError(
                f"unknown meter counter {counter!r}; known: {', '.join(sorted(_COUNTERS))}")
        with self._lock:
            setattr(self, counter, getattr(self, counter) + value)

    # -- reading ---------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since `start()`; 0.0 if the meter was never started."""
        return 0.0 if self.t0 == 0.0 else time.time() - self.t0

    def snapshot(self) -> MeterSnapshot:
        """A consistent read of every counter, taken under the lock."""
        with self._lock:
            u = self.usage
            return MeterSnapshot(
                elapsed_s=0.0 if self.t0 == 0.0 else time.time() - self.t0,
                rollouts=self.rollouts,
                rollout_seconds=self.rollout_seconds,
                eval_seconds=self.eval_seconds,
                merge_seconds=self.merge_seconds,
                calls=u.calls,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                model_seconds=u.seconds,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                cache_inflight_joins=self.cache_inflight_joins,
                stale_considered=self.stale_considered,
                stale_discarded=self.stale_discarded,
                sandbox_wait_s=self.sandbox_wait_s,
                sandbox_setup_s=self.sandbox_setup_s,
                sandboxes_created=self.sandboxes_created,
                sandboxes_reused=self.sandboxes_reused,
                sandbox_failures=self.sandbox_failures,
                env_mismatch=self.env_mismatch,
            )

    def summary(self) -> str:
        s = self.snapshot()
        out = [f"{s.rollouts} rollouts in {s.elapsed_s:.1f}s", self.usage.summary()]
        if s.cache_hits or s.cache_misses:
            out.append(f"cache {s.cache_hits}/{s.cache_hits + s.cache_misses} hit")
        if s.stale_considered:
            out.append(f"stale {s.stale_discarded}/{s.stale_considered}")
        if s.sandboxes_created:
            out.append(f"{s.sandboxes_created} sandboxes, "
                       f"{s.sandbox_wait_s:.1f}s waiting")
        return " | ".join(out)


def measured(fn: Callable[..., T], meter: Optional[Meter]) -> Callable[..., T]:
    """Wrap an actor so each invocation lands in ``meter.usage``.

    ``agents.metered`` does the same job for a ``Completion`` (``prompt -> str``);
    the engine's actors are ``run(rendered, task)`` and
    ``propose(rendered, task, output, reward)``, which that signature cannot
    accept. Same accounting, arbitrary arity.

    **Records the call, not the phase.** ``run`` is invoked from two places --
    a worker exploring, and `_Runtime.eval_one` scoring -- so timing the phase
    here would put evaluation seconds into whichever counter the wrapper was
    given. Each phase times itself; this only counts what the actor cost.

    Returns ``fn`` unchanged when ``meter`` is ``None``, so the un-instrumented
    path stays exactly one function call deep.
    """
    if meter is None:
        return fn

    def call(*args: Any, **kwargs: Any) -> T:
        t0 = time.time()
        try:
            out = fn(*args, **kwargs)
        except Exception:
            meter.usage.record(seconds=time.time() - t0, failed=True)
            raise
        meter.usage.record(seconds=time.time() - t0)
        return out

    # Preserve the signature so `_check_callable` still sees the real arity: it
    # binds `(None,) * n` against `inspect.signature`, and a bare `*args` wrapper
    # would make every arity check pass, including the typo the check exists for.
    try:
        import functools
        call = functools.wraps(fn)(call)  # type: ignore[assignment]
    except (AttributeError, TypeError):   # builtins / C callables
        pass
    return call
