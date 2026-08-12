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

**Three counters are wall-clock on one thread, not sums.** ``merge_seconds`` and
``merge_gate_seconds`` are measured on the single merger (or, synchronously, the
single driver), so they are directly comparable against ``elapsed_s`` -- and
``merge_gate_seconds`` is a *subset* of ``merge_seconds``, not a term to add to
it. ``worker_starved_seconds`` is a sum across workers like
``rollout_seconds``. Together they answer the one question none of the others
can: **did the gate stop anyone from working?** ``eval_seconds`` cannot, because
it sums across the evaluation pool -- eight threads scoring for one second each
is 8.0 there whether the run had eight seconds of gate or one.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

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
    merge_gate_seconds: float = 0.0
    worker_starved_seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_inflight_joins: int = 0
    stale_considered: int = 0
    stale_discarded: int = 0
    redispatched: int = 0
    duplicates_dropped: int = 0
    cas_conflicts: int = 0
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
    "merge_gate_seconds", "worker_starved_seconds",
    "cache_hits", "cache_misses", "cache_inflight_joins",
    "stale_considered", "stale_discarded",
    "redispatched", "duplicates_dropped", "cas_conflicts",
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

    #: Wall-clock the merger was **busy**, on the one thread that merges: the
    #: async merger's sweeps, or the synchronous driver's post-barrier work. Read
    #: against `elapsed_s` it is merger occupancy, which is the quantity a second
    #: merger (or an evaluation stage of its own) would relieve.
    #:
    #: Declared since the first version of this module and written by nobody
    #: until now, so every published run reported `0.0` -- indistinguishable from
    #: "merging was free", which is the opposite of what the profile says.
    merge_seconds: float = 0.0
    #: The part of `merge_seconds` spent blocked on **evaluation**: the staleness
    #: filter's re-verification, the acceptance gate, and the round's held-out
    #: measurement. A **subset** of `merge_seconds`, never a term to add to it.
    #:
    #: The ratio is the whole question: at `merge_gate_seconds / merge_seconds`
    #: near 1 the merger is an evaluation stage wearing a merger's name, and
    #: moving evaluation off it is worth doing. Near 0 it is not, whatever the
    #: profile of somebody else's implementation says.
    merge_gate_seconds: float = 0.0
    #: Summed across workers, like `rollout_seconds`: time a worker spent held at
    #: the barrier-free path's backpressure gate, unable to start a rollout
    #: because more than `async_ratio` cards were already waiting to be merged.
    #:
    #: This is the number that says whether a busy merger **cost** anything. A
    #: merger at 90% occupancy with no starved workers has hidden itself
    #: perfectly behind the rollouts; the same merger with workers idling on it
    #: is the serial-stage bottleneck. Zero on the synchronous path by
    #: construction -- there the round barrier idles every worker for exactly
    #: `merge_seconds`, and a counter would only restate it.
    worker_starved_seconds: float = 0.0

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

    #: Task-level recovery. `redispatched` counts tasks sent out again after
    #: their worker was presumed lost; `duplicates_dropped` counts results that
    #: arrived for a task already answered -- a worker that was only *presumed*
    #: dead, finishing after its work had been given to somebody else.
    #:
    #: The second is the one worth watching. Dropping the duplicate is correct
    #: and invisible, so without a counter a re-dispatch policy that is
    #: needlessly aggressive looks exactly like one that is well tuned.
    redispatched: int = 0
    duplicates_dropped: int = 0
    #: Commits that lost a compare-and-swap race and had to rebase. Zero with one
    #: writer, by construction; non-zero is how much contention a multi-writer
    #: configuration is actually producing.
    cas_conflicts: int = 0

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

    @contextmanager
    def timed(self, counter: str) -> Iterator[None]:
        """Add the wall-clock of a block to ``counter``, exception or not.

        The alternative is `t0 = time.time()` / `try` / `finally` at every site,
        and the phase counters have enough sites -- three inside one merger sweep
        -- that the version which forgets the `finally` gets written eventually.
        A sweep that raises still spent its time, and a phase counter that
        silently omits the failures reads lowest on exactly the runs worth
        looking at.
        """
        if counter not in _COUNTERS:      # fail here, not after the block has run
            raise KeyError(
                f"unknown meter counter {counter!r}; known: {', '.join(sorted(_COUNTERS))}")
        t0 = time.time()
        try:
            yield
        finally:
            self.add(counter, time.time() - t0)

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
                merge_gate_seconds=self.merge_gate_seconds,
                worker_starved_seconds=self.worker_starved_seconds,
                calls=u.calls,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                model_seconds=u.seconds,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                cache_inflight_joins=self.cache_inflight_joins,
                stale_considered=self.stale_considered,
                stale_discarded=self.stale_discarded,
                redispatched=self.redispatched,
                duplicates_dropped=self.duplicates_dropped,
                cas_conflicts=self.cas_conflicts,
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
        if s.merge_seconds:
            out.append(f"merger {s.merge_seconds:.1f}s "
                       f"({s.merge_gate_seconds / s.merge_seconds:.0%} gate)")
        if s.worker_starved_seconds:
            out.append(f"{s.worker_starved_seconds:.1f}s starved")
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
