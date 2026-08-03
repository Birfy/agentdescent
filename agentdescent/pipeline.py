"""Policies shared by the two barrier-free runtimes.

There are two of them -- :func:`~agentdescent.async_evolve.async_evolve` (general,
and the only one a real workload reaches) and
:class:`~agentdescent.async_runtime.AsyncAgentDescent` (the reference runtime the
parallelism results were measured with). They implement the same shape against
different substrates, and for a while they implemented it *independently*: two
measured fixes lived in whichever pipeline they had been written for, and the
repair was to hand-port them rather than to share them. Hand-porting works once.

So the parts that are **policy** rather than plumbing live here. Nothing in this
module touches a ledger, a thread or a task: it is arithmetic over counters,
which is exactly why it can be shared by two pipelines that agree on nothing
else.

Both loops now ask :class:`WorkerHealth` the retirement question rather than
describing it twice: `evolve` had re-grown its own copy (`any_success` plus a
round counter), which is the arrangement this page was written to record.

What is actually shared, so this page cannot claim more than the code does:

* :class:`WorkerHealth` -- both runtimes call ``should_retire`` /
  ``should_warn`` / ``record_success``, which is the rule that was hand-ported
  and the one that matters. ``backoff`` is used by
  :class:`~agentdescent.async_runtime.AsyncAgentDescent`; ``async_evolve``
  deliberately waits a quarter of it and says so at the call site.
* :class:`StallGuard` -- both runtimes count their no-commit sweeps through it.
  Each decides for itself *which* sweeps count (a sweep with no evidence in it
  is neither progress nor a stall, and the two express "no evidence"
  differently), and hands the counting itself here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["WorkerHealth", "StallGuard"]


@dataclass
class WorkerHealth:
    """When a worker should give up, and when it should keep going.

    The rule is not "retire after N failures", and the difference was measured:
    every worker shares one backend, so shedding workers cannot relieve
    throttling -- it only guarantees the run dies. Against an intermittent
    backend at a 1-in-3 failure rate, a blanket rule retired all three workers in
    22 seconds with nothing learned.

    So a worker retires only while **no** worker has ever completed a rollout.
    That case reads as a misconfiguration -- a wrong key, a dead endpoint -- and
    should fail fast and loudly. Once any worker has succeeded, the backend
    demonstrably works, failures are transient, and the run continues on whatever
    evidence it does gather.
    """

    max_errors: int = 3
    #: Has *any* worker ever completed a rollout? Shared across workers on
    #: purpose: it is the whole distinction between "misconfigured" and "flaky".
    any_success: bool = False
    retired: int = 0

    def record_success(self) -> None:
        self.any_success = True

    def should_retire(self, consecutive_failures: int) -> bool:
        return not self.any_success and consecutive_failures >= self.max_errors

    def should_warn(self, consecutive_failures: int) -> bool:
        """A worker that keeps failing *after* the backend has proved itself.

        It will not retire, so without a line here a run finishes clean at a
        fraction of its concurrency and nothing says so."""
        return self.any_success and consecutive_failures == self.max_errors

    def retire(self) -> None:
        self.retired += 1

    def backoff(self, consecutive_failures: int, cap: float = 5.0) -> float:
        """Seconds to wait before this worker's next attempt."""
        return min(2.0 ** max(0, consecutive_failures), cap)


@dataclass
class StallGuard:
    """Backpressure: force a resync when evidence arrives and nothing commits.

    A lag budget alone livelocks when it is wider than the staleness tolerance:
    workers propose against a snapshot too old for the policy to accept, every
    card is discarded, the head never moves -- and because the head never moves,
    the version-based budget never triggers a refresh either. Counting sweeps
    that produced no commit breaks the cycle.

    It also covers **cold start**, where the head has not advanced yet and a
    version-only budget cannot engage at all.
    """

    patience: int = 50
    _idle: int = field(default=0, repr=False)
    forced: int = 0

    def note_sweep(self, committed: bool) -> None:
        self._idle = 0 if committed else self._idle + 1

    def should_force_refresh(self) -> bool:
        return self._idle >= self.patience

    def force(self) -> None:
        """A worker resynced because of the stall; count it and reset."""
        self.forced += 1
        self._idle = 0
