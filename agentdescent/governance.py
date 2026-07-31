"""Layered governance by blast radius (design doc, section 6).

Every Evolvable is *automatically* sorted into one of three layers by its
``blast_radius``.  The layer decides the aggregation protocol and update
frequency (a two-timescale system):

    L2 (fast)   local skills / prompts / few-shot   hours   full async merge
    L1 (slow)   harness / context / learned verifier days    serial + staged rollout
    L0 (frozen) oracle / audit budget / merge perms  human   read-only to the loop

The L1/L2 boundary is *not* a hand-labelled taxonomy: a skill triggered by every
task is pulled into the slow layer automatically by its blast radius, while a
harness patch that only touches one task cluster may ride the fast layer (design
doc, section 3.2).

**L0 is the exception, and deliberately so.** Nothing about a blast radius can
tell you that an artifact is the oracle -- an artifact that can rewrite the thing
that judges it is a structural fact, not a measured one -- so the frozen set is an
explicit list of ids (:data:`FROZEN_IDS`). That is a hand-labelled taxonomy, and
it should be: a verifier that learns to pass itself is exactly what an *estimated*
layer would fail to catch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

from .evolvable import Evolvable


class Layer(IntEnum):
    L2_FAST = 2
    L1_SLOW = 1
    L0_FROZEN = 0


#: The one threshold on blast_radius (a normalized [0, 1] estimate of task-surface
#: impact): at or below it an artifact is L2, above it L1. There used to be a
#: second constant, ``SLOW_MAX = 0.85``, with a comment describing a frozen-layer
#: rule -- ``classify`` never read it, so 0.31 and 0.99 classified identically and
#: the comment documented behaviour that did not exist. L0 is reached by id, not
#: by radius, so one threshold is all there is.
FAST_MAX = 0.30

#: Artifacts the loop may read but never mutate. An id list rather than a radius,
#: because "this is the oracle" is structural (see the module docstring). Names
#: here are reserved: ``evolve(artifact_id="oracle")`` is refused up front.
FROZEN_IDS = frozenset(
    {
        "oracle",
        "audit_budget",
        "merge_permissions",
        "safety_constraints",
    }
)


def classify(artifact: Evolvable) -> Layer:
    """Assign an artifact to a governance layer.

    This is the single definition of the L1/L2 boundary. The aggregator and the
    audit scheduler used to re-derive it from raw floats with a *different*
    threshold (``blast_radius > 0.5``), so an artifact at 0.4 was L1 by governance
    and treated as L2 everywhere it mattered: it got the cold-artifact staleness
    tolerance and no oracle audit at all."""
    if artifact.id in FROZEN_IDS:
        return Layer.L0_FROZEN
    if artifact.blast_radius <= FAST_MAX:
        return Layer.L2_FAST
    return Layer.L1_SLOW


@dataclass
class L1SerialGate:
    """Enforces "at most one L1 diff in evaluation at a time" (section 6).

    L2 artifacts are naturally isolated -- only tasks that trigger them are
    affected -- so they merge fully async.  L1 artifacts have no such isolation,
    so their in-flight changes must be serialized in time to keep attribution
    tractable.

    **Not wired into the shipped runtimes, because they already satisfy it.**
    Every merge decision runs on one thread -- the round barrier in ``evolve()``,
    the single merger thread in ``async_evolve`` and ``AsyncAgentDescent`` -- so at
    most one diff of any layer is ever in evaluation. The guarantee holds by
    construction; this gate is what would enforce it once merges run concurrently
    (a process or host pool), and it is tested in isolation for that day. Treat it
    as a primitive, not as something currently in the path."""

    _in_flight: Dict[str, str] = None  # artifact_id -> diff_id
    #: try_acquire is a check-then-act, so calling it from several threads could
    #: hand the "global L1 lock" to two of them at once -- exactly what this gate
    #: exists to prevent. Guard it for real.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self._in_flight is None:
            self._in_flight = {}

    def try_acquire(self, artifact_id: str, diff_id: str) -> bool:
        with self._lock:
            if self._in_flight:
                # a global L1 lock: at most one L1 diff evaluated anywhere.
                return False
            self._in_flight[artifact_id] = diff_id
            return True

    def release(self, artifact_id: str) -> None:
        with self._lock:
            self._in_flight.pop(artifact_id, None)

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._in_flight)


class GovernanceError(Exception):
    """Raised when the evolution loop tries to mutate a frozen (L0) artifact."""


def assert_mutable(artifact: Evolvable) -> None:
    """Guard invoked before applying any diff (design doc, section 6, L0)."""
    if classify(artifact) is Layer.L0_FROZEN:
        raise GovernanceError(
            f"{artifact.id} is L0-frozen; the evolution loop may only read it"
        )
