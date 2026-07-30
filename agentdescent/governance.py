"""Layered governance by blast radius (design doc, section 6).

Every Evolvable is *automatically* sorted into one of three layers by its
``blast_radius``.  The layer decides the aggregation protocol and update
frequency (a two-timescale system):

    L2 (fast)   local skills / prompts / few-shot   hours   full async merge
    L1 (slow)   harness / context / learned verifier days    serial + staged rollout
    L0 (frozen) oracle / audit budget / merge perms  human   read-only to the loop

The key design point is that this is *not* a hand-labelled artifact taxonomy: a
skill triggered by every task is pulled into the slow layer automatically, while
a harness patch that only touches one task cluster may ride the fast layer
(design doc, section 3.2).
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


# Thresholds on blast_radius (a normalized [0, 1] estimate of task-surface impact).
FAST_MAX = 0.30
SLOW_MAX = 0.85
# Above SLOW_MAX an artifact is a candidate for the frozen layer *only* if it is
# also declared structural (see ``FROZEN_IDS``); otherwise it stays L1.
FROZEN_IDS = frozenset(
    {
        "oracle",
        "audit_budget",
        "merge_permissions",
        "safety_constraints",
    }
)


def classify(artifact: Evolvable) -> Layer:
    """Assign an artifact to a governance layer from its blast radius."""
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
    tractable."""

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
