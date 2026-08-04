"""One ceiling, several processes.

`SandboxPool` bounds what a *process* creates. Two runs on one machine each
respect their own limit and together exceed the machine's, which is the same
mistake `max_concurrency` and `eval_concurrency` made before they shared a gate --
one level up.

The coordination point already exists. Every sandbox carries a lease file naming
its owner and when it was last renewed, and `reap` already reads those to decide
what is abandoned. Counting them answers a different question with the same data:
**how many sandboxes are alive on this machine right now**.

So this is not a server. It is the lease directory, read.

**Quota before capacity.** A pool that only enforces a ceiling lets whoever asks
first take everything, and a long run starves a short one indefinitely. Each
holder is guaranteed a floor -- `capacity // holders`, at least one -- and may use
more only while others are under theirs. Without that, sharing a ceiling is worse
than not sharing: the runs interfere and none of them can tell.

**Placement is reuse, seen from the machine.** `SandboxSpec.reuse` already prefers
a warm sandbox within a process. Across processes the same idea is worth more,
because a warm one carries the toolchain another run has already paid to install
-- and `code_runner` pays that per rollout.

What this does *not* do is cross machines. Nothing here assumes one, but nothing
here has been run on two either, and the difference between "the design permits
it" and "it works" is exactly what this file's tests are for.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .metrics import Meter
from .policies import SandboxSpec
from .sandbox import LEASE_FILE, SandboxPool, WorkspaceProvider

__all__ = ["SharedSandboxPool", "live_leases"]


def live_leases(root: str, now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Every lease under ``root`` that somebody is still renewing.

    The same file the reaper reads, asked a different question. Unreadable and
    expired leases are simply not counted: a lease nobody is renewing is not
    holding a sandbox, whatever the directory still contains.
    """
    now = time.time() if now is None else now
    out: List[Dict[str, Any]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name, LEASE_FILE)
        try:
            with open(path, encoding="utf-8") as fh:
                lease = json.load(fh)
        except (OSError, ValueError):
            continue
        ttl = float(lease.get("ttl", 300.0))
        if now - float(lease.get("renewed_at", 0.0)) <= ttl:
            out.append(lease)
    return out


class SharedSandboxPool(SandboxPool):
    """A pool whose ceiling is the machine's, not this process's.

    Admission asks the lease directory how many sandboxes are alive anywhere,
    rather than how many this process has made. Everything else -- release,
    revoke, reuse, the leak counter -- is inherited unchanged, because the
    lifetime rules do not become different rules when a second process appears.
    """

    def __init__(self, provider: Optional[WorkspaceProvider] = None, *,
                 root: str, capacity: int = 8, holders: int = 1,
                 meter: Optional[Meter] = None, poll: float = 0.05) -> None:
        os.makedirs(root, exist_ok=True)
        # The parent's own ceiling is the machine's: a process must never be
        # allowed past it by its local bookkeeping alone.
        super().__init__(provider or WorkspaceProvider(), max_sandboxes=capacity,
                         meter=meter)
        self.root = root
        self.capacity = capacity
        self.holders = max(1, holders)
        self.poll = poll

    # -- admission ------------------------------------------------------------

    @property
    def floor(self) -> int:
        """What this holder is guaranteed regardless of who else is asking."""
        return max(1, self.capacity // self.holders)

    def _machine_state(self) -> Tuple[int, int]:
        """(alive on this machine, alive and ours)."""
        leases = live_leases(self.root)
        mine = os.getpid()
        ours = sum(1 for l in leases
                   if (l.get("owner") or {}).get("pid") == mine)
        return len(leases), ours

    def _may_admit(self) -> bool:
        total, ours = self._machine_state()
        if ours < self.floor:
            # Under our floor: admitted even when the machine is busy, because a
            # guarantee that yields whenever somebody else is greedy is not one.
            return True
        return total < self.capacity

    def acquire(self, spec: SandboxSpec):  # type: ignore[override]
        waited0 = time.time()
        while not self._may_admit():
            time.sleep(self.poll)
        # The spec's workspace goes under the shared root, or the lease it writes
        # is somewhere nobody is counting.
        placed = spec if spec.workspace_root == self.root else _rooted(spec, self.root)
        # Admission waiting happens before `super().acquire`, so it is not in the
        # wait that records itself there. Counted here, into the same total: a
        # reader asking "how long did this run spend waiting for a sandbox" wants
        # one number, and whether the queue was inside this process or on the
        # machine is a different question.
        if self.meter is not None:
            self.meter.add("sandbox_wait_s", time.time() - waited0)
        return super().acquire(placed)

    def stats(self) -> Dict[str, int]:  # type: ignore[override]
        total, ours = self._machine_state()
        return dict(super().stats(), machine_in_use=total, ours=ours,
                    capacity=self.capacity, floor=self.floor)


def _rooted(spec: SandboxSpec, root: str) -> SandboxSpec:
    from dataclasses import replace
    return replace(spec, workspace_root=root)
