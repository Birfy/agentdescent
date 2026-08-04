"""One ceiling, several processes — with several processes.

`SandboxPool` bounds what a *process* creates. Two runs on one machine each
respect their own limit and together exceed the machine's: the same mistake
`max_concurrency` and `eval_concurrency` made before they shared a gate, one
level up.

Every assertion here uses real processes. A threading version would pass against
a pool that counts only its own sandboxes, which is precisely the thing being
fixed.
"""

import json
import multiprocessing as mp
import os
import time

import pytest

from agentdescent.policies import SandboxSpec
from agentdescent.sandbox import LEASE_FILE, SandboxPool, WorkspaceProvider
from agentdescent.sandbox_shared import SharedSandboxPool, live_leases

CTX = mp.get_context("spawn")


def _hold(root, capacity, holders, count, hold_for, out):
    """One process taking `count` sandboxes in turn, reporting what it saw."""
    pool = SharedSandboxPool(root=root, capacity=capacity, holders=holders,
                             poll=0.01)
    peak = 0
    try:
        for _ in range(count):
            with pool.lease(SandboxSpec()):
                peak = max(peak, len(live_leases(root)))
                time.sleep(hold_for)
    finally:
        pool.close()
    out.put(peak)


def _hold_open(root, capacity, holders, seconds, started, out):
    """Take one and keep it, so another process has to contend for the rest."""
    pool = SharedSandboxPool(root=root, capacity=capacity, holders=holders,
                             poll=0.01)
    try:
        with pool.lease(SandboxSpec()):
            started.set()
            time.sleep(seconds)
        out.put("released")
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# The ceiling is the machine's
# ---------------------------------------------------------------------------


def test_two_processes_share_one_ceiling(tmp_path):
    """The failure a per-process pool cannot see.

    Each process obeys its own limit; together they exceed the machine's. Here
    the limit is read from the lease directory, so both are counting the same
    thing."""
    root = str(tmp_path / "pool")
    out = CTX.Queue()
    procs = [CTX.Process(target=_hold, args=(root, 2, 2, 6, 0.03, out))
             for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0

    peaks = [out.get() for _ in procs]
    assert max(peaks) <= 2, (
        f"{max(peaks)} sandboxes were alive at once under a ceiling of 2")


def test_a_per_process_pool_would_not_have_caught_it(tmp_path):
    """The control. Two ordinary pools, each capped at 2, reach 4 together.

    Without this the test above proves only that a number was respected, not
    that the number means what it should."""
    root = str(tmp_path / "unshared")
    os.makedirs(root, exist_ok=True)
    pools = [SandboxPool(WorkspaceProvider(), max_sandboxes=2) for _ in range(2)]
    held = []
    try:
        for pool in pools:
            for _ in range(2):
                held.append((pool, pool.acquire(SandboxSpec(workspace_root=root))))
        assert len(held) == 4, "two pools of two did not reach four"
    finally:
        for pool, sandbox in held:
            pool.release(sandbox)
        for pool in pools:
            pool.close()


# ---------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------


def test_a_holder_is_not_starved_by_a_greedy_neighbour(tmp_path):
    """A ceiling alone lets whoever asks first take everything.

    The greedy process holds its sandbox for the whole test; the other must
    still get its floor rather than waiting behind it."""
    root = str(tmp_path / "pool")
    started, out = CTX.Event(), CTX.Queue()
    greedy = CTX.Process(target=_hold_open, args=(root, 2, 2, 1.5, started, out))
    greedy.start()
    try:
        assert started.wait(timeout=60), "the greedy holder never started"
        pool = SharedSandboxPool(root=root, capacity=2, holders=2, poll=0.01)
        try:
            t0 = time.time()
            with pool.lease(SandboxSpec()):
                waited = time.time() - t0
            assert waited < 1.0, (
                f"waited {waited:.2f}s behind a neighbour while under our floor")
        finally:
            pool.close()
    finally:
        greedy.join(timeout=60)


def test_the_floor_is_at_least_one(tmp_path):
    """More holders than capacity still has to admit somebody, or nothing runs."""
    pool = SharedSandboxPool(root=str(tmp_path / "p"), capacity=2, holders=8)
    try:
        assert pool.floor == 1
        with pool.lease(SandboxSpec()):
            pass
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Eviction: a holder that dies
# ---------------------------------------------------------------------------


def _die_holding(root, capacity, started):
    pool = SharedSandboxPool(root=root, capacity=capacity, holders=1, poll=0.01)
    pool.acquire(SandboxSpec())        # taken, never released
    started.set()
    time.sleep(60)


def test_a_dead_holders_capacity_comes_back(tmp_path):
    """A process killed mid-run leaves a lease nobody renews. The machine must
    not lose that slot permanently."""
    root = str(tmp_path / "pool")
    started = CTX.Event()
    victim = CTX.Process(target=_die_holding, args=(root, 1, started))
    victim.start()
    try:
        assert started.wait(timeout=60)
        assert len(live_leases(root)) == 1
        victim.kill()
        victim.join(timeout=30)

        # The lease is still on disk and still counted -- it has not expired yet.
        assert len(live_leases(root)) == 1

        # Expire it the way time would, and the slot is free again.
        for name in os.listdir(root):
            path = os.path.join(root, name, LEASE_FILE)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                lease = json.load(fh)
            lease["renewed_at"] = time.time() - 10 * float(lease.get("ttl", 300.0))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(lease, fh)
        assert live_leases(root) == []

        pool = SharedSandboxPool(root=root, capacity=1, holders=1, poll=0.01)
        try:
            with pool.lease(SandboxSpec()):
                pass                    # admitted: the dead holder's slot returned
        finally:
            pool.close()
    finally:
        if victim.is_alive():
            victim.kill()


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_an_unreadable_lease_is_not_counted(tmp_path):
    """A half-written or corrupt lease is not holding a sandbox, whatever the
    directory still contains. Counting it would leak capacity forever."""
    root = str(tmp_path / "pool")
    os.makedirs(os.path.join(root, "ws-broken"), exist_ok=True)
    with open(os.path.join(root, "ws-broken", LEASE_FILE), "w") as fh:
        fh.write("{not json")
    assert live_leases(root) == []


def test_stats_report_the_machine_not_just_this_process(tmp_path):
    root = str(tmp_path / "pool")
    pool = SharedSandboxPool(root=root, capacity=4, holders=2)
    try:
        with pool.lease(SandboxSpec()):
            s = pool.stats()
            assert s["machine_in_use"] == 1 and s["ours"] == 1
            assert s["capacity"] == 4 and s["floor"] == 2
    finally:
        pool.close()
