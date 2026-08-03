"""Sandboxes as a managed resource: bounded, owned, and reclaimable.

A rollout that evolves a *directory* needs that directory to exist on disk, and
`runners.py` has always made one per call. What it has not had is an owner. The
workspace is created inside the `run` closure and deleted by that closure's
`finally`, which means:

* nothing bounds how many exist at once -- `max_concurrency` worker threads and
  `eval_concurrency` scoring threads each make their own, and neither knows about
  the other, so the real ceiling is their sum;
* the `finally` does not run when the thread is abandoned (a round timeout leaves
  the straggler running, and Python cannot cancel a thread) or when the process
  is killed;
* what *does* clean up after a killed process is a sweep that deletes anything in
  `$TMPDIR` older than six hours -- and "old enough" is not "unused".

This module makes the lifetime explicit. One pool, one ceiling, one release path,
and a lease file so a directory can say who owns it.

**Why leases rather than reference counts.** The interesting failures are the
ones where the owner is already gone: an abandoned thread whose `finally` will
never run, a process that took a SIGKILL. A counter kept in the dead process
cannot help. A lease written *into the sandbox* can: the owner renews it while it
is working, and any other process can later read it, see that nobody has renewed
it, and clean up.

**Why the default is still a fresh workspace per rollout.** `code_runner`'s
frozen-file overlay assumes a clean directory: materialise the candidate, write
the pristine files back over it, then run the tests. A workspace carrying the
previous candidate's writes breaks that guarantee, and the symptom is not a
crash -- it is a score that looks entirely ordinary and is wrong. Reuse has to be
asked for.

The only provider here is the local-directory one, which reproduces today's
behaviour exactly. Process-per-worker and remote/container providers implement
the same `SandboxProvider` protocol and land with the work that needs them; there
are no stubs for them here, because a provider that exists but does not isolate
is worse than one that does not exist.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from .metrics import Meter
from .policies import SandboxSpec

__all__ = [
    "LEASE_FILE",
    "LocalWorkspaceSandbox",
    "SandboxLeak",
    "SandboxPool",
    "WorkspaceProvider",
]

#: Written into every sandbox root. Named so a human who finds one in `$TMPDIR`
#: can tell what made it and whether anything still owns it.
LEASE_FILE = ".agentdescent-lease.json"

_WS_PREFIX = "agentdescent-ws-"

#: This process, for lease ownership. The token distinguishes two runs that
#: happen to get the same pid after a restart -- without it, a stale lease from a
#: recycled pid looks alive forever.
_HOST = socket.gethostname()
_TOKEN = uuid.uuid4().hex[:12]


class SandboxLeak(RuntimeError):
    """A sandbox was never returned. Only raised by tests asking for strictness."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


@dataclass
class LocalWorkspaceSandbox:
    """A throwaway directory on this machine -- what a rollout has always got.

    Isolation here is a trimmed environment, `HOME` and `TMPDIR` pointing inside
    the workspace, and a hard timeout. It is **not** a security boundary: the
    code still runs as your user, with your network. That is unchanged by this
    module, and saying otherwise because there is now a file called `sandbox.py`
    would be the worst thing this refactor could do.
    """

    id: str
    root: str
    fingerprint: str
    lease_id: str = ""
    acquired_at: float = 0.0


class WorkspaceProvider:
    """Provisions :class:`LocalWorkspaceSandbox` -- `mkdtemp`, plus a lease file."""

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self._n = 0
        self._lock = threading.Lock()

    def acquire(self, spec: SandboxSpec) -> LocalWorkspaceSandbox:
        root = tempfile.mkdtemp(prefix=_WS_PREFIX, dir=spec.workspace_root)
        with self._lock:
            self._n += 1
            n = self._n
        sb = LocalWorkspaceSandbox(
            id=f"{_HOST}-{os.getpid()}-{n}", root=root,
            fingerprint=spec.fingerprint(), lease_id=uuid.uuid4().hex,
            acquired_at=time.time())
        self._write_lease(sb)
        return sb

    def release(self, sandbox: LocalWorkspaceSandbox) -> None:
        shutil.rmtree(sandbox.root, ignore_errors=True)

    def reset(self, sandbox: LocalWorkspaceSandbox) -> None:
        """Return a reused sandbox to an empty state.

        Everything except the lease goes. Skipping this is how a reused workspace
        leaks one candidate's files into the next one's score."""
        for name in os.listdir(sandbox.root):
            if name == LEASE_FILE:
                continue
            path = os.path.join(sandbox.root, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # -- leases ---------------------------------------------------------------

    def _write_lease(self, sandbox: LocalWorkspaceSandbox) -> None:
        payload = {
            "lease_id": sandbox.lease_id,
            "owner": {"host": _HOST, "pid": os.getpid(), "token": _TOKEN},
            "ttl": self.ttl,
            "renewed_at": time.time(),
            "fingerprint": sandbox.fingerprint,
        }
        try:
            with open(os.path.join(sandbox.root, LEASE_FILE), "w",
                      encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError:
            pass          # a lease we cannot write is a reclaim we cannot do later

    def renew(self, sandbox: LocalWorkspaceSandbox) -> None:
        self._write_lease(sandbox)

    def reap(self, older_than: Optional[float] = None,
             root: Optional[str] = None) -> int:
        """Delete workspaces nobody is renewing any more.

        Replaces "older than six hours" as the test for abandonment, in both
        directions. Age never meant unused: a directory's mtime does not change
        while a process works *inside* it, so a long rollout looked abandoned,
        while on a shared `$TMPDIR` a live run's workspace looked collectable to
        every other run on the machine. Renewal is a statement by the owner, and
        its absence is the thing worth acting on.

        A live owner is given the benefit of the doubt once: between one and two
        TTLs, an expired lease whose pid is still running is left alone (its
        renewal may simply be late). Past that it goes regardless -- otherwise a
        recycled pid would keep a dead run's workspace alive forever.

        Best-effort and silent, like the sweep it replaces: reclaiming disk must
        never be able to fail a run.
        """
        ttl = self.ttl if older_than is None else older_than
        base = root or tempfile.gettempdir()
        now, removed = time.time(), 0
        try:
            names = os.listdir(base)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(_WS_PREFIX):
                continue
            path = os.path.join(base, name)
            try:
                if not self._collectable(path, ttl, now):
                    continue
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            except OSError:
                continue
        return removed

    def _collectable(self, path: str, ttl: float, now: float) -> bool:
        try:
            with open(os.path.join(path, LEASE_FILE), encoding="utf-8") as fh:
                lease = json.load(fh)
        except (OSError, ValueError):
            # No readable lease: either a workspace from before this existed, or
            # one whose owner died mid-write. Fall back to age, which is what the
            # old sweep did, and be generous about it.
            try:
                return now - os.path.getmtime(path) > max(ttl, 3600.0) * 2
            except OSError:
                return False
        age = now - float(lease.get("renewed_at", 0.0))
        lease_ttl = float(lease.get("ttl", ttl))
        if age <= lease_ttl:
            return False                          # somebody is still renewing it
        owner = lease.get("owner") or {}
        if age <= lease_ttl * 2 and owner.get("host") == _HOST:
            pid = owner.get("pid")
            if isinstance(pid, int) and _pid_alive(pid):
                return False                      # a late renewal, probably
        return True


@dataclass
class _Lease:
    sandbox: LocalWorkspaceSandbox
    spec: SandboxSpec


class SandboxPool:
    """The single gate on how many sandboxes exist at once.

    Rollouts and evaluations both come here. They are configured separately --
    `max_concurrency` and `eval_concurrency` mean different things -- but they
    consume the same machine, so the ceiling is one number and not their sum.

    `acquire` blocks when the pool is full. Failing instead would push retry
    logic into every caller, and growing instead would make the ceiling a
    suggestion.
    """

    def __init__(self, provider: Optional[WorkspaceProvider] = None, *,
                 max_sandboxes: int = 8, meter: Optional[Meter] = None) -> None:
        if max_sandboxes < 1:
            raise ValueError(f"max_sandboxes must be >= 1, got {max_sandboxes}")
        self.provider = provider or WorkspaceProvider()
        self.max_sandboxes = max_sandboxes
        self.meter = meter
        self._cond = threading.Condition()
        self._live: Dict[str, _Lease] = {}         # lease_id -> lease
        self._idle: Dict[str, LocalWorkspaceSandbox] = {}   # fingerprint -> warm
        self._counts = {"created": 0, "reused": 0, "failures": 0,
                        "reclaimed": 0, "leaked": 0, "waiters": 0}

    # -- the one way in and out -----------------------------------------------

    @contextmanager
    def lease(self, spec: Optional[SandboxSpec] = None) -> Iterator[LocalWorkspaceSandbox]:
        """Take a sandbox, use it, give it back. Every path gives it back.

        Normal return, exception, timeout, cancellation, and a straggler thread
        that nobody is waiting for any more -- the last of those is why this is a
        context manager around a `finally` rather than an acquire/release pair
        that callers are trusted to match."""
        sandbox = self.acquire(spec or SandboxSpec())
        try:
            yield sandbox
        finally:
            self.release(sandbox)

    def acquire(self, spec: SandboxSpec) -> LocalWorkspaceSandbox:
        t0 = time.time()
        with self._cond:
            self._counts["waiters"] += 1
            try:
                while len(self._live) >= self.max_sandboxes:
                    self._cond.wait()
            finally:
                self._counts["waiters"] -= 1
            warm = self._idle.pop(spec.fingerprint(), None) if spec.reuse else None
            placeholder = uuid.uuid4().hex
            self._live[placeholder] = None        # hold the slot while provisioning

        waited = time.time() - t0
        setup0 = time.time()
        try:
            if warm is not None:
                self.provider.reset(warm)
                self.provider.renew(warm)
                sandbox = warm
                reused = True
            else:
                sandbox = self.provider.acquire(spec)
                reused = False
        except Exception:
            with self._cond:
                self._live.pop(placeholder, None)
                self._counts["failures"] += 1
                self._cond.notify()
            # Provisioning failure is not model failure. Conflating them makes an
            # image pull or a full disk look like a broken backend, and the
            # retirement rule fires hardest before any worker has succeeded --
            # which is exactly when provisioning fails.
            if self.meter is not None:
                self.meter.add("sandbox_failures")
            raise

        with self._cond:
            self._live.pop(placeholder, None)
            self._live[sandbox.lease_id] = _Lease(sandbox, spec)
            self._counts["reused" if reused else "created"] += 1

        if self.meter is not None:
            self.meter.add("sandbox_wait_s", waited)
            self.meter.add("sandbox_setup_s", time.time() - setup0)
            self.meter.add("sandboxes_reused" if reused else "sandboxes_created")
        return sandbox

    def release(self, sandbox: LocalWorkspaceSandbox) -> None:
        """Give a sandbox back. Idempotent: releasing twice is not an error."""
        with self._cond:
            lease = self._live.pop(sandbox.lease_id, None)
            if lease is None:
                return
            keep = lease.spec.reuse
            if keep:
                self._idle[sandbox.fingerprint] = sandbox
            self._cond.notify()
        if not keep:
            self.provider.release(sandbox)

    def revoke(self, lease_id: str) -> bool:
        """Take a sandbox back from an owner that is not going to return it.

        For the straggler abandoned at a round boundary and, later, the worker
        that stopped answering: the thread is still running and its `finally`
        may never execute, so somebody else has to free the slot."""
        with self._cond:
            lease = self._live.pop(lease_id, None)
            if lease is None:
                return False
            self._counts["reclaimed"] += 1
            self._cond.notify()
        self.provider.release(lease.sandbox)
        return True

    # -- observation ----------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return dict(self._counts, in_use=len(self._live), idle=len(self._idle))

    def close(self, strict: bool = False) -> None:
        """Release everything still out. ``strict`` turns a leak into an error.

        Tests use ``strict``; a real run does not, because a leaked sandbox is a
        reason to reclaim disk, not a reason to lose the artifact."""
        with self._cond:
            outstanding = list(self._live.values())
            self._live.clear()
            idle = list(self._idle.values())
            self._idle.clear()
            self._counts["leaked"] += len(outstanding)
            self._cond.notify_all()
        for lease in outstanding:
            self.provider.release(lease.sandbox)
        for sandbox in idle:
            self.provider.release(sandbox)
        if strict and outstanding:
            raise SandboxLeak(
                f"{len(outstanding)} sandbox(es) were never released: "
                + ", ".join(l.sandbox.id for l in outstanding[:5]))
