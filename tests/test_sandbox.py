"""A sandbox is a resource, so it needs an owner, a bound and a way to be reclaimed.

Today a workspace is created inside the `run` closure, deleted by that closure's
`finally`, and reclaimed -- if the process died before the `finally` ran -- by a
sweep over `$TMPDIR` that deletes anything old enough. Each of those three has a
hole, and the first two are reachable on a single machine:

* the `finally` does not run when the thread is abandoned (a round timeout) or
  the process is killed;
* `subprocess.run(timeout=)` kills the process it started and not the ones *that*
  process started, so a timed-out `pytest` or `pip install` leaves children
  behind -- still running, still holding a workspace that has been deleted;
* "old enough" is not "unused": on a shared `$TMPDIR` it can delete a directory
  another live run is still using.

The first test in this file fails against the code as it was written; it is here
first on purpose.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from agentdescent.evolution import Task
from agentdescent.filetree import canonical
from agentdescent.runners import TEST_FAILURE_MARKER, code_runner


def _alive(pid: int) -> bool:
    """Is this pid still running? (Signal 0 checks without delivering.)"""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _reap(pid: int) -> None:
    for sig in (15, 9):
        try:
            os.kill(pid, sig)
        except OSError:
            return
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# The bug that exists today
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_a_timed_out_gate_leaves_no_grandchildren(tmp_path):
    """Killing the child is not enough: the gate's own children outlive it.

    `test_cmd` here starts a detached sleeper and then blocks past the timeout.
    `subprocess.run(timeout=)` kills the shell, reaps it, and returns -- while the
    sleeper keeps running, holding a workspace that has just been deleted. On a
    real gate that sleeper is `pytest` or `pip`, and eight workers leave eight of
    them per timed-out round.
    """
    pidfile = tmp_path / "sleeper.pid"
    spawner = textwrap.dedent(f"""
        import os, subprocess, sys, time
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pidfile)!r}, "w").write(str(p.pid))
        time.sleep(120)          # outlive the timeout so the gate is killed
    """)

    run = code_runner([sys.executable, "main.py"],
                      test_cmd=[sys.executable, "-c", spawner],
                      timeout=2.0)
    out = run(canonical({"main.py": "print('x')"}), Task("t", "hi"))
    assert out.startswith(TEST_FAILURE_MARKER) and "timed out" in out

    deadline = time.time() + 5.0
    while not pidfile.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pidfile.exists(), "the gate never started its child"
    pid = int(pidfile.read_text())

    try:
        # give the kill a moment to propagate through the process group
        for _ in range(40):
            if not _alive(pid):
                break
            time.sleep(0.05)
        assert not _alive(pid), (
            f"pid {pid} outlived the gate that started it: killing the direct "
            "child leaves its own children running, holding a workspace that "
            "has already been deleted")
    finally:
        _reap(pid)


# ---------------------------------------------------------------------------
# The pool: bounded, released on every path, observable
# ---------------------------------------------------------------------------

from agentdescent.policies import SandboxSpec                       # noqa: E402
from agentdescent.metrics import Meter                              # noqa: E402
from agentdescent.sandbox import (                                  # noqa: E402
    LEASE_FILE, SandboxLeak, SandboxPool, WorkspaceProvider,
)


def test_a_sandbox_is_released_on_every_exit_path(tmp_path):
    """Normal return, exception, and an inner `return` all have to give it back."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2)
    roots = []

    with pool.lease() as sb:
        roots.append(sb.root)
    with pytest.raises(ValueError):
        with pool.lease() as sb:
            roots.append(sb.root)
            raise ValueError("boom")

    assert pool.stats()["in_use"] == 0
    assert all(not os.path.exists(r) for r in roots)
    pool.close(strict=True)


def test_the_pool_blocks_rather_than_failing_or_growing(tmp_path):
    """16 rollouts through a pool of 2: never more than 2 at once, all 16 finish.

    The alternatives are worse in both directions -- failing pushes retry logic
    into every caller, growing makes the ceiling a suggestion."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2)
    peak, live, lock = [0], [0], threading.Lock()
    done = []

    def worker():
        with pool.lease() as sb:
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.02)
            with lock:
                live[0] -= 1
            done.append(sb.id)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(done) == 16, "the pool dropped work instead of queueing it"
    assert peak[0] <= 2, f"{peak[0]} sandboxes existed at once, ceiling was 2"
    pool.close(strict=True)


def test_one_ceiling_covers_both_kinds_of_work():
    """Rollouts and evaluations are configured separately and share a machine.

    Two independently-bounded pools would put the real limit at their sum, which
    is what happens today: `max_concurrency` and `eval_concurrency` do not know
    about each other."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=3)
    peak, live, lock = [0], [0], threading.Lock()

    def work():
        with pool.lease():
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.02)
            with lock:
                live[0] -= 1

    rollouts = [threading.Thread(target=work) for _ in range(6)]
    evals = [threading.Thread(target=work) for _ in range(6)]
    for t in rollouts + evals:
        t.start()
    for t in rollouts + evals:
        t.join(timeout=30)
    assert peak[0] <= 3
    pool.close(strict=True)


def test_a_pool_of_one_does_not_deadlock():
    """Degrading to serial is fine; deadlocking is the likely failure here."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=1)
    done = []

    def work():
        with pool.lease():
            time.sleep(0.01)
            done.append(1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert len(done) == 4, "a pool of one serialised into a deadlock"
    pool.close(strict=True)


def test_workspaces_cannot_see_each_other():
    """"One workspace per call" as a concurrent assertion: 32 rollouts each write
    the same filename and none of them observes another's."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=8)
    bad = []

    def worker(i):
        with pool.lease() as sb:
            path = os.path.join(sb.root, "answer.txt")
            with open(path, "w") as fh:
                fh.write(str(i))
            time.sleep(0.01)
            with open(path) as fh:
                if fh.read() != str(i):
                    bad.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not bad, f"{len(bad)} rollouts saw another's file"
    pool.close(strict=True)


def test_revoke_takes_a_sandbox_back_from_an_owner_that_will_not_return_it():
    """The abandoned straggler: its thread is still running and its `finally`
    may never execute, so the slot has to be freeable from outside."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=1)
    sb = pool.acquire(SandboxSpec())
    assert pool.stats()["in_use"] == 1

    assert pool.revoke(sb.lease_id) is True
    assert pool.stats()["in_use"] == 0
    assert pool.stats()["reclaimed"] == 1
    assert not os.path.exists(sb.root)
    assert pool.revoke(sb.lease_id) is False        # idempotent
    # the slot is genuinely free again
    with pool.lease():
        pass
    pool.close(strict=True)


def test_reuse_is_opt_in_and_cleans_up_between_uses():
    """A workspace carrying the last candidate's files produces a score that
    looks ordinary and is wrong, so reuse has to reset."""
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=1)
    spec = SandboxSpec(reuse=True)

    with pool.lease(spec) as first:
        with open(os.path.join(first.root, "leftover.txt"), "w") as fh:
            fh.write("previous candidate")
        root = first.root

    with pool.lease(spec) as second:
        assert second.root == root, "reuse=True should hand back the warm one"
        assert not os.path.exists(os.path.join(second.root, "leftover.txt"))
    assert pool.stats()["reused"] == 1
    pool.close(strict=True)


def test_a_fresh_sandbox_is_the_default():
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2)
    with pool.lease() as a:
        first = a.root
    with pool.lease() as b:
        assert b.root != first
    assert pool.stats()["reused"] == 0
    pool.close(strict=True)


def test_provisioning_failure_is_counted_apart_from_model_failure():
    """An image that will not pull is not a backend that does not work, and the
    retirement rule fires hardest before any worker has succeeded -- which is
    exactly when provisioning fails."""
    class Broken(WorkspaceProvider):
        def acquire(self, spec):
            raise OSError("no space left on device")

    meter = Meter()
    pool = SandboxPool(Broken(), max_sandboxes=2, meter=meter)
    for _ in range(3):
        with pytest.raises(OSError):
            pool.acquire(SandboxSpec())

    assert pool.stats()["failures"] == 3
    assert meter.snapshot().sandbox_failures == 3
    assert pool.stats()["in_use"] == 0, "a failed acquire must not hold the slot"
    with pytest.raises(OSError):                    # the pool is still usable
        pool.acquire(SandboxSpec())


def test_the_pool_reports_what_it_is_doing():
    meter = Meter()
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2, meter=meter)
    with pool.lease():
        pass
    s = meter.snapshot()
    assert s.sandboxes_created == 1
    assert s.sandbox_setup_s > 0.0
    assert s.sandbox_wait_s >= 0.0
    assert set(pool.stats()) >= {"in_use", "idle", "created", "reused",
                                 "failures", "reclaimed", "leaked", "waiters"}
    pool.close(strict=True)


def test_close_reports_a_leak_when_asked():
    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2)
    pool.acquire(SandboxSpec())            # deliberately never released
    with pytest.raises(SandboxLeak):
        pool.close(strict=True)


# ---------------------------------------------------------------------------
# Leases: reclaim what is abandoned, keep what is alive
# ---------------------------------------------------------------------------


def test_an_abandoned_sandbox_is_reclaimed(tmp_path):
    prov = WorkspaceProvider(ttl=0.2)
    sb = prov.acquire(SandboxSpec(workspace_root=str(tmp_path)))
    assert os.path.exists(sb.root)

    # nobody renews it, and its owner is not around
    lease = os.path.join(sb.root, LEASE_FILE)
    data = json.load(open(lease))
    data["renewed_at"] = time.time() - 100
    data["owner"]["pid"] = 999999            # a pid that is not running
    json.dump(data, open(lease, "w"))

    assert prov.reap(root=str(tmp_path)) == 1
    assert not os.path.exists(sb.root)


def test_a_sandbox_that_is_still_being_renewed_is_left_alone(tmp_path):
    """The half the age-based sweep could not get right: a live run's workspace
    does not change mtime while work happens inside it, so on a shared `$TMPDIR`
    it looked collectable to every other run on the machine."""
    prov = WorkspaceProvider(ttl=0.2)
    sb = prov.acquire(SandboxSpec(workspace_root=str(tmp_path)))
    for _ in range(5):
        time.sleep(0.1)
        prov.renew(sb)
        assert prov.reap(root=str(tmp_path)) == 0
        assert os.path.exists(sb.root)
    prov.release(sb)


def test_a_recycled_pid_does_not_keep_a_dead_run_alive_forever(tmp_path):
    """Our own pid is alive, so the one-TTL grace applies; past two it goes.

    Without the upper bound, a lease whose pid happened to be reused would be
    protected indefinitely -- the leak the pid check was supposed to prevent,
    inverted."""
    prov = WorkspaceProvider(ttl=0.2)
    sb = prov.acquire(SandboxSpec(workspace_root=str(tmp_path)))
    lease = os.path.join(sb.root, LEASE_FILE)

    data = json.load(open(lease))
    data["owner"]["pid"] = os.getpid()          # definitely alive
    data["renewed_at"] = time.time() - 0.3      # expired, within one extra TTL
    json.dump(data, open(lease, "w"))
    assert prov.reap(root=str(tmp_path)) == 0, "a late renewal deserves one grace"

    data["renewed_at"] = time.time() - 10.0     # long past two TTLs
    json.dump(data, open(lease, "w"))
    assert prov.reap(root=str(tmp_path)) == 1
    assert not os.path.exists(sb.root)


def test_a_workspace_with_no_lease_falls_back_to_age(tmp_path):
    """Directories from before this existed still have to be collectable."""
    prov = WorkspaceProvider(ttl=0.2)
    old = tmp_path / "agentdescent-ws-legacy"
    old.mkdir()
    assert prov.reap(root=str(tmp_path)) == 0        # recent: left alone
    ancient = time.time() - 3 * 24 * 3600
    os.utime(old, (ancient, ancient))
    assert prov.reap(root=str(tmp_path)) == 1


# ---------------------------------------------------------------------------
# The runners lease instead of making their own
# ---------------------------------------------------------------------------


def test_a_runner_returns_its_workspace_even_when_the_agent_raises():
    """The failure path has to unwind through the pool like the success path."""
    from agentdescent.agents import AgentError
    from agentdescent.runners import tree_runner
    from agentdescent.sandbox import SandboxPool, WorkspaceProvider

    class Exploding:
        def __call__(self, prompt): raise AgentError("nope")
        def in_workspace(self, path): return self

    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=1)
    run = tree_runner(Exploding(), name="s", sandbox_pool=pool)
    for _ in range(3):
        with pytest.raises(AgentError):
            run(canonical({"rules.md": "hi"}), Task("t", "q"))
    assert pool.stats()["in_use"] == 0, "a raising rollout kept its sandbox"
    pool.close(strict=True)


def test_a_shared_pool_puts_two_runners_under_one_ceiling():
    """Two runners with their own pools have a ceiling of the sum; that is the
    situation `max_concurrency` and `eval_concurrency` are in today."""
    from agentdescent.runners import code_runner
    from agentdescent.sandbox import SandboxPool, WorkspaceProvider

    pool = SandboxPool(WorkspaceProvider(), max_sandboxes=2)
    peak, live, lock = [0], [0], threading.Lock()

    def probe(_rendered, _task):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.03)
        with lock:
            live[0] -= 1
        return "x"

    rollout = code_runner([sys.executable, "-c", "print('x')"], sandbox_pool=pool)
    scorer = code_runner([sys.executable, "-c", "print('x')"], sandbox_pool=pool)
    tree = canonical({"main.py": "print('x')"})

    def use(runner):
        with pool.lease() as sb:
            probe(None, None)

    threads = [threading.Thread(target=use, args=(r,))
               for r in (rollout, scorer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert peak[0] <= 2, f"{peak[0]} at once through a shared pool capped at 2"
    pool.close(strict=True)


def test_workspace_root_is_on_both_public_runners(tmp_path):
    """The runners are the public API a directory run hands `evolve()`, so
    placing a workspace has to be possible from each of them."""
    import inspect
    from agentdescent import code_runner, tree_runner

    for fn in (tree_runner, code_runner):
        params = inspect.signature(fn).parameters
        assert "workspace_root" in params, f"{fn.__name__} still cannot place a workspace"
        assert "sandbox_pool" in params


def test_a_runner_puts_its_workspace_where_it_was_told(tmp_path):
    from agentdescent.runners import code_runner

    run = code_runner([sys.executable, "-c",
                       "import os;print(os.getcwd())"],
                      workspace_root=str(tmp_path))
    out = run(canonical({"main.py": "print(1)"}), Task("t", "q"))
    assert str(tmp_path) in out, f"workspace was not under {tmp_path}: {out}"
