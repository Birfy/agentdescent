"""Budgets and queues must actually bound things.

Each of these was a counter or a comment rather than a real limit: the oracle
budget was decremented but never enforced, the audit queue grew without bound
(and re-sorted itself on every submit), and the async runtime's threads were
non-daemon, so an overrunning run blocked interpreter exit.
"""

import threading
import time

from agentdescent.scheduler import AuditScheduler
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget


class _Art:
    pass


def _verifier(budget: int):
    calls = {"n": 0}

    def eval_fn(artifact, tasks):
        calls["n"] += 1
        return 1.0

    v = ThreeLayerVerifier(eval_fn=eval_fn, held_out=list(range(10)),
                           budget=VerifierBudget(oracle_calls_remaining=budget))
    return v, calls


def test_oracle_budget_actually_caps_spending():
    """Past the budget, oracle_eval must not run another full held-out sweep."""
    for budget in (0, 2, 5):
        v, _ = _verifier(budget)
        for _ in range(8):
            v.oracle_eval(_Art())
        assert v.budget.oracle_calls_used == min(budget, 8)
        assert v.budget.oracle_calls_remaining >= 0


def test_oracle_eval_degrades_instead_of_raising_when_exhausted():
    v, _ = _verifier(0)
    assert isinstance(v.oracle_eval(_Art()), float)      # falls back to the cheap layer


def test_audit_queue_is_bounded():
    a = AuditScheduler(max_queued=100)
    for i in range(5000):
        a.submit(f"d{i}", "art", 0.5, (i % 97) / 97)
    assert len(a._items) <= 200          # cap, with the 2x amortisation headroom
    assert a.dropped > 0                 # and it reports what it shed


def test_audit_submit_is_not_quadratic():
    """A full sort per submit made long runs quadratic; heappush is O(log n)."""
    a = AuditScheduler()
    t0 = time.time()
    for i in range(4000):
        a.submit(f"a{i}", "art", 0.5, 0.5)
    first = time.time() - t0
    t0 = time.time()
    for i in range(4000, 8000):
        a.submit(f"b{i}", "art", 0.5, 0.5)
    second = time.time() - t0
    assert second < max(first * 4, 0.5), f"submit cost grew: {first:.3f}s -> {second:.3f}s"


def test_audit_pops_highest_priority_first():
    a = AuditScheduler()
    a.submit("low", "art", 0.1, 0.1)
    a.submit("high", "art", 0.9, 0.9)
    a.submit("mid", "art", 0.5, 0.5)
    assert a.pop().diff_id == "high"


def test_audit_scheduler_is_thread_safe():
    a = AuditScheduler()

    def submit_many(n):
        for i in range(n):
            a.submit(f"d{threading.get_ident()}-{i}", "art", 0.5, 0.5)

    threads = [threading.Thread(target=submit_many, args=(200,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(a._items) + a.dropped == 800


def test_async_runtime_threads_are_daemon():
    """Non-daemon threads kept the interpreter alive past run()'s return."""
    import inspect

    from agentdescent import async_runtime

    src = inspect.getsource(async_runtime.AsyncAgentDescent.run)
    assert src.count("daemon=True") >= 2, "worker and aggregator threads must be daemon"


def test_l1_serial_gate_admits_exactly_one_under_contention():
    """The gate is documented as a global L1 lock but was a check-then-act on a
    plain dict, so two threads could both believe they hold it."""
    from agentdescent.governance import L1SerialGate

    for _ in range(20):
        gate = L1SerialGate()
        winners = []

        def contend(i):
            if gate.try_acquire(f"a{i}", f"d{i}"):
                winners.append(i)

        threads = [threading.Thread(target=contend, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(winners) == 1, f"{len(winners)} threads acquired the L1 lock"


def test_resume_queue_is_thread_safe():
    """Async workers push checkpoints concurrently; pop is a check-then-act."""
    from agentdescent.scheduler import ResumeItem, ResumeQueue

    q = ResumeQueue()

    def push_many(n):
        for i in range(200):
            q.push(ResumeItem(task_id=f"t{n}-{i}", turn=i, conversation=[],
                              external_handle=None, version_at_checkpoint={}))

    threads = [threading.Thread(target=push_many, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drained = 0
    while q.pop() is not None:
        drained += 1
    assert drained == 800
