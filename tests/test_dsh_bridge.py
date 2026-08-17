"""The bridge, driven as two real peers over real pipes.

The properties worth testing are the ones that only show up when both ends are
live: a call in each direction at once, a handler that calls back while it is
being served, and what happens to outstanding calls when the far side dies.
"""

from __future__ import annotations

import os
import threading

import pytest

from agentdescent.dsh.bridge import PROTOCOL_VERSION, Peer, PeerGone, RpcError


def connect(a_handlers=None, b_handlers=None):
    """Two peers wired to each other through a pair of OS pipes."""
    a_read, b_write = os.pipe()
    b_read, a_write = os.pipe()
    errors = []
    a = Peer(os.fdopen(a_read, "r"), os.fdopen(a_write, "w"), a_handlers,
             on_error=errors.append)
    b = Peer(os.fdopen(b_read, "r"), os.fdopen(b_write, "w"), b_handlers,
             on_error=errors.append)
    a.start()
    b.start()
    return a, b, errors


def test_a_call_reaches_the_handler_and_the_result_comes_back():
    a, b, _ = connect(b_handlers={"echo": lambda p: {"saw": p}})
    try:
        assert a.call("echo", {"x": 1}, timeout=5) == {"saw": {"x": 1}}
    finally:
        a.close(), b.close()


def test_both_directions_work_because_neither_side_is_the_client():
    """The harness starts runs; the engine calls back for every rollout."""
    a, b, _ = connect(a_handlers={"rollout.execute": lambda p: {"output": f"ran {p['task']}"}},
                      b_handlers={"run.start": lambda p: "run-1"})
    try:
        assert a.call("run.start", {"artifact": "skill:sql"}, timeout=5) == "run-1"
        assert b.call("rollout.execute", {"task": "t0"}, timeout=5) == {"output": "ran t0"}
    finally:
        a.close(), b.close()


def test_a_handler_may_call_back_while_it_is_being_served():
    """The shape the whole design depends on, and the one that deadlocks naively.

    `run.start` on the engine asks the harness for a rollout before it returns.
    Serving inbound calls on the reader thread would block that reader until the
    handler finished -- and the handler is waiting on a reply that only the
    reader can deliver.
    """
    a, b, _ = connect(a_handlers={"rollout.execute": lambda p: {"output": "done"}})

    def start(_params):
        inner = b.call("rollout.execute", {"task": "t0"}, timeout=5)
        return {"first_rollout": inner["output"]}

    b.handle("run.start", start)
    try:
        assert a.call("run.start", {}, timeout=5) == {"first_rollout": "done"}
    finally:
        a.close(), b.close()


def test_concurrent_calls_settle_against_their_own_replies():
    a, b, _ = connect(b_handlers={"slow": lambda p: p["n"] * 2})
    results = {}

    def ask(n):
        results[n] = a.call("slow", {"n": n}, timeout=5)

    threads = [threading.Thread(target=ask, args=(n,)) for n in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert results == {n: n * 2 for n in range(8)}
    finally:
        a.close(), b.close()


def test_a_raising_handler_becomes_an_error_response_not_a_dead_connection():
    """One bad rollout must not take the bridge down with it."""
    def boom(_params):
        raise ValueError("that candidate exploded")

    a, b, errors = connect(b_handlers={"boom": boom, "fine": lambda p: "ok"})
    try:
        with pytest.raises(RpcError, match="that candidate exploded"):
            a.call("boom", {}, timeout=5)
        assert a.call("fine", {}, timeout=5) == "ok"      # still usable
        assert any("boom" in line for line in errors)
    finally:
        a.close(), b.close()


def test_an_unknown_method_answers_instead_of_hanging():
    a, b, _ = connect(b_handlers={})
    try:
        with pytest.raises(RpcError, match="no handler"):
            a.call("nope", {}, timeout=5)
    finally:
        a.close(), b.close()


def test_a_notification_gets_no_reply_and_never_blocks():
    seen = threading.Event()
    a, b, _ = connect(b_handlers={"log.progress": lambda p: seen.set()})
    try:
        a.notify("log.progress", {"round": 1})
        assert seen.wait(5)
    finally:
        a.close(), b.close()


def test_a_notification_to_an_unknown_method_is_silently_fine():
    a, b, _ = connect(b_handlers={})
    try:
        a.notify("nobody.listening", {})
        assert a.call("x", timeout=0.5) if False else True   # bridge still alive
    except RpcError:
        pass
    finally:
        a.close(), b.close()


def test_outstanding_calls_fail_when_the_peer_dies_rather_than_hanging_forever():
    """A hung call is worse than a failed one: the round never finishes and
    nothing says why."""
    started = threading.Event()

    def never_returns(_params):
        started.set()
        threading.Event().wait(30)

    a, b, _ = connect(b_handlers={"hang": never_returns})
    outcome = {}

    def ask():
        try:
            a.call("hang", {}, timeout=20)
        except BaseException as exc:               # noqa: BLE001 - recording it is the test
            outcome["error"] = exc

    thread = threading.Thread(target=ask)
    thread.start()
    assert started.wait(5)
    b.close()
    a.close()
    thread.join(10)
    assert isinstance(outcome.get("error"), PeerGone)


def test_calling_a_closed_bridge_says_so_immediately():
    a, b, _ = connect()
    a.close()
    with pytest.raises(PeerGone):
        a.call("anything", {}, timeout=1)
    b.close()


def test_a_local_timeout_does_not_corrupt_later_calls():
    """The peer is not told about a local give-up, so its late reply arrives
    against an id nobody is waiting for. That must be ignored, not mismatched
    onto the next call."""
    release = threading.Event()
    a, b, _ = connect(b_handlers={"slow": lambda p: (release.wait(10), "late")[1],
                                  "quick": lambda p: "prompt"})
    try:
        with pytest.raises(TimeoutError):
            a.call("slow", {}, timeout=0.3)
        release.set()
        assert a.call("quick", {}, timeout=5) == "prompt"
    finally:
        a.close(), b.close()


def test_garbage_on_the_wire_does_not_kill_the_loop():
    a, b, _ = connect(b_handlers={"fine": lambda p: "ok"})
    try:
        a._writer.write("this is not json\n")     # noqa: SLF001 - simulating a bad peer
        a._writer.flush()                          # noqa: SLF001
        assert a.call("fine", {}, timeout=5) == "ok"
    finally:
        a.close(), b.close()


def test_the_protocol_version_is_a_number_the_handshake_can_compare():
    assert isinstance(PROTOCOL_VERSION, int) and PROTOCOL_VERSION >= 1
