"""The wire between the harness and the engine: NDJSON-RPC, both directions.

The plugin runs in the harness process (TypeScript); the optimiser runs here
(Python). Neither is the client. The harness starts runs and reads the ledger;
the engine calls *back* for every rollout, because the whole point of the design
is that the harness is the runtime -- the thing being measured is the real
harness, with its real tools, sandbox and approval policy.

So this is a **symmetric peer**, not a server. Both sides may call; both sides
answer. One line is one JSON object, which is the framing the harness's own
Python SDK already speaks, so the ecosystem has one wire vocabulary rather than
two.

**What does not cross.** Functions. Every factory in this package returns a
closure and none of them pickle, which is why :mod:`agentdescent.workspec`
exists: work is described as a :class:`~agentdescent.workspec.Ref` -- a dotted
target plus JSON config, resolved on the far side by the far side's own code.
The bridge carries that description, never the thing itself.

**Failures are values until they are not.** A handler that raises becomes an
error *response*, so one bad rollout does not take the connection down. A broken
pipe is different in kind: the peer is gone, every outstanding call is failed at
once rather than left hanging forever, and the loop stops.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any, Callable, Dict, IO, Mapping, Optional

__all__ = ["Peer", "RpcError", "PeerGone", "PROTOCOL_VERSION"]

#: Bumped when a method's shape changes incompatibly. The handshake compares it
#: and refuses rather than guessing -- a version mismatch that negotiates
#: silently produces wrong measurements, which is worse than not starting.
PROTOCOL_VERSION = 1

#: JSON-RPC-ish codes. Not the full spec: this is a private wire between two
#: processes we ship together, and the useful part is the shape, not the registry.
PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


class RpcError(RuntimeError):
    """The peer answered with an error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.data = code, message, data


class PeerGone(RuntimeError):
    """The peer went away -- EOF, a closed pipe, or an explicit stop.

    Raised at every outstanding call the moment it is noticed, rather than
    letting them wait for a reply that cannot arrive.
    """


class Peer:
    """One end of the bridge.

    ``handlers`` maps a method name to ``(params) -> result``. Reading runs on
    its own thread; :meth:`call` blocks the calling thread until the reply
    arrives, so a rollout can be requested from inside the engine's worker
    without that worker owning an event loop.
    """

    def __init__(self, reader: IO[str], writer: IO[str],
                 handlers: Optional[Mapping[str, Callable[[Any], Any]]] = None,
                 *, on_error: Optional[Callable[[str], None]] = None) -> None:
        self._reader, self._writer = reader, writer
        self._handlers: Dict[str, Callable[[Any], Any]] = dict(handlers or {})
        self._on_error = on_error or (lambda line: print(line, file=sys.stderr))
        self._write_lock = threading.Lock()
        self._pending: Dict[int, "_Pending"] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------

    def handle(self, method: str, fn: Callable[[Any], Any]) -> None:
        """Register a handler. Late registration is fine; the loop reads it live."""
        self._handlers[method] = fn

    def start(self) -> "Peer":
        """Begin reading. Idempotent."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._read_loop, name="dsh-bridge",
                                            daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop, and fail every outstanding call rather than leaving it hanging.

        Closes the **writer**, and deliberately not the reader. The read loop is
        blocked in ``read()``, and closing a buffered reader from another thread
        waits on the very lock that blocked read holds -- so "tidying up" the
        reader here deadlocks `close()` itself. Closing the writer is safe and
        does the same job from the other end: the peer sees EOF, its loop exits,
        it closes its own writer, and this side's reader unblocks in turn.
        """
        if self._closed.is_set():
            return
        self._closed.set()
        with self._write_lock:
            try:
                self._writer.close()
            except Exception:                     # noqa: BLE001 - teardown is best-effort
                pass
        self._fail_all(PeerGone("bridge closed locally"))

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the peer goes away. True if it did."""
        return self._closed.wait(timeout)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def __enter__(self) -> "Peer":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- outbound ----------------------------------------------------------

    def notify(self, method: str, params: Any = None) -> None:
        """Fire and forget. Used for progress, which must never block a rollout."""
        self._send({"method": method, "params": params})

    def call(self, method: str, params: Any = None,
             timeout: Optional[float] = None) -> Any:
        """Call the peer and wait for its answer.

        ``timeout`` is a local give-up, not a cancellation: the peer is not told.
        Anything that needs the far side to actually stop needs its own method.
        """
        if self._closed.is_set():
            raise PeerGone(f"cannot call {method!r}: the bridge is closed")
        with self._pending_lock:
            self._next_id += 1
            call_id = self._next_id
            pending = _Pending()
            self._pending[call_id] = pending
        self._send({"id": call_id, "method": method, "params": params})
        if not pending.done.wait(timeout):
            with self._pending_lock:
                self._pending.pop(call_id, None)
            raise TimeoutError(f"{method!r} did not answer within {timeout}s")
        if isinstance(pending.error, BaseException):
            raise pending.error
        return pending.result

    # -- inbound -----------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            for line in self._reader:
                if self._closed.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                self._dispatch(line)
        except (OSError, ValueError) as exc:      # closed pipe mid-read
            self._on_error(f"bridge read failed: {type(exc).__name__}: {exc}")
        finally:
            self._closed.set()
            self._fail_all(PeerGone("the peer closed the bridge"))

    def _dispatch(self, line: str) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            self._send({"error": {"code": PARSE_ERROR, "message": "not JSON"}})
            return
        if not isinstance(message, dict):
            return
        if "method" in message:
            self._serve(message)
        else:
            self._settle(message)

    def _serve(self, message: Dict[str, Any]) -> None:
        method, call_id = message.get("method"), message.get("id")
        handler = self._handlers.get(str(method))
        if handler is None:
            if call_id is not None:
                self._send({"id": call_id, "error": {
                    "code": METHOD_NOT_FOUND, "message": f"no handler for {method!r}"}})
            return
        # Each inbound call gets a thread: a handler that itself calls back --
        # the harness asking for a rollout while the engine asks for a score --
        # would otherwise deadlock against the single reader.
        threading.Thread(target=self._run_handler, args=(handler, message, call_id),
                         name=f"dsh-bridge-{method}", daemon=True).start()

    def _run_handler(self, handler: Callable[[Any], Any], message: Dict[str, Any],
                     call_id: Any) -> None:
        try:
            result = handler(message.get("params"))
        except Exception as exc:                  # noqa: BLE001 - a value, not a crash
            self._on_error(f"handler {message.get('method')!r} raised: "
                           f"{traceback.format_exc().rstrip()}")
            if call_id is not None:
                self._send({"id": call_id, "error": {
                    "code": INTERNAL_ERROR, "message": f"{type(exc).__name__}: {exc}"}})
            return
        if call_id is not None:
            self._send({"id": call_id, "result": result})

    def _settle(self, message: Dict[str, Any]) -> None:
        call_id = message.get("id")
        with self._pending_lock:
            pending = self._pending.pop(call_id, None)
        if pending is None:                        # a reply to a timed-out call
            return
        error = message.get("error")
        if isinstance(error, dict):
            pending.error = RpcError(int(error.get("code", INTERNAL_ERROR)),
                                     str(error.get("message", "")), error.get("data"))
        else:
            pending.result = message.get("result")
        pending.done.set()

    # -- plumbing ----------------------------------------------------------

    def _send(self, message: Mapping[str, Any]) -> None:
        line = json.dumps(message, separators=(",", ":"), default=str) + "\n"
        with self._write_lock:
            try:
                self._writer.write(line)
                self._writer.flush()
            except (OSError, ValueError) as exc:
                self._closed.set()
                raise PeerGone(f"write failed: {type(exc).__name__}: {exc}") from exc

    def _fail_all(self, error: BaseException) -> None:
        with self._pending_lock:
            outstanding, self._pending = list(self._pending.values()), {}
        for pending in outstanding:
            pending.error = error
            pending.done.set()


class _Pending:
    __slots__ = ("done", "result", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None
