"""Borrow the host agent's own model, through MCP sampling.

A worker agent is the host's CLI as a subprocess, so it can be pointed at a
model with ``extra_args`` or handed the user's whole setup with
``isolate: false``. Neither reaches the thing people actually ask for: *"use the
model my agent is already configured with"* -- the one the session is running,
with the session's authentication, chosen in the host's UI.

MCP has exactly that. ``sampling/createMessage`` lets a **server** ask its
**client** to run a completion; the client picks the model, applies its own
policy, and (in every host that implements it) shows the user what was asked.
No key, no model name, no second bill.

The obstacle is that a run does not live in the server. ``start`` returns in
milliseconds and the evolution proceeds in a **detached process** that outlives
the tool call and often the server itself -- which is the whole point, and is
also why the run cannot simply call ``session.create_message``: it has no
session, and the object is not picklable, sendable or re-creatable from a
child.

So this module is a bridge with the two halves on either side of that gap:

* :class:`SamplingBridge` runs **in the server process**, on loopback, holding
  the live session. It turns an HTTP request into ``create_message`` on the
  server's event loop and the reply back into text.
* :func:`host_model` runs **in the run's process** and is an ordinary
  :data:`~agentdescent.agents.Completion` -- ``prompt -> text`` over stdlib
  HTTP -- so it drops into ``agent`` or ``reflect`` like any other model.

What this buys is real but bounded, and the bound is structural rather than an
implementation gap:

* **The run is tied to the session.** Close the agent and the bridge goes with
  it; calls then fail with a message saying so. A background evolution meant to
  outlive your session should not use the host's model.
* **The host must implement sampling.** Many do not. :func:`bridge_for_session`
  asks the client -- via the capability it declared at initialise -- and returns
  ``None`` rather than wiring up something that would fail on first use.
* **Throughput is the host's.** Every rollout is a request the host serialises
  through one session, so a run with eight workers does not get eight streams.

The token is not decoration: any process on the machine can reach a loopback
port, and this one spends the user's model budget.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

__all__ = [
    "SAMPLING_TOKEN_ENV",
    "SAMPLING_URL_ENV",
    "SamplingBridge",
    "SamplingError",
    "bridge_for_session",
    "host_model",
]

#: Where the run's process finds the bridge. Set by the MCP server on the runs
#: it launches, so a spec only has to name `host_model` -- an address in a spec
#: would be stale the moment the server restarted.
SAMPLING_URL_ENV = "AGENTDESCENT_SAMPLING_URL"
SAMPLING_TOKEN_ENV = "AGENTDESCENT_SAMPLING_TOKEN"

#: A single sampling call's ceiling. Sampling goes through the host's UI and
#: often its approval flow, so a request that hangs must not hang a rollout for
#: the round's whole timeout.
DEFAULT_TIMEOUT = 300.0


class SamplingError(RuntimeError):
    """The host could not, or would not, run the completion."""


# ---------------------------------------------------------------------------
# the run's side: an ordinary Completion
# ---------------------------------------------------------------------------


def host_model(*, max_tokens: int = 4096, system: Optional[str] = None,
               timeout: float = DEFAULT_TIMEOUT,
               url: Optional[str] = None,
               token: Optional[str] = None) -> Callable[[str], str]:
    """The host agent's own model, as a :data:`~agentdescent.agents.Completion`.

    Usable anywhere a model is -- ``agent`` for a ``text`` kind, or ``reflect``
    behind a CLI worker, which is the common case: the rollouts run in the host
    CLI and the reflection runs on the session's model, and the whole run needs
    no provider key::

        {"agent":   {"ref": "claude_code"},
         "reflect": {"ref": "host_model"}}

    ``url`` and ``token`` default to the environment the MCP server sets on the
    runs it launches, so a spec normally passes neither. Raises
    :class:`SamplingError` when there is no bridge -- a run started from the
    shell has no session to borrow, and saying so beats a connection error.
    """
    endpoint = url or os.environ.get(SAMPLING_URL_ENV)
    secret = token or os.environ.get(SAMPLING_TOKEN_ENV, "")

    def complete(prompt: str) -> str:
        if not endpoint:
            raise SamplingError(
                "no MCP sampling bridge in this process: `host_model` borrows the "
                "model of the agent session that started the run, so it works "
                "only for a run launched by the MCP server from a host that "
                "supports sampling. From a shell, name a model instead "
                "(`claude_code`, `openai_compatible`, ...).")
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                           "system": system}).encode()
        req = urllib.request.Request(
            endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {secret}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                payload = json.loads(fh.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise SamplingError(f"the host refused the sampling request "
                                f"(HTTP {e.code}): {detail}") from None
        except urllib.error.URLError as e:
            # The overwhelmingly likely cause, and one no retry will fix.
            raise SamplingError(
                f"the sampling bridge at {endpoint} is gone -- the agent session "
                f"that started this run has probably closed. ({e.reason})") from None
        if payload.get("error"):
            raise SamplingError(str(payload["error"]))
        return payload.get("text", "")

    return complete


# ---------------------------------------------------------------------------
# the server's side: loopback HTTP in front of a live session
# ---------------------------------------------------------------------------


class SamplingBridge:
    """A loopback endpoint that turns HTTP into ``sampling/createMessage``.

    ``ask(prompt, max_tokens, system) -> str`` is called on an HTTP worker
    thread, so it must be safe to call from one -- :func:`bridge_for_session`
    builds the version that hops onto the server's event loop.
    """

    def __init__(self, ask: Callable[[str, int, Optional[str]], str], *,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        self._ask = ask
        self.token = secrets.token_urlsafe(32)
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
                if self.path.rstrip("/") != "/sample":
                    return bridge._reply(self, 404, {"error": "not found"})
                # Constant-time, and checked before the body is read: an
                # unauthenticated caller should not get to spend anything, and
                # `!=` on a secret leaks its prefix under timing.
                sent = self.headers.get("Authorization", "")
                if not secrets.compare_digest(sent, f"Bearer {bridge.token}"):
                    return bridge._reply(self, 401, {"error": "bad token"})
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, TypeError):
                    return bridge._reply(self, 400, {"error": "malformed request"})
                try:
                    text = bridge._ask(str(req.get("prompt", "")),
                                       int(req.get("max_tokens") or 4096),
                                       req.get("system"))
                except Exception as e:  # noqa: BLE001 - the run needs the reason
                    return bridge._reply(self, 200, {"error": f"{type(e).__name__}: {e}"})
                return bridge._reply(self, 200, {"text": text})

            def log_message(self, *_a: Any) -> None:
                """Silence. stdout is the MCP protocol stream on this process."""

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="agentdescent-sampling", daemon=True)
        self._thread.start()

    @staticmethod
    def _reply(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/sample"

    def env(self) -> Dict[str, str]:
        """What a launched run needs to find this bridge."""
        return {SAMPLING_URL_ENV: self.url, SAMPLING_TOKEN_ENV: self.token}

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "SamplingBridge":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def bridge_for_session(session: Any, loop: Any, *,
                       timeout: float = DEFAULT_TIMEOUT) -> Optional[SamplingBridge]:
    """A bridge onto ``session``, or ``None`` if this client cannot sample.

    ``loop`` is the event loop the session is served on -- tools run on it, so
    ``asyncio.get_running_loop()`` inside a tool is the one to pass. Every
    request is scheduled back onto it, because the session's transport is not
    thread-safe and the HTTP handlers are threads.

    The capability check is the client's own declaration from ``initialize``, so
    a host that never implemented sampling gets no bridge rather than a run that
    fails on its first reflection.
    """
    import asyncio

    from mcp.types import ClientCapabilities, SamplingCapability, SamplingMessage, TextContent

    try:
        supported = session.check_client_capability(
            ClientCapabilities(sampling=SamplingCapability()))
    except Exception:  # noqa: BLE001 - an SDK that cannot answer is a "no"
        return None
    if not supported:
        return None

    def ask(prompt: str, max_tokens: int, system: Optional[str]) -> str:
        coro = session.create_message(
            messages=[SamplingMessage(role="user",
                                      content=TextContent(type="text", text=prompt))],
            max_tokens=max_tokens, system_prompt=system)
        result = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)
        content = getattr(result, "content", None)
        text = getattr(content, "text", None)
        if text is None:
            raise SamplingError(
                f"the host returned {type(content).__name__} rather than text; "
                "a reflector needs text")
        return text

    return SamplingBridge(ask)
