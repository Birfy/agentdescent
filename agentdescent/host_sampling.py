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
    "HOST_CLIS",
    "HOST_CLI_ENV",
    "SAMPLING_TOKEN_ENV",
    "SAMPLING_URL_ENV",
    "SamplingBridge",
    "SamplingError",
    "bridge_for_session",
    "host_cli_for_client",
    "host_model",
]

#: Where the run's process finds the bridge. Set by the MCP server on the runs
#: it launches, so a spec only has to name `host_model` -- an address in a spec
#: would be stale the moment the server restarted.
SAMPLING_URL_ENV = "AGENTDESCENT_SAMPLING_URL"
SAMPLING_TOKEN_ENV = "AGENTDESCENT_SAMPLING_TOKEN"

#: Which host is on the other end of this server, when sampling is unavailable.
#: Also set by the MCP server on the runs it launches.
HOST_CLI_ENV = "AGENTDESCENT_HOST_CLI"

#: `clientInfo.name` -> the factory in `agentdescent.agents` that runs that
#: host's CLI. The names are what each host actually sends at `initialize`,
#: captured from a logging shim rather than guessed:
#:
#:     claude-code      2.1.261    caps: roots, elicitation
#:     opencode         1.18.29    caps: roots
#:     dsh-mcp-client   0.0.1      caps: (none)
#:
#: None of the three declares sampling, which is the whole reason this fallback
#: exists. Codex is matched on a substring because it never opens a session for
#: `codex mcp list`, so its name could not be captured the same way.
HOST_CLIS: Dict[str, str] = {
    "claude-code": "claude_code",
    "opencode": "opencode",
    "dsh-mcp-client": "dsh",
    "dsh": "dsh",
    "codex": "codex",
}

#: A single sampling call's ceiling. Sampling goes through the host's UI and
#: often its approval flow, so a request that hangs must not hang a rollout for
#: the round's whole timeout.
DEFAULT_TIMEOUT = 300.0


class SamplingError(RuntimeError):
    """The host could not, or would not, run the completion."""


# ---------------------------------------------------------------------------
# the run's side: an ordinary Completion
# ---------------------------------------------------------------------------


def host_cli_for_client(client_name: Optional[str]) -> Optional[str]:
    """Which `agentdescent.agents` factory runs *this* host's CLI, if any.

    Exact match first, then a substring, so a host that renames itself
    ``codex-cli`` or ``opencode-nightly`` still resolves.
    """
    name = (client_name or "").strip().lower()
    if not name:
        return None
    if name in HOST_CLIS:
        return HOST_CLIS[name]
    for key, factory in HOST_CLIS.items():
        if key in name:
            return factory
    return None


def host_model(*, max_tokens: int = 4096, system: Optional[str] = None,
               timeout: float = DEFAULT_TIMEOUT,
               url: Optional[str] = None,
               token: Optional[str] = None,
               cli: Optional[str] = None) -> Callable[[str], str]:
    """The host agent's own model, as a :data:`~agentdescent.agents.Completion`.

    Usable anywhere a model is -- ``agent`` for a ``text`` kind, or ``reflect``
    behind a CLI worker, which is the common case: the rollouts run in the host
    CLI and the reflection runs on the session's model, and the whole run needs
    no provider key::

        {"agent":   {"ref": "claude_code"},
         "reflect": {"ref": "host_model"}}

    Two routes, tried in order, because **no host measured so far implements
    sampling** -- Claude Code, OpenCode and DeepSeek Harness all connect without
    it, so a `host_model` that only spoke sampling would have been a feature
    nobody could use:

    1. **the live session**, over sampling, when the host supports it. The model
       is the one running the session, with its authentication and policy;
    2. **the host's own CLI**, otherwise -- `claude`, `codex`, `dsh` or
       `opencode`, whichever host started this server, run with the user's real
       configuration (``isolate=False``) so it uses the model and login they
       have set up.

    The second is not the first: it starts a fresh CLI process rather than
    borrowing the session, so it follows the user's *configured* model rather
    than whatever the session switched to, and it costs what that CLI costs.
    It is, though, the thing people mean by "use my agent's model", and it needs
    no key either.

    ``url``, ``token`` and ``cli`` default to the environment the MCP server
    sets on the runs it launches, so a spec normally passes none of them. Raises
    :class:`SamplingError` when neither route is available -- a run started from
    the shell has no host at all, and saying so beats a connection error.
    """
    endpoint = url or os.environ.get(SAMPLING_URL_ENV)
    secret = token or os.environ.get(SAMPLING_TOKEN_ENV, "")
    fallback = cli or os.environ.get(HOST_CLI_ENV)

    def via_cli(prompt: str) -> str:
        from . import agents

        factory = getattr(agents, fallback, None)
        if factory is None:
            raise SamplingError(f"{HOST_CLI_ENV}={fallback!r} names no known host CLI")
        # isolate=False on purpose: the point is the user's configured model and
        # their login, both of which live in the config directory an isolated
        # worker is pointed away from.
        return factory(isolate=False)(prompt)

    def complete(prompt: str) -> str:
        if not endpoint:
            if fallback:
                return via_cli(prompt)
            raise SamplingError(
                "no host to borrow a model from in this process: `host_model` "
                "uses the agent session that started the run -- its model over "
                "MCP sampling, or its CLI -- so it works only for a run launched "
                "by the MCP server. From a shell, name a model instead "
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
            # The session outliving the run is the expected end of a bridge. If
            # the host also has a CLI, that is a better answer than failing.
            if fallback:
                return via_cli(prompt)
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

    try:
        from mcp.types import (
            ClientCapabilities, SamplingCapability, SamplingMessage, TextContent)
    except ImportError:
        # No SDK is the extreme case of the rule below: an SDK that cannot
        # answer the capability question is a "no", and one that is not
        # installed cannot answer at all. Raising here instead turned a host
        # without sampling into a crash, and it is reachable from a test suite
        # installed with `[dev]` alone -- which is what CI installs.
        return None

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
