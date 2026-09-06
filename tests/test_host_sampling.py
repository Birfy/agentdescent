"""Borrowing the host agent's model: the bridge, and the run that crosses it.

The thing under test is a gap. `start` returns in milliseconds and the run
proceeds in a detached process, so the process that has the MCP session is never
the process that needs a model. Everything here is about whether that gap is
crossed correctly and refused clearly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import pytest

from agentdescent.host_sampling import (
    SAMPLING_TOKEN_ENV,
    SAMPLING_URL_ENV,
    SamplingBridge,
    SamplingError,
    bridge_for_session,
    host_model,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _post(url, body, token=None, method="POST"):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method=method,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=10) as fh:
        return json.loads(fh.read().decode())


# ---------------------------------------------------------------------------
# The run's side
# ---------------------------------------------------------------------------


def test_without_a_bridge_it_says_which_case_this_is(monkeypatch):
    """A run started from the shell has no session to borrow, and a connection
    error would send the reader looking for a network problem."""
    monkeypatch.delenv(SAMPLING_URL_ENV, raising=False)
    with pytest.raises(SamplingError) as e:
        host_model()("anything")
    assert "MCP server" in str(e.value)
    assert "claude_code" in str(e.value)          # and what to use instead


def test_a_closed_session_is_reported_as_a_closed_session(monkeypatch):
    """The session outliving the run is the expected failure, not an exception."""
    monkeypatch.setenv(SAMPLING_URL_ENV, "http://127.0.0.1:9/sample")  # discard port
    monkeypatch.setenv(SAMPLING_TOKEN_ENV, "irrelevant")
    with pytest.raises(SamplingError) as e:
        host_model(timeout=5)("anything")
    assert "has probably closed" in str(e.value)


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


def test_the_bridge_round_trips_a_prompt():
    seen = []

    def ask(prompt, max_tokens, system):
        seen.append((prompt, max_tokens, system))
        return "the host's answer"

    with SamplingBridge(ask) as bridge:
        model = host_model(url=bridge.url, token=bridge.token,
                           max_tokens=99, system="be brief")
        assert model("hello") == "the host's answer"
    assert seen == [("hello", 99, "be brief")]


def test_the_bridge_refuses_a_caller_without_the_token():
    """Any process on the machine can reach a loopback port, and this one spends
    the user's model budget."""
    with SamplingBridge(lambda *_: "should not happen") as bridge:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(bridge.url, {"prompt": "x"})
        assert e.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(bridge.url, {"prompt": "x"}, token="wrong")
        assert e.value.code == 401
        # And nothing else is served.
        base = bridge.url.rsplit("/", 1)[0]
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/anything", {}, token=bridge.token)
        assert e.value.code == 404


def test_a_failing_host_reaches_the_run_as_a_reason():
    """The run must be able to say *why* its reflector produced nothing."""
    def ask(*_a):
        raise RuntimeError("the user rejected the sampling request")

    with SamplingBridge(ask) as bridge:
        with pytest.raises(SamplingError) as e:
            host_model(url=bridge.url, token=bridge.token)("hello")
    assert "the user rejected" in str(e.value)


def test_the_bridge_serves_workers_concurrently():
    """n_workers rollouts reflect at once; a serialised bridge would stall them."""
    import threading

    def ask(prompt, *_a):
        time.sleep(0.2)
        return prompt.upper()

    with SamplingBridge(ask) as bridge:
        model = host_model(url=bridge.url, token=bridge.token)
        out = {}
        threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, model(f"p{i}")))
                   for i in range(4)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0
    assert out == {i: f"P{i}" for i in range(4)}
    # Four 0.2s calls in well under 0.8s means they overlapped. The bound is
    # loose on purpose: this asserts concurrency, not a latency budget.
    assert elapsed < 0.6, elapsed


def test_the_cli_route_is_what_makes_this_usable_at_all():
    """No host measured implements sampling -- Claude Code, OpenCode and dsh all
    connect without it -- so the fallback is not a nicety, it is the feature.

    The names are the ones each host really sends at `initialize`, captured from
    a logging shim in front of the server.
    """
    from agentdescent.host_sampling import HOST_CLI_ENV, host_cli_for_client
    from agentdescent.mcp import Tools

    assert host_cli_for_client("claude-code") == "claude_code"
    assert host_cli_for_client("opencode") == "opencode"
    assert host_cli_for_client("dsh-mcp-client") == "dsh"
    assert host_cli_for_client("codex-cli") == "codex"        # substring
    assert host_cli_for_client("some-other-editor") is None

    t = Tools()
    t.host_cli = "claude_code"
    assert t.host_model_env() == {HOST_CLI_ENV: "claude_code"}


def test_host_model_falls_back_to_the_hosts_cli(monkeypatch):
    """With no bridge but a host CLI named, the reflection goes through it."""
    import agentdescent.agents as agents
    from agentdescent.host_sampling import HOST_CLI_ENV

    monkeypatch.delenv(SAMPLING_URL_ENV, raising=False)
    monkeypatch.setenv(HOST_CLI_ENV, "claude_code")
    seen = {}

    def fake_claude_code(**kw):
        seen.update(kw)
        return lambda prompt: f"answered: {prompt}"

    monkeypatch.setattr(agents, "claude_code", fake_claude_code)
    assert host_model()("think") == "answered: think"
    # The user's configured model and login live in the config directory an
    # isolated worker is pointed away from, so this route must not isolate.
    assert seen == {"isolate": False}


def test_a_dead_bridge_falls_back_rather_than_failing_the_run(monkeypatch):
    """The session closing is the expected end of a bridge, not of the run."""
    import agentdescent.agents as agents
    from agentdescent.host_sampling import HOST_CLI_ENV

    monkeypatch.setenv(SAMPLING_URL_ENV, "http://127.0.0.1:9/sample")
    monkeypatch.setenv(SAMPLING_TOKEN_ENV, "irrelevant")
    monkeypatch.setenv(HOST_CLI_ENV, "claude_code")
    monkeypatch.setattr(agents, "claude_code", lambda **kw: lambda p: "from the CLI")
    assert host_model(timeout=5)("think") == "from the CLI"


def test_a_client_without_sampling_gets_no_bridge():
    """Most hosts do not implement sampling. Returning None beats wiring up
    something that fails on the run's first reflection."""
    class Session:
        def check_client_capability(self, _cap):
            return False

    assert bridge_for_session(Session(), loop=None) is None

    class Ancient:
        pass                                     # an SDK that cannot even answer

    assert bridge_for_session(Ancient(), loop=None) is None


# ---------------------------------------------------------------------------
# The whole gap, through the real SDK
# ---------------------------------------------------------------------------


SUBPROCESS_PROBE = r'''
import asyncio, json, os, sys, tempfile
sys.path.insert(0, %(root)r)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CreateMessageResult, TextContent
from agentdescent import demo

CALLS = []

async def sampling_callback(context, params):
    prompt = params.messages[0].content.text
    CALLS.append(prompt)
    return CreateMessageResult(role="assistant", model="probe", stopReason="endTurn",
                               content=TextContent(type="text",
                                                   text=demo.offline_reflector(prompt)))

async def main():
    root, store = tempfile.mkdtemp(), tempfile.mkdtemp()
    spec = demo.build(root)
    spec["reflect"] = {"ref": "host_model"}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agentdescent.cli", "--store", store, "mcp"],
        env={**os.environ, "PYTHONPATH": %(root)r})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w, sampling_callback=sampling_callback) as s:
            await s.initialize()
            out = json.loads((await s.call_tool("start", {"spec": spec})).content[0].text)
            for _ in range(180):
                await asyncio.sleep(0.5)
                st = json.loads((await s.call_tool(
                    "status", {"run_id": out["run_id"]})).content[0].text)
                if st.get("state") in ("done", "error", "cancelled"):
                    break
            print(json.dumps({"available": out.get("host_model_available"),
                              "why": out.get("host_model_unavailable"),
                              "state": st.get("state"),
                              "reward": st.get("best_reward"),
                              "host_calls": len(CALLS)}))
asyncio.run(main())
'''


def test_a_detached_run_really_reflects_on_the_hosts_model():
    """The end to end: server -> bridge -> child env -> HTTP -> session -> client.

    Run in a subprocess because it needs its own event loop and a stdio client,
    and the assertion that matters is not "the bridge answered" but that the
    *detached* run -- a different process, started and outliving the tool call --
    learned the fix using a model only the client has.
    """
    pytest.importorskip("mcp")
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "probe.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(SUBPROCESS_PROBE % {"root": ROOT})
        out = subprocess.run([sys.executable, script], capture_output=True, text=True,
                             timeout=300, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["available"] is True, got
    assert got["host_calls"] > 0, f"the run never asked the host for a completion: {got}"
    assert got["state"] == "done" and got["reward"] == 1.0, got


def test_start_says_when_the_host_cannot_lend_its_model():
    """Without this the run just proposes nothing, which reads as "it learned
    nothing" rather than "it could not ask"."""
    from agentdescent.mcp import Tools

    t = Tools()
    assert t.host_model_env() == {}
    t.bridge, t.bridge_error = None, "this host did not declare the sampling capability"
    with tempfile.TemporaryDirectory() as store:
        t.store = store
        out = t.start({"kind": "text", "target": "x", "data": {"inline": []},
                       "score": "exact", "agent": {"ref": "echo"}})
    assert out.get("host_model_available") is False or out.get("ok") is False
