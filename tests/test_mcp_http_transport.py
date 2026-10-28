"""`agentdescent mcp --transport streamable-http` (#560)."""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from agentdescent.cli import build_parser, main
from agentdescent.mcp import is_loopback_host


def test_help_lists_transport_flags_and_stdio_stays_default():
    a = build_parser().parse_args(["mcp"])
    assert a.transport == "stdio"
    assert a.host == "127.0.0.1" and a.port == 8000 and a.allow_remote is False
    a = build_parser().parse_args(["mcp", "--transport", "streamable-http", "--port", "9"])
    assert a.transport == "streamable-http" and a.port == 9


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True),
    ("127.0.0.2", True), ("0.0.0.0", False), ("::", False), ("10.0.0.5", False),
    ("example.com", False),
])
def test_is_loopback_host(host, expected):
    assert is_loopback_host(host) is expected


def test_non_loopback_host_without_allow_remote_exits_nonzero(capsys):
    rc = main(["mcp", "--transport", "streamable-http", "--host", "0.0.0.0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--allow-remote" in err and "0.0.0.0" in err


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Resp:
    def __init__(self, status, headers, text):
        self.status_code, self.headers, self.text = status, headers, text


def _post(url, body, session=None):
    import urllib.error
    import urllib.request
    headers = {"Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}
    if session:
        headers["mcp-session-id"] = session
        headers["mcp-protocol-version"] = "2025-06-18"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return _Resp(r.status, hdrs, r.read().decode())
    except urllib.error.HTTPError as e:
        return _Resp(e.code, {k.lower(): v for k, v in e.headers.items()},
                     e.read().decode(errors="replace"))


def _rpc_result(resp):
    text = resp.text
    if "text/event-stream" in resp.headers.get("content-type", ""):
        data = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        text = data[-1]
    return json.loads(text)["result"]


def test_streamable_http_initialize_and_list_tools(tmp_path):
    pytest.importorskip("mcp")
    port = _free_port()
    env = dict(os.environ, AGENTDESCENT_HOME=str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentdescent", "--store", str(tmp_path / "runs"),
         "mcp", "--transport", "streamable-http", "--port", str(port)],
        cwd=str(tmp_path), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        deadline = time.time() + 30
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                if proc.poll() is not None or time.time() > deadline:
                    raise AssertionError(proc.stderr.read().decode(errors="replace"))
                time.sleep(0.2)
        resp = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}})
        assert resp.status_code == 200, resp.text
        init = _rpc_result(resp)
        assert init["serverInfo"]["name"] == "agentdescent"
        session = resp.headers.get("mcp-session-id")
        _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
        resp = _post(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session)
        assert resp.status_code == 200, resp.text
        names = {t["name"] for t in _rpc_result(resp)["tools"]}
        assert {"list_algorithms", "doctor", "plan"} <= names
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


class _Fake1xSettings:
    def __init__(self):
        self.host, self.port = "127.0.0.1", 8000
        self.transport_security = object()  # the loopback-only pin mcp 1.x sets


class _Fake1xServer:
    """mcp 1.x FastMCP shape: no host/port on run_streamable_http_async."""

    def __init__(self):
        self.settings = _Fake1xSettings()
        self.ran = None

    async def run_streamable_http_async(self):  # pragma: no cover - not awaited
        pass

    def run(self, transport):
        self.ran = transport


@pytest.mark.parametrize("host,keeps_pin", [("127.0.0.1", True), ("0.0.0.0", False)])
def test_mcp1x_non_loopback_drops_loopback_host_pin(monkeypatch, host, keeps_pin):
    """mcp 1.x pins allowed Host headers to loopback at construction; a remote
    bind must drop it or every non-loopback request is answered 421."""
    import agentdescent.mcp as m

    fake = _Fake1xServer()
    monkeypatch.setattr(m, "build_server", lambda store=None: fake)
    m.serve(transport="streamable-http", host=host, port=9)
    assert fake.ran == "streamable-http"
    assert fake.settings.host == host and fake.settings.port == 9
    assert (fake.settings.transport_security is not None) is keeps_pin


def test_cli_mcp_defaults_match_module():
    """Argparse cannot import mcp.py (circular); keep the literals locked (#589)."""
    from agentdescent import mcp as m
    a = build_parser().parse_args(["mcp"])
    assert a.host == m.DEFAULT_HTTP_HOST and a.port == m.DEFAULT_HTTP_PORT


class _FakePre18Server:
    """mcp 1.2–1.7 shape: no run_streamable_http_async, run only accepts stdio/sse."""

    def __init__(self):
        self.settings = _Fake1xSettings()
        self.ran = None

    def run(self, transport):  # pragma: no cover - must not be reached
        self.ran = transport
        raise AssertionError(f"run({transport!r}) should not be called without streamable-http")


def test_serve_rejects_streamable_http_on_pre_1_8_sdk(monkeypatch):
    import agentdescent.mcp as m

    fake = _FakePre18Server()
    monkeypatch.setattr(m, "build_server", lambda store=None: fake)
    with pytest.raises(RuntimeError, match=r"streamable-http needs mcp>=1\.8"):
        m.serve(transport="streamable-http", host="127.0.0.1", port=9)
    assert fake.ran is None


def test_cmd_mcp_streamable_http_on_pre_1_8_exits_with_actionable_message(monkeypatch, capsys):
    import agentdescent.mcp as m

    fake = _FakePre18Server()
    monkeypatch.setattr(m, "build_server", lambda store=None: fake)
    rc = main(["mcp", "--transport", "streamable-http", "--port", "9"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "mcp>=1.8" in err and "streamable-http" in err
