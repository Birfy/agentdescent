"""A reflector timeout names the endpoint, model, seconds and fix (#467).

Before: a reasoning model behind ``openai_compatible`` (Ark ``doubao-seed-code``
thinks by default) ran past the 120 s read timeout and the only thing stored in
``propose_error`` was ``TimeoutError: The read operation timed out``.
"""

import socket
import threading
import urllib.error

import pytest

from agentdescent import agents
from agentdescent.agents import anthropic_compatible, openai_compatible

from tests.test_propose_errors import _run


@pytest.fixture
def silent_base(monkeypatch):
    """An endpoint that accepts the connection and never answers."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held, stop = [], threading.Event()

    def accept():
        srv.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                held.append(conn)
            except OSError:
                continue

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    monkeypatch.setenv("SILENT_KEY", "k")
    yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    stop.set()
    t.join(1)
    for c in held:
        c.close()
    srv.close()


def test_openai_compatible_read_timeout_is_actionable(silent_base):
    comp = openai_compatible("doubao-seed-code", api_key_env="SILENT_KEY",
                             base_url_env="UNSET_SILENT_BASE",
                             default_base_url=silent_base, timeout=0.3, retries=1)
    with pytest.raises(RuntimeError) as ei:
        comp("hi")
    msg = str(ei.value)
    assert silent_base in msg and "'doubao-seed-code'" in msg
    assert "0.3s" in msg and '"stream": true' in msg and '"timeout"' in msg
    assert "thinking" in msg
    assert agents._is_timeout(ei.value.__cause__)


def test_streaming_timeout_does_not_suggest_stream_again(silent_base):
    comp = openai_compatible("m", api_key_env="SILENT_KEY", base_url_env="UNSET_SILENT_BASE",
                             default_base_url=silent_base, timeout=0.3, retries=1,
                             stream=True)
    with pytest.raises(RuntimeError, match=r"did not answer within 0\.3s") as ei:
        comp("hi")
    assert '"stream": true' not in str(ei.value)


def test_anthropic_compatible_read_timeout_is_actionable(silent_base):
    comp = anthropic_compatible("m", api_key_env="SILENT_KEY", base_url_env="UNSET_SILENT_BASE",
                                default_base_url=silent_base, timeout=0.3, retries=1)
    with pytest.raises(RuntimeError, match=r"did not answer within 0\.3s for model 'm'") as ei:
        comp("hi")
    assert silent_base in str(ei.value)


def test_connect_timeout_wrapped_in_urlerror_is_recognised(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))
    monkeypatch.setattr(agents.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILENT_KEY", "k")
    comp = openai_compatible("m", api_key_env="SILENT_KEY", base_url_env="UNSET_SILENT_BASE",
                             default_base_url="https://h/v3", timeout=5, retries=1)
    with pytest.raises(RuntimeError, match=r"https://h/v3 did not answer within 5s"):
        comp("hi")


def test_other_urlerror_is_unchanged(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))
    monkeypatch.setattr(agents.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILENT_KEY", "k")
    comp = openai_compatible("m", api_key_env="SILENT_KEY", base_url_env="UNSET_SILENT_BASE",
                             default_base_url="https://h/v3", timeout=5, retries=1)
    with pytest.raises(urllib.error.URLError):
        comp("hi")


def test_timeout_message_reaches_propose_error(silent_base):
    comp = openai_compatible("doubao-seed-code", api_key_env="SILENT_KEY",
                             base_url_env="UNSET_SILENT_BASE",
                             default_base_url=silent_base, timeout=0.2, retries=1)
    res, _ = _run(lambda task: comp("reflect"), rounds=1)
    errs = [h.propose_error for h in res.history if h.propose_error]
    assert errs and errs[0].startswith("RuntimeError: ")
    assert "did not answer within 0.2s" in errs[0] and "doubao-seed-code" in errs[0]
