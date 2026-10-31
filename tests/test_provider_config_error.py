"""A deterministic provider refusal stops the run on the first call (#635).

A wrong model name (404 ``UnsupportedModel``) or a key for another endpoint
(401) answers every later request the same way. Before the fix the engine
treated it as a transient and spent a full round of identical failing calls.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentdescent import ProviderConfigError, diagnose_payload, runstore
from agentdescent.agents import openai_compatible, anthropic_compatible
from agentdescent.evolvespec import EvolveSpec

OK = {"choices": [{"message": {"content": "Be helpful and answer with the word ok."}}],
      "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class _Handler(BaseHTTPRequestHandler):
    script: list = []      # status codes to answer, in order; the last one repeats
    hits = 0
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with type(self).lock:
            i = type(self).hits
            type(self).hits += 1
        code = type(self).script[min(i, len(type(self).script) - 1)]
        if code == 200:
            body = json.dumps(OK).encode()
        elif code == 404:
            body = json.dumps({"error": {"code": "UnsupportedModel",
                                         "message": "model not available"}}).encode()
        else:
            body = json.dumps({"error": {"code": "nope", "message": str(code)}}).encode()
        self.send_response(code)
        if code == 429:
            self.send_header("Retry-After", "0")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D401
        pass


@pytest.fixture
def endpoint(monkeypatch):
    _Handler.hits = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "sk-not-real")
    monkeypatch.setenv("FAKE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")

    def use(*codes):
        _Handler.script = list(codes)
        _Handler.hits = 0
    yield use
    server.shutdown()
    server.server_close()


def _spec(retries=3, **evolve):
    model = {"ref": "openai_compatible", "model": "m", "api_key_env": "FAKE_KEY",
             "base_url_env": "FAKE_BASE_URL", "retries": retries}
    return EvolveSpec(
        kind="text", target="You are a helpful assistant.",
        data={"inline": [{"prompt": f"q{i}", "gold": "zzz"} for i in range(8)]},
        agent=model, reflect=dict(model),
        evolve={"rounds": 2, "n_workers": 4, "seed": 0, "patience": 10, **evolve})


def _run(tmp_path, spec):
    store = str(tmp_path / "runs")
    rd = runstore.create(spec.to_dict(), store=store)
    try:
        runstore.execute(rd)
    except Exception:  # noqa: BLE001 - the status is what we assert on
        pass
    return rd, store


def test_404_unsupported_model_is_not_retried(endpoint):
    endpoint(404)
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL")
    with pytest.raises(ProviderConfigError) as info:
        c("hi")
    assert _Handler.hits == 1
    assert "does not serve model 'm'" in str(info.value)
    assert isinstance(info.value, RuntimeError)


def test_401_names_the_api_key_variable(endpoint):
    endpoint(401)
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL")
    with pytest.raises(ProviderConfigError, match="FAKE_KEY"):
        c("hi")
    assert _Handler.hits == 1


def test_anthropic_compatible_fails_fast_too(endpoint):
    endpoint(403)
    c = anthropic_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL")
    with pytest.raises(ProviderConfigError, match="FAKE_KEY"):
        c("hi")
    assert _Handler.hits == 1


def test_a_plain_400_is_still_an_ordinary_error(endpoint):
    endpoint(400)
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL",
                          retries=1)
    with pytest.raises(RuntimeError) as info:
        c("hi")
    assert not isinstance(info.value, ProviderConfigError)


def test_rate_limit_still_retries_and_succeeds(endpoint, monkeypatch):
    import agentdescent.agents as agents
    monkeypatch.setattr(agents.time, "sleep", lambda s: None)
    endpoint(429, 429, 200)
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL")
    assert c("hi")
    assert _Handler.hits == 3


@pytest.mark.parametrize("code", [404, 401])
def test_a_text_run_stops_on_the_first_refusal(tmp_path, endpoint, code):
    endpoint(code)
    rd, store = _run(tmp_path, _spec())
    st = rd.status()
    assert st.state == "failed", st
    # Only the calls already in flight when the first refusal came back.
    assert _Handler.hits <= 4, _Handler.hits
    assert "ProviderConfigError" in (st.error or "") or \
        ("FAKE_KEY" if code == 401 else "does not serve model") in (st.error or ""), st.error
    report = diagnose_payload(rd.run_id if hasattr(rd, "run_id") else st.run_id,
                              store=store)
    assert report.verdict == "run-failed"
    assert not any("log.txt" in r for r in report.recommendations)


def test_a_text_run_survives_rate_limits(tmp_path, endpoint, monkeypatch):
    import agentdescent.agents as agents
    monkeypatch.setattr(agents.time, "sleep", lambda s: None)
    endpoint(429, 429, 200)
    rd, _ = _run(tmp_path, _spec(rounds=1, n_workers=1))
    assert rd.status().state == "done", rd.status()
