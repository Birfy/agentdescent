"""A quota-exhausted 429 fails on the first call (#653).

Ark ``AccountQuotaExceeded`` and OpenAI ``insufficient_quota`` used to map
to ``RateLimited`` and burn minutes of backoff. They are permanent until a
human acts, so they must raise ``QuotaExhausted`` like other
``ProviderConfigError``s. Real throttling keeps the retry path.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentdescent import QuotaExhausted, ProviderConfigError, diagnose_payload, runstore
from agentdescent.agents import RateLimited, openai_compatible, anthropic_compatible
from agentdescent.evolvespec import EvolveSpec

OK = {"choices": [{"message": {"content": "ok"}}],
      "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

ARK_QUOTA = {
    "error": {
        "code": "AccountQuotaExceeded",
        "message": ("You have exceeded the weekly usage quota. "
                    "It will reset at 2026-11-02 00:00:00 +0800 CST."),
        "type": "TooManyRequests",
    }
}
OPENAI_QUOTA = {
    "error": {
        "code": "insufficient_quota",
        "message": "You exceeded your current quota, please check your plan.",
        "type": "insufficient_quota",
    }
}
RATE_LIMIT = {
    "error": {
        "code": "rate_limit_exceeded",
        "message": "Rate limit reached for requests.",
        "type": "too_many_requests",
    }
}


class _Handler(BaseHTTPRequestHandler):
    body: bytes = b"{}"
    code: int = 429
    retry_after: str = ""
    hits = 0
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with type(self).lock:
            type(self).hits += 1
        if type(self).code == 200:
            payload = json.dumps(OK).encode()
        else:
            payload = type(self).body
        self.send_response(type(self).code)
        if type(self).retry_after:
            self.send_header("Retry-After", type(self).retry_after)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # noqa: D401
        pass


@pytest.fixture
def endpoint(monkeypatch):
    _Handler.hits = 0
    _Handler.code = 429
    _Handler.body = b"{}"
    _Handler.retry_after = ""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "sk-not-real")
    monkeypatch.setenv("FAKE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")

    def use(*, code=429, body=None, retry_after=""):
        _Handler.code = code
        _Handler.body = json.dumps(body if body is not None else {}).encode()
        _Handler.retry_after = retry_after
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
    except Exception:  # noqa: BLE001 - status is what we assert on
        pass
    return rd, store


@pytest.mark.parametrize("body,marker", [
    (ARK_QUOTA, "AccountQuotaExceeded"),
    (OPENAI_QUOTA, "insufficient_quota"),
])
def test_quota_429_fails_on_first_attempt(endpoint, body, marker):
    endpoint(body=body)
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL",
                          retries=3)
    with pytest.raises(QuotaExhausted) as info:
        c("hi")
    assert _Handler.hits == 1, "with_retries must not back off a quota 429"
    assert marker in str(info.value)
    assert "quota for this model is used up" in str(info.value)
    assert "FAKE_KEY" in str(info.value)
    assert isinstance(info.value, ProviderConfigError)
    assert isinstance(info.value, RuntimeError)


def test_rate_limit_exceeded_still_retries(endpoint, monkeypatch):
    import agentdescent.agents as agents
    monkeypatch.setattr(agents.time, "sleep", lambda s: None)
    # First two are throttling; third succeeds — same path as before #653.
    class _Scripted(_Handler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            with type(self).lock:
                i = type(self).hits
                type(self).hits += 1
            if i < 2:
                payload = json.dumps(RATE_LIMIT).encode()
                code, retry = 429, "0"
            else:
                payload = json.dumps(OK).encode()
                code, retry = 200, ""
            self.send_response(code)
            if retry:
                self.send_header("Retry-After", retry)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    # Replace the fixture server with a scripted one for this case.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import os
    os.environ["FAKE_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
    _Scripted.hits = 0
    try:
        c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL",
                              retries=3)
        assert c("hi") == "ok"
        assert _Scripted.hits == 3
    finally:
        server.shutdown()
        server.server_close()


def test_rate_limit_exceeded_raises_rate_limited_not_quota(endpoint):
    endpoint(body=RATE_LIMIT, retry_after="5")
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL",
                          retries=1)
    with pytest.raises(RateLimited) as info:
        c("hi")
    assert not isinstance(info.value, QuotaExhausted)
    assert info.value.retry_after == 5.0


def test_anthropic_compatible_quota_fails_fast_too(endpoint):
    endpoint(body=ARK_QUOTA)
    c = anthropic_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL",
                             retries=3)
    with pytest.raises(QuotaExhausted):
        c("hi")
    assert _Handler.hits == 1


def test_engine_stops_after_one_quota_call(tmp_path, endpoint):
    endpoint(body=ARK_QUOTA)
    rd, store = _run(tmp_path, _spec(rounds=2, n_workers=4))
    st = rd.status()
    assert st.state == "failed", st
    # Only in-flight calls when the first refusal landed — not minutes of retries.
    assert _Handler.hits <= 4, _Handler.hits
    assert "QuotaExhausted" in (st.error or "") or \
        "AccountQuotaExceeded" in (st.error or ""), st.error
    assert "quota for this model is used up" in (st.error or ""), st.error
    report = diagnose_payload(rd.run_id if hasattr(rd, "run_id") else st.run_id,
                              store=store)
    assert report.verdict == "run-failed"
    assert any("quota" in r.lower() for r in report.recommendations)
    assert not any("doctor" in r for r in report.recommendations)
