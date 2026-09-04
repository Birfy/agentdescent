"""Tests for the provider-agnostic agent/inference layer."""

import io

import pytest

from agentdescent.agents import claude, echo, from_callable, with_retries


def test_echo_returns_prompt_or_transform():
    assert echo()("hello") == "hello"
    assert echo(str.upper)("hello") == "HELLO"


def test_from_callable_is_identity():
    fn = lambda p: p[::-1]
    assert from_callable(fn)("abc") == "cba"


def test_with_retries_recovers_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    out = with_retries(flaky, attempts=3, backoff=0, sleep=lambda s: None)("x")
    assert out == "ok" and calls["n"] == 3


def test_with_retries_reraises_after_exhaustion():
    def always_fail(prompt):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retries(always_fail, attempts=2, backoff=0, sleep=lambda s: None)("x")


def test_claude_adapter_wiring_with_injected_client():
    # a fake anthropic client, so no network / no real key is needed.
    class _Block:
        type = "text"
        text = "answer"

    class _Msg:
        content = [_Block()]

    class _Messages:
        def create(self, **kw):
            assert kw["model"] == "claude-haiku-4-5"
            return _Msg()

    class _Client:
        messages = _Messages()

    complete = claude(model="claude-haiku-4-5", client=_Client())
    assert complete("hi") == "answer"


# ---------------------------------------------------------------------------
# Every blocking boundary has a bound
# ---------------------------------------------------------------------------


def test_claude_bounds_a_single_request():
    """A hung endpoint must fail, not stall the run.

    `claude()` was the one blocking boundary in the package with no timeout:
    `_git` has 120s, `_CliAgent` 600s, `runners._sh` takes one per call, and
    `openai_compatible` has had 120s all along. Without one the SDK's 600s
    default applies, the SDK retries internally, and `with_retries` retries that
    -- so one logical call against a stalled endpoint blocks for over half an
    hour while the log says nothing.

    Measured before the fix: a GEPA run sat 51 minutes on a single round with
    1.07s of CPU and one ESTABLISHED socket.
    """
    from agentdescent.agents import claude

    seen = {}

    class _Messages:
        def create(self, **kw):
            seen.update(kw)
            return type("M", (), {"content": [], "usage": None})()

    class _Client:
        messages = _Messages()

    claude(model="m", client=_Client(), retries=1)("hi")
    assert "timeout" in seen, "claude() sent no timeout; a hung request stalls forever"
    assert seen["timeout"] == 120.0

    claude(model="m", client=_Client(), retries=1, timeout=7.5)("hi")
    assert seen["timeout"] == 7.5, "an explicit timeout must win"


def test_both_provider_adapters_bound_a_request():
    """The two adapters are alternatives for the same job, so a caller should not
    have to know which one silently has no bound."""
    import inspect

    from agentdescent.agents import claude, openai_compatible

    for fn in (claude, openai_compatible):
        assert "timeout" in inspect.signature(fn).parameters, fn.__name__

# ---------------------------------------------------------------------------
# Streaming: the transport a reasoning model behind a gateway needs
# ---------------------------------------------------------------------------


class _FakeStream:
    """A minimal stand-in for the file object `urlopen` returns on an SSE body."""

    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


SSE_LINES = [
    'data: {"choices":[{"delta":{"reasoning_content":"thinking"},"index":0}]}\n',
    'data: {"choices":[{"delta":{"content":"Hello "},"index":0}]}\n',
    "\n",
    "event: ping\n",
    "data: {not json at all}\n",
    'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop",'
    '"index":0}],"usage":{"prompt_tokens":10,"completion_tokens":5,'
    '"total_tokens":15}}\n',
    "data: [DONE]\n",
]


def test_the_sse_reader_rebuilds_the_reply_and_survives_junk():
    """One malformed frame is a provider quirk, not a reason to lose the answer."""
    from agentdescent.agents import _read_sse

    data = _read_sse(_FakeStream(SSE_LINES))
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 15


def test_streaming_asks_for_usage_and_still_records_it(monkeypatch):
    """`stream=True` must not quietly turn token accounting off.

    A streamed response carries usage only if the request asked for it, so the
    adapter sends `stream_options.include_usage` and reads the final chunk. An
    endpoint that ignores the field reports zero rather than failing.
    """
    import json

    from agentdescent.agents import Usage, openai_compatible
    import agentdescent.agents as agents

    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data.decode())
        sent["timeout"] = timeout
        return _FakeStream(SSE_LINES)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents.urllib.request, "urlopen", fake_urlopen)
    meter = Usage()
    reply = openai_compatible(model="m", usage=meter, retries=1, stream=True)("hi")
    assert reply == "Hello world"
    assert sent["body"]["stream"] is True
    assert sent["body"]["stream_options"] == {"include_usage": True}
    assert meter.total_tokens == 15


def test_the_plain_path_is_untouched_by_the_streaming_option(monkeypatch):
    """Streaming is opt-in; the default request must not grow a `stream` field."""
    import io
    import json

    from agentdescent.agents import openai_compatible
    import agentdescent.agents as agents

    sent = {}

    class _Plain(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data.decode())
        return _Plain(json.dumps({
            "choices": [{"message": {"content": "plain"}}],
            "usage": {"total_tokens": 3},
        }).encode())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents.urllib.request, "urlopen", fake_urlopen)
    assert openai_compatible(model="m", retries=1)("hi") == "plain"
    assert "stream" not in sent["body"]


def test_a_rate_limit_backs_off_on_its_own_schedule():
    """Retrying a 429 in half a second is three attempts spent in two seconds.

    Measured against a throttling endpoint: the generic schedule burned every
    attempt inside two seconds, the engine retired each worker after three
    consecutive failures, and a twenty-expansion run ended with an empty tree.
    """
    from agentdescent.agents import RateLimited, with_retries

    slept = []

    def limited(_prompt):
        raise RateLimited("429 too many requests")

    with pytest.raises(RateLimited):
        with_retries(limited, attempts=3, sleep=slept.append)("hi")
    assert slept == [5.0, 10.0], slept

    def generic(_prompt):
        raise RuntimeError("connection reset")

    other = []
    with pytest.raises(RuntimeError):
        with_retries(generic, attempts=3, sleep=other.append)("hi")
    assert other == [0.5, 1.0], "a transport blip must not wait like a limiter"


def test_retry_after_wins_and_is_capped():
    """The provider knows better than the schedule -- up to a point."""
    from agentdescent.agents import RateLimited, with_retries

    slept = []
    with pytest.raises(RateLimited):
        with_retries(lambda _p: (_ for _ in ()).throw(
            RateLimited("429", retry_after=23.0)), attempts=2,
            sleep=slept.append)("hi")
    assert slept == [23.0]

    capped = []
    with pytest.raises(RateLimited):
        with_retries(lambda _p: (_ for _ in ()).throw(
            RateLimited("429", retry_after=9999.0)), attempts=2,
            sleep=capped.append, max_sleep=60.0)("hi")
    assert capped == [60.0], "a hostile Retry-After must not park a worker"


def test_a_429_is_raised_as_a_rate_limit_with_its_header_read(monkeypatch):
    """Only the status code separates "slow down" from "you are misconfigured"."""
    import urllib.error

    from agentdescent.agents import RateLimited, openai_compatible
    import agentdescent.agents as agents

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests",
            {"Retry-After": "12"}, io.BytesIO(b'{"error":"rate limit"}'))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agents.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RateLimited) as caught:
        openai_compatible(model="m", retries=1)("hi")
    assert caught.value.retry_after == 12.0
    assert "429" in str(caught.value)

    def fake_400(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"bad model"}'))

    monkeypatch.setattr(agents.urllib.request, "urlopen", fake_400)
    with pytest.raises(RuntimeError) as plain:
        openai_compatible(model="m", retries=1)("hi")
    assert not isinstance(plain.value, RateLimited)
