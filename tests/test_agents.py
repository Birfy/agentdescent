"""Tests for the provider-agnostic agent/inference layer."""

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
