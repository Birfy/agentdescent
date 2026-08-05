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
