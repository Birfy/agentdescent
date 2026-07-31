"""Cost accounting.

`Completion = Callable[[str], str]` discards the token counts the providers
actually return, so a run of an expensive model reported nothing about what it
cost. `Usage` collects calls, real tokens, model wall-clock and failures.
"""

import os
import threading

import pytest

from agentdescent.agents import Usage, echo, metered, with_retries


def test_metered_counts_calls_and_time():
    u = Usage()
    m = metered(echo(str.upper), u)
    for _ in range(3):
        assert m("hi") == "HI"
    assert u.calls == 3
    assert u.seconds >= 0.0
    assert u.failures == 0


def test_metered_records_failures_and_reraises():
    u = Usage()

    def boom(prompt):
        raise RuntimeError("provider down")

    m = metered(boom, u)
    for _ in range(2):
        try:
            m("hi")
        except RuntimeError:
            pass
    assert u.calls == 2 and u.failures == 2


def test_metered_counts_every_retry_attempt():
    """Retries are real calls -- they cost money and must show up."""
    u = Usage()
    attempts = {"n": 0}

    def flaky(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    robust = with_retries(metered(flaky, u), attempts=5, backoff=0.0)
    assert robust("hi") == "ok"
    assert u.calls == 3 and u.failures == 2


def test_total_tokens_and_cost():
    u = Usage()
    u.record(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert u.total_tokens == 1_500_000
    # $0.30 per 1M prompt, $1.20 per 1M completion
    assert abs(u.estimated_cost(0.30, 1.20) - (0.30 + 0.60)) < 1e-9


def test_usage_is_thread_safe():
    """Workers share one Usage; increments must not be lost."""
    u = Usage()

    def hammer():
        for _ in range(500):
            u.record(prompt_tokens=1, completion_tokens=2)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert u.calls == 4000
    assert u.prompt_tokens == 4000 and u.completion_tokens == 8000


def test_summary_is_readable():
    u = Usage()
    u.record(prompt_tokens=12, completion_tokens=34, seconds=1.5)
    s = u.summary()
    assert "1 calls" in s and "12" in s and "34" in s


def test_adapters_retry_transient_failures_by_default():
    """A plain `completion(...)` call outside the engine used to have no cover.

    Measured on a real run: one `RemoteDisconnected` during an example's final
    held-out evaluation discarded the whole run's result.
    """
    import itertools

    from agentdescent.agents import openai_compatible

    calls = itertools.count()

    def flaky_urlopen(*a, **kw):
        if next(calls) < 2:
            raise OSError("Remote end closed connection without response")
        raise AssertionError("reached the third attempt")   # proves it retried

    import agentdescent.agents as agents
    real = agents.urllib.request.urlopen
    agents.urllib.request.urlopen = flaky_urlopen
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        model = openai_compatible(model="m", retries=3)
        with pytest.raises(AssertionError, match="third attempt"):
            model("hi")
    finally:
        agents.urllib.request.urlopen = real


def test_retries_can_be_switched_off():
    from agentdescent.agents import openai_compatible
    import inspect
    assert inspect.signature(openai_compatible).parameters["retries"].default == 3
    assert inspect.signature(
        __import__("agentdescent.agents", fromlist=["claude"]).claude
    ).parameters["retries"].default == 3
