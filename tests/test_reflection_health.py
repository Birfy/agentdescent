"""The reflector must not fail silently.

A reasoning model spends its token budget on reasoning before emitting anything,
so too small a `max_tokens` returns an *empty* completion rather than a short
answer. The engine reads an empty proposal as "nothing worth changing" and the run
then looks like the framework cannot learn, while the reflector never actually
spoke. Measured on deepseek-v4-flash: at max_tokens=1024 four of eight reflection
prompts came back empty; at 3000, none did.
"""

import warnings

from agentdescent.agents import claude, echo, openai_compatible
from agentdescent.evolution import LLMAgent, Task


def _agent(reply):
    return LLMAgent(echo(lambda p: reply))


def test_empty_completion_warns_with_the_likely_cause():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _agent("").propose("art", Task(id="t", prompt="p"), "out", 0.0)
    assert out is None
    msg = " ".join(str(x.message) for x in w)
    assert "empty completion" in msg
    assert "max_tokens" in msg, "the warning must name the likely fix"


def test_whitespace_only_counts_as_empty():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _agent("   \n\t ").propose("a", Task(id="t", prompt="p"), "o", 0.0) is None
    assert any("empty completion" in str(x.message) for x in w)


def test_a_literal_none_answer_is_not_a_warning():
    """"NONE" is the documented way to say no rule helps -- a real answer."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _agent("NONE").propose("a", Task(id="t", prompt="p"), "o", 0.0) is None
    assert not [x for x in w if "empty completion" in str(x.message)]


def test_a_real_proposal_passes_through_unwarned():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _agent("Always answer in cents.").propose(
            "a", Task(id="t", prompt="p"), "o", 0.0)
    assert out == "Always answer in cents."
    assert not w


def test_the_warning_does_not_repeat_on_every_call():
    """A starved reflector fails every time; the warning must not spam the log."""
    agent = _agent("")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(20):
            agent.propose("a", Task(id="t", prompt="p"), "o", 0.0)
    assert 1 <= len([x for x in w if "empty completion" in str(x.message)]) <= 3


def test_default_max_tokens_is_generous_enough_for_a_reasoning_model():
    """1024 starved deepseek-v4-flash half the time; billing is per token
    generated, not per cap, so a high default costs nothing."""
    import inspect

    for fn in (claude, openai_compatible):
        default = inspect.signature(fn).parameters["max_tokens"].default
        assert default >= 4096, f"{fn.__name__} default max_tokens={default} is too low"
