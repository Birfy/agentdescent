"""Three things a caller hits directly, each of which failed quietly.

* a provider that answered `content: null` broke the one contract in the package
  (`Completion` is prompt -> str) and the failure was then *retried as a backend
  transient*;
* flipping `asynchronous=True` turned an unbounded run into a 20-second one, and
  the truncated result was indistinguishable from a converged one;
* the train/held-out split is positional, so grouped data holds out one end of
  the file and every gate in the run is then measured against a different
  distribution.
"""

import io
import json
import os
import urllib.error
import urllib.request
import warnings

import pytest

import agentdescent.evolution as ev
from agentdescent.agents import openai_compatible
from agentdescent.evolution import LLMAgent, SingleSlot, Task, evolve


# -- the provider contract ------------------------------------------------------


def _stub_response(payload, monkeypatch):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **kw: _Resp(json.dumps(payload).encode()))


def test_a_reasoning_model_that_returns_null_content_still_returns_a_string(monkeypatch):
    """`Completion` is `prompt -> str`. DeepSeek's reasoner and GLM's thinking
    modes answer with `content: null` and the visible text in `reasoning_content`
    when the budget goes to reasoning."""
    _stub_response({"choices": [{"message": {"role": "assistant", "content": None,
                                             "reasoning_content": "..."}}]}, monkeypatch)
    out = openai_compatible(model="deepseek-reasoner", retries=1)("hi")
    assert out == "", f"expected a string, got {out!r}"


def test_null_content_surfaces_as_the_empty_completion_warning(monkeypatch):
    """It used to raise `'NoneType' object has no attribute 'strip'` from inside
    `LLMAgent`, which the engine caught and retried as a *backend transient* --
    diagnosing a systematic model/parameter mismatch as a flaky endpoint.

    `LLMAgent.propose` already has the right diagnosis for this; it just never
    got the chance to run.
    """
    _stub_response({"choices": [{"message": {"content": None}}]}, monkeypatch)
    agent = LLMAgent(openai_compatible(model="deepseek-reasoner", retries=1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert agent.propose("skill", Task(id="1", prompt="q"), "out", 0.0) is None
    assert any("empty completion" in str(w.message) for w in caught), \
        "the reflector's own empty-completion diagnosis never fired"


def test_an_http_error_carries_the_provider_s_message(monkeypatch):
    """The body is the only useful part, and it lives on `e.read()`."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def boom(*a, **kw):
        raise urllib.error.HTTPError(
            "u", 429, "Too Many Requests", {},
            io.BytesIO(b'{"error":{"message":"rate limit: retry in 12s"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as excinfo:
        openai_compatible(model="m", retries=1)("hi")
    assert "429" in str(excinfo.value)
    assert "retry in 12s" in str(excinfo.value), "the provider's message was discarded"


def test_a_200_with_no_choices_names_the_endpoint(monkeypatch):
    """Some proxies answer HTTP 200 with `{"error": ...}`."""
    _stub_response({"error": {"message": "model not found"}}, monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        openai_compatible(model="nope", retries=1)("hi")
    assert "no choices" in str(excinfo.value) and "nope" in str(excinfo.value)


def test_extra_kwargs_reach_the_request_body(monkeypatch):
    """`claude()` takes **create_kwargs; this had no way to set temperature."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sent = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def capture(req, **kw):
        sent.update(json.loads(req.data))
        return _Resp(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", capture)
    openai_compatible(model="m", retries=1, temperature=0)("hi")
    assert sent.get("temperature") == 0


# -- the silent async redefinitions ---------------------------------------------


def _tasks(n=8):
    return [Task(id=str(i), prompt=str(i), meta={"gold": "y"}) for i in range(n)]


def _quick(**kw):
    return evolve(_tasks(), lambda t, o: 1.0, run=lambda r, t: "y",
                  propose=lambda *a: None, strategy=SingleSlot(initial_value="v"),
                  n_workers=2, held_out_frac=0.5, **kw)


def test_async_says_that_an_unset_max_seconds_becomes_twenty():
    """`None` means "no limit" on one path and "20 seconds" on the other."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _quick(rounds=1, asynchronous=True)
    assert any("20.0 seconds" in str(w.message) for w in caught), \
        "flipping one boolean silently bounded the run"


def test_async_says_that_rounds_changes_meaning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _quick(rounds=3, asynchronous=True, max_seconds=1.0)
    assert any("rollouts" in str(w.message) and "rounds=3" in str(w.message)
               for w in caught)


def test_stop_reason_tells_convergence_from_a_spent_budget():
    """`error` is None for both, and `history` has entries for both."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converged = _quick(rounds=20, target_reward=0.5)
        exhausted = _quick(rounds=2)
    assert converged.stop_reason == "target_reward"
    assert exhausted.stop_reason == "rounds"
    assert converged.error is exhausted.error is None, \
        "premise: `error` cannot distinguish these two"


def test_stop_reason_survives_save_and_load(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _quick(rounds=20, target_reward=0.5)
    path = tmp_path / "r.json"
    res.save(str(path))
    from agentdescent.evolution import EvolutionResult
    assert EvolutionResult.load(str(path)).stop_reason == "target_reward"


# -- the positional split -------------------------------------------------------


def _split(tasks, **kw):
    eng = ev._build_engine(
        tasks, lambda t, o: 1.0, agent=None, run=lambda r, t: "y",
        propose=lambda *a: None, strategy=SingleSlot(initial_value="v"),
        initial_state=None, blast_radius=0.2, artifact_id="a", held_out_frac=0.4,
        repo_path=None, agg_config=None, staleness_policy=None,
        aggregator_factory=None, oracle_budget=10, **kw)
    return eng.train, eng.held_out


def _grouped(n=20):
    """A dataset ordered by class -- FiNER, SWE-bench and friends all are."""
    return [Task(id=str(i), prompt="a" if i < 12 else "b", meta={"gold": "y"})
            for i in range(n)]


def test_an_ordered_dataset_holds_out_one_end_of_the_file_by_default():
    """Documented, not fixed by default: the split is positional on purpose, so
    `Dataset.val_frac` keeps meaning "exactly this Dataset's val"."""
    _, held_out = _split(_grouped())
    assert {t.prompt for t in held_out} == {"b"}, \
        "premise: the tail of a grouped file is one class"


def test_shuffle_makes_the_held_out_set_representative():
    _, held_out = _split(_grouped(), shuffle=True, seed=7)
    assert {t.prompt for t in held_out} == {"a", "b"}, \
        "every gate in the run is measured on this set"


def test_shuffling_is_deterministic_given_a_seed():
    a, _ = _split(_grouped(), shuffle=True, seed=3)
    b, _ = _split(_grouped(), shuffle=True, seed=3)
    c, _ = _split(_grouped(), shuffle=True, seed=4)
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.id for t in a] != [t.id for t in c]


def test_a_tiny_held_out_set_is_called_out():
    """At 1 item `final_reward` is 0.0 or 1.0 and nothing else."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _split([Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(4)])
    assert any("held-out set has only" in str(w.message) for w in caught)
