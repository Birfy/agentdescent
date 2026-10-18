"""A spec-built model adapter reports into the run's ``Usage`` (#405).

``runstore.execute`` makes its own ``Usage`` and hands it to ``compose`` ->
``evolve``; the engine's meter reads tokens from it. The adapters a spec builds
(``openai_compatible``, ``claude``) only learn their ``Usage`` through their
constructor, so before the fix a spec run reported ``tokens=0`` and
``evolve.max_tokens`` could never fire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentdescent import runstore
from agentdescent.agents import Usage
from agentdescent.evolvespec import EvolveSpec, compose
from agentdescent.workspec import Ref

PROMPT_TOKENS, COMPLETION_TOKENS = 30, 20


RECEIVED = []


def make_counting_model(*, usage=None, tag="model"):
    """A spec-nameable factory that, like ``openai_compatible``, takes ``usage=``."""
    RECEIVED.append((tag, usage))
    return lambda prompt: "ok"


def make_cli_like(*, usage=None):
    RECEIVED.append(("cli", usage))
    return lambda prompt: "ok"


class _Fake(BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self):  # noqa: N802
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({
            "choices": [{"message": {"content": "Be helpful and answer with the word ok."}}],
            "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D401
        pass


@pytest.fixture
def fake_endpoint(monkeypatch):
    _Fake.hits = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("FAKE_KEY", "sk-not-real")
    monkeypatch.setenv("FAKE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield
    server.shutdown()
    server.server_close()


def _spec(**evolve):
    model = {"ref": "openai_compatible", "model": "m", "api_key_env": "FAKE_KEY",
             "base_url_env": "FAKE_BASE_URL", "retries": 1}
    return EvolveSpec(
        kind="text", target="You are a helpful assistant.",
        data={"inline": [{"prompt": f"q{i}", "gold": "zzz"} for i in range(8)]},
        agent=model, reflect=dict(model),
        evolve={"rounds": 4, "n_workers": 2, "seed": 0, "patience": 10, **evolve})


def _run(tmp_path, spec):
    rd = runstore.create(spec.to_dict(), store=str(tmp_path / "runs"))
    runstore.execute(rd)
    return rd.status()


# -- the injection itself -----------------------------------------------------


def test_compose_hands_the_run_usage_to_agent_reflect_and_nested_refs():
    RECEIVED.clear()
    u = Usage()
    spec = EvolveSpec(
        kind="text", target="x", data={"inline": [{"prompt": "q", "gold": "ok"}] * 8},
        agent={"ref": "tests.test_spec_usage:make_counting_model", "tag": "agent"},
        reflect={"ref": "tests.test_spec_usage:make_counting_model", "tag": "reflect"},
        policies={"reflective_merge": {
            "ref": "agentdescent.fusion:reflective_merge",
            "complete": {"ref": "tests.test_spec_usage:make_counting_model", "tag": "merge"}}},
        allow=["tests."])
    compose(spec, usage=u)
    assert {tag: got.target is u for tag, got in RECEIVED} == {
        "agent": True, "reflect": True, "merge": True}


def test_without_a_run_usage_nothing_is_injected():
    RECEIVED.clear()
    spec = EvolveSpec(
        kind="text", target="x", data={"inline": [{"prompt": "q", "gold": "ok"}] * 8},
        agent={"ref": "tests.test_spec_usage:make_counting_model", "tag": "agent"},
        allow=["tests."])
    compose(spec)                                     # what `plan` does
    assert RECEIVED == [("agent", None)]


def test_which_factories_get_the_usage():
    from agentdescent import agents
    from agentdescent.workspec import _takes_usage

    assert _takes_usage(agents.openai_compatible) and _takes_usage(agents.claude)
    # Calls and seconds only: sharing the run's Usage would double-count calls
    # (and halve `max_calls`) for no tokens.
    assert not _takes_usage(agents.cli_agent)
    assert not _takes_usage(agents.claude_code)       # **kwargs wrapper
    assert not _takes_usage(lambda: None)


def test_a_cli_agent_ref_is_left_alone():
    RECEIVED.clear()
    from agentdescent.workspec import resolve_ref

    resolve_ref(Ref("tests.test_spec_usage:make_cli_like", {}), ("tests.",), usage=Usage())
    assert RECEIVED[0][1] is not None                 # a factory that asks for one gets it
    agent = resolve_ref(Ref("agentdescent.agents:cli_agent", {"command": ["true"]}),
                        ("agentdescent.",), usage=Usage())
    assert agent.usage is None


# -- end to end through the run store -----------------------------------------


def test_status_tokens_count_what_the_adapters_were_billed(tmp_path, fake_endpoint):
    st = _run(tmp_path, _spec())
    assert st.state == "done", st
    assert st.tokens > 0
    # every request the fake endpoint answered was billed, none twice ...
    assert st.tokens == _Fake.hits * (PROMPT_TOKENS + COMPLETION_TOKENS)
    # ... and `calls` is still one per actor invocation, not per provider
    # attempt on top: max_calls and usd_per_call budgets keep their meaning.
    assert st.calls == _Fake.hits


def test_evolve_max_tokens_stops_the_run(tmp_path, fake_endpoint):
    st = _run(tmp_path, _spec(max_tokens=PROMPT_TOKENS + COMPLETION_TOKENS))
    assert st.state == "done", st
    assert st.stop_reason == "max_tokens", st
    assert st.round < 4


def test_token_sink_adds_tokens_and_nothing_else():
    u = Usage()
    sink = u.token_sink()
    sink.record(prompt_tokens=3, completion_tokens=4, seconds=9.0, failed=True)
    sink.record(seconds=1.0, failed=True)            # a failed attempt reports no tokens
    assert (u.prompt_tokens, u.completion_tokens) == (3, 4)
    assert (u.calls, u.seconds, u.failures) == (0, 0.0, 0)


def test_the_python_api_still_shares_the_usage_directly(fake_endpoint):
    from agentdescent.agents import openai_compatible

    u = Usage()
    c = openai_compatible("m", api_key_env="FAKE_KEY", base_url_env="FAKE_BASE_URL", usage=u)
    c("hi")
    assert (u.calls, u.prompt_tokens, u.completion_tokens) == (1, PROMPT_TOKENS, COMPLETION_TOKENS)
