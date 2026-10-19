"""A spec's ``base_url`` / ``api_key`` must not ride into the request body (#422).

``openai_compatible`` / ``anthropic_compatible`` take ``**create_kwargs`` and put
every unknown keyword in the JSON body, so ``"base_url": "https://..."`` in a spec
was accepted, ignored, and the call went to the default host with the other
provider's key in ``Authorization``. Offline: ``urlopen`` is faked, no network."""

from __future__ import annotations

import io
import json

import pytest

import agentdescent.agents as agents
from agentdescent import cli
from agentdescent.agents import anthropic_compatible, openai_compatible
from agentdescent.cli import _unusable_refs
from agentdescent.evolvespec import EvolveSpec
from agentdescent.mcp import Tools

OPENAI = "openai_compatible"
ANTHROPIC = "agentdescent.agents:anthropic_compatible"
ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
       "ARK_API_KEY", "ARK_BASE_URL", "MY_OPENAI_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def _spec_dict(tmp_path, **blocks):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n' * 8, encoding="utf-8")
    target = tmp_path / "prompt.txt"
    target.write_text("hi\n", encoding="utf-8")
    return {"kind": "text", "target": str(target),
            "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
            "score": "contains", "agent": {"ref": "echo"}, **blocks}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(monkeypatch, reply, sent):
    def urlopen(req, timeout=None):
        sent.append((req.full_url, json.loads(req.data.decode()), dict(req.header_items())))
        return _Resp(json.dumps(reply).encode())

    monkeypatch.setattr(agents.urllib.request, "urlopen", urlopen)


# -- the adapters refuse connection parameters that are not theirs ----------------


@pytest.mark.parametrize("factory", [openai_compatible, anthropic_compatible])
@pytest.mark.parametrize("name", ["base_url", "api_base", "api_url", "base", "url", "endpoint"])
def test_a_url_keyword_names_the_real_spelling(factory, name):
    with pytest.raises(TypeError) as e:
        factory("m", **{name: "https://x"})
    msg = str(e.value)
    assert f"`{name}`" in msg and "default_base_url" in msg and "base_url_env" in msg


@pytest.mark.parametrize("factory", [openai_compatible, anthropic_compatible])
@pytest.mark.parametrize("name", ["api_key", "apikey", "key"])
def test_a_key_keyword_names_api_key_env_and_never_echoes_the_value(factory, name):
    with pytest.raises(TypeError) as e:
        factory("m", **{name: "sk-secret-value"})
    assert f"`{name}`" in str(e.value) and "api_key_env" in str(e.value)
    assert "sk-secret-value" not in str(e.value)


def test_ordinary_body_extras_still_pass_through(monkeypatch):
    sent = []
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _fake_urlopen(monkeypatch, {"choices": [{"message": {"content": "o"}}]}, sent)
    assert openai_compatible("m", retries=1, temperature=0,
                             thinking={"type": "disabled"})("hi") == "o"
    _fake_urlopen(monkeypatch, {"content": [{"type": "text", "text": "a"}]}, sent)
    assert anthropic_compatible("m", retries=1, temperature=0,
                                thinking={"type": "disabled"})("hi") == "a"
    for _, body, _ in sent:
        assert body["temperature"] == 0 and body["thinking"] == {"type": "disabled"}


def test_the_documented_spellings_reach_the_right_host_with_the_right_key(monkeypatch):
    sent = []
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    _fake_urlopen(monkeypatch, {"choices": [{"message": {"content": "o"}}]}, sent)
    openai_compatible("m", retries=1, api_key_env="ARK_API_KEY",
                      default_base_url="https://ark.example/api/v3")("hi")
    url, body, headers = sent[0]
    assert url == "https://ark.example/api/v3/chat/completions"
    assert headers["Authorization"] == "Bearer ark-key"
    assert not {"base_url", "api_key"} & set(body)
    # base_url_env overrides default_base_url
    monkeypatch.setenv("ARK_BASE_URL", "https://other.example/v1")
    openai_compatible("m", retries=1, api_key_env="ARK_API_KEY", base_url_env="ARK_BASE_URL",
                      default_base_url="https://ark.example/api/v3")("hi")
    assert sent[1][0] == "https://other.example/v1/chat/completions"


# -- plan / evolve / MCP start refuse the spec, by field --------------------------


@pytest.mark.parametrize("ref", [OPENAI, ANTHROPIC])
@pytest.mark.parametrize("field", ["agent", "reflect"])
def test_plan_refuses_a_base_url_on_the_cli(tmp_path, capsys, ref, field):
    spec = _spec_dict(tmp_path, **{field: {"ref": ref, "model": "m", "base_url": "https://x"}})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert cli.main(["plan", str(path)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert field in out["error"] and "`base_url`" in out["error"]
    assert "default_base_url" in out["error"] and "base_url_env" in out["error"]


@pytest.mark.parametrize("ref", [OPENAI, ANTHROPIC])
def test_plan_refuses_a_base_url_over_mcp(tmp_path, ref):
    spec = _spec_dict(tmp_path, reflect={"ref": ref, "model": "m", "base_url": "https://x"})
    out = Tools(str(tmp_path / "runs")).plan(spec)
    assert out["ok"] is False and "`base_url`" in out["error"]
    assert "default_base_url" in out["error"]


def test_evolve_refuses_before_creating_a_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    spec = _spec_dict(tmp_path, agent={"ref": OPENAI, "model": "m", "api_key": "sk-x"})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert cli.main(["--store", str(tmp_path / "runs"), "evolve", str(path)]) == 2
    err = capsys.readouterr().err
    assert "`api_key`" in err and "api_key_env" in err and "sk-x" not in err
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_plan_still_accepts_the_documented_spelling(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec = _spec_dict(tmp_path, reflect={
        "ref": OPENAI, "model": "m", "api_key_env": "ARK_API_KEY", "temperature": 0,
        "default_base_url": "https://ark.example/api/v3"})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert cli.main(["plan", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and not [w for w in out["warnings"] if "openai_compatible" in w]


# -- plan warns when a non-OpenAI key would go to api.openai.com ------------------


def _spec(tmp_path, **block):
    return EvolveSpec.from_dict(_spec_dict(
        tmp_path, reflect={"ref": OPENAI, "model": "m", **block}))


def test_a_non_openai_key_with_no_endpoint_warns_about_api_openai_com(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    [warning] = _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY"))
    assert warning.startswith("reflect: `openai_compatible` would send ARK_API_KEY to ")
    assert "api.openai.com" in warning and "default_base_url" in warning
    assert "OPENAI_BASE_URL" in warning


def test_the_warning_names_the_blocks_own_base_url_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    [warning] = _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY",
                                     base_url_env="ARK_BASE_URL"))
    assert "ARK_BASE_URL" in warning and "OPENAI_BASE_URL" not in warning


def test_it_reaches_plan_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec_dict(tmp_path, reflect={
        "ref": OPENAI, "model": "m", "api_key_env": "ARK_API_KEY"})))
    assert cli.main(["plan", str(path)]) == 0
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("api.openai.com" in w and "ARK_API_KEY" in w for w in warnings), warnings


def test_no_warning_when_an_endpoint_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    # the spec names the endpoint ...
    assert _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY",
                                default_base_url="https://ark.example/v3")) == []
    # ... or the environment does (and wins over the default)
    monkeypatch.setenv("ARK_BASE_URL", "https://ark.example/v3")
    assert _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY",
                                base_url_env="ARK_BASE_URL")) == []
    assert _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY",
                                base_url_env="ARK_BASE_URL",
                                default_base_url="https://api.openai.com/v1")) == []


def test_no_warning_for_an_openai_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MY_OPENAI_KEY", "k")
    assert _unusable_refs(_spec(tmp_path)) == []
    assert _unusable_refs(_spec(tmp_path, api_key_env="MY_OPENAI_KEY")) == []


def test_an_unset_key_keeps_only_its_own_warning(tmp_path):
    [warning] = _unusable_refs(_spec(tmp_path, api_key_env="ARK_API_KEY"))
    assert "needs ARK_API_KEY" in warning and "would send" not in warning
