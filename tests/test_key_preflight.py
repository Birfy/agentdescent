"""`plan` / `evolve` / MCP `start` warn about the key the adapter will actually
read: the block's own ``api_key_env``, not a hard-coded ``OPENAI_API_KEY`` (#406)."""

from __future__ import annotations

import json
import sys

import pytest

from agentdescent import cli
from agentdescent.cli import _unusable_refs
from agentdescent.evolvespec import EvolveSpec

KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "X_KEY", "X_URL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    # the `anthropic` package is a separate preflight item; pretend it is there
    real = cli.doctor_report
    monkeypatch.setattr(cli, "doctor_report", lambda: {
        **real(), "optional": {**real()["optional"], "anthropic": True}})


def _spec(tmp_path, reflect=None, agent=None):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n' * 8, encoding="utf-8")
    target = tmp_path / "prompt.txt"
    target.write_text("hi\n", encoding="utf-8")
    return EvolveSpec.from_dict({
        "kind": "text", "target": str(target),
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains", "agent": agent or {"ref": "echo"},
        **({"reflect": reflect} if reflect is not None else {})})


@pytest.mark.parametrize("ref,default,other", [
    ("openai_compatible", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
])
@pytest.mark.parametrize("field", ["reflect", "agent"])
def test_the_named_key_is_the_one_checked(tmp_path, monkeypatch, ref, default, other, field):
    block = {"ref": ref, "model": "m", "api_key_env": "X_KEY"}
    if ref == "openai_compatible":
        # an endpoint of its own: a non-OpenAI key with none configured is the
        # separate "would send X_KEY to api.openai.com" warning (#422)
        block["default_base_url"] = "https://x.example/v1"
    spec = (_spec(tmp_path, reflect=block) if field == "reflect"
            else _spec(tmp_path, agent=block))

    # false alarm: X_KEY set, the default key not -> quiet
    monkeypatch.setenv("X_KEY", "k")
    assert _unusable_refs(spec) == []

    # missed failure: X_KEY unset, an unrelated default key set -> warns, naming X_KEY
    monkeypatch.delenv("X_KEY")
    monkeypatch.setenv(default, "unrelated")
    warnings = _unusable_refs(spec)
    assert len(warnings) == 1 and "X_KEY" in warnings[0], warnings
    assert default not in warnings[0], warnings
    assert warnings[0].startswith(f"{field}: `{ref}`")


def test_the_hint_names_the_base_url_variable_the_spec_set(tmp_path):
    spec = _spec(tmp_path, reflect={"ref": "openai_compatible", "model": "m",
                                    "api_key_env": "X_KEY", "base_url_env": "X_URL"})
    [warning] = _unusable_refs(spec)
    assert "needs X_KEY (and X_URL for a non-OpenAI endpoint)" in warning
    assert "OPENAI" not in warning


@pytest.mark.parametrize("block", ["openai_compatible",
                                   {"ref": "openai_compatible", "model": "m"},
                                   {"ref": "openai_compatible", "model": "m", "api_key_env": ""},
                                   {"ref": "openai_compatible", "model": "m", "api_key_env": 7}])
def test_default_names_keep_todays_warning_and_silence(tmp_path, monkeypatch, block):
    spec = _spec(tmp_path, reflect=block)
    [warning] = _unusable_refs(spec)
    assert warning == (
        "reflect: `openai_compatible` needs OPENAI_API_KEY (and OPENAI_BASE_URL for a "
        "non-OpenAI endpoint); without it every proposal raises and the run fails after "
        "two rounds with nothing learned")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert _unusable_refs(spec) == []


def test_claude_default_key_is_unchanged(tmp_path, monkeypatch):
    spec = _spec(tmp_path, reflect={"ref": "claude"})
    [warning] = _unusable_refs(spec)
    assert warning.endswith("and needs ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert _unusable_refs(spec) == []


def test_plan_reports_the_named_key(tmp_path, monkeypatch, capsys):
    spec = _spec(tmp_path, reflect={"ref": "openai_compatible", "model": "m",
                                    "api_key_env": "X_KEY",
                                    "default_base_url": "https://x.example/v1"})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    assert cli.main(["plan", str(path)]) in (0, 1)
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("X_KEY" in w for w in warnings), warnings
    monkeypatch.setenv("X_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY")
    cli.main(["plan", str(path)])
    assert not [w for w in json.loads(capsys.readouterr().out)["warnings"]
                if "openai_compatible" in w]


# -- the claude adapter honours the api_key_env the preflight now checks ----------


def _fake_anthropic(monkeypatch, seen):
    class _Messages:
        def create(self, **kw):
            seen["create"] = kw
            return type("M", (), {"content": [type("B", (), {"text": "hi", "type": "text"})()], "usage": None})()

    class Anthropic:
        def __init__(self, **kw):
            seen["init"] = kw
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", type(sys)("anthropic"))
    sys.modules["anthropic"].Anthropic = Anthropic


def test_claude_reads_the_key_named_by_api_key_env(monkeypatch):
    from agentdescent.agents import claude

    seen = {}
    _fake_anthropic(monkeypatch, seen)
    monkeypatch.setenv("X_KEY", "sk-from-x")
    assert claude("m", api_key_env="X_KEY", retries=1)("p") == "hi"
    assert seen["init"] == {"max_retries": 0, "api_key": "sk-from-x"}
    assert "api_key_env" not in seen["create"]       # it used to be sent in the request body


def test_claude_without_api_key_env_is_unchanged_and_unset_key_is_clear(monkeypatch):
    from agentdescent.agents import claude

    seen = {}
    _fake_anthropic(monkeypatch, seen)
    claude("m", retries=1)("p")
    assert seen["init"] == {"max_retries": 0}
    with pytest.raises(RuntimeError, match="set X_KEY"):
        claude("m", api_key_env="X_KEY", retries=1)("p")
