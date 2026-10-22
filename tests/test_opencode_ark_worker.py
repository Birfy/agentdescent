"""The OpenCode + gateway-key worker path (#455).

A machine whose only worker is OpenCode and whose only key is ``ARK_API_KEY``
gets ``{"ref": "opencode"}`` from ``init``. Isolated, that worker cannot read the
user's ``opencode.json`` and, with no ``-m``, picks OpenCode's built-in default
model for the provider -- which a Coding-only Ark account may not have activated.
``init`` and ``doctor`` must point at the documented ``-m`` + ``OPENCODE_CONFIG``
recipe before the first rollout fails; ``plan`` already does.
"""

from __future__ import annotations

import json

import pytest

from agentdescent import cli
from agentdescent.integrations import FORWARDED_ENV

_CLEAR = (*FORWARDED_ENV, "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
          "DEEPSEEK_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _CLEAR:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)


def _init(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "default_agent_ref", lambda: ("opencode", True))
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# x")
    assert cli.main(["init", str(skill), "--out", "spec.json", "--model", "m",
                     "--base-url", "https://ark.example/api/v3"]) == 0
    out = capsys.readouterr().out
    return json.loads((tmp_path / "spec.json").read_text()), out


def test_init_warns_that_bare_opencode_is_not_coding_ready(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys)
    assert spec["agent"] == {"ref": "opencode"}
    [warning] = [ln for ln in out.splitlines() if "OPENCODE_CONFIG" in ln]
    assert warning.startswith("warning:")
    for word in ("ARK_API_KEY", "default model", "-m", "docs/plugins.md"):
        assert word in warning, word


def test_init_is_quiet_for_a_plain_openai_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    _, out = _init(tmp_path, monkeypatch, capsys)
    assert "OPENCODE_CONFIG" not in out


def test_a_configured_worker_gets_no_advice():
    keys = {"ARK_API_KEY": True}
    configured = {"ref": "opencode",
                  "extra_args": ["-m", "ark/deepseek-v4-flash"],
                  "env": {"OPENCODE_CONFIG": "/abs/providers.json"}}
    assert cli._opencode_worker_advice(configured, keys) is None
    assert cli._opencode_worker_advice({"ref": "opencode"}, keys)


def test_doctor_hints_the_worker_recipe(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    real_which = cli.shutil.which
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda name: "/fake/opencode" if name == "opencode" else real_which(name))
    rep = cli.doctor_report()
    [hint] = [h for h in rep["hints"] if "OPENCODE_CONFIG" in h]
    for word in ("opencode is on PATH", "ARK_API_KEY", "default model",
                 "docs/plugins.md"):
        assert word in hint, word
