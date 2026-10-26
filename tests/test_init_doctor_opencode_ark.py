"""``init`` / ``doctor`` name the OpenCode Coding worker recipe when only a gateway key is set (#455).

``plan`` already warns (#462). ``init`` used to write a silent ``{"ref": "opencode"}``
and ``doctor`` only mentioned the ``openai_compatible`` reflect route.
"""

from __future__ import annotations

import json
import shutil

import pytest

from agentdescent import cli
from agentdescent.integrations import FORWARDED_ENV, provider_key_names

_KEYS = tuple(provider_key_names()) + ("OPENAI_API_KEY", "OPENAI_BASE_URL",
                                        "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in set(_KEYS) | set(FORWARDED_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)


def _with_opencode(monkeypatch):
    real_which = shutil.which

    def which(name, *a, **k):
        if name == "opencode":
            return "/fake/opencode"
        return real_which(name, *a, **k)

    monkeypatch.setattr(shutil, "which", which)


def _init(tmp_path, monkeypatch, capsys, *extra):
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "my-skill"
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# x")
    rows = "".join(json.dumps({"prompt": f"q{i}", "gold": "a"}) + "\n" for i in range(8))
    (tmp_path / "cases.jsonl").write_text(rows)
    assert cli.main(["init", str(skill), "--data", str(tmp_path / "cases.jsonl"),
                     "--agent", "opencode", "--out", "spec.json", *extra]) == 0
    out = capsys.readouterr().out
    return json.loads((tmp_path / "spec.json").read_text()), out


def test_init_warns_on_bare_opencode_with_only_ark(tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys)
    assert spec["agent"] == {"ref": "opencode"}
    assert "OPENCODE_CONFIG" in out and "docs/plugins.md" in out
    assert "default model" in out or "Volcengine" in out
    assert "Export OPENAI_API_KEY" not in out


def test_init_stays_quiet_about_opencode_when_openai_key_is_set(
        tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    _, out = _init(tmp_path, monkeypatch, capsys, "--model", "m")
    assert "OPENCODE_CONFIG" not in out
    assert "Volcengine" not in out


def test_starter_warnings_skip_configured_opencode_recipe(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    keys = cli.doctor_report()["provider_keys_present"]
    block = {"ref": "opencode", "extra_args": ["-m", "ark/deepseek-v4-flash"],
             "env": {"OPENCODE_CONFIG": "/abs/providers.json"}}
    assert cli._opencode_configured(block)
    assert cli._opencode_provider_remedy(block, keys) is None
    lines = cli._starter_model_warnings(
        {"agent": block,
         "reflect": {"ref": "openai_compatible", "model": "m",
                     "api_key_env": "ARK_API_KEY",
                     "default_base_url": "https://x.example/v1"}},
        "spec.json")
    assert not any("bare `opencode`" in line for line in lines)


def test_doctor_hints_the_opencode_coding_worker_path(monkeypatch):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "k")
    hints = cli.doctor_report()["hints"]
    worker = [h for h in hints if "docs/plugins.md" in h and "OPENCODE_CONFIG" in h]
    assert len(worker) == 1, hints
    assert "ark/deepseek-v4-flash" in worker[0]
    assert any("openai_compatible" in h and "api_key_env" in h for h in hints)


def test_doctor_skips_worker_hint_without_opencode_cli(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    real_which = shutil.which

    def which(name, *a, **k):
        if name == "opencode":
            return None
        return real_which(name, *a, **k)

    monkeypatch.setattr(shutil, "which", which)
    hints = cli.doctor_report()["hints"]
    assert not any("OPENCODE_CONFIG" in h for h in hints)
    assert any("openai_compatible" in h for h in hints)


def test_doctor_keeps_non_ark_gateway_recipe_generic(monkeypatch):
    # A DeepSeek key is not an Ark Coding account: no Volcengine, no ark/ model.
    _with_opencode(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    hints = cli.doctor_report()["hints"]
    worker = [h for h in hints if "OPENCODE_CONFIG" in h]
    assert len(worker) == 1, hints
    assert "<provider>/<model>" in worker[0]
    assert "{env:DEEPSEEK_API_KEY}" in worker[0]
    assert "Volcengine" not in worker[0] and "ark/" not in worker[0]


def test_init_keeps_non_ark_gateway_recipe_generic(tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    _, out = _init(tmp_path, monkeypatch, capsys)
    assert "bare `opencode`" in out and "<provider>/<model>" in out
    assert "Volcengine" not in out and "ark/deepseek-v4-flash" not in out


def test_init_and_doctor_agree_when_openai_and_ark_are_both_set(
        tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ARK_API_KEY", "k")
    hints = cli.doctor_report()["hints"]
    doctor_warns = any("OPENCODE_CONFIG" in h for h in hints)
    _, out = _init(tmp_path, monkeypatch, capsys)
    init_warns = "bare `opencode`" in out
    assert doctor_warns and init_warns
    assert "ark/deepseek-v4-flash" in out


_ARK = "https://ark.cn-beijing.volces.com/api/coding/v3"


def test_init_wires_model_and_base_url_into_opencode_worker(tmp_path, monkeypatch, capsys):
    """#485: with --model/--base-url given, the worker gets the Coding recipe too."""
    _with_opencode(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "secret-value")
    spec, out = _init(tmp_path, monkeypatch, capsys, "--model", "deepseek-v4-flash",
                      "--base-url", _ARK, "--key-env", "ARK_API_KEY")
    agent = spec["agent"]
    assert agent["extra_args"] == ["-m", "ark/deepseek-v4-flash"]
    cfg_path = agent["env"]["OPENCODE_CONFIG"]
    assert cfg_path == str(tmp_path / "spec.opencode-providers.json")
    cfg = json.loads(open(cfg_path).read())
    ark = cfg["provider"]["ark"]
    assert ark["options"] == {"baseURL": _ARK, "apiKey": "{env:ARK_API_KEY}"}
    assert "deepseek-v4-flash" in ark["models"]
    assert "secret-value" not in open(cfg_path).read()
    assert cli._opencode_configured(agent)
    assert "is bare `opencode`" not in out
    assert spec["reflect"]["model"] == "deepseek-v4-flash"


def test_init_opencode_recipe_infers_key_env_from_environment(tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, _ = _init(tmp_path, monkeypatch, capsys, "--model", "m", "--base-url", _ARK)
    cfg = json.loads(open(spec["agent"]["env"]["OPENCODE_CONFIG"]).read())
    assert cfg["provider"]["ark"]["options"]["apiKey"] == "{env:ARK_API_KEY}"


def test_init_opencode_recipe_uses_custom_provider_for_other_endpoints(
        tmp_path, monkeypatch, capsys):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    spec, _ = _init(tmp_path, monkeypatch, capsys, "--model", "m",
                    "--base-url", "http://localhost:8000/v1")
    assert spec["agent"]["extra_args"] == ["-m", "custom/m"]
    cfg = json.loads(open(spec["agent"]["env"]["OPENCODE_CONFIG"]).read())
    assert cfg["provider"]["custom"]["options"]["apiKey"] == "{env:OPENAI_API_KEY}"


@pytest.mark.parametrize("extra", [(), ("--model", "m"), ("--base-url", _ARK)])
def test_init_keeps_bare_opencode_without_both_flags(tmp_path, monkeypatch, capsys, extra):
    _with_opencode(monkeypatch)
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys, *extra)
    assert spec["agent"] == {"ref": "opencode"}
    assert not (tmp_path / "spec.opencode-providers.json").exists()
    assert "OPENCODE_CONFIG" in out
