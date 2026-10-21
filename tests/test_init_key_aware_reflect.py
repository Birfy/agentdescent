"""``init`` writes a reflector matched to the keys that are set, never a guessed model (#451).

``doctor`` said "use ARK_API_KEY with api_key_env", ``init`` wrote
``{"ref": "openai_compatible", "model": "deepseek-v4-flash"}`` (OPENAI_API_KEY at
api.openai.com) and warned "Export OPENAI_API_KEY", and with a real OpenAI key
``plan`` passed a model that endpoint does not serve.
"""

import json

import pytest

from agentdescent import cli
from agentdescent.evolvespec import EvolveSpec, load_spec
from agentdescent.integrations import provider_key_names

_KEYS = tuple(provider_key_names()) + ("OPENAI_API_KEY", "OPENAI_BASE_URL",
                                        "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)
    # A worker CLI "on PATH" for plan's gate; the reflector is what is under test.
    real = cli.doctor_report

    def fake():
        rep = real()
        return {**rep, "agent_clis": {**rep["agent_clis"], "opencode": "/fake/opencode"}}

    monkeypatch.setattr(cli, "doctor_report", fake)


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


def _plan(tmp_path):
    return cli.plan_payload(load_spec(str(tmp_path / "spec.json")))


def test_ark_only_names_ark_and_guesses_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys)
    assert spec["reflect"] == {"ref": "openai_compatible", "model": cli.PLACEHOLDER_MODEL,
                               "api_key_env": "ARK_API_KEY",
                               "default_base_url": cli.PLACEHOLDER_URL}
    assert "ARK_API_KEY" in out and "Export OPENAI_API_KEY" not in out
    assert "deepseek-v4-flash" not in json.dumps(spec)
    payload = _plan(tmp_path)
    assert payload["ok"] is False
    assert "reflect.model" in payload["error"] and "reflect.default_base_url" in payload["error"]


def test_openai_key_gets_no_model_that_api_openai_com_does_not_serve(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys)
    assert spec["reflect"] == {"ref": "openai_compatible", "model": cli.PLACEHOLDER_MODEL}
    assert "placeholder" in out and "--model" in out
    payload = _plan(tmp_path)
    assert payload["ok"] is False and "reflect.model" in payload["error"]


def test_filled_in_values_plan_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    spec, out = _init(tmp_path, monkeypatch, capsys, "--model", "doubao-seed-code",
                      "--base-url", "https://ark.example/api/v3")
    assert spec["reflect"] == {"ref": "openai_compatible", "model": "doubao-seed-code",
                               "api_key_env": "ARK_API_KEY",
                               "default_base_url": "https://ark.example/api/v3"}
    assert "warning" not in out
    payload = _plan(tmp_path)
    assert payload["ok"] is True, payload.get("error")
    assert not [w for w in payload["warnings"] if w.startswith("reflect:")]


def test_key_env_flag_wins_and_no_key_still_warns(tmp_path, monkeypatch, capsys):
    spec, out = _init(tmp_path, monkeypatch, capsys, "--model", "m")
    assert spec["reflect"] == {"ref": "openai_compatible", "model": "m"}
    assert "OPENAI_API_KEY is not set" in out
    spec, _ = _init(tmp_path, monkeypatch, capsys, "--model", "m", "--key-env", "MY_KEY",
                    "--base-url", "https://x.example/v1")
    assert spec["reflect"]["api_key_env"] == "MY_KEY"


def test_kind_text_agent_follows_the_same_rule(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("be terse")
    assert cli.main(["init", str(prompt), "--out", "spec.json"]) == 0
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["agent"]["api_key_env"] == "ARK_API_KEY"
    assert spec["agent"]["model"] == cli.PLACEHOLDER_MODEL


def test_plan_reflect_key_warning_lists_the_key_that_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n' * 8)
    target = tmp_path / "prompt.txt"
    target.write_text("hi\n")
    spec = EvolveSpec.from_dict({
        "kind": "text", "target": str(target),
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains", "agent": {"ref": "echo"},
        "reflect": {"ref": "openai_compatible", "model": "m"}})
    [warning] = cli._unusable_refs(spec)
    assert "needs OPENAI_API_KEY" in warning
    assert "ARK_API_KEY is set" in warning and "api_key_env" in warning
