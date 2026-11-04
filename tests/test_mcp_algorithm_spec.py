"""MCP ``algorithm_spec`` = ``algorithms run --json`` for hosts (#723)."""
from __future__ import annotations

import json
from pathlib import Path

from agentdescent import cli
from agentdescent.mcp import TOOL_DESCRIPTIONS, Tools

_FLAGS = dict(model="deepseek-v4-flash",
              base_url="https://ark.cn-beijing.volces.com/api/coding/v3")


def _prompt_and_cases(tmp_path: Path, n: int = 8):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("be terse\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(n)))
    return prompt, cases


def test_builds_plans_and_returns_a_startable_spec(tmp_path):
    prompt, cases = _prompt_and_cases(tmp_path)
    r = Tools(None).algorithm_spec("GEPA", str(prompt), str(cases),
                                   params={"seed": 3}, rounds=1, workers=1, **_FLAGS)
    assert r["ok"] is True, r
    out = tmp_path / "agentdescent.gepa.spec.json"
    assert r["spec_path"] == str(out) and out.exists()
    assert r["algorithm"]["name"] == "gepa" and r["algorithm"]["params"]["seed"] == 3
    assert r["estimate"]["rounds"] == 1 and r["estimate"]["n_workers"] == 1
    spec = r["spec"]
    assert spec["algorithm"] == {"name": "gepa", "params": {"seed": 3}}
    assert spec["target"] == str(prompt) and spec["data"]["path"] == str(cases)
    assert "start" in r["next"]
    assert not any(k.startswith("_") for k in r)
    # The returned dict plans the same way through the MCP plan tool.
    assert Tools(None).plan(spec)["ok"] is True


def test_relative_paths_are_absolutised_and_default_out_is_beside_target(
        tmp_path, monkeypatch):
    prompt, cases = _prompt_and_cases(tmp_path)
    monkeypatch.chdir(tmp_path)
    r = Tools(None).algorithm_spec("rrsi", "prompt.md", "cases.jsonl", **_FLAGS)
    assert r["ok"] is True, r
    assert r["spec_path"] == str(tmp_path / "agentdescent.rrsi.spec.json")
    assert r["spec"]["target"] == str(prompt)


def test_directory_target_spec_is_not_written_inside_it(tmp_path):
    skill = tmp_path / "myskill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    _, cases = _prompt_and_cases(tmp_path)
    r = Tools(None).algorithm_spec("archive_selection", str(skill), str(cases),
                                   agent="claude_code")
    assert r.get("spec_path") == str(tmp_path / "agentdescent.archive_selection.spec.json")
    assert sorted(p.name for p in skill.iterdir()) == ["SKILL.md"]


def test_refusals_match_the_cli_and_write_nothing(tmp_path):
    prompt, cases = _prompt_and_cases(tmp_path)
    t = Tools(None)
    assert "unknown algorithm" in t.algorithm_spec("nope", str(prompt))["error"]
    assert "needs a directory target" in t.algorithm_spec(
        "evoskill", str(prompt), str(cases))["error"]
    assert "no parameter" in t.algorithm_spec(
        "gepa", str(prompt), str(cases), params={"bogus": 1})["error"]
    assert "params must be an object" in t.algorithm_spec(
        "gepa", str(prompt), str(cases), params=[1])["error"]
    assert "--kind must be one of" in t.algorithm_spec(
        "gepa", str(prompt), str(cases), kind="nope")["error"]
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_missing_cases_names_the_field_not_a_positional(tmp_path):
    prompt, _ = _prompt_and_cases(tmp_path)
    r = Tools(None).algorithm_spec("gepa", str(prompt), str(tmp_path / "nope.jsonl"),
                                   **_FLAGS)
    assert r["ok"] is False and "does not exist" in r["error"]
    assert "third argument" not in r["error"] and "`cases`" in r["error"]
    assert "spec" not in r       # no path string masquerading as the spec dict
    assert r["spec_path"].endswith("agentdescent.gepa.spec.json")


def test_same_object_as_algorithms_run_json(tmp_path, capsys):
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "cli.spec.json"
    assert cli.main(["--json", "algorithms", "run", "gepa", str(prompt), str(cases),
                     "--model", _FLAGS["model"], "--base-url", _FLAGS["base_url"],
                     "--out", str(out)]) == 0
    from_cli = json.loads(capsys.readouterr().out)
    r = Tools(None).algorithm_spec("gepa", str(prompt), str(cases),
                                   out=str(tmp_path / "mcp.spec.json"), **_FLAGS)
    for key in ("ok", "tasks", "algorithm", "estimate", "warnings", "notes"):
        assert r[key] == from_cli[key], key


def test_description_is_written_for_the_model():
    text = TOOL_DESCRIPTIONS["algorithm_spec"]
    assert "algorithms run" in text and "start" in text and "never guess" in text


def test_params_object_is_passed_through_not_re_split(tmp_path):
    # #729: params used to be re-encoded as `--param k=v` strings, so a key
    # with `=` was split and an empty key's error named the CLI flag.
    prompt, cases = _prompt_and_cases(tmp_path)
    t = Tools(None)
    r = t.algorithm_spec("gepa", str(prompt), str(cases),
                         params={"seed=9": 3}, **_FLAGS)
    assert r["ok"] is False and "'seed=9'" in r["error"], r
    assert "expected int" not in r["error"]
    r = t.algorithm_spec("gepa", str(prompt), str(cases), params={"": 1}, **_FLAGS)
    assert r["ok"] is False and "params key" in r["error"]
    assert "--param" not in r["error"]
    r = t.algorithm_spec("gepa", str(prompt), str(cases),
                         params={"seed": 1, " seed ": 2}, **_FLAGS)
    assert r["ok"] is False and "more than once" in r["error"]
    assert "--param" not in r["error"]
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))
    # Names are stripped like the CLI's `--param`.
    r = t.algorithm_spec("gepa", str(prompt), str(cases),
                         params={" seed ": 4}, rounds=1, workers=1, **_FLAGS)
    assert r["ok"] is True and r["spec"]["algorithm"]["params"] == {"seed": 4}, r
