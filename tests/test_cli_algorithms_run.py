"""``agentdescent algorithms run`` = init + plan in one step (#674)."""
from __future__ import annotations

import json
from pathlib import Path

from agentdescent import cli


def _prompt_and_cases(tmp_path: Path, n: int = 8):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("be terse\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(n)))
    return prompt, cases


def test_algorithms_run_writes_spec_and_plans(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "out.spec.json"
    assert cli.main([
        "algorithms", "run", "gepa", str(prompt), str(cases),
        "--model", "deepseek-v4-flash",
        "--base-url", "https://ark.cn-beijing.volces.com/api/coding/v3",
        "--out", str(out),
    ]) == 0
    captured = capsys.readouterr()
    assert f"wrote {out}" in captured.out
    assert "next: agentdescent evolve" in captured.out
    assert str(out) in captured.out
    spec = json.loads(out.read_text())
    assert spec["algorithm"] == "gepa"
    assert spec["kind"] == "text"
    assert spec["data"]["path"] == str(cases)
    # Did not start a run.
    assert not (tmp_path / ".agentdescent" / "runs").exists()
    home_runs = Path.home() / ".agentdescent" / "runs"
    # No new claim: just ensure the verb did not create under cwd store.
    assert "started" not in captured.out.lower()


def test_algorithms_run_default_out_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    assert cli.main([
        "algorithms", "run", "gepa", str(prompt), str(cases),
        "--model", "m", "--base-url", "https://example.test/v1",
    ]) == 0
    out = tmp_path / "agentdescent.gepa.spec.json"
    assert out.exists()
    assert json.loads(out.read_text())["algorithm"] == "gepa"
    assert f"next: agentdescent evolve {out.name}" in capsys.readouterr().out


def test_algorithms_run_json_one_object(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "s.json"
    assert cli.main([
        "--json", "algorithms", "run", "gepa", str(prompt), str(cases),
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["spec_path"] == str(out)
    assert payload["algorithm"]["name"] == "gepa"
    assert payload["next"] == f"agentdescent evolve {out}"
    assert payload["tasks"] >= 4


def test_algorithms_run_unknown_name(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    assert cli.main([
        "algorithms", "run", "nope", str(prompt), str(cases),
    ]) == 2
    err = capsys.readouterr().err
    assert "unknown algorithm 'nope'" in err
    assert "agentdescent algorithms list" in err
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_algorithms_run_unknown_name_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    assert cli.main(["--json", "algorithms", "run", "nope", str(prompt)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"ok", "error"}
    assert payload["ok"] is False
    assert "unknown algorithm 'nope'" in payload["error"]
    assert "agentdescent algorithms list" in payload["error"]


def test_algorithms_run_kind_mismatch(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    assert cli.main([
        "algorithms", "run", "gepa", str(skill), str(cases),
        "--model", "m",
    ]) == 2
    err = capsys.readouterr().err
    assert "applies to kind" in err or "kind" in err
    assert "archive_selection" in err or "evoskill" in err


def test_algorithms_run_kind_mismatch_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    assert cli.main(["--json", "algorithms", "run", "gepa", str(skill)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "kind" in payload["error"]


def test_algorithms_list_and_show_unchanged(capsys):
    assert cli.main(["algorithms", "list"]) == 0
    listed = capsys.readouterr().out
    assert "gepa" in listed and "rrsi" in listed
    assert cli.main(["algorithms", "show", "gepa"]) == 0
    assert "gepa" in capsys.readouterr().out
