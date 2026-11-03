"""Re-running init / ``algorithms run`` keeps the replaced spec as ``.bak`` (#725)."""
from __future__ import annotations

import json
from pathlib import Path

from agentdescent import cli


def _prompt_and_cases(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("be terse\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    return prompt, cases


def _run(prompt, cases, out, *extra):
    return cli.main(["algorithms", "run", "gepa", str(prompt), str(cases),
                     "--model", "m", "--base-url", "https://example.test/v1",
                     "--out", str(out), *extra])


def test_identical_rerun_writes_no_backup(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "s.json"
    assert _run(prompt, cases, out) == 0
    assert _run(prompt, cases, out) == 0
    assert not (tmp_path / "s.json.bak").exists()
    assert "previous copy" not in capsys.readouterr().out


def test_edited_spec_is_backed_up_and_named(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "s.json"
    assert _run(prompt, cases, out) == 0
    edited = json.loads(out.read_text())
    edited["evolve"]["rounds"] = 42
    out.write_text(json.dumps(edited))
    capsys.readouterr()
    assert _run(prompt, cases, out, "--rounds", "1") == 0
    text = capsys.readouterr().out
    bak = tmp_path / "s.json.bak"
    assert json.loads(bak.read_text())["evolve"]["rounds"] == 42
    assert json.loads(out.read_text())["evolve"]["rounds"] == 1
    assert f"previous copy is saved as {bak}" in text


def test_init_json_reports_backup_note(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "i.json"
    out.write_text("not json at all")
    assert cli.main(["--json", "init", str(prompt), "--data", str(cases),
                     "--model", "m", "--base-url", "https://example.test/v1",
                     "--out", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (tmp_path / "i.json.bak").read_text() == "not json at all"
    assert any("i.json.bak" in n for n in payload["notes"])
