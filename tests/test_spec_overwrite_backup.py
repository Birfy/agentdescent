"""Re-running init / ``algorithms run`` keeps the replaced spec as ``.bak-N`` (#725, #727)."""
from __future__ import annotations

import json
import shutil
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
    assert not (tmp_path / "s.json.bak-0").exists()
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
    bak = tmp_path / "s.json.bak-0"
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
    assert (tmp_path / "i.json.bak-0").read_text() == "not json at all"
    assert any("i.json.bak-0" in n for n in payload["notes"])


def test_second_replacement_keeps_first_backup(tmp_path, capsys, monkeypatch):
    """A fixed `.bak` slot lost the first hand edit on the next re-run (#727)."""
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "s.json"
    assert _run(prompt, cases, out) == 0
    for rounds in (41, 42):
        edited = json.loads(out.read_text())
        edited["evolve"]["rounds"] = rounds
        out.write_text(json.dumps(edited))
        assert _run(prompt, cases, out, "--rounds", "1") == 0
    assert json.loads((tmp_path / "s.json.bak-0").read_text())["evolve"]["rounds"] == 41
    assert json.loads((tmp_path / "s.json.bak-1").read_text())["evolve"]["rounds"] == 42
    assert not (tmp_path / "s.json.bak").exists()


def _init_opencode(prompt, cases, out):
    skill = prompt.parent / "my-skill"
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# x")
    return cli.main(["init", str(skill), "--data", str(cases), "--agent", "opencode",
                     "--model", "m", "--base-url", "https://example.test/v1",
                     "--out", str(out)])


def test_opencode_providers_file_is_backed_up_when_different(tmp_path, capsys, monkeypatch):
    """The provider-only config next to the spec was overwritten silently (#727)."""
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/fake/opencode"
                        if name == "opencode" else real_which(name, *a, **k))
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "o.json"
    assert _init_opencode(prompt, cases, out) == 0
    providers = tmp_path / "o.opencode-providers.json"
    assert providers.is_file()
    capsys.readouterr()
    # identical re-run: no backup, no note
    assert _init_opencode(prompt, cases, out) == 0
    assert not list(tmp_path.glob("o.opencode-providers.json.bak*"))
    assert "previous copy" not in capsys.readouterr().out
    providers.write_text('{"hand": "edited"}')
    assert _init_opencode(prompt, cases, out) == 0
    text = capsys.readouterr().out
    bak = tmp_path / "o.opencode-providers.json.bak-0"
    assert json.loads(bak.read_text()) == {"hand": "edited"}
    assert f"previous copy is saved as {bak}" in text
