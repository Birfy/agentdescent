"""Missing PATH must not be guessed as skill_dir (#706)."""
from __future__ import annotations

import json
from pathlib import Path

from agentdescent import cli
from agentdescent.evolvespec import looks_like_path_target, load_spec


def test_looks_like_path_target():
    assert looks_like_path_target("nope.txt")
    assert looks_like_path_target("prompts/a.md")
    assert looks_like_path_target("./x")
    assert looks_like_path_target("~/prompt.md")
    assert not looks_like_path_target("Answer briefly.")
    assert not looks_like_path_target("be terse")


def test_guess_kind_refuses_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        cli.guess_kind("nope.txt")
    except ValueError as e:
        assert "does not exist" in str(e) and "nope.txt" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_algorithms_run_missing_path_plain(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["algorithms", "run", "gepa", "nope.txt"]) == 2
    err = capsys.readouterr().err
    assert "does not exist" in err and "nope.txt" in err
    assert "skill_dir" not in err
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_algorithms_run_missing_path_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--json", "algorithms", "run", "gepa", "nope.txt"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "does not exist" in payload["error"]
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_init_missing_path_plain(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "z.json"
    assert cli.main(["init", "nope.txt", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert not out.exists()


def test_init_missing_path_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "z.json"
    assert cli.main(["--json", "init", "nope.txt", "--out", str(out)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": False, "error": payload["error"]}
    assert "does not exist" in payload["error"]
    assert not out.exists()


def test_algorithms_run_kind_text_missing_warns(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    out = tmp_path / "x3.json"
    assert cli.main([
        "algorithms", "run", "gepa", "nope.txt", str(cases),
        "--kind", "text",
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 0
    captured = capsys.readouterr()
    assert out.exists()
    assert "is not a file" in captured.out or "is not a file" in captured.err
    assert "prompt text" in captured.out or "prompt text" in captured.err


def test_plan_text_missing_path_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    spec = {
        "version": 1, "kind": "text", "target": "missing.txt",
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains",
        "agent": {"ref": "openai_compatible", "model": "m",
                  "default_base_url": "https://example.test/v1",
                  "api_key_env": "OPENAI_API_KEY"},
        "evolve": {"rounds": 2, "n_workers": 1},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    payload = cli.plan_payload(load_spec(str(path)))
    assert payload["ok"] is True
    assert any("missing.txt" in w and "prompt text" in w for w in payload["warnings"])


def test_plan_text_inline_no_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    spec = {
        "version": 1, "kind": "text", "target": "Answer briefly.",
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains",
        "agent": {"ref": "openai_compatible", "model": "m",
                  "default_base_url": "https://example.test/v1",
                  "api_key_env": "OPENAI_API_KEY"},
        "evolve": {"rounds": 2, "n_workers": 1},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    payload = cli.plan_payload(load_spec(str(path)))
    assert payload["ok"] is True
    assert not any("prompt text" in w for w in payload["warnings"])


def test_plan_skill_dir_missing_says_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    spec = {
        "version": 1, "kind": "skill_dir", "target": str(tmp_path / "absent-skill"),
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains",
        "agent": {"ref": "opencode"},
        "reflect": {"ref": "openai_compatible", "model": "m",
                    "default_base_url": "https://example.test/v1",
                    "api_key_env": "OPENAI_API_KEY"},
        "evolve": {"rounds": 2, "n_workers": 1},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    try:
        cli.plan_payload(load_spec(str(path)))
    except Exception as e:  # SpecError
        msg = str(e)
        assert "does not exist" in msg
        assert "is not a directory" not in msg
    else:
        raise AssertionError("expected SpecError")
