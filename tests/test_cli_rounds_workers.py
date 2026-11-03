"""``--rounds`` / ``--workers`` on ``algorithms run`` and ``init`` (#707)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdescent import cli

LLM = ["--model", "m", "--base-url", "https://example.test/v1"]


def _prompt_and_cases(tmp_path: Path, n: int = 8):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("be terse\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(n)))
    return prompt, cases


def _run_json(capsys, argv):
    code = cli.main(["--json", *argv])
    obj = json.loads(capsys.readouterr().out)
    return code, obj


def test_algorithms_run_rounds_workers_shrink_the_plan(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    base = ["algorithms", "run", "gepa", str(prompt), str(cases), *LLM]
    code, full = _run_json(capsys, [*base, "--out", str(tmp_path / "full.json")])
    assert code == 0, full
    small_out = tmp_path / "small.json"
    code, small = _run_json(capsys, [*base, "--out", str(small_out),
                                     "--rounds", "1", "--workers", "1"])
    assert code == 0, small
    spec = json.loads(small_out.read_text())
    assert spec["evolve"] == {"rounds": 1, "n_workers": 1}
    assert json.loads((tmp_path / "full.json").read_text())["evolve"] == {
        "rounds": 8, "n_workers": 8}
    assert (small["estimate"]["agent_calls_upper_bound"]
            < full["estimate"]["agent_calls_upper_bound"])


def test_algorithms_run_plain_hint_only_without_flags(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    base = ["algorithms", "run", "gepa", str(prompt), str(cases), *LLM]
    assert cli.main([*base, "--out", str(tmp_path / "a.json")]) == 0
    assert "--rounds 1 --workers 1" in capsys.readouterr().out
    assert cli.main([*base, "--out", str(tmp_path / "b.json"),
                     "--rounds", "1", "--workers", "1"]) == 0
    assert "quick trial" not in capsys.readouterr().out


def test_init_rounds_workers(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "init.json"
    code, obj = _run_json(capsys, ["init", str(prompt), "--data", str(cases), *LLM,
                                   "--out", str(out), "--rounds", "2", "--workers", "3"])
    assert code == 0, obj
    assert json.loads(out.read_text())["evolve"] == {"rounds": 2, "n_workers": 3}


def test_init_only_rounds_keeps_kind_workers_and_policy_slot_budget(
        tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "slot.json"
    code, obj = _run_json(capsys, ["init", "selection", "--kind", "policy_slot", *LLM,
                                   "--out", str(out), "--rounds", "1"])
    assert code == 0, obj
    evolve = json.loads(out.read_text())["evolve"]
    assert evolve == {"rounds": 1, "n_workers": 2, "max_seconds": 5400}


@pytest.mark.parametrize("verb", ["init", "run"])
@pytest.mark.parametrize("flag", ["--rounds", "--workers"])
def test_zero_is_a_usage_error_in_json(tmp_path, capsys, monkeypatch, verb, flag):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "bad.json"
    if verb == "init":
        argv = ["init", str(prompt), "--data", str(cases)]
    else:
        argv = ["algorithms", "run", "gepa", str(prompt), str(cases)]
    code, obj = _run_json(capsys, [*argv, *LLM, "--out", str(out), flag, "0"])
    assert code == 2
    assert obj["ok"] is False and flag in obj["error"]
    assert not out.exists()


def test_zero_plain_is_exit_2(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    assert cli.main(["algorithms", "run", "gepa", str(prompt), str(cases), *LLM,
                     "--rounds", "0"]) == 2
    assert "--rounds must be an integer >= 1" in capsys.readouterr().err
