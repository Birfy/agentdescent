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
    assert "'gepa' applies to kind ['text'], not 'skill_dir'" in err
    assert "evoskill" in err
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_algorithms_run_kind_mismatch_speaks_run_not_init(tmp_path, capsys, monkeypatch):
    # The name is positional: no phantom --algorithm flag, and the one-file
    # route is `algorithms run ... --kind text`, not `init` (#702).
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    assert cli.main(["algorithms", "run", "gepa", str(skill)]) == 2
    err = capsys.readouterr().err
    assert "--algorithm" not in err
    assert "agentdescent init" not in err
    assert f"agentdescent algorithms run gepa {skill / 'SKILL.md'} --kind text" in err
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_algorithms_run_kind_mismatch_on_file_says_it_needs_a_directory(
        tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, _cases = _prompt_and_cases(tmp_path)
    assert cli.main(["--json", "algorithms", "run", "evoskill", str(prompt)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"ok", "error"} and payload["ok"] is False
    assert payload["error"].startswith("algorithm 'evoskill' applies to kind")
    assert "--algorithm" not in payload["error"]
    assert "is a file, so its kind is 'text'" in payload["error"]
    assert "needs a directory target" in payload["error"]


def test_algorithms_run_guessed_dir_kind_names_kind_flag(tmp_path, capsys, monkeypatch):
    # A directory's kind is only a guess; say so and name --kind (#702).
    from agentdescent import algorithms
    from agentdescent.algorithms import Algorithm, Overrides

    monkeypatch.chdir(tmp_path)
    algorithms.register(Algorithm(name="_agent_dir_only", summary="s",
                                  kinds=("agent_dir",), fidelity="preset",
                                  params=(), build=lambda ctx, p: Overrides()),
                        replace=True)
    try:
        skill = tmp_path / "sk"
        skill.mkdir()
        assert cli.main(["algorithms", "run", "_agent_dir_only", str(skill)]) == 2
        err = capsys.readouterr().err
        assert f"kind 'skill_dir' was guessed from {skill}" in err
        assert "pass --kind (one of ['agent_dir'])" in err
        # An explicit --kind is not a guess: no such hint.
        assert cli.main(["algorithms", "run", "_agent_dir_only", str(skill),
                         "--kind", "skill_dir"]) == 2
        assert "was guessed" not in capsys.readouterr().err
    finally:
        algorithms._REGISTRY.pop("_agent_dir_only")


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


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "sk"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    return skill


def test_algorithms_run_plan_failure_still_reports_written_spec(tmp_path, capsys, monkeypatch):
    # #688: the spec is on disk even when plan fails; plain mode must say so.
    # #692: a missing cases path the user passed is named in the tip.
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    missing = tmp_path / "missing.jsonl"
    assert cli.main([
        "algorithms", "run", "gepa", str(prompt), str(missing),
        "--model", "m", "--base-url", "https://example.test/v1",
    ]) == 2
    captured = capsys.readouterr()
    assert "wrote agentdescent.gepa.spec.json" in captured.out
    assert "does not exist" in captured.err
    assert "check the cases path you passed" in captured.err
    assert "next: fix agentdescent.gepa.spec.json, then agentdescent plan" in captured.out
    assert "agentdescent evolve" not in captured.out
    assert (tmp_path / "agentdescent.gepa.spec.json").exists()


def test_algorithms_run_group_kind_validated(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    assert cli.main(["algorithms", "--kind", "bogus", "run", "gepa", str(prompt)]) == 2
    err = capsys.readouterr().err
    assert "--kind must be one of" in err and "'bogus'" in err
    assert not list(tmp_path.glob("agentdescent.*.spec.json"))


def test_algorithms_run_group_kind_validated_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    assert cli.main(["--json", "algorithms", "--kind", "bogus", "run", "gepa",
                     str(prompt)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "--kind must be one of" in payload["error"]


def test_algorithms_run_agent_passthrough(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = _skill(tmp_path)
    _, cases = _prompt_and_cases(tmp_path)
    out = tmp_path / "s.json"
    rc = cli.main(["algorithms", "run", "evoskill", str(skill), str(cases),
                   "--agent", "dsh", "--out", str(out)])
    # Plan may still exit 2 when `dsh` is not on PATH; the agent must be written.
    assert rc in (0, 2)
    assert out.exists()
    capsys.readouterr()
    spec = json.loads(out.read_text())
    assert spec["agent"] == {"ref": "dsh"}
    assert spec["algorithm"] == "evoskill"


def test_algorithms_run_agent_mismatch_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = _skill(tmp_path)
    out = tmp_path / "s.json"
    assert cli.main(["algorithms", "run", "evoskill", str(skill),
                     "--agent", "openai_compatible", "--out", str(out)]) == 2
    assert "--agent openai_compatible" in capsys.readouterr().err
    assert not out.exists()

def test_algorithms_run_omitted_cases_tips_third_arg(tmp_path, capsys, monkeypatch):
    # #692: default eval/cases.jsonl missing → tip the optional third argument.
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    out = tmp_path / "o.json"
    assert cli.main([
        "algorithms", "run", "gepa", str(prompt),
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 2
    captured = capsys.readouterr()
    assert f"wrote {out}" in captured.out
    assert "does not exist" in captured.err
    assert "pass a cases.jsonl as the third argument" in captured.err
    assert "agentdescent algorithms run --help" in captured.err
    assert out.exists()


def test_algorithms_run_omitted_cases_tips_third_arg_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("x\n")
    out = tmp_path / "o.json"
    assert cli.main([
        "--json", "algorithms", "run", "gepa", str(prompt),
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["spec"] == str(out)
    # #694: same keys as the failed-plan shape, and never `evolve` next.
    assert payload["spec_path"] == str(out)
    assert payload["next"] == f"agentdescent plan {out}"
    assert "pass a cases.jsonl as the third argument" in payload["error"]
    assert "does not exist" in payload["error"]


def test_algorithms_run_failed_plan_json_next_is_plan_not_evolve(
        tmp_path, capsys, monkeypatch):
    # #694: plan refuses the spec (too few cases); `next` must not say evolve.
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path, n=1)
    out = tmp_path / "s.json"
    assert cli.main([
        "--json", "algorithms", "run", "gepa", str(prompt), str(cases),
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "at least 4 tasks" in payload["error"]
    assert payload["spec_path"] == str(out)
    assert payload["next"] == f"agentdescent plan {out}"
    assert out.exists()


def test_algorithms_run_failed_plan_plain_next_is_plan(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path, n=1)
    out = tmp_path / "s.json"
    assert cli.main([
        "algorithms", "run", "gepa", str(prompt), str(cases),
        "--model", "m", "--base-url", "https://example.test/v1",
        "--out", str(out),
    ]) == 2
    captured = capsys.readouterr()
    assert f"wrote {out}" in captured.out
    assert "at least 4 tasks" in captured.err
    assert f"next: fix {out}, then agentdescent plan {out}" in captured.out
    assert "agentdescent evolve" not in captured.out


def test_algorithms_run_resolves_a_case_variant_to_the_registered_name(
        tmp_path, capsys, monkeypatch):
    # #696: `algorithms run GEPA` failed as unknown; it now writes the
    # canonical name into the spec and the default file name.
    monkeypatch.chdir(tmp_path)
    prompt, cases = _prompt_and_cases(tmp_path)
    assert cli.main([
        "algorithms", "run", "GEPA", str(prompt), str(cases),
        "--model", "m", "--base-url", "https://example.test/v1",
    ]) == 0
    out = tmp_path / "agentdescent.gepa.spec.json"
    assert out.exists()
    assert json.loads(out.read_text())["algorithm"] == "gepa"
    assert f"next: agentdescent evolve {out.name}" in capsys.readouterr().out
