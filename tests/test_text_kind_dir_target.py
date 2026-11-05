"""kind=text refuses an existing directory as its target (#755).

Before: `--kind text` on an existing directory only warned "is not a file",
`plan` said ok, the run evolved the path *string* as the prompt, `apply
--dry-run` claimed it would write the directory and `apply` traced back on
IsADirectoryError. A missing path keeps the #706 warning (tested there).
"""
from __future__ import annotations

import json

import pytest

from agentdescent import cli
from agentdescent.evolvespec import SpecError, compose, load_spec

from tests.test_runstore_cli import _cli, store  # noqa: F401  (fixture)


def _cases(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(
        json.dumps({"prompt": f"q{i}", "gold": f"a{i}"}) + "\n" for i in range(8)))
    return cases


def _skill_dir(tmp_path):
    d = tmp_path / "csv-total"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("skill\n", encoding="utf-8")
    return d


def _flags():
    return ["--kind", "text", "--model", "m", "--base-url", "https://example.test/v1"]


def test_algorithms_run_refuses_a_directory_for_kind_text(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = _skill_dir(tmp_path)
    out = tmp_path / "spec.json"
    code = cli.main(["algorithms", "run", "gepa", str(d), str(_cases(tmp_path)),
                     *_flags(), "--out", str(out)])
    err = capsys.readouterr().err
    assert code == 2
    assert "is a directory" in err and "SKILL.md" in err and "skill_dir" in err
    assert not out.exists()


def test_init_json_refuses_a_directory_for_kind_text(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = _skill_dir(tmp_path)
    out = tmp_path / "spec.json"
    code = cli.main(["--json", "init", str(d), "--data", str(_cases(tmp_path)),
                     *_flags(), "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["ok"] is False
    assert "is a directory" in payload["error"]
    assert not out.exists()


@pytest.mark.parametrize("algorithm", [None, "gepa", "ace", "dgm"])
def test_plan_refuses_a_hand_written_text_spec_over_a_directory(tmp_path, monkeypatch,
                                                                algorithm):
    monkeypatch.chdir(tmp_path)
    d = _skill_dir(tmp_path)
    spec = {"version": 1, "kind": "text", "target": str(d),
            "data": {"path": str(_cases(tmp_path)), "prompt": "prompt", "gold": "gold"},
            "score": "contains",
            "agent": {"ref": "openai_compatible", "model": "m",
                      "default_base_url": "https://example.test/v1",
                      "api_key_env": "OPENAI_API_KEY"},
            "evolve": {"rounds": 1, "n_workers": 1}}
    if algorithm:
        spec["algorithm"] = algorithm
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(SpecError, match="is a directory"):
        compose(load_spec(str(path)))
    code, out = _cli("--json", "plan", str(path))
    assert code != 0 and json.loads(out)["ok"] is False


def test_a_text_file_target_still_plans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = _skill_dir(tmp_path)
    out = tmp_path / "spec.json"
    assert cli.main(["algorithms", "run", "gepa", str(d / "SKILL.md"),
                     str(_cases(tmp_path)), *_flags(), "--out", str(out)]) == 0
    assert out.exists()


def test_apply_to_a_directory_is_an_error_not_a_traceback(store, tmp_path):
    from tests.test_evolvespec import _rows_file

    target = tmp_path / "prompt.txt"
    target.write_text("original\n", encoding="utf-8")
    spec = {"kind": "text", "target": str(target),
            "data": {"path": _rows_file(str(tmp_path)), "prompt": "prompt", "gold": "gold"},
            "score": "contains", "agent": "echo",
            "reflect": {"ref": "tests.test_evolvespec:text_reflect", "call": False},
            "allow": ["tests."],
            "evolve": {"rounds": 1, "n_workers": 1, "seed": 0}}
    path = tmp_path / "text-spec.json"
    path.write_text(json.dumps(spec))
    code, out = _cli("--store", store, "--json", "evolve", str(path))
    assert code == 0, out
    rid = json.loads(out)["run_id"]
    d = _skill_dir(tmp_path)
    before = sorted(p.name for p in d.rglob("*"))
    for extra in (["--dry-run"], []):
        code, out = _cli("--store", store, "--json", "apply", rid, "--to", str(d), *extra)
        payload = json.loads(out)
        assert code == 2 and payload["ok"] is False, out
        assert "is a directory" in payload["error"] and "--to FILE" in payload["error"]
    assert sorted(p.name for p in d.rglob("*")) == before
    assert not list(tmp_path.glob("csv-total.bak-*"))
