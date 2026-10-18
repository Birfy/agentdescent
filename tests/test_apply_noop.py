"""`apply` of a run that changed nothing is a reported no-op, not a backup (#396)."""

import json
import os
import stat

import pytest

from agentdescent import cli
from agentdescent.evolution import EvolutionResult
from agentdescent.filetree import canonical, load_tree

from tests.test_runstore_cli import _cli, _spec_file, store  # noqa: F401  (fixture)


def _result(state):
    return EvolutionResult(state=state, rendered=canonical(state), final_reward=0.0,
                           history=[], ledger_log=[])


def _tree(tmp_path, files):
    root = tmp_path / "tree"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return str(root)


def _baks(path):
    parent = os.path.dirname(path)
    return sorted(n for n in os.listdir(parent) if ".bak-" in n)


# -- EvolutionResult.write_to (the directory path of `apply`, and `show`'s plan) --


def test_identical_tree_is_a_noop_with_no_backup_and_no_writes(tmp_path):
    files = {"SKILL.md": "a\n", "references/rules.md": "b\n"}
    dest = _tree(tmp_path, files)
    before = {p: os.stat(os.path.join(dest, p)).st_mtime_ns for p in files}
    for dry in (True, False):
        plan = _result(files).write_to(dest, dry_run=dry)
        assert plan["noop"] is True
        assert plan["written"] == [] and plan["backup"] == []
        assert plan["unchanged"] == ["SKILL.md", "references/rules.md"]
    assert _baks(dest) == []
    assert {p: os.stat(os.path.join(dest, p)).st_mtime_ns for p in files} == before


def test_only_the_changed_file_is_written_and_backed_up(tmp_path):
    dest = _tree(tmp_path, {"SKILL.md": "a\n", "rules.md": "old\n"})
    state = {"SKILL.md": "a\n", "rules.md": "new\n", "added.md": "x\n"}
    dry = _result(state).write_to(dest, dry_run=True)
    assert dry["written"] == ["added.md", "rules.md"] and dry["unchanged"] == ["SKILL.md"]
    assert dry["noop"] is False and _baks(dest) == []
    plan = _result(state).write_to(dest)
    assert plan["written"] == dry["written"] and plan["unchanged"] == ["SKILL.md"]
    assert plan["noop"] is False and len(plan["backup"]) == 1
    assert load_tree(plan["backup"][0])["rules.md"] == "old\n"
    assert load_tree(dest) == state


def test_extra_files_alone_do_not_make_an_apply_worth_doing(tmp_path):
    dest = _tree(tmp_path, {"rules.md": "x\n", "notes.md": "mine\n"})
    plan = _result({"rules.md": "x\n"}).write_to(dest)
    assert plan["noop"] is True and plan["extra"] == ["notes.md"] and _baks(dest) == []
    # ...but pruning them is a change
    plan = _result({"rules.md": "x\n"}).write_to(dest, prune=True)
    assert plan["noop"] is False and plan["deleted"] == ["notes.md"] and plan["backup"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec bit")
def test_a_script_that_lost_its_exec_bit_is_still_a_change(tmp_path):
    dest = _tree(tmp_path, {"scripts/run.sh": "echo hi\n"})
    os.chmod(os.path.join(dest, "scripts/run.sh"), 0o644)
    plan = _result({"scripts/run.sh": "echo hi\n"}).write_to(dest)
    assert plan["written"] == ["scripts/run.sh"] and plan["noop"] is False
    assert os.stat(os.path.join(dest, "scripts/run.sh")).st_mode & stat.S_IXUSR
    assert _result({"scripts/run.sh": "echo hi\n"}).write_to(dest)["noop"] is True


def test_a_missing_target_is_all_written(tmp_path):
    dest = str(tmp_path / "fresh")
    plan = _result({"a.md": "1\n"}).write_to(dest)
    assert plan["written"] == ["a.md"] and plan["unchanged"] == [] and plan["noop"] is False
    assert plan["backup"] == [] and load_tree(dest) == {"a.md": "1\n"}


# -- the CLI / MCP path: cli.apply_payload -------------------------------------


def _evolved(store, tmp_path):
    path, spec = _spec_file(tmp_path)
    code, out = _cli("--store", store, "--json", "evolve", path)
    assert code == 0, out
    return json.loads(out)["run_id"], spec


def test_apply_after_the_target_already_matches_reports_noop(store, tmp_path, capsys):
    rid, spec = _evolved(store, tmp_path)
    # first apply really changes rules.md (forward -> reverse), and keeps a backup
    first = json.loads(_cli("--store", store, "apply", rid)[1])
    assert first["noop"] is False and "rules.md" in first["written"] and first["backup"]
    baks = _baks(spec.target)
    assert len(baks) == 1
    capsys.readouterr()
    # the target now equals the run's tree: nothing to do, no second backup
    for argv in (["apply", rid, "--dry-run"], ["apply", rid]):
        code, out = _cli("--store", store, *argv)
        plan = json.loads(out)
        assert code == 0
        assert plan["noop"] is True and plan["written"] == [] and plan["backup"] == []
        assert "rules.md" in plan["unchanged"]
        assert f"nothing to apply: {spec.target} already matches run {rid}" in (
            capsys.readouterr().err)
    assert _baks(spec.target) == baks
    # `show` reports the same judgement
    shown = cli.show_payload(rid, store=store)
    assert shown["apply_plan"]["noop"] is True and shown["apply_plan"]["written"] == []


def test_apply_payload_is_the_one_judgement_the_mcp_tool_uses(store, tmp_path):
    rid, spec = _evolved(store, tmp_path)
    plan = cli.apply_payload(rid, store=store, dry_run=True)
    assert plan["noop"] is False and plan["written"] == ["rules.md"]
    assert _baks(spec.target) == []                       # a dry run never backs up
    cli.apply_payload(rid, store=store)
    assert cli.apply_payload(rid, store=store)["noop"] is True


def _text_run(store, tmp_path, target_text):
    from tests.test_evolvespec import _rows_file

    target = tmp_path / "prompt.txt"
    target.write_text(target_text, encoding="utf-8")
    spec = {"kind": "text", "target": str(target),
            "data": {"path": _rows_file(str(tmp_path)), "prompt": "prompt", "gold": "gold"},
            "score": "contains", "agent": "echo",
            "reflect": {"ref": "tests.test_evolvespec:text_reflect", "call": False},
            "allow": ["tests."], "evolve": {"rounds": 1, "n_workers": 1, "seed": 0}}
    path = tmp_path / "text-spec.json"
    path.write_text(json.dumps(spec))
    code, out = _cli("--store", store, "--json", "evolve", str(path))
    assert code == 0, out
    return json.loads(out)["run_id"], target


def test_text_target_identical_to_the_run_is_a_noop_without_a_bak(store, tmp_path):
    rid, target = _text_run(store, tmp_path, "original\n")
    rendered = cli.show_payload(rid, store=store)["rendered"]
    target.write_text(rendered, encoding="utf-8")
    for dry in (True, False):
        plan = cli.apply_payload(rid, store=store, dry_run=dry)
        assert plan["noop"] is True and plan["unchanged"] == [str(target)]
        assert not plan.get("written") and plan.get("backup", []) == []
    assert _baks(str(target)) == []
    assert cli.show_payload(rid, store=store)["apply_plan"]["noop"] is True


def test_text_target_that_differs_is_written_with_a_backup(store, tmp_path):
    rid, target = _text_run(store, tmp_path, "original\n")
    target.write_text("edited since the run\n", encoding="utf-8")
    dry = cli.apply_payload(rid, store=store, dry_run=True)
    assert dry["noop"] is False and dry["would_write"] == str(target)
    assert _baks(str(target)) == []
    plan = cli.apply_payload(rid, store=store)
    assert plan["noop"] is False and plan["written"] == [str(target)]
    assert plan["backup"] == [str(target) + ".bak-0"]
    assert open(plan["backup"][0]).read() == "edited since the run\n"


def test_crlf_files_compare_verbatim(tmp_path):
    """`load_tree` keeps CRLF, so a run over a CRLF directory has CRLF state; the
    comparison must not read the target in universal-newline mode."""
    files = {"SKILL.md": "a\r\nb\r\n", "rules.md": "x\ny\n"}
    dest = str(tmp_path / "crlf")
    os.makedirs(dest)
    for rel, text in files.items():
        with open(os.path.join(dest, rel), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    assert load_tree(dest) == files
    for dry in (True, False):
        plan = _result(files).write_to(dest, dry_run=dry)
        assert plan["noop"] is True and plan["written"] == [] and plan["backup"] == []
    assert _baks(dest) == []
    # a real change to one CRLF file is still detected, and written verbatim
    changed = {**files, "SKILL.md": "a\r\nc\r\n"}
    plan = _result(changed).write_to(dest)
    assert plan["written"] == ["SKILL.md"] and len(plan["backup"]) == 1
    assert load_tree(dest) == changed


def test_text_target_with_crlf_is_compared_verbatim(store, tmp_path):
    rid, target = _text_run(store, tmp_path, "line one\r\nline two\r\n")
    rendered = cli.show_payload(rid, store=store)["rendered"]
    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(rendered)
    assert cli.apply_payload(rid, store=store)["noop"] is True
    assert _baks(str(target)) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec bits")
def test_a_script_with_only_user_exec_is_not_what_apply_leaves(tmp_path):
    dest = _tree(tmp_path, {"scripts/run.sh": "echo hi\n"})
    os.chmod(os.path.join(dest, "scripts/run.sh"), 0o744)          # u+x, not g+x
    plan = _result({"scripts/run.sh": "echo hi\n"}).write_to(dest)
    assert plan["written"] == ["scripts/run.sh"] and plan["noop"] is False
    assert _result({"scripts/run.sh": "echo hi\n"}).write_to(dest)["noop"] is True


def test_an_empty_artifact_is_a_noop_and_creates_nothing(tmp_path):
    dest = str(tmp_path / "missing")
    plan = _result({}).write_to(dest)
    assert plan["noop"] is True and plan["written"] == [] and not os.path.exists(dest)


def test_text_dry_run_plan_has_a_written_field(store, tmp_path):
    rid, target = _text_run(store, tmp_path, "original\n")
    same = cli.apply_payload(rid, store=store, dry_run=True)
    assert same["written"] == [] and same["noop"] is True
    target.write_text("edited since the run\n", encoding="utf-8")
    diff = cli.apply_payload(rid, store=store, dry_run=True)
    assert diff["written"] == [str(target)] and diff["would_write"] == str(target)
