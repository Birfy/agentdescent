"""The run store and the CLI, end to end and offline.

A run is a directory plus a detached process, so the tests here actually detach:
`launch` starts `python -m agentdescent.cli run`, the test polls `status.json`
until the child reports, and `cancel` is checked against a deliberately slow
agent. Everything uses the stub workspace agent from `test_evolvespec.py`.
"""

import json
import threading
from pathlib import Path
import os
import subprocess
import sys
import time

import pytest

from agentdescent import cli, runstore
from agentdescent.evolution import RoundInfo
from agentdescent.runstore import RunStoreError, RunStatus

from tests.test_evolvespec import _WORDS, _dir_spec  # noqa: F401  (shared fixtures)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = str(tmp_path / "runs")
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    return s


def _spec_file(tmp_path, **extra):
    spec = _dir_spec(str(tmp_path), **extra)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return str(path), spec


def _wait(rd, states=("done", "failed", "cancelled"), timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = rd.status()
        if st.state in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"run did not finish: {rd.status()}")


def _cli(*args, **kw):
    """Run the CLI in-process; returns (exit code, stdout)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(list(args))
    return code, buf.getvalue()


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def test_root_honours_agentdescent_home(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path))
    assert runstore.root() == str(tmp_path / "runs")


def test_create_writes_spec_and_status(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    assert rd.spec_dict()["kind"] == "skill_dir"
    st = rd.status()
    assert st.state == "created" and st.rounds == 3 and st.kind == "skill_dir"
    with pytest.raises(RunStoreError, match="already exists"):
        runstore.create(spec.to_dict(), run_id=rd.run_id, store=store)
    with pytest.raises(RunStoreError, match="no run"):
        runstore.get("nope", store=store)


def test_status_is_replaced_atomically(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.update_status(round=2, best_reward=0.5)
    assert not [f for f in os.listdir(rd.path) if f.endswith(".tmp")]
    assert rd.status().round == 2


def _round(reward=0.5, n=0):
    return RoundInfo(round=n, held_out_reward=reward, n_items=1,
                     committed=0, rejected=0)


def test_append_round_writes_null_not_nan_for_nonfinite_reward(store, tmp_path):
    """A NaN/inf reward must not land as the non-portable NaN token (#271)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.append_round(_round(float("nan")))
    rd.append_round(_round(float("inf"), n=1))
    raw = Path(rd.rounds_path).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    rows = rd.rounds()
    assert len(rows) == 2
    assert rows[0]["held_out_reward"] is None
    assert rows[1]["held_out_reward"] is None
    # Strict JSON: browsers and jq parse every line.
    for line in raw.splitlines():
        json.loads(line)


def test_rounds_skips_a_torn_line_and_keeps_the_rest(store, tmp_path):
    """One corrupt JSONL line must not hide every other round (#271)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.append_round(_round(0.25, n=0))
    with open(rd.rounds_path, "a", encoding="utf-8") as fh:
        fh.write('{"round": 1, "held_out_rew')          # torn
        fh.write("\n")
    rd.append_round(_round(0.75, n=2))
    rows = rd.rounds()
    assert [r["round"] for r in rows] == [0, 2]
    assert [r["held_out_reward"] for r in rows] == [0.25, 0.75]


def test_append_after_a_torn_tail_without_newline_keeps_the_next_round(
        store, tmp_path):
    """A killed writer leaves no newline; the next append must not glue on (#271)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.append_round(_round(0.25, n=0))
    with open(rd.rounds_path, "a", encoding="utf-8") as fh:
        fh.write('{"round": 1, "held_out_rew')          # torn, no newline
    rd.append_round(_round(0.75, n=2))
    rows = rd.rounds()
    assert [r["round"] for r in rows] == [0, 2]
    assert [r["held_out_reward"] for r in rows] == [0.25, 0.75]


def test_rounds_survives_a_tear_inside_a_multibyte_character(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.append_round(_round(0.25, n=0))
    with open(rd.rounds_path, "ab") as fh:
        fh.write('{"round": 1, "note": "é'.encode("utf-8")[:-1])
    rd.append_round(_round(0.75, n=2))
    assert [r["round"] for r in rd.rounds()] == [0, 2]


def test_text_status_and_watch_print_null_rewards(store, tmp_path):
    """A non-finite reward is stored as null; text status/watch must not crash."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rid = os.path.basename(rd.path)
    rd.append_round(_round(float("nan"), n=0))
    rd.append_round(_round(0.5, n=1))
    code, out = _cli("--store", store, "status", rid)
    assert code == 0, out
    assert "reward=n/a" in out and "reward=0.500" in out
    code, out = _cli("--store", store, "watch", rid, "--interval", "0")
    assert "round   0  reward=n/a" in out and "reward=0.500" in out


def test_finite_or_none_and_sanitize_strip_nonfinite_floats():
    """Helpers used by append_round / on_round / the panel (#271)."""
    assert runstore._finite_or_none(1.5) == 1.5
    assert runstore._finite_or_none(float("nan")) is None
    assert runstore._finite_or_none(float("-inf")) is None
    assert runstore._finite_or_none("x") == "x"
    cleaned = runstore._sanitize_jsonable(
        {"best": float("nan"), "nested": [1.0, float("inf")], "ok": 0.2})
    assert cleaned == {"best": None, "nested": [1.0, None], "ok": 0.2}
    # Order-dependence that status used to inherit from max(best):
    assert max([1.0, float("nan")]) == 1.0
    assert str(max([float("nan"), 1.0])) == "nan"


def test_status_and_panel_payload_stay_strict_json_after_nan_rounds(store, tmp_path):
    """Finite-only best tracking + sanitize keep status/panel portable (#271)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    best = []
    for n, reward in enumerate(
            (float("nan"), 0.4, float("inf"), 0.9)):
        rd.append_round(_round(reward, n=n))
        finite = runstore._finite_or_none(reward)
        if finite is not None:
            best.append(finite)
        rd.update_status(
            round=n + 1, last_reward=finite,
            best_reward=max(best) if best else None)
    st = rd.status()
    assert st.best_reward == 0.9 and st.last_reward == 0.9
    raw_status = Path(rd.status_path).read_text()
    assert "NaN" not in raw_status and "Infinity" not in raw_status
    body = json.dumps(
        runstore._sanitize_jsonable(
            {**st.to_dict(), "rounds": rd.rounds()}),
        default=str, allow_nan=False)
    json.loads(body)
    assert "NaN" not in body and "Infinity" not in body

class _RaiseOnStr:
    def __str__(self):
        raise TypeError("cannot stringify for status")


def test_unencodable_status_write_leaves_no_orphan_tmp(store, tmp_path):
    """A serializer refusal must not leave status.json.<uuid>.tmp behind (#267)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    before = set(os.listdir(rd.path))
    with pytest.raises(TypeError, match="cannot stringify"):
        rd.update_status(error=_RaiseOnStr())
    after = set(os.listdir(rd.path))
    assert after == before
    assert not [f for f in after if f.endswith(".tmp")]
    # Prior status remains readable and unchanged.
    assert rd.status().state == "created"


def test_failed_replace_still_removes_the_sibling_tmp(store, tmp_path, monkeypatch):
    """os.replace failure must clean the sibling, not leave an orphan (#267)."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(runstore.os, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        rd.update_status(round=9, best_reward=0.9)
    assert not [f for f in os.listdir(rd.path) if f.endswith(".tmp")]
    # Old status untouched.
    assert rd.status().round == 0

    monkeypatch.setattr(runstore.os, "replace", real_replace)
    rd.update_status(round=3)
    assert rd.status().round == 3
    assert not [f for f in os.listdir(rd.path) if f.endswith(".tmp")]


def test_encode_refusal_never_creates_a_tmp(tmp_path):
    """json.dumps must run before open(tmp); a raise leaves zero new files (#267)."""
    path = str(tmp_path / "status.json")
    before = set(os.listdir(tmp_path))
    with pytest.raises(TypeError, match="cannot stringify"):
        runstore._write_json(path, {"error": _RaiseOnStr()})
    assert set(os.listdir(tmp_path)) == before
    assert not os.path.exists(path)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_a_dead_pid_reads_as_failed_not_running(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    rd.update_status(state="running", pid=2 ** 22 - 1)     # not a live process
    st = rd.status()
    assert st.state == "failed" and "without reporting" in st.error


def test_execute_in_process_keeps_the_directory_current(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    result = runstore.execute(rd)
    assert result.error is None and result.final_reward == 1.0
    st = rd.status()
    assert st.state == "done" and st.stop_reason and st.best_reward == 1.0
    assert st.round == len(result.history) and st.calls > 0
    assert rd.rounds()[0]["round"] == 0
    assert rd.result().final_reward == 1.0
    assert os.path.exists(os.path.join(rd.tree_path, "rules.md"))
    assert os.path.isdir(rd.ledger_path)


def test_execute_reports_a_spec_error_in_status(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    d = spec.to_dict()
    d["score"] = "sideways"
    rd = runstore.create(d, store=store)
    with pytest.raises(Exception):
        runstore.execute(rd)
    st = rd.status()
    assert st.state == "failed" and "SpecError" in st.error


def test_launch_detaches_and_the_child_reports(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    st = runstore.launch(rd)
    assert st.state == "running" and st.pid
    with pytest.raises(RunStoreError, match="already running"):
        runstore.launch(rd)
    st = _wait(rd)
    assert st.state == "done", rd.log_tail()
    assert rd.result().final_reward == 1.0
    assert [s.run_id for s in runstore.list_runs(store=store)] == [rd.run_id]


def test_cancel_kills_the_run_and_its_workers(store, tmp_path):
    slow = {"ref": "cli_agent",
            "command": [sys.executable, "-c", "import time,sys; time.sleep(60); print(sys.argv[1])"]}
    _, spec = _spec_file(tmp_path, agent=slow, evolve={"rounds": 5, "n_workers": 1})
    rd = runstore.create(spec.to_dict(), store=store)
    st = runstore.launch(rd)
    time.sleep(2.0)                                # let it spawn a worker
    assert rd.status().state == "running"
    st = runstore.cancel(rd.run_id, store=store, grace=2.0)
    assert st.state == "cancelled"
    assert not RunStatus(run_id="x", pid=st.pid).alive()
    # nothing in the group survived: no sleeping python owned by that session
    out = subprocess.run(["ps", "-o", "pid=,sid=,args="], capture_output=True, text=True).stdout
    assert not [l for l in out.splitlines() if "time.sleep(60)" in l], out


def test_resume_continues_on_the_same_ledger(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    runstore.execute(rd)
    n_before = len(rd.rounds())
    st = runstore.resume(rd.run_id, store=store)
    assert st.state == "running"
    st = _wait(rd)
    assert st.state == "done", rd.log_tail()
    assert len(rd.rounds()) > n_before          # appended, not restarted
    assert os.path.isdir(rd.ledger_path)


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_init_guesses_the_kind(tmp_path):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# x")
    assert cli.starter_spec(str(skill))["kind"] == "skill_dir"
    code = tmp_path / "bot"
    (code / "tests").mkdir(parents=True)
    (code / "main.py").write_text("print(1)")
    assert cli.starter_spec(str(code))["kind"] == "agent_code"
    plugin = tmp_path / "plug"
    (plugin / ".claude-plugin").mkdir(parents=True)
    s = cli.starter_spec(str(plugin))
    assert s["kind"] == "plugin" and s["host"] == "claude_code"
    prompt = tmp_path / "p.md"
    prompt.write_text("be terse")
    assert cli.starter_spec(str(prompt))["kind"] == "text"
    agents = tmp_path / ".claude" / "agents" / "reviewer"
    agents.mkdir(parents=True)
    assert cli.starter_spec(str(agents))["kind"] == "agent_dir"


def test_init_writes_a_spec_that_plan_can_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "s"
    skill.mkdir()
    code, out = _cli("init", str(skill), "--out", "spec.json")
    assert code == 0 and os.path.exists("spec.json")
    # plan fails cleanly on the placeholder data path, naming the field
    code, out = _cli("--json", "plan", "spec.json")
    assert code == 2 and "does not exist" in out


def test_plan_reports_estimate_and_notes(store, tmp_path):
    path, _ = _spec_file(tmp_path)
    code, out = _cli("--json", "plan", path)
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] and payload["tasks"] == len(_WORDS)
    assert payload["estimate"]["agent_calls_upper_bound"] > 0
    assert payload["evolve_kwargs"]["blast_radius"] == 0.2
    assert payload["evolve_kwargs"]["strategy"] == "FileTree"


def test_plan_rejects_fewer_than_four_tasks(tmp_path):
    """plan must fail closed on the same 4-task rule evolve raises (#343)."""
    from agentdescent.cli import plan_payload
    from agentdescent.evolvespec import EvolveSpec

    rows = [{"prompt": f"q{i}", "gold": f"a{i}"} for i in range(3)]
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Answer.\n")
    spec = EvolveSpec.from_dict({
        "kind": "text", "target": str(prompt),
        "data": {"path": str(cases)}, "score": "contains",
        "agent": "echo", "reflect": "echo",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0},
    })
    payload = plan_payload(spec)
    assert payload["ok"] is False
    assert payload["tasks"] == 3
    assert "need at least 4 tasks to split train/held-out" in payload["error"]

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    code, out = _cli("--json", "plan", str(spec_path))
    assert code == 2
    assert json.loads(out)["ok"] is False


def test_plan_ok_at_four_tasks_with_soft_note_below_eight(tmp_path):
    from agentdescent.cli import plan_payload
    from agentdescent.evolvespec import EvolveSpec, compose

    rows = [{"prompt": f"q{i}", "gold": f"a{i}"} for i in range(4)]
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Answer.\n")
    spec = EvolveSpec.from_dict({
        "kind": "text", "target": str(prompt),
        "data": {"path": str(cases)}, "score": "contains",
        "agent": "echo", "reflect": "echo",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0},
    })
    payload = plan_payload(spec)
    assert payload["ok"] is True and payload["tasks"] == 4
    assert any("8-20 is a workable minimum" in n for n in payload["notes"])
    # compose itself no longer soft-notes below 4 (that is a hard plan error)
    few = EvolveSpec.from_dict({
        "kind": "text", "target": str(prompt),
        "data": {"inline": rows[:1]}, "score": "contains",
        "agent": "echo", "reflect": "echo",
        "evolve": {"rounds": 1},
    })
    notes = compose(few).notes
    assert not any("8-20 is a workable minimum" in n for n in notes)


def test_evolve_status_show_apply_round_trip(store, tmp_path):
    path, spec = _spec_file(tmp_path)
    code, out = _cli("--store", store, "--json", "evolve", path)
    assert code == 0, out
    rid = json.loads(out)["run_id"]

    code, out = _cli("--store", store, "--json", "status", rid)
    assert code == 0
    st = json.loads(out)
    assert st["state"] == "done" and st["recent_rounds"]

    code, out = _cli("--store", store, "--json", "status")
    assert json.loads(out)[0]["run_id"] == rid

    code, out = _cli("--store", store, "--json", "show", rid)
    payload = json.loads(out)
    assert payload["final_reward"] == 1.0
    assert "-MODE: forward" in payload["diff"] and "+MODE: reverse" in payload["diff"]
    assert payload["apply_plan"]["written"] == ["rules.md"]

    # apply: dry run first, then for real, backed up
    code, out = _cli("--store", store, "apply", rid, "--dry-run")
    assert json.loads(out)["written"] == ["rules.md"]
    with open(os.path.join(spec.target, "rules.md")) as fh:
        assert "forward" in fh.read()
    code, out = _cli("--store", store, "apply", rid)
    plan = json.loads(out)
    assert plan["backup"]
    with open(os.path.join(spec.target, "rules.md")) as fh:
        assert "MODE: reverse" in fh.read()


def test_applying_a_text_target_twice_keeps_both_backups(store, tmp_path):
    """`.bak` was a single slot, so the second apply destroyed the first backup.

    Apply, hand-edit, apply again: the surviving `.bak` held the hand-edit and
    the original the first apply had saved was gone -- from the command whose
    whole promise is that it keeps one. The directory path has always numbered
    them; this is that, for a file.
    """
    from tests.test_evolvespec import _rows_file

    target = tmp_path / "prompt.txt"
    target.write_text("original\n", encoding="utf-8")
    # kind=text needs a plain Completion, not a WorkspaceAgent (#340)
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

    # The run committed nothing, so it evolved the text it started from; applying
    # that over an identical file is a no-op (#396) and makes no backup. Edit the
    # target after the run so there is something for `apply` to overwrite.
    target.write_text("original\n", encoding="utf-8")
    assert json.loads(_cli("--store", store, "apply", rid)[1])["backup"] == []
    target.write_text("edited since the run\n", encoding="utf-8")
    first = json.loads(_cli("--store", store, "apply", rid)[1])["backup"]
    assert first == [str(target) + ".bak-0"]
    assert open(first[0]).read() == "edited since the run\n"

    target.write_text("hand-edited\n", encoding="utf-8")
    second = json.loads(_cli("--store", store, "apply", rid)[1])["backup"]
    assert second == [str(target) + ".bak-1"], second

    # the point: the first backup is still the original, untouched
    assert open(first[0]).read() == "edited since the run\n"
    assert open(second[0]).read() == "hand-edited\n"


def test_evolve_detach_then_watch(store, tmp_path):
    path, _ = _spec_file(tmp_path)
    code, out = _cli("--store", store, "--json", "evolve", path, "--detach")
    assert code == 0, out
    rid = json.loads(out)["run_id"]
    code, out = _cli("--store", store, "watch", rid, "--interval", "0.2")
    assert code == 0, out
    assert "round   0" in out and "done" in out


def test_status_brief_prints_only_live_runs(store, tmp_path):
    path, _ = _spec_file(tmp_path)
    _cli("--store", store, "evolve", path)
    code, out = _cli("--store", store, "status", "--brief")
    assert code == 0 and out == ""


def test_spec_errors_exit_2_with_the_field(store, tmp_path):
    path, spec = _spec_file(tmp_path)
    d = spec.to_dict()
    d["agent"] = "echo"
    with open(path, "w") as fh:
        json.dump(d, fh)
    code, out = _cli("--json", "plan", path)
    assert code == 2 and "WorkspaceAgent" in out


def test_doctor_reports_without_failing():
    rep = cli.doctor_report()
    assert set(rep) >= {"agent_clis", "provider_keys_present", "problems", "run_store"}
    assert rep["agent_clis"]["git"]


def test_console_script_entry_point_is_declared():
    with open(os.path.join(ROOT, "pyproject.toml")) as fh:
        cfg = fh.read()
    assert 'agentdescent = "agentdescent.cli:main"' in cfg
    assert 'mcp = ["mcp' in cfg


def test_module_entry_point_runs_as_a_subprocess():
    proc = subprocess.run([sys.executable, "-m", "agentdescent.cli", "--help"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0 and "evolve" in proc.stdout


def test_a_detached_run_started_with_relative_paths_finds_its_data(store, tmp_path, monkeypatch):
    """The regression: the child's cwd is the run directory, not the caller's.

    A spec with `data.path: cases.jsonl` planned cleanly in the parent and then
    died in the detached child with "cases.jsonl does not exist"."""
    path, spec = _spec_file(tmp_path)
    d = spec.to_dict()
    d["target"] = os.path.relpath(d["target"], str(tmp_path))
    d["data"] = {**d["data"], "path": os.path.relpath(d["data"]["path"], str(tmp_path))}
    with open(path, "w") as fh:
        json.dump(d, fh)
    monkeypatch.chdir(tmp_path)

    code, out = _cli("--store", store, "--json", "evolve", "spec.json", "--detach")
    assert code == 0, out
    rd = runstore.get(json.loads(out)["run_id"], store=store)
    st = _wait(rd)
    assert st.state == "done", rd.log_tail()
    # the stored spec is the absolute one, so the run is re-runnable from anywhere
    assert os.path.isabs(rd.spec_dict()["data"]["path"])


def test_the_rollout_counter_is_not_a_sum_of_sums(tmp_path):
    """`RoundInfo.rollouts` is cumulative by definition, and the run store was
    adding it to the previous total. Measured on a 630-round run: 1,270 calls
    and 397,530 rollouts -- 630x631, the triangular number. It is one of four
    numbers `status` shows someone deciding whether to cancel.
    """
    from agentdescent import runstore
    from agentdescent.demo import build

    spec = build(str(tmp_path))          # writes the skill and the cases too
    spec["evolve"] = {"rounds": 6, "n_workers": 2, "seed": 0,
                      "target_reward": 2.0, "patience": 99}
    rd = runstore.create(spec, store=str(tmp_path / "store"))
    runstore.execute(rd)

    rounds = rd.rounds()
    assert rounds, "the run recorded no rounds"
    st = rd.status()
    # The last round's cumulative count is the run's count -- not the sum of
    # every round's running total.
    assert st.rollouts == rounds[-1]["rollouts"], (
        st.rollouts, [r["rollouts"] for r in rounds])
    assert st.rollouts <= st.calls, (st.rollouts, st.calls)


# ---------------------------------------------------------------------------
# budget_usd needs usd_per_call (#342)
# ---------------------------------------------------------------------------


def test_require_budget_pair_raises_clearly():
    with pytest.raises(RunStoreError, match="budget_usd needs usd_per_call"):
        runstore.require_budget_pair(5.0, None)
    with pytest.raises(RunStoreError, match="budget_usd needs usd_per_call"):
        runstore.require_budget_pair(5.0, 0.0)
    runstore.require_budget_pair(None, None)          # no budget: fine
    runstore.require_budget_pair(5.0, 0.02)           # both set: fine


def test_evolve_budget_without_usd_per_call_creates_no_run(store, tmp_path):
    path, _ = _spec_file(tmp_path)
    before = set(os.listdir(store)) if os.path.isdir(store) else set()
    code, out = _cli("--store", store, "--json", "evolve", path, "--budget", "5")
    assert code == 2
    # stderr is not captured by _cli; the exit code + no new run dir is the
    # acceptance criterion. Re-run via main with stderr captured for the message.
    import io
    from contextlib import redirect_stderr, redirect_stdout
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code2 = cli.main(["--store", store, "evolve", path, "--budget", "5"])
    assert code2 == 2 and "budget_usd needs usd_per_call" in err.getvalue()
    after = set(os.listdir(store)) if os.path.isdir(store) else set()
    assert after == before


def test_evolve_detach_budget_without_usd_per_call_creates_no_run(store, tmp_path):
    path, _ = _spec_file(tmp_path)
    before = set(os.listdir(store)) if os.path.isdir(store) else set()
    import io
    from contextlib import redirect_stderr, redirect_stdout
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = cli.main(["--store", store, "evolve", path, "--detach", "--budget", "5"])
    assert code == 2 and "budget_usd needs usd_per_call" in err.getvalue()
    after = set(os.listdir(store)) if os.path.isdir(store) else set()
    assert after == before


def test_execute_budget_without_price_writes_status_error(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    with pytest.raises(RunStoreError, match="budget_usd needs usd_per_call"):
        runstore.execute(rd, budget_usd=5.0, usd_per_call=None)
    st = rd.status()
    assert st.state == "failed"
    assert "budget_usd needs usd_per_call" in (st.error or "")


def test_status_surfaces_error_from_log_when_process_dies(store, tmp_path):
    """A child that died after writing only log.txt must not read as
    'process ended without reporting' when the log names the real failure."""
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    # Pretend a detached child marked itself running, then died, leaving the
    # clear message only in log.txt (the pre-fix detach/MCP failure mode).
    # A pid that cannot be alive on this machine (not 0: kill(0,0) hits the group).
    rd.update_status(state="running", pid=2_000_000_000, started=time.time())
    with open(rd.log_path, "w", encoding="utf-8") as fh:
        fh.write("error: budget_usd needs usd_per_call: the engine counts calls, "
                 "and a tool-using agent's price per call is not known in advance\n")
    st = rd.status()
    assert st.state == "failed"
    assert "budget_usd needs usd_per_call" in (st.error or "")
    assert "process ended without reporting" not in (st.error or "")


def test_launch_refuses_budget_without_usd_per_call(store, tmp_path):
    _, spec = _spec_file(tmp_path)
    rd = runstore.create(spec.to_dict(), store=store)
    with pytest.raises(RunStoreError, match="budget_usd needs usd_per_call"):
        runstore.launch(rd, budget_usd=5.0)
    assert rd.status().state == "created"


# ---------------------------------------------------------------------------
# missing CLI binary → plan/evolve fail closed (#346)
# kind=text + CLI WorkspaceAgent is refused at compose (#340); PATH checks use
# skill_dir so the missing-binary gate stays covered for directory kinds.
# ---------------------------------------------------------------------------


def _fake_doctor_clis(monkeypatch, *, present=(), absent=None):
    """Patch doctor_report so named agent binaries look present or missing."""
    import agentdescent.cli as cli_mod

    absent = set(absent or ("claude", "codex", "dsh", "opencode"))
    present = set(present)
    real = cli_mod.doctor_report

    def fake():
        rep = real()
        clis = dict(rep["agent_clis"])
        for name in ("claude", "codex", "dsh", "opencode"):
            if name in present:
                clis[name] = clis.get(name) or f"/fake/{name}"
            elif name in absent:
                clis[name] = None
        return {**rep, "agent_clis": clis}

    monkeypatch.setattr(cli_mod, "doctor_report", fake)


def _dir_spec_with_cli_agent(tmp_path, agent, *, sub="cli"):
    """skill_dir + named CLI agent — compose succeeds; PATH gate can fire (#346)."""
    root = tmp_path / sub
    root.mkdir(exist_ok=True)
    return _dir_spec(str(root), agent=agent, reflect="echo",
                     evolve={"rounds": 1, "n_workers": 1, "seed": 0})


def _text_spec_with_agent(tmp_path, agent, n_tasks=8):
    from agentdescent.evolvespec import EvolveSpec

    rows = [{"prompt": f"q{i}", "gold": f"a{i}"} for i in range(n_tasks)]
    cases = tmp_path / "cases-cli.jsonl"
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    prompt = tmp_path / "prompt-cli.txt"
    prompt.write_text("Answer.\n")
    return EvolveSpec.from_dict({
        "kind": "text", "target": str(prompt),
        "data": {"path": str(cases)}, "score": "contains",
        "agent": agent, "reflect": "echo",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0},
    })


def test_plan_rejects_missing_agent_cli(tmp_path, monkeypatch):
    """plan must fail closed when claude_code's binary is not on PATH (#346)."""
    from agentdescent.cli import plan_payload, require_cli_on_path
    from agentdescent.evolvespec import SpecError

    _fake_doctor_clis(monkeypatch, present=(), absent=("claude", "codex", "dsh", "opencode"))
    spec = _dir_spec_with_cli_agent(tmp_path, {"ref": "claude_code"})
    payload = plan_payload(spec)
    assert payload["ok"] is False
    err = payload["error"]
    assert "claude_code" in err and "claude" in err and "on PATH" in err
    assert "install" in err or "openai_compatible" in err
    # missing binary is a hard error, not a soft warning
    assert not any("needs `claude` on PATH" in w for w in payload["warnings"])

    with pytest.raises(SpecError, match="claude_code"):
        require_cli_on_path(spec)

    # string short name is checked the same way
    spec_str = _dir_spec_with_cli_agent(tmp_path, "codex", sub="cli-codex")
    payload2 = plan_payload(spec_str)
    assert payload2["ok"] is False and "codex" in payload2["error"]


def test_plan_ok_when_named_cli_is_on_path(tmp_path, monkeypatch):
    _fake_doctor_clis(monkeypatch, present=("claude",),
                      absent=("codex", "dsh", "opencode"))
    from agentdescent.cli import plan_payload, require_cli_on_path

    spec = _dir_spec_with_cli_agent(tmp_path, {"ref": "claude_code", "isolate": False})
    payload = plan_payload(spec)
    assert payload["ok"] is True
    assert "error" not in payload
    require_cli_on_path(spec)  # does not raise

    # non-CLI agents stay ok regardless of CLI PATH (kind=text + echo)
    _fake_doctor_clis(monkeypatch, present=(), absent=("claude", "codex", "dsh", "opencode"))
    echo_spec = _text_spec_with_agent(tmp_path, "echo")
    assert plan_payload(echo_spec)["ok"] is True
    require_cli_on_path(echo_spec)


def test_plan_suggests_present_cli_when_default_missing(tmp_path, monkeypatch):
    _fake_doctor_clis(monkeypatch, present=("opencode",),
                      absent=("claude", "codex", "dsh"))
    from agentdescent.cli import plan_payload

    payload = plan_payload(_dir_spec_with_cli_agent(tmp_path, {"ref": "claude_code"}))
    assert payload["ok"] is False
    assert "`opencode`" in payload["error"]


def test_evolve_refuses_missing_cli_without_creating_a_run(store, tmp_path, monkeypatch):
    _fake_doctor_clis(monkeypatch, present=(), absent=("claude", "codex", "dsh", "opencode"))
    spec = _dir_spec_with_cli_agent(tmp_path, {"ref": "claude_code"})
    path = tmp_path / "missing-cli-spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    before = set(os.listdir(store)) if os.path.isdir(store) else set()
    import io
    from contextlib import redirect_stderr, redirect_stdout
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = cli.main(["--store", store, "--json", "evolve", str(path), "--detach"])
    assert code == 2
    assert "claude" in err.getvalue() and "on PATH" in err.getvalue()
    after = set(os.listdir(store)) if os.path.isdir(store) else set()
    assert after == before


def test_plan_rejects_text_with_cli_workspace_agent(tmp_path, monkeypatch):
    """kind=text + claude_code fails at compose even when the CLI is on PATH (#340)."""
    from agentdescent.evolvespec import SpecError, compose

    _fake_doctor_clis(monkeypatch, present=("claude", "codex", "dsh", "opencode"))
    spec = _text_spec_with_agent(tmp_path, {"ref": "claude_code", "isolate": False})
    with pytest.raises(SpecError, match=r"agent:.*kind='text'.*WorkspaceAgent"):
        compose(spec)
    # cmd_plan / MCP plan catch SpecError → ok: false (plan_payload raises)
    import io
    from contextlib import redirect_stdout
    path = tmp_path / "text-cli-spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["--json", "plan", str(path)])
    assert code == 2
    out = json.loads(buf.getvalue())
    assert out["ok"] is False and "WorkspaceAgent" in out["error"] and "agent" in out["error"]


# ---------------------------------------------------------------------------
# in-flight status during a long actor call (#358)
# ---------------------------------------------------------------------------


_SLOW_REFLECT_GATE = threading.Event()
_SLOW_REFLECT_ENTERED = threading.Event()


def _slow_stub_reflect(prompt: str) -> str:
    """Blocks until the test releases the gate, so status can observe pending."""
    from tests.test_evolvespec import stub_reflect
    _SLOW_REFLECT_ENTERED.set()
    if not _SLOW_REFLECT_GATE.wait(timeout=30):
        raise TimeoutError("test gate was not released")
    return stub_reflect(prompt)


def test_status_line_shows_pending_phase():
    st = RunStatus(run_id="r1", state="running", round=0, rounds=1,
                   calls=0, pending_calls=1, phase="propose",
                   kind="skill_dir", target="/tmp/x")
    line = cli._status_line(st)
    assert "calls=0" in line
    assert "pending=1" in line
    assert "phase=propose" in line
    # quiet when idle
    idle = RunStatus(run_id="r1", state="running", calls=3)
    assert "pending=" not in cli._status_line(idle)


def test_execute_writes_pending_while_propose_is_in_flight(store, tmp_path):
    """status.json must show pending/phase before the first propose returns."""
    _SLOW_REFLECT_GATE.clear()
    _SLOW_REFLECT_ENTERED.clear()
    _, spec = _spec_file(
        tmp_path,
        reflect={"ref": "tests.test_runstore_cli:_slow_stub_reflect", "call": False},
        evolve={"rounds": 1, "n_workers": 1, "seed": 0},
    )
    # allow the test module as a reflect target
    d = spec.to_dict()
    d["allow"] = list(dict.fromkeys((d.get("allow") or []) + ["tests."]))
    rd = runstore.create(d, store=store)

    errors = []

    def runner():
        try:
            runstore.execute(rd)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert _SLOW_REFLECT_ENTERED.wait(timeout=30), "propose never started"

    # Poll until the progress hook has rewritten status.json
    deadline = time.time() + 10
    st = None
    while time.time() < deadline:
        st = rd.status()
        if st.pending_calls >= 1:
            break
        time.sleep(0.05)
    assert st is not None and st.pending_calls >= 1, st
    assert st.phase == "propose"
    assert st.state == "running"
    # A prior run() may already have recorded completed calls; the bug is that
    # status showed no in-flight signal at all during a long propose (#358).

    payload = cli.status_payload(rd.run_id, store=store)
    assert payload["pending_calls"] >= 1 and payload["phase"] == "propose"

    _SLOW_REFLECT_GATE.set()
    t.join(timeout=120)
    assert not t.is_alive(), "execute did not finish"
    assert not errors, errors
    done = rd.status()
    assert done.state == "done"
    assert done.pending_calls == 0 and done.phase is None
    assert done.calls > 0


def test_update_status_serializes_concurrent_rewrites(tmp_path, monkeypatch):
    """``update_status`` is a read-modify-replace; actor-thread progress
    callbacks overlap ``on_round`` in async runs, so a delayed writer must
    not restore the document it read before a newer write (#358 review)."""
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    rd = runstore.RunDir(str(run_dir))
    rd.update_status(state="running")

    real_read = runstore._read_json
    first_read = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def slow_first_read(path):
        out = real_read(path)
        calls["n"] += 1
        if calls["n"] == 1:
            first_read.set()
            assert release.wait(5)
        return out

    monkeypatch.setattr(runstore, "_read_json", slow_first_read)
    slow = threading.Thread(target=lambda: rd.update_status(calls=1,
                                                            pending_calls=1))
    slow.start()
    assert first_read.wait(5)
    fast = threading.Thread(
        target=lambda: runstore.RunDir(str(run_dir)).update_status(round=5))
    fast.start()
    time.sleep(0.2)  # unserialized, `fast` would finish its write here
    release.set()
    slow.join(5)
    fast.join(5)
    st = rd.status()
    assert (st.round, st.calls, st.pending_calls) == (5, 1, 1)
