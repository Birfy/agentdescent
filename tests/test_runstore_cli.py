"""The run store and the CLI, end to end and offline.

A run is a directory plus a detached process, so the tests here actually detach:
`launch` starts `python -m agentdescent.cli run`, the test polls `status.json`
until the child reports, and `cancel` is checked against a deliberately slow
agent. Everything uses the stub workspace agent from `test_evolvespec.py`.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from agentdescent import cli, runstore
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
