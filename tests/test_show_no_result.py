"""`show` on a run with no result still says what is known (#466).

A failed, cancelled or still-running run used to show only ``note: no result
yet`` -- dropping the finished rounds, the reflector error and the recorded
rollouts, which are exactly what a host agent needs to diagnose it.
"""

import io
import json
import os
from contextlib import redirect_stdout

from agentdescent import cli, demo, runstore
from agentdescent.mcp import Tools

_WRONG = '''
import json

def keep_wrong(prompt):
    return "<EDITS>" + json.dumps({
        "rationale": "keep it",
        "edits": [{"path": "references/rules.md", "content": "COLUMN:  id\\n"}],
    }) + "</EDITS>"
'''


def _failed_run(tmp_path, monkeypatch, *, keep_rounds=True, keep_records=True):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    (tmp_path / "wrong_reflector_466.py").write_text(_WRONG)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = demo.build(str(tmp_path / "demo"))
    spec["reflect"] = {"ref": "wrong_reflector_466:keep_wrong", "call": False}
    spec["allow"] = ["wrong_reflector_466"]
    spec["evolve"] = dict(spec["evolve"], rounds=1)
    rd = runstore.create(spec)
    runstore.execute(rd)
    # The run dies before it can write its result: what a killed process leaves.
    os.remove(rd.result_path)
    rounds = rd.rounds()
    with open(rd.rounds_path, "w") as fh:
        for r in rounds if keep_rounds else []:
            r = dict(r, propose_errors=1,
                     propose_error="TimeoutError: The read operation timed out")
            fh.write(json.dumps(r) + "\n")
    if not keep_records:
        import shutil
        shutil.rmtree(rd.trajectories_path)
    rd.update_status(state="failed", error="process ended without reporting",
                     propose_errors=1 if keep_rounds else 0,
                     propose_error=("TimeoutError: The read operation timed out"
                                    if keep_rounds else None))
    return rd


def test_failed_run_shows_history_error_and_samples(tmp_path, monkeypatch):
    rd = _failed_run(tmp_path, monkeypatch)
    p = cli.show_payload(rd.run_id)
    assert p["note"] == "no result yet"
    assert p["state"] == "failed"
    assert p["rounds"] >= 1 and p["history"]
    assert p["history"][0]["propose_error"].startswith("TimeoutError")
    assert "held_out_reward" in p["history"][0]
    assert p["propose_errors"] == 1
    assert p["propose_error"].startswith("TimeoutError")
    assert p["samples"] and all(s["of"] == "any" for s in p["samples"])
    assert "diff" not in p and "apply_plan" not in p


def test_mcp_show_matches_and_stays_under_the_cap(tmp_path, monkeypatch):
    rd = _failed_run(tmp_path, monkeypatch)
    out = Tools(None).show(rd.run_id)
    want = cli.show_payload(rd.run_id)
    for k in ("note", "history", "propose_error", "samples"):
        assert out[k] == want[k]
    assert len(json.dumps(out["samples"]).encode()) <= cli.SHOW_SAMPLES_MAX_BYTES


def test_text_mode_prints_rounds_and_samples(tmp_path, monkeypatch):
    rd = _failed_run(tmp_path, monkeypatch)

    class A:
        run_id, store, diff, json = rd.run_id, None, False, False

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.cmd_show(A()) == 0
    text = buf.getvalue()
    assert "note: no result yet" in text
    assert "rounds so far:" in text and "TimeoutError" in text
    assert "lowest-reward rollouts (output vs gold):" in text


def test_nothing_recorded_still_just_says_no_result(tmp_path, monkeypatch):
    rd = _failed_run(tmp_path, monkeypatch, keep_rounds=False, keep_records=False)
    p = cli.show_payload(rd.run_id)
    assert p["note"] == "no result yet"
    for k in ("rounds", "history", "samples", "propose_error", "propose_errors"):
        assert k not in p
