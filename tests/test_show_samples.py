"""`show` says why a run's reward is what it is (#452).

A run that learns nothing used to leave no rollout answers behind: `show` had
the reward and the outcomes, never what the agent actually replied. The run
store now keeps the engine's rollout records and `show` lists the worst ones
of the final artifact next to their gold.
"""

import io
import json
import os
from contextlib import redirect_stdout

from agentdescent import cli, demo, runstore
from agentdescent.mcp import Tools

# A reflector that keeps the wrong column: every candidate scores like the base,
# so the run learns nothing -- the case `show` has to explain.
_WRONG = '''
import json

def keep_wrong(prompt):
    return "<EDITS>" + json.dumps({
        "rationale": "keep it",
        "edits": [{"path": "references/rules.md", "content": "COLUMN:  id\\n"}],
    }) + "</EDITS>"
'''


def _run(tmp_path, monkeypatch, **spec_changes):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    (tmp_path / "wrong_reflector_452.py").write_text(_WRONG)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = demo.build(str(tmp_path / "demo"))
    spec["reflect"] = {"ref": "wrong_reflector_452:keep_wrong", "call": False}
    spec["allow"] = ["wrong_reflector_452"]
    spec["evolve"] = dict(spec["evolve"], rounds=1)
    spec.update(spec_changes)
    rd = runstore.create(spec)
    runstore.execute(rd)
    return rd


def _run_rejected(tmp_path, monkeypatch, **spec_changes):
    """A correct base + the wrong reflector: the candidate is a real regression.

    The failing-run fixture above is the right one for `samples`, but an
    equivalent-text candidate may sometimes commit on the gate's posterior, so
    it cannot pin `rejected` deterministically. Starting from the correct rule
    makes rejection a measured regression, not a tie.
    """
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    (tmp_path / "wrong_reflector_452.py").write_text(_WRONG)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = demo.build(str(tmp_path / "demo"))
    rules = tmp_path / "demo" / demo.DEMO_SKILL / "references" / "rules.md"
    rules.write_text("COLUMN: amount\n")
    # Make the correct base fail on a few cases, so a reflector is asked for an
    # edit (the engine only proposes on failures). `+1000` cannot be an id sum.
    cases = tmp_path / "demo" / "cases.jsonl"
    rows = [json.loads(line) for line in cases.read_text().splitlines()]
    for i in (0, 3, 6, 9):
        rows[i]["gold"] = str(int(rows[i]["gold"]) + 1000)
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    spec["reflect"] = {"ref": "wrong_reflector_452:keep_wrong", "call": False}
    spec["allow"] = ["wrong_reflector_452"]
    spec["evolve"] = dict(spec["evolve"], rounds=1, target_reward=1.1)
    spec.update(spec_changes)
    rd = runstore.create(spec)
    runstore.execute(rd)
    return rd


def test_show_lists_failing_rollouts_with_output_and_gold(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch)
    payload = cli.show_payload(rd.run_id)
    assert payload["final_reward"] < 1.0
    samples = payload["samples"]
    assert 0 < len(samples) <= cli.SHOW_SAMPLES
    gold = {str(i): r["gold"] for i, r in enumerate(
        json.loads(l) for l in (tmp_path / "demo" / "cases.jsonl").read_text().splitlines())}
    for s in samples:
        assert s["reward"] == 0.0
        assert s["output"] and s["gold"] == gold[s["task_id"]]
        assert s["output"] != s["gold"]          # the agent summed `id`, visibly
        assert s["of"] == "final"
    assert len({s["task_id"] for s in samples}) == len(samples)


def test_mcp_show_returns_the_same_samples_under_the_cap(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch)
    out = Tools(None).show(rd.run_id)
    assert out["samples"] == cli.show_payload(rd.run_id)["samples"]
    assert len(json.dumps(out["samples"]).encode()) <= cli.SHOW_SAMPLES_MAX_BYTES


def test_text_mode_prints_a_short_block(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch)

    class A:
        run_id, store, diff, json = rd.run_id, None, False, False

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.cmd_show(A()) == 0
    text = buf.getvalue()
    assert "lowest-reward rollouts (output vs gold):" in text
    assert "gold=" in text and "reward=0.000" in text


def test_record_rollouts_false_writes_nothing_and_show_is_unchanged(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch, record_rollouts=False)
    assert not os.path.exists(rd.trajectories_path)
    assert "samples" not in cli.show_payload(rd.run_id)


def test_samples_are_truncated_and_capped(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch)
    from agentdescent.evolvespec import EvolveSpec
    from agentdescent.trajectory import Trajectory, TrajectoryStore

    store = TrajectoryStore(rd.trajectories_path)
    for i in range(20):
        store.record(Trajectory(artifact_id="x", task_id=f"long{i}", env_fingerprint="",
                                rendered=f"r{i}", output="9" * 5000, reward=-1.0))
    spec = EvolveSpec.from_dict(rd.spec_dict())
    samples = cli.show_samples(rd, spec, rendered=None, n=50, max_bytes=2000)
    assert samples and all(len(s["output"]) <= cli.SHOW_SAMPLE_CHARS + 3 for s in samples)
    assert len(json.dumps(samples).encode()) <= 2000


def test_a_run_without_records_shows_no_samples_key(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(rd.trajectories_path)
    assert "samples" not in cli.show_payload(rd.run_id)


# -- rejected candidates: what the reflector tried (#452) -----------------------


def test_show_lists_rejected_candidates_with_files_and_diff(tmp_path, monkeypatch):
    rd = _run_rejected(tmp_path, monkeypatch)
    assert os.path.exists(rd.candidates_path)
    payload = cli.show_payload(rd.run_id)
    rejected = payload["rejected"]
    assert 0 < len(rejected) <= cli.SHOW_REJECTED
    for rec in rejected:
        assert rec["outcome"] and rec["outcome"] != "committed"
        assert rec["round"] == 0
        assert "references/rules.md" in rec["files"]
        assert "references/rules.md" in rec["diff"]
        assert rec["rationale"] == "keep it"
        assert 0.0 <= rec["gated_reward"] <= 1.0


def test_mcp_show_returns_the_same_rejected_under_the_cap(tmp_path, monkeypatch):
    rd = _run_rejected(tmp_path, monkeypatch)
    out = Tools(None).show(rd.run_id)
    assert out["rejected"] == cli.show_payload(rd.run_id)["rejected"]
    assert len(json.dumps(out["rejected"]).encode()) <= cli.SHOW_REJECTED_MAX_BYTES


def test_text_mode_prints_the_rejected_block(tmp_path, monkeypatch):
    rd = _run_rejected(tmp_path, monkeypatch)

    class A:
        run_id, store, diff, json = rd.run_id, None, False, False

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.cmd_show(A()) == 0
    text = buf.getvalue()
    assert "rejected candidates (what the reflector tried):" in text
    assert "references/rules.md" in text and "rationale: keep it" in text


def test_record_rollouts_false_writes_neither_diagnostic_file(tmp_path, monkeypatch):
    rd = _run(tmp_path, monkeypatch, record_rollouts=False)
    assert not os.path.exists(rd.trajectories_path)
    assert not os.path.exists(rd.candidates_path)
    payload = cli.show_payload(rd.run_id)
    assert "samples" not in payload and "rejected" not in payload


def test_rejected_records_are_bounded(tmp_path, monkeypatch):
    rd = _run_rejected(tmp_path, monkeypatch)
    with open(rd.candidates_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert 0 < len(records) <= runstore.CANDIDATES_PER_ROUND
    for rec in records:
        assert len(rec["diff"]) <= runstore.CANDIDATE_DIFF_CHARS + 3
        assert len(rec["files"]) <= 20
