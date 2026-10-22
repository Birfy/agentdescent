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
