"""Seed held-out reward surfaces on result / CLI / status (#708)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Optional

from agentdescent import EvolveSpec, Task, cli, demo, evolve, runstore
from agentdescent.evolution import EvolutionResult, Strategy


class _Slot:
    """One-slot strategy: empty seed scores 0; proposing GOAL makes it score 1."""

    def initial(self):
        return {}

    def render(self, state):
        return state.get("v", "")

    def to_diff(self, state, proposal, author, base_version, target):
        from agentdescent.evolvable import Diff
        text = (proposal or "").strip()
        if not text or state.get("v") == text:
            return None
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": text}, author=author)


assert isinstance(_Slot(), Strategy)


class _Agent:
    def solve(self, skill_text, task):
        return skill_text if skill_text else "wrong"

    def propose(self, skill_text, task, output, reward) -> Optional[str]:
        return task.meta["expected"]


def test_evolve_records_initial_and_final_reward():
    tasks = [Task(id=f"t{i}", prompt="x", meta={"expected": "GOAL"})
             for i in range(10)]
    reward = lambda t, o: 1.0 if o == t.meta["expected"] else 0.0
    result = evolve(tasks, reward, agent=_Agent(), strategy=_Slot(),
                    rounds=3, n_workers=1, held_out_frac=0.4,
                    self_verify=False)
    assert result.error is None, result.error
    assert result.initial_reward == 0.0
    assert result.final_reward == 1.0


def test_result_save_load_round_trips_initial_reward(tmp_path):
    path = tmp_path / "r.json"
    r = EvolutionResult(
        state={"a": "1"}, rendered="a", final_reward=1.0,
        history=[], ledger_log=[], initial_reward=0.25)
    r.save(str(path))
    assert EvolutionResult.load(str(path)).initial_reward == 0.25
    raw = json.loads(path.read_text())
    del raw["initial_reward"]
    path.write_text(json.dumps(raw))
    assert EvolutionResult.load(str(path)).initial_reward is None


def test_runstore_status_and_show_carry_initial_reward(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    store = str(tmp_path / "runs")
    work = tmp_path / "work"
    spec_dict = demo.build(str(work))
    spec_dict["evolve"] = dict(spec_dict.get("evolve") or {}, rounds=2, n_workers=1)
    spec = EvolveSpec.from_dict(spec_dict)
    rd = runstore.create(spec.to_dict(), store=store)
    result = runstore.execute(rd)
    assert result.error is None, result.error
    assert result.initial_reward is not None
    assert result.initial_reward < result.final_reward == 1.0
    st = rd.status()
    assert st.initial_reward == result.initial_reward

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["--store", store, "status", rd.run_id, "--json"]) == 0
    stj = json.loads(buf.getvalue())
    assert stj["initial_reward"] == result.initial_reward

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["--store", store, "show", rd.run_id, "--no-diff"]) == 0
    out = buf.getvalue()
    assert f"initial_reward: {result.initial_reward}" in out
    assert f"final_reward: {result.final_reward}" in out


def test_demo_prints_initial_to_final_lift(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["--store", str(tmp_path / "runs"), "demo",
                         "--dir", str(tmp_path / "work")])
    out = buf.getvalue()
    assert code == 0, out
    assert "held-out reward: 0.000 -> 1.000" in out, out
