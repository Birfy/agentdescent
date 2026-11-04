"""A seed that already hits the target is not reported as learned (#732)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from agentdescent import cli, demo, diagnose_payload
from agentdescent.diagnose import SEED_SOLVED_HINT, seed_already_solved


def _patch(monkeypatch, payload):
    import agentdescent.cli as cli_mod
    monkeypatch.setattr(cli_mod, "show_payload",
                        lambda run_id, store=None, **kw: payload)


def test_seed_at_target_gets_its_own_verdict(monkeypatch):
    _patch(monkeypatch, {"final_reward": 1.0, "initial_reward": 1.0,
                         "stop_reason": "target_reward",
                         "outcomes": {"gepa-select": 1}})
    report = diagnose_payload("r")
    assert report.verdict == "seed-already-solved"
    assert report.evidence["initial_reward"] == 1.0
    text = " ".join(report.recommendations)
    assert "apply" not in text
    assert "cases the seed gets wrong" in text


def test_improved_run_is_still_solved(monkeypatch):
    _patch(monkeypatch, {"final_reward": 1.0, "initial_reward": 0.4,
                         "stop_reason": "target_reward",
                         "outcomes": {"committed": 1}})
    assert diagnose_payload("r").verdict == "solved"


def test_old_run_without_initial_reward_is_unchanged(monkeypatch):
    _patch(monkeypatch, {"final_reward": 1.0, "stop_reason": "target_reward",
                         "outcomes": {"committed": 1}})
    assert diagnose_payload("r").verdict == "solved"


def test_predicate_edges():
    assert seed_already_solved(0.8, 0.8, "target_reward")
    assert not seed_already_solved(0.8, 0.8, "rounds")
    assert seed_already_solved(1.0, 1.0, None)
    assert not seed_already_solved(None, 1.0, "target_reward")
    assert not seed_already_solved(0.5, 1.0, "target_reward")
    assert not seed_already_solved("x", 1.0, None)


class _Result:
    def __init__(self, init, final, stop):
        self.initial_reward, self.final_reward, self.stop_reason = init, final, stop
        self.error = None

    def outcomes(self):
        return {"gepa-select": 1}


def _run_evolve(tmp_path, monkeypatch, result, *extra):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    spec = tmp_path / "spec.json"
    spec_dict = demo.build(str(tmp_path / "work"))
    spec.write_text(json.dumps(spec_dict))
    monkeypatch.setattr(cli.runstore, "execute", lambda rd, **kw: result)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["--store", str(tmp_path / "runs"), "evolve",
                         str(spec), *extra])
    return code, buf.getvalue()


def test_evolve_prints_hint_when_seed_already_solved(tmp_path, monkeypatch):
    code, out = _run_evolve(tmp_path, monkeypatch,
                            _Result(1.0, 1.0, "target_reward"))
    assert code == 0, out
    assert "held-out reward: 1.000 -> 1.000" in out
    assert SEED_SOLVED_HINT in out


def test_evolve_json_carries_note(tmp_path, monkeypatch):
    code, out = _run_evolve(tmp_path, monkeypatch,
                            _Result(1.0, 1.0, "target_reward"), "--json")
    assert code == 0, out
    assert json.loads(out)["note"] == SEED_SOLVED_HINT


def test_evolve_has_no_hint_after_real_improvement(tmp_path, monkeypatch):
    code, out = _run_evolve(tmp_path, monkeypatch,
                            _Result(0.0, 1.0, "target_reward"), "--json")
    assert code == 0, out
    assert "note" not in json.loads(out)
    code, out = _run_evolve(tmp_path, monkeypatch, _Result(0.0, 1.0, "target_reward"))
    assert SEED_SOLVED_HINT not in out
