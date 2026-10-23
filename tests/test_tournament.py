"""The algorithm tournament has to run real arms and report a measured winner."""

from __future__ import annotations

import json

import pytest

from agentdescent import (AlgorithmTournamentReport, cli, run_tournament,
                          TournamentReport)
import agentdescent.tournament as tournament
from agentdescent.rule_tournament import TournamentReport as RuleTournamentReport
from tests.test_evolvespec import _dir_spec


def test_two_arms_run_under_the_same_spec_and_report_a_winner(tmp_path):
    spec = _dir_spec(str(tmp_path))
    report = run_tournament(spec, ["default", "archive_selection"],
                            seeds=(0,), rounds=1,
                            repo_root=str(tmp_path / "repos"))
    assert report.kind == "skill_dir"
    assert [e.algorithm for e in report.arms] == ["default", "archive_selection"]
    assert all(e.error is None for e in report.arms)
    assert {a.algorithm for a in report.algorithms} == {"default", "archive_selection"}
    assert report.arms[0].outcomes or report.arms[1].outcomes
    assert not (tmp_path / "repos").exists() or not any((tmp_path / "repos").iterdir())


def test_an_unknown_or_wrong_kind_algorithm_is_refused(tmp_path):
    spec = _dir_spec(str(tmp_path))
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_tournament(spec, ["nope"], seeds=(0,))
    with pytest.raises(ValueError, match="applies to kind"):
        run_tournament(spec, ["rrsi"], seeds=(0,))
    with pytest.raises(ValueError, match="at least one algorithm"):
        run_tournament(spec, [], seeds=(0,))


def test_cli_json_prints_the_measured_ranking(tmp_path, capsys):
    spec = _dir_spec(str(tmp_path))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    code = cli.main(["--json", "tournament", str(path),
                     "--algorithm", "default",
                     "--algorithm", "archive_selection",
                     "--rounds", "1"])
    assert code in (0, 1)          # 1 means the two tied, which is a valid answer
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "skill_dir"
    assert {e["algorithm"] for e in out["arms"]} == {"default", "archive_selection"}
    assert {a["algorithm"] for a in out["algorithms"]} == {"default", "archive_selection"}


def test_the_new_report_does_not_shadow_rule_tournament():
    assert TournamentReport is RuleTournamentReport
    assert AlgorithmTournamentReport is not TournamentReport


class _FakeResult:
    def __init__(self, reward, calls=5, error=None):
        self.final_reward = reward
        self.stop_reason = "rounds"
        self.error = error
        self.usage = type("U", (), {"calls": calls})()
        self.outcomes = lambda: {"committed": 1}


class _FakeComp:
    def __init__(self, reward, calls=5):
        self.tasks = []
        self.notes = ()
        self._result = _FakeResult(reward, calls)

    def run(self):
        return self._result


def test_winner_is_the_best_algorithm_across_seeds_not_the_luckiest_run(
        tmp_path, monkeypatch):
    rewards = {("default", 0): 1.0, ("default", 1): 0.2, ("default", 2): 0.2,
               ("archive_selection", 0): 0.9,
               ("archive_selection", 1): 0.9,
               ("archive_selection", 2): 0.9}

    def fake_compose(arm, repo_path=None):
        seed = int(arm.evolve["seed"])
        return _FakeComp(rewards[(arm.algorithm or "default", seed)])

    monkeypatch.setattr(tournament, "compose", fake_compose)
    spec = _dir_spec(str(tmp_path))
    report = tournament.run_tournament(
        spec, ["default", "archive_selection"], seeds=(0, 1, 2))
    assert report.winner == "archive_selection"
    assert [a.algorithm for a in report.algorithms] == \
        ["archive_selection", "default"]
    assert report.algorithms[0].mean_reward == pytest.approx(0.9)
    assert report.algorithms[0].min_reward == 0.9
    assert report.algorithms[1].max_reward == 1.0


def test_a_tie_is_reported_as_a_tie_not_a_win(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "compose",
                        lambda arm, repo_path=None: _FakeComp(1.0, calls=7))
    spec = _dir_spec(str(tmp_path))
    report = tournament.run_tournament(
        spec, ["default", "archive_selection"], seeds=(0, 1))
    assert report.winner is None
    assert report.tied == ("archive_selection", "default")
    assert "no measured difference" in report.reason


def test_default_keeps_the_specs_own_algorithm(tmp_path, monkeypatch):
    seen = []

    def fake_compose(arm, repo_path=None):
        seen.append(arm.algorithm)
        return _FakeComp(1.0)

    monkeypatch.setattr(tournament, "compose", fake_compose)
    spec = _dir_spec(str(tmp_path), algorithm="archive_selection")
    tournament.run_tournament(spec, ["default"], seeds=(0,))
    assert seen == ["archive_selection"]


def test_bad_rounds_or_max_seconds_fail_before_any_run(tmp_path):
    spec = _dir_spec(str(tmp_path))
    with pytest.raises(ValueError, match="rounds"):
        run_tournament(spec, ["default"], rounds=0)
    with pytest.raises(ValueError, match="max_seconds"):
        run_tournament(spec, ["default"], max_seconds=0)


def test_dry_run_prices_every_arm(tmp_path):
    spec = _dir_spec(str(tmp_path))
    est = tournament.estimate_tournament(spec, ["default", "archive_selection"],
                                         seeds=(0, 1), rounds=1)
    assert est["total_calls"] > 0
    assert len(est["arms"]) == 2
    assert all(row["total_calls"] == row["calls_per_arm"] * 2
               for row in est["arms"])
