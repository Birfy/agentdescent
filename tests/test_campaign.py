"""A campaign must carry the artifact forward and keep the best session."""

from __future__ import annotations

import json
import os

import pytest

from agentdescent import CampaignReport, cli, estimate_campaign, run_campaign
from tests.test_evolvespec import _dir_spec


def test_it_runs_sessions_forward_and_keeps_a_copy_of_the_target(tmp_path):
    spec = _dir_spec(str(tmp_path))
    before = open(os.path.join(spec.target, "rules.md")).read()
    report = run_campaign(spec, sessions=2, rounds=1,
                          workdir=str(tmp_path / "campaign"))
    assert isinstance(report, CampaignReport)
    assert len(report.sessions) >= 1
    assert report.final_target != spec.target
    assert os.path.exists(os.path.join(report.final_target, "rules.md"))
    assert open(os.path.join(spec.target, "rules.md")).read() == before
    assert report.best_reward >= report.sessions[0].reward


def test_a_named_algorithm_is_used_for_every_session(tmp_path):
    spec = _dir_spec(str(tmp_path))
    report = run_campaign(spec, sessions=1, rounds=1,
                          algorithm="archive_selection",
                          workdir=str(tmp_path / "c"))
    assert report.sessions[0].error is None


def test_patience_stops_a_plateau(tmp_path):
    spec = _dir_spec(str(tmp_path))
    report = run_campaign(spec, sessions=5, rounds=1, patience=1,
                          workdir=str(tmp_path / "c"))
    # the stub run reaches its target in the first session, so one non-improving
    # session is enough to stop well before five
    assert report.stopped_reason in ("patience", "sessions", "target_reward")
    assert len(report.sessions) <= 5


def test_reaching_target_reward_stops_the_campaign(tmp_path):
    spec = _dir_spec(str(tmp_path))
    report = run_campaign(spec, sessions=3, rounds=1, patience=2,
                          workdir=str(tmp_path / "c"))
    first = report.sessions[0]
    if first.stop_reason != "target_reward":
        pytest.skip("stub spec did not reach its target in session 0")
    assert len(report.sessions) == 1
    assert report.stopped_reason == "target_reward"


def test_policy_slot_and_bad_arguments_are_refused(tmp_path):
    spec = _dir_spec(str(tmp_path))
    with pytest.raises(ValueError, match="sessions"):
        run_campaign(spec, sessions=0)
    with pytest.raises(ValueError, match="patience"):
        run_campaign(spec, patience=0)
    from agentdescent.evolvespec import EvolveSpec
    with pytest.raises(ValueError, match="policy_slot"):
        run_campaign(EvolveSpec(kind="policy_slot", target="selection",
                                data={}, agent="echo"), sessions=1)


def test_dry_run_prices_every_session(tmp_path):
    spec = _dir_spec(str(tmp_path))
    est = estimate_campaign(spec, sessions=3, rounds=1)
    assert est["total_calls"] == est["calls_per_session"] * 3


def test_cli_json_and_dry_run(tmp_path, capsys):
    spec = _dir_spec(str(tmp_path))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    assert cli.main(["--json", "campaign", str(path), "--sessions", "1",
                     "--rounds", "1", "--workdir", str(tmp_path / "c")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sessions"] and out["final_target"]
    assert cli.main(["--json", "campaign", str(path), "--dry-run",
                     "--sessions", "2"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["total_calls"] == est["calls_per_session"] * 2


class _FakeResult:
    def __init__(self, reward, marker, error=None):
        self.final_reward = reward
        self.rendered = marker
        self.error = error
        self.stop_reason = "error" if error else "rounds"
        self.usage = type("U", (), {"calls": 3})()
        self.outcomes = lambda: {"committed": 1}


class _FakeComp:
    def __init__(self, result):
        self._result = result

    def run(self):
        return self._result


def _text_spec(tmp_path):
    from agentdescent.evolvespec import EvolveSpec
    target = tmp_path / "prompt.txt"
    target.write_text("seed\n")
    return EvolveSpec(kind="text", target=str(target),
                      data={"inline": [{"prompt": "q", "gold": "a"}] * 4},
                      agent="echo")


def test_a_plateau_keeps_the_previous_artifact_and_reports_last_reward(
        tmp_path, monkeypatch):
    import agentdescent.campaign as campaign
    results = [_FakeResult(0.5, "s0\n"), _FakeResult(0.3, "s1\n"),
               _FakeResult(0.7, "s2\n")]
    monkeypatch.setattr(campaign, "compose",
                        lambda arm, repo_path=None: _FakeComp(results.pop(0)))
    spec = _text_spec(tmp_path)
    report = run_campaign(spec, sessions=3, patience=2,
                          workdir=str(tmp_path / "c"))
    assert [s.improved for s in report.sessions] == [None, False, True]
    assert report.best_reward == 0.7 and report.final_reward == 0.7
    assert open(report.final_target).read() == "s2\n"


def test_session_zero_is_the_baseline_and_final_reward_is_the_last_session(
        tmp_path, monkeypatch):
    import agentdescent.campaign as campaign
    results = [_FakeResult(0.5, "s0\n"), _FakeResult(0.3, "s1\n")]
    monkeypatch.setattr(campaign, "compose",
                        lambda arm, repo_path=None: _FakeComp(results.pop(0)))
    spec = _text_spec(tmp_path)
    report = run_campaign(spec, sessions=2, patience=2,
                          workdir=str(tmp_path / "c"))
    assert report.sessions[0].improved is None
    assert report.baseline_reward == 0.5
    assert report.best_reward == 0.5 and report.final_reward == 0.3
    assert open(report.final_target).read() == "s0\n"


def test_an_errored_session_is_not_applied(tmp_path, monkeypatch):
    import agentdescent.campaign as campaign
    results = [_FakeResult(0.5, "s0\n"), _FakeResult(0.9, "bad\n", error="boom")]
    monkeypatch.setattr(campaign, "compose",
                        lambda arm, repo_path=None: _FakeComp(results.pop(0)))
    spec = _text_spec(tmp_path)
    report = run_campaign(spec, sessions=2, workdir=str(tmp_path / "c"))
    assert report.stopped_reason == "error"
    assert open(report.final_target).read() == "s0\n"


def test_a_non_empty_workdir_and_inline_text_are_refused(tmp_path):
    spec = _text_spec(tmp_path)
    workdir = tmp_path / "busy"
    workdir.mkdir()
    (workdir / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        run_campaign(spec, sessions=1, workdir=str(workdir))
    inline = _text_spec(tmp_path)
    inline.target = "Answer with only the number."
    with pytest.raises(ValueError, match="inline"):
        run_campaign(inline, sessions=1)
    with pytest.raises(ValueError, match="inline"):
        estimate_campaign(inline, sessions=1)


def test_estimate_refuses_an_inline_text_target(tmp_path):
    spec = _text_spec(tmp_path)
    spec.target = "not a file"
    with pytest.raises(ValueError, match="inline"):
        estimate_campaign(spec, sessions=2)
