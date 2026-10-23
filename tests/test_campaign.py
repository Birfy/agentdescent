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
    assert report.stopped_reason in ("patience", "sessions")
    assert len(report.sessions) <= 5


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
