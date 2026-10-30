"""Diagnosis must turn show's evidence into one next step."""

from __future__ import annotations

import pytest

from agentdescent import cli, diagnose_payload


def _patch(monkeypatch, payload):
    import agentdescent.cli as cli_mod
    monkeypatch.setattr(cli_mod, "show_payload",
                        lambda run_id, store=None, **kw: payload)


def test_agent_failure_evidence_is_named(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.0, "outcomes": {"below-threshold": 2},
        "samples": [{"task_id": "t", "output": "21", "gold": "321", "reward": 0.0}]})
    report = diagnose_payload("r")
    assert report.verdict == "agent-failures"
    assert report.evidence["sample"]["gold"] == "321"
    assert "samples" in " ".join(report.recommendations) or \
        "skill" in " ".join(report.recommendations)


def test_all_matching_samples_is_a_scoring_mismatch(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.0, "outcomes": {},
        "samples": [{"task_id": "t", "output": "321", "gold": "321",
                     "reward": 0.0}]})
    assert diagnose_payload("r").verdict == "scoring-mismatch"


def test_reflector_error_wins_over_samples(monkeypatch):
    _patch(monkeypatch, {"final_reward": 0.0, "propose_errors": 3,
                         "propose_error": "KeyError: answers",
                         "samples": [{"output": "x", "gold": "y", "reward": 0.0}]})
    report = diagnose_payload("r")
    assert report.verdict == "reflector-error"
    assert "doctor" in " ".join(report.recommendations)


def test_rejected_candidates_are_gate_rejections(monkeypatch):
    _patch(monkeypatch, {"final_reward": 0.4, "outcomes": {"below-threshold": 2},
                         "rejected": [{"outcome": "below-threshold"}]})
    assert diagnose_payload("r").verdict == "gate-rejected"


def test_run_failure_and_solved(monkeypatch):
    _patch(monkeypatch, {"error": "boom"})
    assert diagnose_payload("r").verdict == "run-failed"
    _patch(monkeypatch, {"final_reward": 1.0, "outcomes": {"committed": 1}})
    assert diagnose_payload("r").verdict == "solved"


def test_cli_json(monkeypatch, capsys):
    _patch(monkeypatch, {"final_reward": 1.0, "outcomes": {"committed": 1}})
    assert cli.main(["--json", "diagnose", "r"]) == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "solved"


def test_a_committed_partial_run_is_not_called_no_improvement(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.75, "outcomes": {"committed": 1},
        "samples": [{"output": "297", "gold": "999", "reward": 0.0}]})
    report = diagnose_payload("r")
    assert report.verdict == "partial-improvement"
    assert "committed 1" in report.summary


def test_gate_rejection_wins_over_the_worst_sample_rule(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.4, "outcomes": {"below-threshold": 5},
        "rejected": [{"outcome": "below-threshold"}],
        "samples": [{"output": "x", "gold": "y", "reward": 0.0}]})
    assert diagnose_payload("r").verdict == "gate-rejected"


def test_running_and_cancelled_runs_are_named(monkeypatch):
    _patch(monkeypatch, {"state": "running", "note": "no result yet"})
    assert diagnose_payload("r").verdict == "running"
    _patch(monkeypatch, {"state": "cancelled", "stop_reason": "cancelled"})
    assert diagnose_payload("r").verdict == "cancelled"


def test_truncated_samples_do_not_claim_a_scoring_mismatch(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.0, "outcomes": {},
        "samples": [{"output": "same...", "gold": "same...", "reward": 0.0,
                     "output_truncated": True, "gold_truncated": True}]})
    assert diagnose_payload("r").verdict != "scoring-mismatch"


def test_missing_gold_and_missing_reward_are_not_failures(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.6, "outcomes": {},
        "samples": [{"output": "x", "gold": None, "reward": None}]})
    assert diagnose_payload("r").verdict == "no-improvement"


def test_named_algorithm_outcomes_count_as_adopted(monkeypatch):
    """GEPA/SkillOpt/DGM record adopt under their own category, not committed."""
    for category in ("gepa-select", "skillopt-accept", "dgm-select"):
        _patch(monkeypatch, {
            "final_reward": 0.75, "outcomes": {category: 3},
            "samples": [{"output": "297", "gold": "999", "reward": 0.0}]})
        report = diagnose_payload("r")
        assert report.verdict == "partial-improvement", category
        assert "committed 3" in report.summary


def test_history_committed_wins_over_empty_committed_outcome(monkeypatch):
    _patch(monkeypatch, {
        "final_reward": 0.75,
        "outcomes": {"gepa-select": 3},
        "history": [
            {"round": 1, "committed": 1, "rejected": 0},
            {"round": 2, "committed": 1, "rejected": 0},
            {"round": 3, "committed": 1, "rejected": 0},
        ],
        "samples": [{"output": "297", "gold": "999", "reward": 0.0}]})
    report = diagnose_payload("r")
    assert report.verdict == "partial-improvement"
    assert report.evidence["committed"] == 3


def test_adopted_count_helper_prefers_history():
    from agentdescent.diagnose import adopted_count
    assert adopted_count(history=[{"committed": 1}, {"committed": 2}],
                         outcomes={"committed": 9}) == 3
    assert adopted_count(outcomes={"gepa-select": 2, "below-threshold": 5}) == 2
    assert adopted_count(outcomes={"below-threshold": 5}) == 0
    assert adopted_count(history=[], outcomes={"skillopt-accept": 1}) == 1
