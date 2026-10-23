"""The safety case has to be stricter than any one trust mechanism alone."""

from __future__ import annotations

import json
import time

import pytest

from agentdescent import ConsolidationStore, EvolutionResult, problem_descriptor, safety_report
from agentdescent import cli


def _archive(store, run_id, slot, reward, *, exam=None):
    store.archive(
        EvolutionResult(state={"v": "x"}, rendered="x", final_reward=reward,
                        history=[], ledger_log=[]),
        run_id=run_id, strategy="SourceSlot", config={"dataset": "safety"},
        artifact_id="meta-rule", problem=problem_descriptor(slot))
    if exam is not None:
        store.mark_exam(run_id, exam=exam)
    time.sleep(0.005)


class _Audit:
    def __init__(self, biases):
        self.biases = biases

    def gain_bias(self, run_id):
        return self.biases.get(run_id)


def test_an_empty_archive_is_caution(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    report = safety_report(store)
    assert report.verdict == "caution"
    assert any(c.name == "archive" and c.status == "warn" for c in report.checks)


def test_passed_rules_with_a_clean_audit_are_safe(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="passed")
    _archive(store, "r2", "task_sampler", 0.6, exam="passed")
    report = safety_report(store, audit=_Audit({"r1": 0.01, "r2": -0.02}))
    assert report.verdict == "safe", report.reason


def test_missing_audit_is_caution_not_pass(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="passed")
    report = safety_report(store)
    assert report.verdict == "caution"
    assert any(c.name == "audit_bias" and c.status == "warn" for c in report.checks)


def test_a_biased_audit_is_unsafe(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="passed")
    report = safety_report(store, audit=_Audit({"r1": 0.6}), bias_threshold=0.1)
    assert report.verdict == "unsafe"
    assert any(c.name == "audit_bias" and c.status == "fail" for c in report.checks)


def test_flagged_or_demoted_rules_are_not_seedable(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.9, exam="demoted")
    _archive(store, "r2", "selection", 0.8, exam="flagged")
    report = safety_report(store, audit=_Audit({"r1": 0.0, "r2": 0.0}))
    assert report.verdict == "unsafe"
    assert any(c.name == "seedable_best" and c.status == "fail" for c in report.checks)


def test_an_unexamined_best_is_unsafe_under_the_default_passed_bar(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8)
    report = safety_report(store, audit=_Audit({"r1": 0.0}))
    assert report.verdict == "unsafe"
    assert any(c.name == "seedable_best" and c.status == "fail" for c in report.checks)
    # ...and only a warning when the caller asks for the no-bar behaviour
    loose = safety_report(store, audit=_Audit({"r1": 0.0}), min_exam=None)
    assert loose.verdict == "caution"


def test_only_flagged_rules_are_seedable_at_flagged_but_not_passed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="flagged")
    strict = safety_report(store, audit=_Audit({"r1": 0.0}))
    assert strict.verdict == "unsafe"
    loose = safety_report(store, audit=_Audit({"r1": 0.0}), min_exam="flagged")
    assert loose.verdict == "caution"


def test_an_unknown_exam_status_is_unsafe(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="passed")
    store.mark_exam("r1", exam="faild")
    report = safety_report(store, audit=_Audit({"r1": 0.0}))
    assert report.verdict == "unsafe"
    assert any(c.name == "exam_status" and c.status == "fail" for c in report.checks)


def test_a_demoted_biased_rule_does_not_fail_the_safety_case(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "bad", "selection", 0.9, exam="demoted")
    _archive(store, "good", "selection", 0.8, exam="passed")
    report = safety_report(store, audit=_Audit({"bad": 0.9, "good": 0.01}))
    assert report.verdict == "caution"      # exam warning, but not unsafe
    assert any(c.name == "audit_bias" and c.status == "pass" for c in report.checks)


def test_cli_json_and_missing_path(tmp_path, capsys):
    store = ConsolidationStore(str(tmp_path / "a"))
    _archive(store, "r1", "selection", 0.8, exam="passed")
    assert cli.main(["--json", "safety", "--archive",
                     str(tmp_path / "a")]) == 1      # caution: no audit
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "caution"
    assert cli.main(["safety", "--archive", str(tmp_path / "typo")]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "typo").exists()
    assert cli.main(["safety", "--archive", str(tmp_path / "a"),
                     "--audit", str(tmp_path / "no-audit.jsonl")]) == 2
    assert "audit store" in capsys.readouterr().err


def test_invalid_bias_threshold_or_min_exam_is_refused(tmp_path):
    store = ConsolidationStore(str(tmp_path / "a"))
    with pytest.raises(ValueError, match="bias_threshold"):
        safety_report(store, bias_threshold=-1)
    with pytest.raises(ValueError, match="min_exam"):
        safety_report(store, min_exam="flagd")
