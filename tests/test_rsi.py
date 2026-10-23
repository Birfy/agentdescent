"""The longitudinal RSI report has to tell rising from plateaued honestly."""

from __future__ import annotations

import json
import time

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    problem_descriptor,
    rsi_report,
)
from agentdescent import cli
from agentdescent.mcp import Tools


def _archive(store, run_id, slot, reward):
    store.archive(
        EvolutionResult(state={"value": "x"}, rendered="x", final_reward=reward,
                        history=[], ledger_log=[]),
        run_id=run_id, strategy="SourceSlot", config={"dataset": "rsi-demo"},
        artifact_id="meta-rule", problem=problem_descriptor(slot))
    time.sleep(0.01)      # keep created_at strictly ordered


def test_a_rising_slot_is_called_improving(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.2, 0.4, 0.7)):
        _archive(store, f"r{i}", "selection", reward)
    report = rsi_report(store)
    row = report.slots["selection"]
    assert report.verdict == "improving"
    assert row.first_reward == 0.2 and row.best_reward == 0.7
    assert row.gain == 0.7 - 0.2
    assert row.best_run_id == "r2" and row.records_since_best == 0
    assert row.trusted == 0 and row.unexamined == 3


def test_a_plateau_is_called_stagnant(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.8, 0.4, 0.3)):
        _archive(store, f"r{i}", "selection", reward)
    report = rsi_report(store, stagnation_window=2)
    row = report.slots["selection"]
    assert report.verdict == "stagnant"
    assert row.best_reward == 0.8 and row.records_since_best == 2


def test_one_rising_and_one_flat_slot_is_mixed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.2, 0.5, 0.8)):
        _archive(store, f"a{i}", "selection", reward)
    for i, reward in enumerate((0.8, 0.4, 0.3)):
        _archive(store, f"b{i}", "task_sampler", reward)
    report = rsi_report(store, stagnation_window=2)
    assert report.verdict == "mixed"
    assert report.improving_slots == ("selection",)
    assert report.stagnant_slots == ("task_sampler",)


def test_an_empty_archive_is_insufficient(tmp_path):
    store = ConsolidationStore(str(tmp_path / "empty"))
    report = rsi_report(store)
    assert report.verdict == "insufficient" and report.slots == {}


def test_a_plateau_after_an_early_gain_is_stagnant(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.2, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)):
        _archive(store, f"r{i}", "selection", reward)
    report = rsi_report(store, stagnation_window=3)
    assert report.slots["selection"].verdict == "stagnant"
    assert report.verdict == "stagnant"


def test_a_demoted_best_cannot_drive_an_improving_verdict(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive(store, "a", "selection", 0.2)
    _archive(store, "b", "selection", 0.9)
    store.mark_exam("b", exam="demoted")
    row = rsi_report(store).slots["selection"]
    assert row.best_reward == 0.2 and row.best_run_id == "a"
    assert row.demoted == 1
    assert row.verdict != "improving"


def test_a_missing_archive_path_errors_without_creating_it(tmp_path):
    missing = tmp_path / "typo"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        rsi_report(str(missing))
    assert not missing.exists()


def test_evidence_and_reallocation_are_serialized(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.2, 0.6)):
        _archive(store, f"r{i}", "selection", reward)
    payload = rsi_report(store).to_dict()
    assert payload["evidence"] is not None
    assert "selection" in payload["evidence"]["slots"]


def test_mixed_reason_comes_from_the_actual_verdicts(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive(store, "a0", "selection", 0.8)
    _archive(store, "a1", "selection", 0.4)
    _archive(store, "a2", "selection", 0.4)
    _archive(store, "a3", "selection", 0.4)
    _archive(store, "b0", "task_sampler", 0.8)
    _archive(store, "b1", "task_sampler", 0.8)
    report = rsi_report(store, stagnation_window=3)
    assert report.verdict == "mixed"
    assert "selection: stagnant" in report.reason
    assert "task_sampler: flat" in report.reason


def test_invalid_parameters_are_refused(tmp_path):
    with pytest.raises(ValueError, match="stagnation_window"):
        rsi_report(str(tmp_path), stagnation_window=0)
    with pytest.raises(ValueError, match="tolerance"):
        rsi_report(str(tmp_path), tolerance=-0.1)


def test_exam_counts_are_included(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive(store, "a", "selection", 0.4)
    _archive(store, "b", "selection", 0.6)
    store.mark_exam("a", exam="demoted")
    store.mark_exam("b", exam="passed")
    row = rsi_report(store).slots["selection"]
    assert row.demoted == 1 and row.trusted == 1 and row.flagged == 0


def test_cli_and_mcp_return_the_same_report(tmp_path, capsys):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i, reward in enumerate((0.2, 0.6)):
        _archive(store, f"r{i}", "selection", reward)
    assert cli.main(["--json", "rsi", "--archive",
                     str(tmp_path / "archive")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "improving"
    mcp = Tools(None).rsi(archive=str(tmp_path / "archive"))
    assert mcp["verdict"] == out["verdict"]
    assert mcp["slots"]["selection"]["best_reward"] == out["slots"]["selection"]["best_reward"]
