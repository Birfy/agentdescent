"""Offline checks for the two local demonstration experiments."""

from __future__ import annotations

import json

import pytest

from experiment import openevolve_program_search as oe
from experiment import textgrad_prompt_optimization as tg


def test_textgrad_answer_parser_is_exact_but_format_tolerant():
    assert tg.score_response(
        "Reasoning...\nAnswer: alpha beta it&t", "alpha beta it&t"
    ) == (True, "alpha beta it&t")
    assert tg.score_response("alpha, beta, gamma", "alpha beta gamma")[0]
    assert not tg.score_response("Answer: beta alpha", "alpha beta")[0]


def test_textgrad_uses_official_positional_splits_and_longest(monkeypatch):
    rows = [
        {"input": f"question {index}", "target": " ".join(["word"] * (index % 11 + 1))}
        for index in range(250)
    ]
    monkeypatch.setattr(tg, "fetch_text", lambda *args, **kwargs: json.dumps({"examples": rows}))
    train, val, test, digest = tg.load_bbh_splits(
        train_size=3, val_size=4, test_size=5, subset="longest"
    )
    assert all(0 <= item.index < 50 for item in train)
    assert all(50 <= item.index < 150 for item in val)
    assert all(150 <= item.index < 250 for item in test)
    assert [item.word_count for item in train] == sorted(
        (item.word_count for item in train), reverse=True
    )
    assert len(digest) == 64


def test_textgrad_dry_run_needs_neither_data_nor_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tg, "fetch_text", lambda *args, **kwargs: pytest.fail("network used"))
    assert tg.main(["--dry-run", "--model", "glm-5.2"]) == 0
    assert "upper-bound calls" in capsys.readouterr().out


def test_openevolve_source_gate_accepts_genome_and_rejects_unsafe_code():
    assert oe.validate_source(oe.INITIAL_PROGRAM)[0]
    assert not oe.validate_source("import os\ndef search_algorithm(*args): return (0, 0)")[0]
    assert not oe.validate_source(
        "def search_algorithm(*args):\n    return (-1.704, 0.678)\n"
    )[0]
    assert not oe.validate_source(
        "def search_algorithm(*args):\n    return open('/etc/passwd').read()\n"
    )[0]


def test_openevolve_official_metric_formula():
    trials = [
        {
            "success": True,
            "x": oe.GLOBAL_MIN_X,
            "y": oe.GLOBAL_MIN_Y,
            "seconds": 0.01,
            "objective_calls": 100,
        }
        for _ in range(10)
    ]
    metrics = oe.official_metrics(trials)
    expected_value = oe.objective_value(oe.GLOBAL_MIN_X, oe.GLOBAL_MIN_Y)
    expected_value_score = 1.0 / (1.0 + abs(expected_value - oe.GLOBAL_MIN_VALUE))
    expected = (0.5 * expected_value_score + 0.3 + 0.2) * 1.5
    assert metrics["combined_score"] == pytest.approx(expected)
    assert metrics["distance_score"] == 1.0
    assert metrics["reliability_score"] == 1.0


def test_openevolve_initial_program_runs_deterministically_in_sandbox():
    first = oe.evaluate_source(
        oe.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    second = oe.evaluate_source(
        oe.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    assert first[0] and second[0]
    assert first[1]["avg_value"] == second[1]["avg_value"]
    assert first[1]["avg_distance"] == second[1]["avg_distance"]
    assert first[1]["avg_objective_calls"] == 40


def test_openevolve_dry_run_needs_no_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert oe.main(["--dry-run", "--model", "glm-5.2"]) == 0
    assert "iterations/calls=6" in capsys.readouterr().out
