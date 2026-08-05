"""Offline contract tests for the OpenEvolve AgentDescent port."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from bench import openevolve_agentdescent as bench
from examples import _openevolve_support as evaluator
from examples import openevolve_program_evolution as port


def test_source_gate_accepts_genome_and_rejects_unsafe_code():
    assert evaluator.validate_source(evaluator.INITIAL_PROGRAM)[0]
    assert not evaluator.validate_source(
        "import os\ndef search_algorithm(*args): return (0, 0)"
    )[0]
    assert not evaluator.validate_source(
        "def search_algorithm(*args):\n    return (-1.704, 0.678)\n"
    )[0]
    assert not evaluator.validate_source(
        "def search_algorithm(*args):\n    return open('/etc/passwd').read()\n"
    )[0]


def test_combined_metric_matches_pinned_upstream_fixture():
    trials = [
        {
            "success": True,
            "x": evaluator.GLOBAL_MIN_X,
            "y": evaluator.GLOBAL_MIN_Y,
            "seconds": 0.01,
            "objective_calls": 100,
        }
        for _ in range(10)
    ]
    metrics = evaluator.combined_metrics(trials)
    # OpenEvolve 411fb59 examples/function_minimization/evaluator.py:190-215.
    assert metrics["combined_score"] == pytest.approx(1.4997641483797084)
    assert metrics["distance_score"] == 1.0
    assert metrics["reliability_score"] == 1.0


def test_map_elites_islands_migrate_and_respect_candidate_cap():
    archive = port.OpenEvolveArchive(
        archive_size=8,
        num_islands=2,
        feature_bins=4,
        exploitation_ratio=1.0,
        migration_interval=2,
        candidate_limit=3,
    )

    def add(code, iteration, island, score, *, baseline=False):
        archive.add_program(
            evaluator.Program(
                evaluator.program_id(code),
                iteration,
                island,
                None,
                code,
                "fixture",
                {"combined_score": score},
                True,
            ),
            baseline=baseline,
        )

    add(evaluator.INITIAL_PROGRAM, 0, 0, 0.1, baseline=True)
    add(evaluator.INITIAL_PROGRAM + "\n# island one\n", 1, 1, 0.2)
    add(evaluator.INITIAL_PROGRAM + "\n# stronger island zero\n", 2, 0, 0.3)

    assert archive.migrations >= 1
    selections = [archive.select_parent() for _ in range(3)]
    assert [selection[1] for selection in selections] == [0, 1, 0]
    assert archive.select_parent() is None


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="Bubblewrap is not installed")
def test_initial_program_runs_deterministically_in_sandbox():
    first = evaluator.evaluate_source(
        evaluator.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    second = evaluator.evaluate_source(
        evaluator.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    assert first[0] and second[0]
    assert first[1]["avg_value"] == second[1]["avg_value"]
    assert first[1]["avg_distance"] == second[1]["avg_distance"]
    assert first[1]["avg_objective_calls"] == 40


def test_dry_run_needs_no_key_network_or_sandbox(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(port, "evaluate_source", lambda *a, **k: pytest.fail("sandbox used"))
    assert port.main(["--dry-run", "--model", "glm-5.2"]) == 0
    output = capsys.readouterr().out
    assert "mode=sync" in output
    assert "no API, dataset, or sandbox" in output


def test_benchmark_dry_run_reports_reserved_calls_without_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert bench.main(["--dry-run", "--repeats", "2", "--iterations", "6"]) == 0
    assert "reserved model calls=36" in capsys.readouterr().out


def test_benchmark_speedups_pair_matching_repeats_and_exclude_missed_targets():
    rows = []
    for repeat, serial_wall, sync_wall, serial_ttq, sync_ttq in (
        (1, 12.0, 6.0, 8.0, 4.0),
        (2, 30.0, 10.0, 9.0, None),
    ):
        for mode, wall, ttq in (
            ("serial", serial_wall, serial_ttq),
            ("sync", sync_wall, sync_ttq),
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "seed": repeat * 100,
                    "mode": mode,
                    "end_to_end_seconds": wall,
                    "time_to_quality_s": ttq,
                }
            )

    wall = bench._paired_speedup(
        rows,
        baseline_mode="serial",
        comparison_mode="sync",
        metric="end_to_end_seconds",
    )
    ttq = bench._paired_speedup(
        rows,
        baseline_mode="serial",
        comparison_mode="sync",
        metric="time_to_quality_s",
    )
    assert wall == {"paired_runs": 2, "speedup_min_median_max": [2.0, 2.5, 3.0]}
    assert ttq == {"paired_runs": 1, "speedup_min_median_max": [2.0, 2.0, 2.0]}


def test_agentdescent_serial_sync_and_async_run_real_port_offline():
    improved = (
        '"""improved"""\n'
        "def search_algorithm(objective, budget, rng, bounds):\n"
        "    return (-1.7, 0.68)\n"
    )

    calls = {"count": 0}

    def complete(prompt):
        calls["count"] += 1
        iteration = re.search(r"Iteration: (\d+)", prompt).group(1)
        return (
            f"<PROGRAM>\n{improved}</PROGRAM>\n"
            f"<CHANGE_SUMMARY>improved candidate {iteration}</CHANGE_SUMMARY>"
        )

    def fake_evaluate(source, *, trials, budget, seed, timeout, max_length):
        is_improved = '"""improved"""' in source
        rows = []
        for index in range(trials):
            x, y = ((-1.7, 0.68) if is_improved else (4.0, 4.0))
            rows.append(
                {
                    "seed": seed + index,
                    "success": True,
                    "x": x,
                    "y": y,
                    "seconds": 0.001,
                    "objective_calls": budget,
                }
            )
        return True, evaluator.combined_metrics(rows), "", rows

    for mode in ("serial", "sync", "async"):
        calls["count"] = 0
        run = port.run_agentdescent_openevolve(
            complete,
            mode=mode,
            iterations=4,
            workers=2,
            task_count=8,
            evaluator=fake_evaluate,
            max_seconds=5.0,
        )
        baseline = run.archive.baseline().metrics["framework_reward"]
        assert run.result.final_reward > baseline
        assert run.result.outcomes().get("committed") == 1
        assert run.result.stale_considered >= 4
        assert run.result.retired_workers == 0
        assert run.archive.summary()["programs_evaluated"] == 2
        assert calls["count"] == 4


def test_resource_limits_are_applied_in_runner_not_threaded_parent():
    parent_source = Path(evaluator.__file__).read_text(encoding="utf-8")
    runner_source = evaluator.RUNNER.read_text(encoding="utf-8")
    assert "preexec_fn" not in parent_source
    assert "start_new_session=True" in parent_source
    assert "resource.setrlimit" in runner_source
