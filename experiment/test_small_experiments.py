"""Offline checks for the local demonstration experiments."""

from __future__ import annotations

import json
import re
import time

import pytest

from experiment import openevolve_program_search as oe
from experiment import algorithm_parallel_async_benchmark as ap
from experiment import parallel_async_time_to_quality as pa
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


def test_parallel_async_tasks_keep_train_and_held_out_categories_aligned():
    tasks = pa.make_tasks([0.01, 0.02, 0.03, 0.04])
    assert [task.meta["category"] for task in tasks[:4]] == [
        task.meta["category"] for task in tasks[4:]
    ]
    assert all(not task.meta["held_out"] for task in tasks[:4])
    assert all(task.meta["held_out"] for task in tasks[4:])
    assert tasks[2].meta["proposal_categories"] == ["c2"]

    groups = pa.make_transferable_proposal_groups(workers=4, slow_workers=1)
    transferable = pa.make_tasks([0.01, 0.01, 0.01, 0.15], groups)
    assert transferable[0].meta["proposal_categories"] == ["c0", "c1", "c2"]
    assert transferable[3].meta["proposal_categories"] == ["c3"]


def test_parallel_and_async_reduce_time_to_quality_on_controlled_latency():
    uniform = [0.04] * 4
    serial = pa.run_sync_observation(
        uniform,
        scenario="test-uniform",
        repeat=0,
        concurrency=1,
        target=1.0,
        rounds=2,
        max_seconds=3.0,
    )
    parallel = pa.run_sync_observation(
        uniform,
        scenario="test-uniform",
        repeat=0,
        concurrency=4,
        target=1.0,
        rounds=2,
        max_seconds=3.0,
    )
    assert parallel.final_reward == serial.final_reward == 1.0
    assert parallel.cost_to_quality_rollouts == serial.cost_to_quality_rollouts == 4
    assert parallel.time_to_quality_s < serial.time_to_quality_s * 0.65

    heavy = [0.01, 0.01, 0.01, 0.15]
    proposal_groups = pa.make_transferable_proposal_groups(workers=4, slow_workers=1)
    sync = pa.run_sync_observation(
        heavy,
        scenario="test-heavy",
        repeat=0,
        concurrency=4,
        target=0.75,
        rounds=2,
        max_seconds=3.0,
        proposal_groups=proposal_groups,
    )
    asynchronous = pa.run_async_observation(
        heavy,
        scenario="test-heavy",
        repeat=0,
        target=0.75,
        rounds=2,
        async_ratio=3,
        max_seconds=3.0,
        proposal_groups=proposal_groups,
    )
    assert sync.final_reward >= 0.75
    assert asynchronous.final_reward >= 0.75
    assert asynchronous.time_to_quality_s < sync.time_to_quality_s * 0.65


def test_algorithm_parallel_async_dry_run_needs_no_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert ap.main(["--dry-run"]) == 0
    assert "upper-bound model calls=48" in capsys.readouterr().out


def test_textgrad_real_algorithm_scheduler_modes_with_fake_model(monkeypatch):
    answers = {
        "train one": "alpha beta",
        "train two": "delta gamma",
        "validate one": "able baker",
        "validate two": "cable zebra",
    }

    def fake_make_completion(args, usage, *, temperature=None):
        def complete(prompt):
            started = time.monotonic()
            time.sleep(0.002)
            if "You are TextualGradientDescent" in prompt:
                response = (
                    "<IMPROVED_VARIABLE>Sort every supplied word lexicographically and "
                    "return the complete ordered list after Answer:.</IMPROVED_VARIABLE>"
                )
            elif "You are the backward engine" in prompt:
                response = "Require a lexicographically sorted list instead of a number."
            elif "Backpropagate the response feedback" in prompt:
                response = "Clarify the output type and compare every character."
            else:
                question = re.search(
                    r"<USER_QUESTION>\s*(.*?)\s*</USER_QUESTION>", prompt, re.S
                ).group(1)
                response = (
                    f"Answer: {answers[question]}"
                    if "Sort every supplied word" in prompt
                    else "Answer: 0"
                )
            usage.record(
                prompt_tokens=1,
                completion_tokens=1,
                seconds=time.monotonic() - started,
            )
            return response

        return complete

    monkeypatch.setattr(ap, "make_completion", fake_make_completion)
    args = ap.build_parser().parse_args(
        [
            "--textgrad-batch-size",
            "2",
            "--textgrad-val-size",
            "2",
            "--textgrad-target-accuracy",
            "0.5",
            "--concurrency",
            "2",
        ]
    )
    train = [
        tg.Example(1, "train one", answers["train one"]),
        tg.Example(2, "train two", answers["train two"]),
    ]
    val = [
        tg.Example(101, "validate one", answers["validate one"]),
        tg.Example(102, "validate two", answers["validate two"]),
    ]

    observations = []
    for mode in ap.MODES:
        result = ap.run_textgrad_mode(
            args,
            mode=mode,
            repeat=0,
            train=train,
            val=val,
        )
        assert result["baseline_quality"] == 0.0
        assert result["final_quality"] == 1.0
        assert result["target_reached"]
        assert result["usage"]["total"]["calls"] == 11
        assert {"baseline_eval", "train_forward", "candidate_eval"} <= set(
            result["stage_summary"]
        )
        observations.append(result)
    replay = ap._textgrad_trace_replay(observations)
    assert replay["quality_target"] == 1.0
    assert (
        replay["modes"]["sync_parallel"]["time_to_quality_s"]
        <= replay["modes"]["serial"]["time_to_quality_s"]
    )
    assert (
        replay["modes"]["async_pipeline"]["time_to_quality_s"]
        <= replay["modes"]["sync_parallel"]["time_to_quality_s"]
    )


def test_openevolve_real_algorithm_scheduler_modes_with_fake_model(monkeypatch):
    def fake_make_completion(args, usage, *, temperature=None):
        def complete(prompt):
            slot = int(re.search(r"Iteration: (\d+)", prompt).group(1))
            usage.record(prompt_tokens=1, completion_tokens=1, seconds=0.001)
            return (
                "<PROGRAM>\n"
                "def search_algorithm(objective, budget, rng, bounds):\n"
                f"    return ({slot}.0, 0.0)\n"
                f"# candidate-{slot}\n"
                "</PROGRAM>\n"
                "<CHANGE_SUMMARY>deterministic test candidate</CHANGE_SUMMARY>"
            )

        return complete

    def fake_evaluate(source, **kwargs):
        match = re.search(r"candidate-(\d+)", source)
        score = 0.5 if match is None else 0.5 + 0.01 * int(match.group(1))
        metrics = {
            "combined_score": score,
            "avg_value": -score,
            "avg_distance": 1.0 - score,
        }
        return True, metrics, "", []

    monkeypatch.setattr(ap, "make_completion", fake_make_completion)
    monkeypatch.setattr(oe, "evaluate_source", fake_evaluate)
    args = ap.build_parser().parse_args(
        [
            "--openevolve-candidates",
            "3",
            "--openevolve-min-score-gain",
            "0.005",
            "--concurrency",
            "3",
        ]
    )

    observations = []
    for mode in ap.MODES:
        result = ap.run_openevolve_mode(args, mode=mode, repeat=0)
        assert result["baseline_quality"] == 0.5
        assert result["final_quality"] == pytest.approx(0.53)
        assert result["target_reached"]
        assert result["usage"]["calls"] == 3
        assert len(result["candidates"]) == 3
        observations.append(result)
    replay = ap._openevolve_trace_replay(observations)
    assert replay["quality_target"] == pytest.approx(0.505)
    assert replay["modes"]["async_pipeline"]["calls_to_quality"] == 1
    assert (
        replay["modes"]["async_pipeline"]["time_to_quality_s"]
        <= replay["modes"]["sync_parallel"]["time_to_quality_s"]
    )
