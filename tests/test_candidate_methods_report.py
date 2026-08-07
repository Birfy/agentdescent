import json

import pytest

from bench.candidate_methods_merge import merge_payloads
from bench.candidate_methods_report import render_report
from examples.candidate_methods.benchmark import summarize
from examples.candidate_methods.runtime import MODES


def _row(mode):
    wall = {"serial": 20.0, "sync_parallel": 10.0, "async_pipeline": 8.0}[mode]
    engine = {"serial": 18.0, "sync_parallel": 9.0, "async_pipeline": 7.0}[mode]
    return {
        "algorithm": "reflexion",
        "fidelity": "mechanism_microport",
        "mode": mode,
        "repeat": 1,
        "seed": 0,
        "wall_seconds": wall,
        "engine_wall_seconds": engine,
        "baseline_quality": 0.0,
        "final_quality": 1.0,
        "quality_gain": 1.0,
        "baseline_validation_quality": 0.0,
        "validation_quality": 1.0,
        "quality_target": 0.75,
        "target_reached": True,
        "time_to_quality_s": wall,
        "candidates": 2,
        "accepted": 2,
        "stale_considered": 2,
        "stale_discarded": 0,
        "stale_rate": 0.0,
        "invalid_candidates": 0,
        "usage": {
            "calls": 6,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model_seconds": wall,
            "failures": 0,
        },
        "actor_usage": {
            "calls": 4,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_seconds": wall,
            "failures": 0,
        },
        "events": [{} for _ in range(6)],
        "framework": {
            "runtime": "async_evolve" if mode == "async_pipeline" else "evolve",
            "rollouts": 2,
        },
        "budget": {
            "reserved_candidates": 2,
            "observed_candidates": 2,
            "observed_rollouts": 2,
            "expected_proposal_calls": 2,
            "observed_proposal_calls": 2,
            "matched": True,
        },
    }


def _payload(status="completed"):
    observations = [_row(mode) for mode in MODES]
    return {
        "status": status,
        "config": {
            "algorithms": ["reflexion"],
            "modes": list(MODES),
            "repeats": 1,
            "seed": 0,
            "model": "glm-5.2",
        },
        "implementation": {"source_sha256": "abc123"},
        "observations": observations,
        "summary": summarize(observations),
    }


def test_report_rejects_an_incomplete_paid_matrix(tmp_path):
    with pytest.raises(ValueError, match="not 'completed'"):
        render_report(_payload("running"), input_path=tmp_path / "run.json")


def test_report_renders_quality_timing_cost_and_fidelity(tmp_path):
    report = render_report(_payload(), input_path=tmp_path / "run.json")
    assert "Reflexion" in report
    assert "0.000 -> 1.000" in report
    assert "2.00x (n=1)" in report
    assert "Provider calls: **18**" in report
    assert "Framework actor calls: **12**" in report
    assert "Algorithm proposal calls: **6**" in report
    assert "mechanism_microport" in report
    assert "no response replay" in report
    assert "evolve(max_concurrency=1)" in report
    assert "Async vs sync engine window" in report
    assert "## Interpretation" in report
    assert "Sync parallel has a median end-to-end win" in report
    assert "| `serial` | 1 | 6 | 4 | 2 | 2 | 150 |" in report


def test_report_rejects_non_framework_observations(tmp_path):
    payload = _payload()
    payload["observations"][0]["framework"]["runtime"] = "thread_pool"
    with pytest.raises(ValueError, match="outside AgentDescent runtimes"):
        render_report(payload, input_path=tmp_path / "run.json")


def test_merge_combines_disjoint_seed_runs_and_recomputes_summary(tmp_path):
    seed_zero = _payload()
    seed_hundred = json.loads(json.dumps(_payload()))
    seed_hundred["config"]["seed"] = 100
    for row in seed_hundred["observations"]:
        row["seed"] = 100
    seed_hundred["summary"] = summarize(seed_hundred["observations"])

    merged = merge_payloads(
        [(tmp_path / "zero.json", seed_zero), (tmp_path / "hundred.json", seed_hundred)],
        expected_seeds=[0, 100],
    )

    assert merged["status"] == "completed"
    assert merged["config"]["repeats"] == 2
    assert merged["config"]["seed_sequence"] == [0, 100]
    assert len(merged["observations"]) == 6
    assert merged["summary"]["totals"]["observations"] == 6
    assert merged["summary"]["totals"]["logical_calls"] == 36
    assert len(merged["source_runs"]) == 2
    assert "events" in merged["artifact"]["omitted_observation_fields"]
    assert all("events" not in row for row in merged["observations"])
    assert all("phase_summary" not in row for row in merged["observations"])
    assert all("history" not in row["framework"] for row in merged["observations"])
    assert {row["repeat"] for row in merged["observations"] if row["seed"] == 100} == {2}


def test_merge_rejects_overlapping_or_different_implementations(tmp_path):
    first = _payload()
    duplicate = json.loads(json.dumps(first))
    with pytest.raises(ValueError, match="expected seeds exactly once"):
        merge_payloads(
            [(tmp_path / "a.json", first), (tmp_path / "b.json", duplicate)],
            expected_seeds=[0, 100],
        )

    different = json.loads(json.dumps(first))
    different["implementation"]["source_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprint differs"):
        merge_payloads(
            [(tmp_path / "a.json", first), (tmp_path / "b.json", different)],
            expected_seeds=[0],
        )
