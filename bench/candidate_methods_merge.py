"""Merge disjoint candidate-method runtime matrices without rerunning seeds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bench.candidate_methods import summarize, utc_now, write_json


RUNTIMES = {
    "serial": "evolve",
    "sync_parallel": "evolve",
    "async_pipeline": "async_evolve",
}
RUN_SPECIFIC_CONFIG = {"repeats", "seed"}
PUBLIC_OBSERVATION_FIELDS = (
    "algorithm",
    "fidelity",
    "mode",
    "repeat",
    "source_repeat",
    "seed",
    "wall_seconds",
    "engine_wall_seconds",
    "baseline_quality",
    "final_quality",
    "quality_gain",
    "baseline_validation_quality",
    "validation_quality",
    "quality_target",
    "target_reached",
    "time_to_quality_s",
    "candidates",
    "accepted",
    "stale_considered",
    "stale_discarded",
    "stale_rate",
    "invalid_candidates",
    "usage",
    "actor_usage",
    "budget",
)
PUBLIC_FRAMEWORK_FIELDS = (
    "runtime",
    "max_concurrency",
    "stop_reason",
    "rollouts",
    "rollout_seconds",
    "eval_seconds",
    "retired_workers",
)


def _seed_sequence(config: Mapping[str, Any]) -> List[int]:
    seed = int(config.get("seed", 0))
    repeats = int(config.get("repeats", 0))
    if repeats < 1:
        raise ValueError("source matrix has no repeats")
    return [seed + repeat * 100 for repeat in range(repeats)]


def _validate_source(path: Path, payload: Mapping[str, Any]) -> List[int]:
    if payload.get("status") != "completed":
        raise ValueError(f"{path}: matrix status is not 'completed'")
    config = payload.get("config")
    observations = payload.get("observations")
    if not isinstance(config, Mapping) or not isinstance(observations, list):
        raise ValueError(f"{path}: invalid matrix structure")

    algorithms = list(config.get("algorithms", ()))
    modes = list(config.get("modes", ()))
    seeds = _seed_sequence(config)
    expected = {
        (algorithm, mode, seed)
        for seed in seeds
        for algorithm in algorithms
        for mode in modes
    }
    actual = {
        (row.get("algorithm"), row.get("mode"), row.get("seed"))
        for row in observations
    }
    if len(observations) != len(expected) or actual != expected:
        raise ValueError(f"{path}: duplicate, missing, or unexpected observations")
    if any(row.get("error") for row in observations):
        raise ValueError(f"{path}: matrix contains failed observations")
    if any(
        row.get("framework", {}).get("runtime") != RUNTIMES.get(row.get("mode"))
        for row in observations
    ):
        raise ValueError(f"{path}: matrix contains non-framework observations")
    if any(not row.get("budget", {}).get("matched") for row in observations):
        raise ValueError(f"{path}: matrix contains budget mismatches")
    if any(int(row.get("usage", {}).get("failures", 0)) for row in observations):
        raise ValueError(f"{path}: matrix contains provider failures")
    return seeds


def _comparable_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in RUN_SPECIFIC_CONFIG
    }


def _compact_observation(row: Mapping[str, Any]) -> Dict[str, Any]:
    compact = {
        key: copy.deepcopy(row[key])
        for key in PUBLIC_OBSERVATION_FIELDS
        if key in row
    }
    framework = row.get("framework", {})
    compact["framework"] = {
        key: copy.deepcopy(framework[key])
        for key in PUBLIC_FRAMEWORK_FIELDS
        if key in framework
    }
    return compact


def merge_payloads(
    sources: Sequence[Tuple[Path, Mapping[str, Any]]],
    *,
    expected_seeds: Sequence[int],
) -> Dict[str, Any]:
    if not sources:
        raise ValueError("at least one source matrix is required")
    seeds = list(expected_seeds)
    if not seeds or seeds != [seeds[0] + index * 100 for index in range(len(seeds))]:
        raise ValueError("expected seeds must be a non-empty, ascending 100-step sequence")

    first_payload = sources[0][1]
    first_config = first_payload.get("config", {})
    first_hash = first_payload.get("implementation", {}).get("source_sha256")
    if not first_hash:
        raise ValueError("source matrix has no implementation fingerprint")

    observations: List[Dict[str, Any]] = []
    source_runs: List[Dict[str, Any]] = []
    observed_seeds: List[int] = []
    for path, payload in sources:
        run_seeds = _validate_source(path, payload)
        if _comparable_config(payload["config"]) != _comparable_config(first_config):
            raise ValueError(f"{path}: source matrix configuration differs")
        source_hash = payload.get("implementation", {}).get("source_sha256")
        if source_hash != first_hash:
            raise ValueError(f"{path}: implementation fingerprint differs")
        observed_seeds.extend(run_seeds)
        observations.extend(copy.deepcopy(payload["observations"]))
        source_runs.append(
            {
                "path": path.as_posix(),
                "seed": int(payload["config"]["seed"]),
                "repeats": int(payload["config"]["repeats"]),
                "seeds": run_seeds,
                "observations": len(payload["observations"]),
                "started_at": payload.get("started_at"),
                "completed_at": payload.get("completed_at"),
                "source_sha256": source_hash,
            }
        )

    if sorted(observed_seeds) != seeds or len(observed_seeds) != len(set(observed_seeds)):
        raise ValueError("source matrices do not cover the expected seeds exactly once")

    algorithms = list(first_config.get("algorithms", ()))
    modes = list(first_config.get("modes", ()))
    algorithm_order = {name: index for index, name in enumerate(algorithms)}
    mode_order = {name: index for index, name in enumerate(modes)}
    observations.sort(
        key=lambda row: (
            int(row["seed"]),
            algorithm_order[str(row["algorithm"])],
            mode_order[str(row["mode"])],
        )
    )
    repeat_by_seed = {seed: index + 1 for index, seed in enumerate(seeds)}
    for row in observations:
        original_repeat = row.get("repeat")
        row["repeat"] = repeat_by_seed[int(row["seed"])]
        if original_repeat != row["repeat"]:
            row["source_repeat"] = original_repeat

    summary = summarize(observations)
    observations = [_compact_observation(row) for row in observations]

    config = copy.deepcopy(first_config)
    config["seed"] = seeds[0]
    config["repeats"] = len(seeds)
    config["seed_sequence"] = seeds
    merged = {
        key: copy.deepcopy(first_payload[key])
        for key in ("experiment", "method", "implementation", "fidelity_classes")
        if key in first_payload
    }
    merged.update(
        {
            "status": "completed",
            "started_at": min(
                str(payload.get("started_at", "")) for _, payload in sources
            ),
            "completed_at": utc_now(),
            "merged_at": utc_now(),
            "artifact": {
                "kind": "compact_publication_matrix",
                "omitted_observation_fields": [
                    "events",
                    "phase_summary",
                    "notes",
                    "framework.history",
                    "framework.outcomes",
                ],
                "note": (
                    "summary logical-call counts were computed before event traces "
                    "and diagnostic detail were removed; source runs retain them"
                ),
            },
            "config": config,
            "source_runs": source_runs,
            "observations": observations,
            "summary": summary,
        }
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sources = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in args.inputs
    ]
    merged = merge_payloads(sources, expected_seeds=args.expected_seeds)
    write_json(args.output, merged)
    print(
        f"wrote {args.output} observations={len(merged['observations'])} "
        f"seeds={merged['config']['seed_sequence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
