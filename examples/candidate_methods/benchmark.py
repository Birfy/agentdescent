"""Live AgentDescent runtime matrix for the remaining issue #74 candidates.

Every observation uses the real ``evolve`` or ``async_evolve`` runtime. The
output keeps aggregate metrics and compact call timing only; it never stores
prompts, responses, generated tasks, learned artifacts, source, or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Usage, claude, openai_compatible

from .environment_methods import make_skillweaver, make_voyager
from .framework import FrameworkMethodResult, PortSpec, run_port
from .inference_methods import (
    make_aflow,
    make_promptbreeder,
    make_reflexion,
    make_self_refine,
)
from .runtime import MODES, Recorder, compact_events, interval, rotate, usage_dict
from .self_edit_methods import make_godel_agent, make_sica
from .self_play_methods import make_absolute_zero, make_agent0, make_r_zero


PortFactory = Callable[[Recorder, int], PortSpec]
ALGORITHMS: Dict[str, Tuple[str, PortFactory]] = {
    "promptbreeder": ("mechanism_microport", make_promptbreeder),
    "aflow": ("mechanism_microport", make_aflow),
    "reflexion": ("mechanism_microport", make_reflexion),
    "self_refine": ("mechanism_microport", make_self_refine),
    "voyager": ("environment_analogue", make_voyager),
    "skillweaver": ("environment_analogue", make_skillweaver),
    "absolute_zero": ("inference_analogue", make_absolute_zero),
    "r_zero": ("inference_analogue", make_r_zero),
    "agent0": ("inference_analogue", make_agent0),
    "sica": ("self_edit_analogue", make_sica),
    "godel_agent": ("self_edit_analogue", make_godel_agent),
}

PROPOSAL_CALLS_PER_CANDIDATE = {
    "promptbreeder": 1,
    "aflow": 1,
    "reflexion": 1,
    "self_refine": 1,
    "voyager": 3,
    "skillweaver": 3,
    "absolute_zero": 1,
    "r_zero": 2,
    "agent0": 1,
    "sica": 1,
    "godel_agent": 1,
}

DEFAULT_OUTPUT = (
    Path(__file__).with_name("results") / "candidate-methods-framework-live.json"
)


def source_fingerprint() -> str:
    """Hash every module defining this matrix, excluding generated results."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=list(ALGORITHMS),
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--provider", choices=("openai", "glm", "claude"), default="openai"
    )
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        choices=("disabled", "enabled", "default"),
        default="disabled",
    )
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--async-ratio", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--shutdown-grace", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def _validate_config(args: argparse.Namespace) -> None:
    if args.workers < 2:
        raise ValueError("workers must be at least 2 for the parallel comparison")
    if args.candidates < args.workers or args.candidates % args.workers:
        raise ValueError("candidates must be divisible by workers and at least workers")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.async_ratio < 0:
        raise ValueError("async-ratio must be non-negative")
    if args.max_tokens < 64 or args.api_timeout <= 0:
        raise ValueError("max-tokens and api-timeout must be positive")
    if args.max_seconds <= 0 or args.shutdown_grace < 0:
        raise ValueError("max-seconds must be positive and shutdown-grace non-negative")


def _validate_environment(args: argparse.Namespace) -> None:
    if args.provider in ("openai", "glm") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("set OPENAI_API_KEY")
    if args.provider == "glm" and not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError("set OPENAI_BASE_URL for --provider glm")


def _completion(args: argparse.Namespace, usage: Usage):
    if args.provider in ("openai", "glm"):
        extra: Dict[str, Any] = {"temperature": args.temperature}
        if (
            args.thinking != "default"
            and (args.provider == "glm" or args.model.lower().startswith("glm"))
        ):
            extra["thinking"] = {"type": args.thinking}
        return openai_compatible(
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            usage=usage,
            **extra,
        )
    return claude(
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.api_timeout,
        usage=usage,
        temperature=args.temperature,
    )


def _paired_speedup(
    observations: Sequence[Dict[str, Any]],
    *,
    baseline_mode: str,
    comparison_mode: str,
    metric: str,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    pairs: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in observations:
        if (
            row.get("error")
            or row.get("mode") not in (baseline_mode, comparison_mode)
            or (algorithm is not None and row.get("algorithm") != algorithm)
        ):
            continue
        key = (str(row["algorithm"]), int(row["seed"]))
        pairs.setdefault(key, {})[str(row["mode"])] = row
    ratios: List[float] = []
    for modes in pairs.values():
        if baseline_mode not in modes or comparison_mode not in modes:
            continue
        baseline = modes[baseline_mode].get(metric)
        comparison = modes[comparison_mode].get(metric)
        if baseline is None or comparison in (None, 0):
            continue
        ratios.append(float(baseline) / float(comparison))
    return {
        "paired_runs": len(ratios),
        "speedup_min_median_max": interval(ratios),
    }


def summarize(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows_by_algorithm: Dict[str, Dict[str, Any]] = {}
    comparisons_by_algorithm: Dict[str, Dict[str, Any]] = {}
    for algorithm in ALGORITHMS:
        modes: Dict[str, Any] = {}
        for mode in MODES:
            rows = [
                row
                for row in observations
                if row.get("algorithm") == algorithm
                and row.get("mode") == mode
                and not row.get("error")
            ]
            reached = [row for row in rows if row.get("target_reached")]
            modes[mode] = {
                "runs": len(rows),
                "target_reached_runs": len(reached),
                "wall_s_min_median_max": interval(
                    float(row["wall_seconds"]) for row in rows
                ),
                "engine_wall_s_min_median_max": interval(
                    float(row["engine_wall_seconds"]) for row in rows
                ),
                "ttq_s_min_median_max": interval(
                    float(row["time_to_quality_s"]) for row in reached
                ),
                "baseline_quality_min_median_max": interval(
                    float(row["baseline_quality"]) for row in rows
                ),
                "final_quality_min_median_max": interval(
                    float(row["final_quality"]) for row in rows
                ),
                "quality_gain_min_median_max": interval(
                    float(row["quality_gain"]) for row in rows
                ),
                "calls_min_median_max": interval(
                    float(row["usage"]["calls"]) for row in rows
                ),
                "tokens_min_median_max": interval(
                    float(row["usage"]["total_tokens"]) for row in rows
                ),
                "accepted": sum(int(row["accepted"]) for row in rows),
                "invalid_candidates": sum(
                    int(row["invalid_candidates"]) for row in rows
                ),
                "stale_considered": sum(
                    int(row["stale_considered"]) for row in rows
                ),
                "stale_discarded": sum(
                    int(row["stale_discarded"]) for row in rows
                ),
            }
        rows_by_algorithm[algorithm] = modes
        comparisons_by_algorithm[algorithm] = {
            "sync_vs_serial_wall": _paired_speedup(
                observations,
                baseline_mode="serial",
                comparison_mode="sync_parallel",
                metric="wall_seconds",
                algorithm=algorithm,
            ),
            "sync_vs_serial_engine_wall": _paired_speedup(
                observations,
                baseline_mode="serial",
                comparison_mode="sync_parallel",
                metric="engine_wall_seconds",
                algorithm=algorithm,
            ),
            "async_vs_sync_wall": _paired_speedup(
                observations,
                baseline_mode="sync_parallel",
                comparison_mode="async_pipeline",
                metric="wall_seconds",
                algorithm=algorithm,
            ),
            "async_vs_sync_engine_wall": _paired_speedup(
                observations,
                baseline_mode="sync_parallel",
                comparison_mode="async_pipeline",
                metric="engine_wall_seconds",
                algorithm=algorithm,
            ),
            "sync_vs_serial_ttq": _paired_speedup(
                observations,
                baseline_mode="serial",
                comparison_mode="sync_parallel",
                metric="time_to_quality_s",
                algorithm=algorithm,
            ),
            "async_vs_sync_ttq": _paired_speedup(
                observations,
                baseline_mode="sync_parallel",
                comparison_mode="async_pipeline",
                metric="time_to_quality_s",
                algorithm=algorithm,
            ),
        }

    successful = [row for row in observations if not row.get("error")]
    comparisons = {
        "definition": (
            "paired speedup = baseline seconds / comparison seconds for one "
            "algorithm and seed"
        ),
        "sync_vs_serial_wall": _paired_speedup(
            observations,
            baseline_mode="serial",
            comparison_mode="sync_parallel",
            metric="wall_seconds",
        ),
        "sync_vs_serial_engine_wall": _paired_speedup(
            observations,
            baseline_mode="serial",
            comparison_mode="sync_parallel",
            metric="engine_wall_seconds",
        ),
        "async_vs_sync_wall": _paired_speedup(
            observations,
            baseline_mode="sync_parallel",
            comparison_mode="async_pipeline",
            metric="wall_seconds",
        ),
        "async_vs_sync_engine_wall": _paired_speedup(
            observations,
            baseline_mode="sync_parallel",
            comparison_mode="async_pipeline",
            metric="engine_wall_seconds",
        ),
        "sync_vs_serial_ttq": _paired_speedup(
            observations,
            baseline_mode="serial",
            comparison_mode="sync_parallel",
            metric="time_to_quality_s",
        ),
        "async_vs_sync_ttq": _paired_speedup(
            observations,
            baseline_mode="sync_parallel",
            comparison_mode="async_pipeline",
            metric="time_to_quality_s",
        ),
    }
    return {
        "algorithms": rows_by_algorithm,
        "algorithm_comparisons": comparisons_by_algorithm,
        "totals": {
            "observations": len(successful),
            "provider_calls": sum(
                int(row["usage"]["calls"]) for row in successful
            ),
            "actor_calls": sum(
                int(row["actor_usage"]["calls"]) for row in successful
            ),
            "logical_calls": sum(
                len(row.get("events", ())) for row in successful
            ),
            "proposal_calls": sum(
                int(row["budget"]["observed_proposal_calls"])
                for row in successful
            ),
            "tokens": sum(
                int(row["usage"]["total_tokens"]) for row in successful
            ),
            "provider_failures": sum(
                int(row["usage"]["failures"]) for row in successful
            ),
            "budget_mismatches": sum(
                not bool(row.get("budget", {}).get("matched"))
                for row in successful
            ),
        },
        "comparisons": comparisons,
    }


def _config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in ("output", "yes", "resume")
    }


def _initial_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "experiment": (
            "issue #74 remaining candidate methods on AgentDescent: "
            "serial vs sync vs async"
        ),
        "status": "running",
        "started_at": utc_now(),
        "method": {
            "runtime": (
                "serial/sync use agentdescent.evolution.evolve; async uses "
                "agentdescent.async_evolve.async_evolve"
            ),
            "provider_calls": "live calls; no replay or synthetic latency",
            "mode_order": "rotated by repeat and algorithm index",
            "quality": "strict disjoint held-out task/environment reward",
            "budget": (
                "equal candidate and algorithm proposal-call budgets; framework "
                "gate calls are measured separately"
            ),
            "privacy": (
                "no prompts, responses, generated tasks, learned artifacts, "
                "source, or credentials in output"
            ),
        },
        "implementation": {
            "source_sha256": source_fingerprint(),
            "scope": "all examples/candidate_methods/*.py files",
        },
        "fidelity_classes": {
            "mechanism_microport": (
                "released/paper control flow on a compact deterministic domain"
            ),
            "environment_analogue": (
                "control flow preserved; unavailable embodied/web environment replaced"
            ),
            "inference_analogue": (
                "self-play roles preserved; verbal memory replaces unavailable RL updates"
            ),
            "self_edit_analogue": (
                "real AST-gated source replacement over a scalar policy surface"
            ),
        },
        "config": _config_payload(args),
        "observations": [],
    }


def _resume_payload(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    result = json.loads(args.output.read_text(encoding="utf-8"))
    stored = result.get("config", {})
    current = _config_payload(args)
    keys = (
        "algorithms",
        "modes",
        "provider",
        "model",
        "workers",
        "candidates",
        "async_ratio",
        "repeats",
        "seed",
    )
    mismatched = [key for key in keys if stored.get(key) != current.get(key)]
    if mismatched:
        raise ValueError(
            "resume configuration differs for: " + ", ".join(mismatched)
        )
    observations = list(result.get("observations", []))
    result["status"] = "running"
    result["resumed_at"] = utc_now()
    result.setdefault("resume_requests", []).append({"at": utc_now()})
    return result, observations


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_config(args)
    _validate_environment(args)
    if args.resume and args.output.exists():
        result, observations = _resume_payload(args)
    else:
        result = _initial_payload(args)
        observations: List[Dict[str, Any]] = []

    expected_source = result["implementation"]["source_sha256"]
    completed = {
        (row.get("algorithm"), row.get("mode"), row.get("seed"))
        for row in observations
        if (
            not row.get("error")
            and row.get("budget", {}).get("matched")
            and int(row.get("usage", {}).get("failures", 0)) == 0
        )
    }
    write_json(args.output, result)

    algorithms = list(args.algorithms)
    for repeat in range(args.repeats):
        run_seed = args.seed + repeat * 100
        for algorithm_index, algorithm in enumerate(rotate(algorithms, repeat)):
            fidelity, factory = ALGORITHMS[algorithm]
            for mode in rotate(args.modes, repeat + algorithm_index):
                if source_fingerprint() != expected_source:
                    result.update(
                        {
                            "status": "source_changed",
                            "source_changed_at": utc_now(),
                            "observations": observations,
                            "summary": summarize(observations),
                        }
                    )
                    write_json(args.output, result)
                    raise RuntimeError(
                        "candidate-method source changed during the matrix; "
                        "start a new output file"
                    )
                key = (algorithm, mode, run_seed)
                if key in completed:
                    print(
                        f"skip completed algorithm={algorithm} mode={mode} "
                        f"seed={run_seed}",
                        flush=True,
                    )
                    continue
                observations = [
                    row
                    for row in observations
                    if not (
                        row.get("algorithm") == algorithm
                        and row.get("mode") == mode
                        and row.get("seed") == run_seed
                    )
                ]
                print(
                    f"start algorithm={algorithm} mode={mode} "
                    f"repeat={repeat + 1}/{args.repeats} seed={run_seed}",
                    flush=True,
                )
                model_usage = Usage()
                recorder = Recorder(_completion(args, model_usage), model_usage)
                try:
                    spec = factory(recorder, run_seed)
                    if spec.name != algorithm or spec.fidelity != fidelity:
                        raise RuntimeError("port factory metadata does not match registry")
                    port_result: FrameworkMethodResult = run_port(
                        spec,
                        recorder,
                        mode=mode,
                        seed=run_seed,
                        workers=args.workers,
                        candidate_budget=args.candidates,
                        async_ratio=args.async_ratio,
                        max_seconds=args.max_seconds,
                        shutdown_grace=args.shutdown_grace,
                    )
                    observation = port_result.compact()
                    observation["repeat"] = repeat + 1
                except Exception as error:
                    observation = {
                        "algorithm": algorithm,
                        "fidelity": fidelity,
                        "mode": mode,
                        "repeat": repeat + 1,
                        "seed": run_seed,
                        "wall_seconds": recorder.elapsed(),
                        "usage": usage_dict(model_usage),
                        "phase_summary": recorder.phase_summary(),
                        "events": compact_events(recorder.events),
                        "error": f"{type(error).__name__}: {str(error)[:500]}",
                    }
                observations.append(observation)
                result["observations"] = observations
                result["summary"] = summarize(observations)
                write_json(args.output, result)
                if observation.get("error"):
                    print(
                        f"failed {algorithm}/{mode}: {observation['error']}",
                        flush=True,
                    )
                else:
                    print(
                        f"done {algorithm}/{mode}: quality="
                        f"{observation['baseline_quality']:.3f}->"
                        f"{observation['final_quality']:.3f} "
                        f"ttq={observation['time_to_quality_s']} "
                        f"wall={observation['wall_seconds']:.2f}s "
                        f"engine={observation['engine_wall_seconds']:.2f}s "
                        f"calls={observation['usage']['calls']} "
                        f"budget={observation['budget']['matched']}",
                        flush=True,
                    )

    errors = sum(bool(row.get("error")) for row in observations)
    budget_mismatches = sum(
        bool(not row.get("error") and not row.get("budget", {}).get("matched"))
        for row in observations
    )
    provider_failures = sum(
        int(row.get("usage", {}).get("failures", 0))
        for row in observations
        if not row.get("error")
    )
    result.update(
        {
            "status": (
                "completed"
                if errors == 0
                and budget_mismatches == 0
                and provider_failures == 0
                else "partial"
            ),
            "completed_at": utc_now(),
            "observations": observations,
            "summary": summarize(observations),
            "errors": errors,
            "budget_mismatches": budget_mismatches,
            "provider_failures": provider_failures,
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_config(args)
    if args.dry_run:
        proposal_calls = (
            args.candidates
            * len(args.modes)
            * args.repeats
            * sum(PROPOSAL_CALLS_PER_CANDIDATE[name] for name in args.algorithms)
        )
        print(
            "Candidate-method AgentDescent matrix dry run: "
            f"algorithms={len(args.algorithms)}, modes={len(args.modes)}, "
            f"repeats={args.repeats}, reserved candidates="
            f"{args.candidates * len(args.algorithms) * len(args.modes) * args.repeats}, "
            f"reserved proposal calls={proposal_calls}, workers={args.workers}, "
            f"model={args.model}"
        )
        print(
            "  serial/sync runtime=evolve; async runtime=async_evolve; "
            "gate/evaluation calls are additional and measured, not estimated"
        )
        for name in args.algorithms:
            per_mode = args.candidates * PROPOSAL_CALLS_PER_CANDIDATE[name]
            print(f"  {name}: {per_mode} reserved proposal calls per mode")
        return 0

    _validate_environment(args)
    if not args.yes and sys.stdin.isatty():
        answer = input("Proceed with paid live candidate-method calls? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted.")
            return 0
    result = run_benchmark(args)
    print(
        f"finished status={result['status']} "
        f"observations={len(result['observations'])} errors={result['errors']} "
        f"budget_mismatches={result['budget_mismatches']} "
        f"provider_failures={result['provider_failures']} output={args.output}"
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
