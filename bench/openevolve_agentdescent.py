"""Live serial/sync/async benchmark for the OpenEvolve AgentDescent port.

All modes run the same model, temperature, mutation-call cap, evaluator seeds,
and objective budget. Results contain summaries and framework metrics only; model
responses and generated source are deliberately omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agentdescent.agents import Usage, claude, openai_compatible

from examples.openevolve_program_evolution import run_agentdescent_openevolve


MODES = ("serial", "sync", "async")
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "openevolve-agentdescent-live.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _rotate(items: Sequence[str], amount: int) -> List[str]:
    if not items:
        return []
    shift = amount % len(items)
    return list(items[shift:]) + list(items[:shift])


def _interval(values: Sequence[float]) -> Optional[List[float]]:
    if not values:
        return None
    return [min(values), statistics.median(values), max(values)]


def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keep reproducibility metrics without generated code or per-trial payloads."""
    keys = (
        "combined_score",
        "avg_value",
        "avg_distance",
        "successful_trials",
        "total_trials",
        "framework_reward",
    )
    return {key: metrics.get(key) for key in keys}


def compact_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    framework = observation["framework"]
    archive = observation["archive"]
    test = observation["test"]
    return {
        "mode": observation["mode"],
        "wall_seconds": observation["wall_seconds"],
        "framework": {
            key: framework[key]
            for key in (
                "baseline_reward",
                "final_reward",
                "quality_gain",
                "stop_reason",
                "error",
                "outcomes",
                "rollouts",
                "rollout_seconds",
                "eval_seconds",
                "stale_considered",
                "stale_discarded",
                "stale_rate",
                "retired_workers",
            )
        },
        "archive": {
            key: archive[key]
            for key in (
                "programs_evaluated",
                "valid_programs",
                "island_cell_counts",
                "island_generations",
                "migrations",
            )
        },
        "test": {
            "baseline": _compact_metrics(test["baseline"]),
            "best": _compact_metrics(test["best"]),
            "combined_score_gain": test["combined_score_gain"],
        },
        "framework_actor_usage": {
            key: observation["actor_usage"][key]
            for key in ("calls", "failures", "model_seconds")
        },
        "quality_target": observation["quality_target"],
        "target_reached": observation["target_reached"],
        "time_to_quality_s": observation["time_to_quality_s"],
        "rollouts_to_quality": observation["rollouts_to_quality"],
    }


def _paired_speedup(
    observations: Sequence[Dict[str, Any]],
    *,
    baseline_mode: str,
    comparison_mode: str,
    metric: str,
) -> Dict[str, Any]:
    paired: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for row in observations:
        if row.get("error") or row.get("mode") not in (baseline_mode, comparison_mode):
            continue
        key = (row.get("repeat"), row.get("seed"))
        paired.setdefault(key, {})[row["mode"]] = row

    ratios = []
    for modes in paired.values():
        if baseline_mode not in modes or comparison_mode not in modes:
            continue
        baseline = modes[baseline_mode].get(metric)
        comparison = modes[comparison_mode].get(metric)
        if baseline is None or comparison in (None, 0):
            continue
        ratios.append(float(baseline) / float(comparison))
    return {
        "paired_runs": len(ratios),
        "speedup_min_median_max": _interval(ratios),
    }


def summarize(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    modes: Dict[str, Any] = {}
    for mode in MODES:
        rows = [row for row in observations if row.get("mode") == mode and not row.get("error")]
        reached = [row for row in rows if row.get("target_reached")]
        modes[mode] = {
            "runs": len(rows),
            "target_reached_runs": len(reached),
            "time_to_quality_s_min_median_max": _interval(
                [float(row["time_to_quality_s"]) for row in reached]
            ),
            "engine_wall_s_min_median_max": _interval(
                [float(row["wall_seconds"]) for row in rows]
            ),
            "end_to_end_s_min_median_max": _interval(
                [float(row["end_to_end_seconds"]) for row in rows]
            ),
            "validation_reward_min_median_max": _interval(
                [float(row["framework"]["final_reward"]) for row in rows]
            ),
            "test_combined_score_min_median_max": _interval(
                [float(row["test"]["best"]["combined_score"]) for row in rows]
            ),
            "test_gain_min_median_max": _interval(
                [float(row["test"]["combined_score_gain"]) for row in rows]
            ),
            "model_calls_min_median_max": _interval(
                [float(row["model_usage"]["calls"]) for row in rows]
            ),
            "tokens_min_median_max": _interval(
                [float(row["model_usage"]["total_tokens"]) for row in rows]
            ),
            "stale_rate_min_median_max": _interval(
                [float(row["framework"]["stale_rate"]) for row in rows]
            ),
            "retired_workers": sum(
                int(row["framework"]["retired_workers"]) for row in rows
            ),
        }

    comparisons: Dict[str, Any] = {
        "definition": (
            "paired speedup = baseline seconds / comparison seconds for the same "
            "repeat and evaluator seed"
        ),
        "sync_vs_serial_end_to_end": _paired_speedup(
            observations,
            baseline_mode="serial",
            comparison_mode="sync",
            metric="end_to_end_seconds",
        ),
        "async_vs_sync_end_to_end": _paired_speedup(
            observations,
            baseline_mode="sync",
            comparison_mode="async",
            metric="end_to_end_seconds",
        ),
        "sync_vs_serial_time_to_quality": _paired_speedup(
            observations,
            baseline_mode="serial",
            comparison_mode="sync",
            metric="time_to_quality_s",
        ),
        "async_vs_sync_time_to_quality": _paired_speedup(
            observations,
            baseline_mode="sync",
            comparison_mode="async",
            metric="time_to_quality_s",
        ),
    }
    return {"modes": modes, "comparisons": comparisons}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "glm", "claude"), default="openai")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"), default="disabled")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--test-trials", type=int, default=6)
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--objective-budget", type=int, default=200)
    parser.add_argument("--archive-size", type=int, default=20)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--feature-bins", type=int, default=4)
    parser.add_argument("--exploitation-ratio", type=float, default=0.7)
    parser.add_argument("--migration-interval", type=int, default=4)
    parser.add_argument("--candidate-timeout", type=float, default=15.0)
    parser.add_argument("--max-code-length", type=int, default=20_000)
    parser.add_argument("--async-ratio", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=120.0,
        help="include in-flight async requests in return time and usage",
    )
    parser.add_argument(
        "--quality-target",
        type=float,
        default=0.8,
        help="fixed normalized validation target; baseline was pre-measured at 0.681-0.729",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.iterations < 1 or args.workers < 1 or args.iterations % args.workers:
        raise ValueError("iterations must be positive and divisible by workers")
    if not 0.0 <= args.quality_target <= 1.0:
        raise ValueError("quality-target must be in [0, 1]")
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


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _validate(args)
    result: Dict[str, Any] = {
        "experiment": "OpenEvolve on AgentDescent: serial vs sync vs async",
        "status": "running",
        "started_at": utc_now(),
        "method": {
            "algorithm": "real OpenEvolve program mutation + MAP-Elites islands",
            "runtime": "AgentDescent evolve for serial/sync and async_evolve for async",
            "provider_calls": "live calls with no trace replay",
            "budget": "equal reserved LLM mutation calls per mode",
            "quality": "validation TTQ plus disjoint evaluator test seeds",
            "order": "mode order rotates each repeat",
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "observations": [],
    }
    write_json(args.output, result)
    observations: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        run_seed = args.seed + repeat * 100
        for mode in _rotate(args.modes, repeat):
            print(f"starting repeat={repeat + 1}/{args.repeats} mode={mode} seed={run_seed}")
            model_usage = Usage()
            actor_usage = Usage()
            started = time.monotonic()
            try:
                run = run_agentdescent_openevolve(
                    _completion(args, model_usage),
                    mode=mode,
                    iterations=args.iterations,
                    workers=args.workers,
                    task_count=args.tasks,
                    test_trials=args.test_trials,
                    held_out_frac=args.held_out_frac,
                    objective_budget=args.objective_budget,
                    archive_size=args.archive_size,
                    islands=args.islands,
                    feature_bins=args.feature_bins,
                    exploitation_ratio=args.exploitation_ratio,
                    migration_interval=args.migration_interval,
                    candidate_timeout=args.candidate_timeout,
                    max_code_length=args.max_code_length,
                    async_ratio=args.async_ratio,
                    max_seconds=args.max_seconds,
                    shutdown_grace=args.shutdown_grace,
                    seed=run_seed,
                    usage=actor_usage,
                    verbose=True,
                )
                observation = compact_observation(run.summary(args.quality_target))
                observation.update(
                    {
                        "repeat": repeat + 1,
                        "seed": run_seed,
                        "end_to_end_seconds": time.monotonic() - started,
                        "model_usage": {
                            "calls": model_usage.calls,
                            "failures": model_usage.failures,
                            "prompt_tokens": model_usage.prompt_tokens,
                            "completion_tokens": model_usage.completion_tokens,
                            "total_tokens": model_usage.total_tokens,
                            "model_seconds": model_usage.seconds,
                        },
                    }
                )
            except Exception as exc:  # keep earlier paid observations on disk
                observation = {
                    "mode": mode,
                    "repeat": repeat + 1,
                    "seed": run_seed,
                    "end_to_end_seconds": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            observations.append(observation)
            result["observations"] = observations
            result["summary"] = summarize(observations)
            write_json(args.output, result)
            if observation.get("error"):
                print(f"failed: {observation['error']}")
            else:
                print(
                    f"completed mode={mode}: validation="
                    f"{observation['framework']['baseline_reward']:.4f}->"
                    f"{observation['framework']['final_reward']:.4f}, "
                    f"test_gain={observation['test']['combined_score_gain']:.4f}, "
                    f"TTQ={observation['time_to_quality_s']}, "
                    f"end_to_end={observation['end_to_end_seconds']:.2f}s"
                )

    errors = sum(bool(row.get("error")) for row in observations)
    result.update(
        {
            "status": "completed" if errors == 0 else "partial",
            "completed_at": utc_now(),
            "observations": observations,
            "summary": summarize(observations),
            "errors": errors,
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        calls = args.repeats * len(args.modes) * args.iterations
        print(
            f"OpenEvolve AgentDescent benchmark dry run: modes={','.join(args.modes)}, "
            f"repeats={args.repeats}, seeds={args.repeats}, "
            f"reserved model calls={calls}, temperature={args.temperature}"
        )
        return 0
    _validate(args)
    if not args.yes and sys.stdin.isatty():
        answer = input("Proceed with paid live benchmark calls? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted.")
            return 0
    result = run_benchmark(args)
    print(
        f"completed: observations={len(result['observations'])}, "
        f"errors={result['errors']}, output={args.output}"
    )
    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
