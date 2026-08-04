"""Measure serial, synchronous-parallel, and asynchronous time-to-quality.

The benchmark uses the real AgentDescent evolution drivers and an I/O-like
``sleep`` inside training rollouts. Held-out scoring is deterministic and cheap,
so the experiment isolates the scheduler: rollout concurrency and the round
barrier are the only meaningful differences between modes.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import EvolutionResult, Task, evolve
from agentdescent.strategies import AppendRules

from experiment._common import REPORT_DIR, utc_now, write_json


@dataclass
class Observation:
    scenario: str
    repeat: int
    mode: str
    concurrency: int
    target_reward: float
    time_to_quality_s: float
    cost_to_quality_rollouts: int
    final_reward: float
    wallclock_s: float
    total_rollouts: int
    rollout_seconds: float
    eval_seconds: float
    throughput_rollouts_s: float
    stop_reason: str
    history_points: int
    stale_considered: int
    stale_discarded: int
    stale_rate: float
    forced_refreshes: int
    retired_workers: int
    outcomes: Dict[str, int]
    timeline: List[Dict[str, float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--uniform-latency", type=float, default=0.08)
    parser.add_argument("--fast-latency", type=float, default=0.05)
    parser.add_argument("--slow-latency", type=float, default=0.40)
    parser.add_argument("--slow-workers", type=int, default=2)
    parser.add_argument("--heavy-tail-target", type=float, default=0.75)
    parser.add_argument(
        "--heavy-tail-pattern",
        choices=("transferable", "incremental"),
        default="transferable",
        help=(
            "transferable lets any fast rollout produce the shared rule that reaches "
            "the target; incremental makes each rollout solve one category"
        ),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--async-ratio", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "parallel-async-time-to-quality-result.json",
    )
    return parser


def make_tasks(
    latencies: Sequence[float],
    proposal_groups: Sequence[Sequence[str]] | None = None,
) -> List[Task]:
    categories = [f"c{index}" for index in range(len(latencies))]
    category_set = set(categories)
    if proposal_groups is None:
        proposal_groups = [[category] for category in categories]
    if len(proposal_groups) != len(latencies):
        raise ValueError("proposal_groups must have one entry per training task")

    tasks: List[Task] = []
    for index, latency in enumerate(latencies):
        proposal_categories = list(proposal_groups[index])
        if not proposal_categories:
            raise ValueError("each training task must propose at least one category")
        unknown = sorted(set(proposal_categories) - category_set)
        if unknown:
            raise ValueError(f"proposal_groups contains unknown categories: {unknown}")
        tasks.append(
            Task(
                id=f"train-{index}",
                prompt=f"learn category {index}",
                meta={
                    "category": f"c{index}",
                    "proposal_categories": proposal_categories,
                    "latency": float(latency),
                    "held_out": False,
                },
            )
        )
    for index in range(len(latencies)):
        tasks.append(
            Task(
                id=f"held-out-{index}",
                prompt=f"check category {index}",
                meta={
                    "category": f"c{index}",
                    "proposal_categories": [],
                    "latency": 0.0,
                    "held_out": True,
                },
            )
        )
    return tasks


def make_transferable_proposal_groups(
    workers: int, slow_workers: int
) -> List[List[str]]:
    """Make every fast task discover the same broadly useful rule."""
    fast_workers = workers - slow_workers
    fast_categories = [f"c{index}" for index in range(fast_workers)]
    return [
        list(fast_categories) if index < fast_workers else [f"c{index}"]
        for index in range(workers)
    ]


def run_actor(rendered: str, task: Task) -> str:
    if not task.meta["held_out"]:
        # sleep models network/tool latency and releases the GIL, like an API call.
        time.sleep(float(task.meta["latency"]))
    token = f"ENABLE_{task.meta['category']}"
    return "correct" if token in rendered else "incorrect"


def reward_actor(task: Task, output: str) -> float:
    return 1.0 if output == "correct" else 0.0


def propose_actor(rendered: str, task: Task, output: str, score: float) -> str:
    return "\n".join(
        f"ENABLE_{category}" for category in task.meta["proposal_categories"]
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 2:
        raise ValueError("workers must be at least 2")
    if args.repeats < 1 or args.rounds < 1:
        raise ValueError("repeats and rounds must be positive")
    if not 0 < args.slow_workers < args.workers:
        raise ValueError("slow-workers must be between 1 and workers - 1")
    if min(args.uniform_latency, args.fast_latency, args.slow_latency) < 0:
        raise ValueError("latencies must be non-negative")
    if args.slow_latency <= args.fast_latency:
        raise ValueError("slow-latency must be greater than fast-latency")
    if not 0.0 < args.heavy_tail_target <= 1.0:
        raise ValueError("heavy-tail-target must be in (0, 1]")
    fast_fraction = (args.workers - args.slow_workers) / args.workers
    if args.heavy_tail_target > fast_fraction:
        raise ValueError(
            "heavy-tail-target must be reachable using only fast workers; "
            f"maximum is {fast_fraction:.3f}"
        )


def _common_kwargs(
    latencies: Sequence[float],
    *,
    target: float,
    seed: int,
    workers: int,
    proposal_groups: Sequence[Sequence[str]] | None = None,
) -> Dict[str, Any]:
    return {
        "tasks": make_tasks(latencies, proposal_groups),
        "reward": reward_actor,
        "run": run_actor,
        "propose": propose_actor,
        "strategy": AppendRules(title="# Learned categories"),
        "held_out_frac": 0.5,
        "target_reward": target,
        "self_verify": False,
        "eval_concurrency": workers,
        "seed": seed,
    }


def _observation(
    result: EvolutionResult,
    *,
    scenario: str,
    repeat: int,
    mode: str,
    concurrency: int,
    target: float,
) -> Observation:
    if result.error:
        raise RuntimeError(f"{mode} run failed: {result.error}")
    reached = result.time_to_quality(target)
    cost = result.cost_to_quality(target)
    if reached is None or cost is None:
        raise RuntimeError(
            f"{mode} failed to reach target={target}; final={result.final_reward}, "
            f"stop={result.stop_reason}"
        )
    return Observation(
        scenario=scenario,
        repeat=repeat,
        mode=mode,
        concurrency=concurrency,
        target_reward=target,
        time_to_quality_s=reached,
        cost_to_quality_rollouts=cost,
        final_reward=result.final_reward,
        wallclock_s=result.wallclock,
        total_rollouts=result.rollouts,
        rollout_seconds=result.rollout_seconds,
        eval_seconds=result.eval_seconds,
        throughput_rollouts_s=(result.rollouts / result.wallclock if result.wallclock else 0.0),
        stop_reason=result.stop_reason,
        history_points=len(result.history),
        stale_considered=result.stale_considered,
        stale_discarded=result.stale_discarded,
        stale_rate=result.stale_rate(),
        forced_refreshes=result.forced_refreshes,
        retired_workers=result.retired_workers,
        outcomes=result.outcomes(),
        timeline=[
            {
                "elapsed_s": point.elapsed_s,
                "reward": point.held_out_reward,
                "rollouts": float(point.rollouts),
            }
            for point in result.history
        ],
    )


def run_sync_observation(
    latencies: Sequence[float],
    *,
    scenario: str,
    repeat: int,
    concurrency: int,
    target: float,
    rounds: int,
    max_seconds: float,
    proposal_groups: Sequence[Sequence[str]] | None = None,
) -> Observation:
    workers = len(latencies)
    result = evolve(
        **_common_kwargs(
            latencies,
            target=target,
            seed=repeat,
            workers=workers,
            proposal_groups=proposal_groups,
        ),
        rounds=rounds,
        n_workers=workers,
        max_concurrency=concurrency,
        max_seconds=max_seconds,
    )
    mode = "serial" if concurrency == 1 else f"sync_parallel_{concurrency}"
    return _observation(
        result,
        scenario=scenario,
        repeat=repeat,
        mode=mode,
        concurrency=concurrency,
        target=target,
    )


def run_async_observation(
    latencies: Sequence[float],
    *,
    scenario: str,
    repeat: int,
    target: float,
    rounds: int,
    async_ratio: int,
    max_seconds: float,
    proposal_groups: Sequence[Sequence[str]] | None = None,
) -> Observation:
    workers = len(latencies)
    result = async_evolve(
        **_common_kwargs(
            latencies,
            target=target,
            seed=repeat,
            workers=workers,
            proposal_groups=proposal_groups,
        ),
        n_workers=workers,
        async_ratio=async_ratio,
        max_seconds=max_seconds,
        max_iters=rounds * workers,
        shutdown_grace=max(latencies) + 0.25,
    )
    return _observation(
        result,
        scenario=scenario,
        repeat=repeat,
        mode="async_no_barrier",
        concurrency=workers,
        target=target,
    )


def _rotate(items: Sequence[Any], amount: int) -> List[Any]:
    items = list(items)
    offset = amount % len(items)
    return items[offset:] + items[:offset]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(observations: Sequence[Observation]) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str], List[Observation]] = {}
    for observation in observations:
        grouped.setdefault((observation.scenario, observation.mode), []).append(observation)
    summary: Dict[str, Any] = {}
    for (scenario, mode), rows in grouped.items():
        t = [row.time_to_quality_s for row in rows]
        wall = [row.wallclock_s for row in rows]
        costs = [row.cost_to_quality_rollouts for row in rows]
        rewards = [row.final_reward for row in rows]
        summary.setdefault(scenario, {})[mode] = {
            "runs": len(rows),
            "concurrency": rows[0].concurrency,
            "target_reward": rows[0].target_reward,
            "time_to_quality_s": {
                "mean": statistics.fmean(t),
                "median": statistics.median(t),
                "min": min(t),
                "max": max(t),
                "p95": _percentile(t, 0.95),
                "population_stddev": statistics.pstdev(t),
            },
            "wallclock_s": {
                "mean": statistics.fmean(wall),
                "median": statistics.median(wall),
            },
            "cost_to_quality_rollouts": {
                "mean": statistics.fmean(costs),
                "median": statistics.median(costs),
                "min": min(costs),
                "max": max(costs),
            },
            "final_reward": {
                "mean": statistics.fmean(rewards),
                "min": min(rewards),
                "max": max(rewards),
            },
            "stale_rate_mean": statistics.fmean(row.stale_rate for row in rows),
            "retired_workers_max": max(row.retired_workers for row in rows),
        }
    return summary


def _median_t(summary: Dict[str, Any], scenario: str, mode: str) -> float:
    return summary[scenario][mode]["time_to_quality_s"]["median"]


def speedups(
    summary: Dict[str, Any],
    workers: int,
    observations: Sequence[Observation],
) -> Dict[str, Any]:
    uniform_serial = _median_t(summary, "uniform_scaling", "serial")
    uniform: Dict[str, float] = {}
    for mode, row in summary["uniform_scaling"].items():
        uniform[mode] = uniform_serial / row["time_to_quality_s"]["median"]

    heavy_serial = _median_t(summary, "heavy_tail", "serial")
    heavy_parallel = _median_t(summary, "heavy_tail", f"sync_parallel_{workers}")
    heavy_async = _median_t(summary, "heavy_tail", "async_no_barrier")
    paired: Dict[int, Dict[str, float]] = {}
    for row in observations:
        if row.scenario != "heavy_tail":
            continue
        paired.setdefault(row.repeat, {})[row.mode] = row.time_to_quality_s
    pairwise = [
        values[f"sync_parallel_{workers}"] / values["async_no_barrier"]
        for values in paired.values()
        if f"sync_parallel_{workers}" in values and "async_no_barrier" in values
    ]
    return {
        "uniform_scaling_vs_serial": uniform,
        "heavy_tail": {
            "sync_parallel_vs_serial": heavy_serial / heavy_parallel,
            "async_vs_serial": heavy_serial / heavy_async,
            "async_vs_sync_parallel": heavy_parallel / heavy_async,
            "paired_async_vs_sync": {
                "runs": len(pairwise),
                "median_speedup": statistics.median(pairwise),
                "min_speedup": min(pairwise),
                "max_speedup": max(pairwise),
                "async_faster_runs": sum(value > 1.0 for value in pairwise),
            },
        },
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    started = time.monotonic()
    observations: List[Observation] = []
    uniform_latencies = [args.uniform_latency] * args.workers
    heavy_latencies = [args.fast_latency] * (args.workers - args.slow_workers) + [
        args.slow_latency
    ] * args.slow_workers
    heavy_proposal_groups = None
    if args.heavy_tail_pattern == "transferable":
        heavy_proposal_groups = make_transferable_proposal_groups(
            args.workers, args.slow_workers
        )
    concurrency_levels = sorted({1, 2, 4, args.workers})
    concurrency_levels = [level for level in concurrency_levels if level <= args.workers]

    result: Dict[str, Any] = {
        "experiment": "AgentDescent parallel/async time-to-quality",
        "status": "running",
        "started_at": utc_now(),
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "workers": args.workers,
            "repeats": args.repeats,
            "uniform_latency_s": args.uniform_latency,
            "fast_latency_s": args.fast_latency,
            "slow_latency_s": args.slow_latency,
            "slow_workers": args.slow_workers,
            "uniform_target_reward": 1.0,
            "heavy_tail_target_reward": args.heavy_tail_target,
            "heavy_tail_pattern": args.heavy_tail_pattern,
            "rounds": args.rounds,
            "max_rollout_budget": args.rounds * args.workers,
            "async_ratio": args.async_ratio,
            "max_seconds": args.max_seconds,
            "self_verify": False,
            "eval_concurrency": args.workers,
        },
        "method": {
            "uniform_artifact": "AppendRules with one rule per category",
            "heavy_tail_artifact": (
                "any fast task proposes one transferable rule covering every fast "
                "category"
                if args.heavy_tail_pattern == "transferable"
                else "each task proposes one category-specific rule"
            ),
            "split": f"{args.workers} train + {args.workers} held-out tasks",
            "quality": "fraction of held-out categories whose rule was learned",
            "latency_scope": "training rollouts only; held-out scorer is deterministic and cheap",
            "serial_definition": (
                f"n_workers={args.workers}, max_concurrency=1; same work units as parallel"
            ),
            "parallel_definition": (
                f"synchronous DataParallel with max_concurrency up to {args.workers}"
            ),
            "async_definition": (
                f"async_evolve with {args.workers} producers and no round barrier"
            ),
        },
        "observations": [],
    }
    write_json(args.output, result)

    for repeat in range(args.repeats):
        for concurrency in _rotate(concurrency_levels, repeat):
            observation = run_sync_observation(
                uniform_latencies,
                scenario="uniform_scaling",
                repeat=repeat,
                concurrency=concurrency,
                target=1.0,
                rounds=args.rounds,
                max_seconds=args.max_seconds,
            )
            observations.append(observation)
            print(
                f"uniform repeat={repeat + 1}/{args.repeats} concurrency={concurrency}: "
                f"TTQ={observation.time_to_quality_s:.4f}s"
            )

        heavy_modes = ["serial", "sync", "async"]
        for mode in _rotate(heavy_modes, repeat):
            if mode == "serial":
                observation = run_sync_observation(
                    heavy_latencies,
                    scenario="heavy_tail",
                    repeat=repeat,
                    concurrency=1,
                    target=args.heavy_tail_target,
                    rounds=args.rounds,
                    max_seconds=args.max_seconds,
                    proposal_groups=heavy_proposal_groups,
                )
            elif mode == "sync":
                observation = run_sync_observation(
                    heavy_latencies,
                    scenario="heavy_tail",
                    repeat=repeat,
                    concurrency=args.workers,
                    target=args.heavy_tail_target,
                    rounds=args.rounds,
                    max_seconds=args.max_seconds,
                    proposal_groups=heavy_proposal_groups,
                )
            else:
                observation = run_async_observation(
                    heavy_latencies,
                    scenario="heavy_tail",
                    repeat=repeat,
                    target=args.heavy_tail_target,
                    rounds=args.rounds,
                    async_ratio=args.async_ratio,
                    max_seconds=args.max_seconds,
                    proposal_groups=heavy_proposal_groups,
                )
            observations.append(observation)
            print(
                f"heavy repeat={repeat + 1}/{args.repeats} mode={observation.mode}: "
                f"TTQ={observation.time_to_quality_s:.4f}s, "
                f"cost={observation.cost_to_quality_rollouts}"
            )

        result["observations"] = [asdict(item) for item in observations]
        result["wall_seconds_so_far"] = time.monotonic() - started
        write_json(args.output, result)

    aggregate = summarize(observations)
    result.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "observations": [asdict(item) for item in observations],
            "summary": aggregate,
            "speedups": speedups(aggregate, args.workers, observations),
            "benchmark_wall_seconds": time.monotonic() - started,
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.dry_run:
        modes_per_repeat = len({1, 2, 4, args.workers}) + 3
        print(
            "parallel/async TTQ dry run: "
            f"workers={args.workers}, repeats={args.repeats}, "
            f"runs={args.repeats * modes_per_repeat}, output={args.output}"
        )
        return 0
    result = run_benchmark(args)
    heavy = result["speedups"]["heavy_tail"]
    uniform = result["speedups"]["uniform_scaling_vs_serial"]
    print(
        "completed: "
        f"sync-{args.workers} uniform speedup={uniform[f'sync_parallel_{args.workers}']:.2f}x; "
        f"async vs sync-heavy={heavy['async_vs_sync_parallel']:.2f}x; "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
