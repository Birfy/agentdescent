"""Benchmark parallel and asynchronous execution on TextGrad and OpenEvolve.

Unlike the scheduler-only benchmark, this script makes real model calls and runs
the two implemented algorithms.  Each mode receives the same data, candidate
budget, quality threshold, and concurrency cap.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from agentdescent.agents import Usage

from experiment import openevolve_program_search as oe
from experiment import textgrad_prompt_optimization as tg
from experiment._common import (
    REPORT_DIR,
    add_model_args,
    confirm_paid_run,
    make_completion,
    require_api_environment,
    sum_usage,
    usage_dict,
    utc_now,
    write_json,
)


MODES = ("serial", "sync_parallel", "async_pipeline")
ALGORITHMS = ("textgrad", "openevolve")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser, max_tokens=1024, temperature=0.0)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--textgrad-batch-size", type=int, default=2)
    parser.add_argument("--textgrad-val-size", type=int, default=3)
    parser.add_argument("--textgrad-target-accuracy", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--textgrad-subset",
        choices=("first", "longest"),
        default="first",
    )
    parser.add_argument("--openevolve-candidates", type=int, default=3)
    parser.add_argument("--openevolve-trials", type=int, default=5)
    parser.add_argument("--openevolve-objective-budget", type=int, default=100)
    parser.add_argument("--openevolve-min-score-gain", type=float, default=0.005)
    parser.add_argument("--openevolve-candidate-timeout", type=float, default=15.0)
    parser.add_argument("--openevolve-max-code-length", type=int, default=20000)
    parser.add_argument("--openevolve-archive-size", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "algorithm-parallel-async-result.json",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1 or args.concurrency < 1:
        raise ValueError("repeats and concurrency must be positive")
    if args.textgrad_batch_size < 2:
        raise ValueError("textgrad-batch-size must be at least 2 to test a pipeline")
    if args.textgrad_val_size < 1:
        raise ValueError("textgrad-val-size must be positive")
    if not 0.0 < args.textgrad_target_accuracy <= 1.0:
        raise ValueError("textgrad-target-accuracy must be in (0, 1]")
    if args.openevolve_candidates < 2:
        raise ValueError("openevolve-candidates must be at least 2")
    if args.openevolve_trials < 1 or args.openevolve_objective_budget < 2:
        raise ValueError("OpenEvolve trials must be positive and budget must be at least 2")
    if args.openevolve_min_score_gain < 0:
        raise ValueError("openevolve-min-score-gain must be non-negative")
    if args.openevolve_archive_size < 1:
        raise ValueError("openevolve-archive-size must be positive")


def _rotate(items: Sequence[str], amount: int) -> List[str]:
    values = list(items)
    offset = amount % len(values)
    return values[offset:] + values[:offset]


def _parallel_map(
    function: Callable[[Any], Any], items: Sequence[Any], concurrency: int
) -> List[Any]:
    with ThreadPoolExecutor(max_workers=min(concurrency, len(items))) as pool:
        return list(pool.map(function, items))


def _recorded_call(
    function: Callable[[], Any],
    *,
    phase: str,
    unit: str,
    started: float,
    events: List[Dict[str, Any]],
    lock: threading.Lock,
) -> Any:
    call_started = time.monotonic()
    success = False
    try:
        value = function()
        success = True
        return value
    finally:
        ended = time.monotonic()
        with lock:
            events.append(
                {
                    "phase": phase,
                    "unit": unit,
                    "started_s": call_started - started,
                    "ended_s": ended - started,
                    "duration_s": ended - call_started,
                    "success": success,
                }
            )


def _stage_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["phase"], []).append(event)
    return {
        phase: {
            "calls": len(rows),
            "wall_span_s": max(row["ended_s"] for row in rows)
            - min(row["started_s"] for row in rows),
            "sum_call_s": sum(row["duration_s"] for row in rows),
            "min_call_s": min(row["duration_s"] for row in rows),
            "max_call_s": max(row["duration_s"] for row in rows),
        }
        for phase, rows in grouped.items()
    }


def _textgrad_eval_payload(results: Sequence[tg.ItemResult]) -> Dict[str, Any]:
    solved = sum(result.correct for result in results)
    return {
        "accuracy": solved / len(results),
        "solved": solved,
        "total": len(results),
        "wilson_95": tg.wilson_interval(solved, len(results)),
        "items": [asdict(result) for result in results],
    }


def run_textgrad_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    repeat: int,
    train: Sequence[tg.Example],
    val: Sequence[tg.Example],
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown execution mode: {mode}")
    usages = {
        "forward": Usage(),
        "response_gradient": Usage(),
        "prompt_gradient": Usage(),
        "tgd_update": Usage(),
    }
    completions = {
        phase: make_completion(args, usage)
        for phase, usage in usages.items()
    }
    started = time.monotonic()
    events: List[Dict[str, Any]] = []
    event_lock = threading.Lock()
    parallel = mode != "serial"

    def invoke(
        completion_phase: str,
        event_phase: str,
        unit: str,
        prompt: str,
    ) -> str:
        return _recorded_call(
            lambda: completions[completion_phase](prompt).strip(),
            phase=event_phase,
            unit=unit,
            started=started,
            events=events,
            lock=event_lock,
        )

    def solve(system_prompt: str, item: tg.Example, phase: str) -> tg.ItemResult:
        response = invoke(
            "forward",
            phase,
            f"{phase}:item-{item.index}",
            tg.solve_prompt(system_prompt, item.question),
        )
        correct, predicted = tg.score_response(response, item.target)
        return tg.ItemResult(item.index, correct, predicted, item.target, response)

    def evaluate(system_prompt: str, items: Sequence[tg.Example], phase: str):
        function = lambda item: solve(system_prompt, item, phase)
        rows = (
            _parallel_map(function, items, args.concurrency)
            if parallel
            else [function(item) for item in items]
        )
        return _textgrad_eval_payload(rows)

    baseline = evaluate(tg.STARTING_PROMPT, val, "baseline_eval")
    baseline_ready_s = time.monotonic() - started
    target = args.textgrad_target_accuracy
    time_to_quality = baseline_ready_s if baseline["accuracy"] >= target else None
    batch = tg._batch_for_step(
        train,
        step=0,
        batch_size=args.textgrad_batch_size,
        seed=args.seed + repeat,
    )

    def response_gradient(pair: Tuple[tg.Example, tg.ItemResult]) -> str:
        item, item_result = pair
        return invoke(
            "response_gradient",
            "response_gradient",
            f"item-{item.index}",
            tg._response_gradient_prompt(item, item_result),
        )

    def prompt_gradient(
        triple: Tuple[tg.Example, tg.ItemResult, str]
    ) -> Dict[str, Any]:
        item, item_result, response_feedback = triple
        feedback = invoke(
            "prompt_gradient",
            "prompt_gradient",
            f"item-{item.index}",
            tg._prompt_gradient_prompt(
                tg.STARTING_PROMPT,
                item,
                item_result,
                response_feedback,
            ),
        )
        return {
            "item": item,
            "result": item_result,
            "response_gradient": response_feedback,
            "prompt_gradient": feedback,
        }

    def full_trajectory(item: tg.Example) -> Dict[str, Any]:
        item_result = solve(tg.STARTING_PROMPT, item, "train_forward")
        feedback = response_gradient((item, item_result))
        return prompt_gradient((item, item_result, feedback))

    if mode == "serial":
        trajectories = [full_trajectory(item) for item in batch]
    elif mode == "sync_parallel":
        batch_results = _parallel_map(
            lambda item: solve(tg.STARTING_PROMPT, item, "train_forward"),
            batch,
            args.concurrency,
        )
        response_feedback = _parallel_map(
            response_gradient,
            list(zip(batch, batch_results)),
            args.concurrency,
        )
        trajectories = _parallel_map(
            prompt_gradient,
            list(zip(batch, batch_results, response_feedback)),
            args.concurrency,
        )
    else:
        trajectories = _parallel_map(full_trajectory, batch, args.concurrency)

    raw_update = invoke(
        "tgd_update",
        "tgd_update",
        "batch-update",
        tg._update_prompt(
            tg.STARTING_PROMPT,
            [entry["prompt_gradient"] for entry in trajectories],
        ),
    )
    candidate = tg.extract_improved_variable(raw_update)
    valid, rejection_reason = tg.validate_candidate(candidate, batch)
    candidate_validation = evaluate(candidate, val, "candidate_eval") if valid else None
    candidate_ready_s = time.monotonic() - started
    accepted = bool(
        valid and candidate_validation["accuracy"] >= baseline["accuracy"]
    )
    if valid and not accepted:
        rejection_reason = (
            f"validation accuracy regressed from {baseline['accuracy']:.6f} "
            f"to {candidate_validation['accuracy']:.6f}"
        )
    final_accuracy = (
        candidate_validation["accuracy"] if accepted else baseline["accuracy"]
    )
    if time_to_quality is None and final_accuracy >= target:
        time_to_quality = candidate_ready_s
    wall = time.monotonic() - started
    usage = sum_usage(usages)
    return {
        "algorithm": "textgrad",
        "mode": mode,
        "repeat": repeat,
        "target_quality": target,
        "target_reached": time_to_quality is not None,
        "time_to_quality_s": time_to_quality,
        "candidate_ready_s": candidate_ready_s,
        "wall_seconds": wall,
        "baseline_quality": baseline["accuracy"],
        "final_quality": final_accuracy,
        "quality_gain": final_accuracy - baseline["accuracy"],
        "baseline_validation": baseline,
        "candidate_validation": candidate_validation,
        "candidate_valid": valid,
        "candidate_accepted": accepted,
        "candidate_rejection_reason": rejection_reason,
        "candidate_prompt": candidate,
        "batch_ids": [item.index for item in batch],
        "gradients": [
            {
                "item_id": entry["item"].index,
                "response_gradient": entry["response_gradient"],
                "prompt_gradient": entry["prompt_gradient"],
            }
            for entry in trajectories
        ],
        "usage": usage,
        "throughput_calls_s": (
            usage["total"]["calls"] / wall if wall else 0.0
        ),
        "stage_summary": _stage_summary(events),
        "events": sorted(events, key=lambda event: event["started_s"]),
    }


def run_openevolve_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    repeat: int,
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown execution mode: {mode}")
    usage = Usage()
    completion = make_completion(args, usage)
    started = time.monotonic()
    events: List[Dict[str, Any]] = []
    event_lock = threading.Lock()
    seed = args.seed + repeat

    def record(function, phase: str, unit: str):
        return _recorded_call(
            function,
            phase=phase,
            unit=unit,
            started=started,
            events=events,
            lock=event_lock,
        )

    baseline_valid, baseline_metrics, baseline_error, baseline_trials = record(
        lambda: oe.evaluate_source(
            oe.INITIAL_PROGRAM,
            trials=args.openevolve_trials,
            budget=args.openevolve_objective_budget,
            seed=seed,
            timeout=args.openevolve_candidate_timeout,
            max_length=args.openevolve_max_code_length,
        ),
        "baseline_evaluation",
        "initial-program",
    )
    if not baseline_valid:
        raise RuntimeError(f"OpenEvolve initial program failed: {baseline_error}")
    initial = oe.Program(
        oe._program_id(oe.INITIAL_PROGRAM),
        0,
        0,
        None,
        oe.INITIAL_PROGRAM,
        "uniform random search baseline",
        baseline_metrics,
        True,
    )
    archive = [initial]
    target = baseline_metrics["combined_score"] + args.openevolve_min_score_gain
    time_to_quality = None
    completion_order: List[Dict[str, Any]] = []
    programs: Dict[int, Tuple[oe.Program, List[Dict[str, Any]]]] = {}

    def generate(slot: int) -> Tuple[int, str, str]:
        raw = record(
            lambda: completion(
                oe._mutation_prompt(
                    initial,
                    initial,
                    initial,
                    iteration=slot + 1,
                    budget=args.openevolve_objective_budget,
                    trials=args.openevolve_trials,
                )
            ).strip(),
            "mutation",
            f"candidate-{slot + 1}",
        )
        code, summary = oe.extract_program(raw)
        return slot, code, summary

    def evaluate(generated: Tuple[int, str, str]):
        slot, code, change_summary = generated
        valid, metrics, error, trials = record(
            lambda: oe.evaluate_source(
                code,
                trials=args.openevolve_trials,
                budget=args.openevolve_objective_budget,
                seed=seed,
                timeout=args.openevolve_candidate_timeout,
                max_length=args.openevolve_max_code_length,
            ),
            "candidate_evaluation",
            f"candidate-{slot + 1}",
        )
        program = oe.Program(
            oe._program_id(code),
            slot + 1,
            slot % max(1, args.openevolve_candidates),
            initial.program_id,
            code,
            change_summary,
            metrics,
            valid,
            error,
        )
        return slot, program, trials

    def pipeline(slot: int):
        return evaluate(generate(slot))

    def commit(item: Tuple[int, oe.Program, List[Dict[str, Any]]]) -> None:
        nonlocal archive, time_to_quality
        slot, program, trials = item
        programs[slot] = (program, trials)
        if program.valid:
            archive = oe.prune_archive(
                [*archive, program], args.openevolve_archive_size
            )
        best = max(archive, key=lambda row: row.metrics["combined_score"])
        elapsed = time.monotonic() - started
        completion_order.append(
            {
                "slot": slot + 1,
                "elapsed_s": elapsed,
                "candidate_score": program.metrics["combined_score"],
                "best_score": best.metrics["combined_score"],
                "valid": program.valid,
            }
        )
        if time_to_quality is None and best.metrics["combined_score"] >= target:
            time_to_quality = elapsed

    slots = list(range(args.openevolve_candidates))
    if mode == "serial":
        for slot in slots:
            commit(pipeline(slot))
    elif mode == "sync_parallel":
        generated = _parallel_map(generate, slots, args.concurrency)
        evaluated = _parallel_map(evaluate, generated, args.concurrency)
        for item in sorted(evaluated, key=lambda row: row[0]):
            commit(item)
    else:
        with ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(slots))
        ) as pool:
            futures = [pool.submit(pipeline, slot) for slot in slots]
            for future in as_completed(futures):
                commit(future.result())

    wall = time.monotonic() - started
    best = max(archive, key=lambda row: row.metrics["combined_score"])
    return {
        "algorithm": "openevolve",
        "mode": mode,
        "repeat": repeat,
        "target_quality": target,
        "target_reached": time_to_quality is not None,
        "time_to_quality_s": time_to_quality,
        "candidate_ready_s": (
            min(row["elapsed_s"] for row in completion_order)
            if completion_order
            else None
        ),
        "wall_seconds": wall,
        "baseline_quality": baseline_metrics["combined_score"],
        "final_quality": best.metrics["combined_score"],
        "quality_gain": best.metrics["combined_score"]
        - baseline_metrics["combined_score"],
        "baseline": oe._serialize_program(initial, include_trials=baseline_trials),
        "best": oe._serialize_program(best),
        "candidates": [
            oe._serialize_program(programs[slot][0], include_trials=programs[slot][1])
            for slot in sorted(programs)
        ],
        "completion_order": completion_order,
        "usage": usage_dict(usage),
        "throughput_calls_s": usage.calls / wall if wall else 0.0,
        "stage_summary": _stage_summary(events),
        "events": sorted(events, key=lambda event: event["started_s"]),
    }


def summarize(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(
            (observation["algorithm"], observation["mode"]), []
        ).append(observation)
    summary: Dict[str, Any] = {}
    for (algorithm, mode), rows in grouped.items():
        reached = [row["time_to_quality_s"] for row in rows if row["target_reached"]]
        walls = [row["wall_seconds"] for row in rows]
        gains = [row["quality_gain"] for row in rows]
        calls = [
            row["usage"]["total"]["calls"]
            if algorithm == "textgrad"
            else row["usage"]["calls"]
            for row in rows
        ]
        tokens = [
            row["usage"]["total"]["total_tokens"]
            if algorithm == "textgrad"
            else row["usage"]["total_tokens"]
            for row in rows
        ]
        summary.setdefault(algorithm, {})[mode] = {
            "runs": len(rows),
            "target_reached_runs": len(reached),
            "time_to_quality_s_median": statistics.median(reached) if reached else None,
            "wall_seconds_median": statistics.median(walls),
            "quality_gain_median": statistics.median(gains),
            "calls_median": statistics.median(calls),
            "tokens_median": statistics.median(tokens),
        }
    for algorithm, modes in summary.items():
        serial = modes.get("serial", {}).get("time_to_quality_s_median")
        for mode, row in modes.items():
            ttq = row["time_to_quality_s_median"]
            row["ttq_speedup_vs_serial"] = serial / ttq if serial and ttq else None
        sync = modes.get("sync_parallel", {}).get("time_to_quality_s_median")
        asynchronous = modes.get("async_pipeline", {}).get("time_to_quality_s_median")
        modes["async_vs_sync_ttq_speedup"] = (
            sync / asynchronous if sync and asynchronous else None
        )
    return summary


def _event_duration(
    observation: Dict[str, Any], phase: str, unit_suffix: str | None = None
) -> float:
    matches = [
        event
        for event in observation["events"]
        if event["phase"] == phase
        and (unit_suffix is None or event["unit"].endswith(unit_suffix))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one event for phase={phase!r}, suffix={unit_suffix!r}; "
            f"found {len(matches)}"
        )
    return float(matches[0]["duration_s"])


def _phase_durations(observation: Dict[str, Any], phase: str) -> List[float]:
    return [
        float(event["duration_s"])
        for event in observation["events"]
        if event["phase"] == phase
    ]


def _textgrad_trace_replay(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    canonical = next(
        (row for row in rows if row["mode"] == "sync_parallel"), rows[0]
    )
    target = min(row["final_quality"] for row in rows)
    if target <= max(row["baseline_quality"] for row in rows):
        raise ValueError("TextGrad trace has no common post-optimization quality target")

    baseline = _phase_durations(canonical, "baseline_eval")
    candidate_eval = _phase_durations(canonical, "candidate_eval")
    update = _event_duration(canonical, "tgd_update")
    per_item: Dict[int, Dict[str, float]] = {}
    for item_id in canonical["batch_ids"]:
        suffix = f"item-{item_id}"
        per_item[item_id] = {
            phase: _event_duration(canonical, phase, suffix)
            for phase in ("train_forward", "response_gradient", "prompt_gradient")
        }

    serial_train = sum(sum(phases.values()) for phases in per_item.values())
    sync_train = sum(
        max(phases[phase] for phases in per_item.values())
        for phase in ("train_forward", "response_gradient", "prompt_gradient")
    )
    async_train = max(sum(phases.values()) for phases in per_item.values())
    serial = sum(baseline) + serial_train + update + sum(candidate_eval)
    sync = max(baseline) + sync_train + update + max(candidate_eval)
    asynchronous = max(baseline) + async_train + update + max(candidate_eval)

    live_common_target: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        reached = row["final_quality"] >= target
        live_common_target[row["mode"]] = {
            "reached": reached,
            "time_to_quality_s": row["candidate_ready_s"] if reached else None,
            "baseline_quality": row["baseline_quality"],
            "final_quality": row["final_quality"],
        }
    return {
        "source_mode": canonical["mode"],
        "source_repeat": canonical["repeat"],
        "quality_target": target,
        "source_model_calls": canonical["usage"]["total"]["calls"],
        "source_tokens": canonical["usage"]["total"]["total_tokens"],
        "method": (
            "critical-path replay of one real TextGrad response/gradient/candidate "
            "trace; every mode receives identical outputs and call durations"
        ),
        "live_observed_at_common_target": live_common_target,
        "modes": {
            "serial": {
                "time_to_quality_s": serial,
                "calls_to_quality": canonical["usage"]["total"]["calls"],
            },
            "sync_parallel": {
                "time_to_quality_s": sync,
                "calls_to_quality": canonical["usage"]["total"]["calls"],
                "speedup_vs_serial": serial / sync,
            },
            "async_pipeline": {
                "time_to_quality_s": asynchronous,
                "calls_to_quality": canonical["usage"]["total"]["calls"],
                "speedup_vs_serial": serial / asynchronous,
                "speedup_vs_sync": sync / asynchronous,
            },
        },
        "components_s": {
            "baseline_serial": sum(baseline),
            "baseline_parallel": max(baseline),
            "train_serial": serial_train,
            "train_sync_parallel": sync_train,
            "train_async_pipeline": async_train,
            "update": update,
            "candidate_eval_serial": sum(candidate_eval),
            "candidate_eval_parallel": max(candidate_eval),
        },
    }


def _openevolve_trace_replay(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    canonical = max(rows, key=lambda row: row["final_quality"])
    target = canonical["target_quality"]
    baseline = _event_duration(canonical, "baseline_evaluation")
    candidates = []
    for slot, candidate in enumerate(canonical["candidates"], 1):
        suffix = f"candidate-{slot}"
        candidates.append(
            {
                "slot": slot,
                "score": candidate["metrics"]["combined_score"],
                "valid": candidate["valid"],
                "mutation_s": _event_duration(canonical, "mutation", suffix),
                "evaluation_s": _event_duration(
                    canonical, "candidate_evaluation", suffix
                ),
            }
        )

    serial_elapsed = baseline
    serial_cost = 0
    serial_cost_to_quality = None
    serial_best = canonical["baseline_quality"]
    serial_ttq = None
    for candidate in candidates:
        serial_elapsed += candidate["mutation_s"] + candidate["evaluation_s"]
        serial_cost += 1
        if candidate["valid"]:
            serial_best = max(serial_best, candidate["score"])
        if serial_ttq is None and serial_best >= target:
            serial_ttq = serial_elapsed
            serial_cost_to_quality = serial_cost

    sync_full = baseline + max(row["mutation_s"] for row in candidates) + max(
        row["evaluation_s"] for row in candidates
    )
    sync_ttq = sync_full if max(row["score"] for row in candidates if row["valid"]) >= target else None

    asynchronous_best = canonical["baseline_quality"]
    asynchronous_ttq = None
    asynchronous_cost = 0
    completion_order = sorted(
        candidates,
        key=lambda row: row["mutation_s"] + row["evaluation_s"],
    )
    for candidate in completion_order:
        asynchronous_cost += 1
        if candidate["valid"]:
            asynchronous_best = max(asynchronous_best, candidate["score"])
        if asynchronous_best >= target:
            asynchronous_ttq = (
                baseline + candidate["mutation_s"] + candidate["evaluation_s"]
            )
            break
    asynchronous_full = baseline + max(
        row["mutation_s"] + row["evaluation_s"] for row in candidates
    )
    if (
        serial_ttq is None
        or serial_cost_to_quality is None
        or sync_ttq is None
        or asynchronous_ttq is None
    ):
        raise ValueError("canonical OpenEvolve trace does not reach its quality target")

    return {
        "source_mode": canonical["mode"],
        "source_repeat": canonical["repeat"],
        "quality_target": target,
        "baseline_quality": canonical["baseline_quality"],
        "final_quality": max(
            canonical["baseline_quality"],
            *(row["score"] for row in candidates if row["valid"]),
        ),
        "source_model_calls": canonical["usage"]["calls"],
        "source_tokens": canonical["usage"]["total_tokens"],
        "method": (
            "critical-path replay of one real OpenEvolve candidate generation and "
            "sandbox-evaluation trace; every mode receives identical programs and durations"
        ),
        "modes": {
            "serial": {
                "time_to_quality_s": serial_ttq,
                "calls_to_quality": serial_cost_to_quality,
                "full_budget_wall_s": serial_elapsed,
            },
            "sync_parallel": {
                "time_to_quality_s": sync_ttq,
                "calls_to_quality": len(candidates),
                "full_budget_wall_s": sync_full,
                "speedup_vs_serial": serial_ttq / sync_ttq,
            },
            "async_pipeline": {
                "time_to_quality_s": asynchronous_ttq,
                "calls_to_quality": asynchronous_cost,
                "full_budget_wall_s": asynchronous_full,
                "speedup_vs_serial": serial_ttq / asynchronous_ttq,
                "speedup_vs_sync": sync_ttq / asynchronous_ttq,
            },
        },
        "candidates": candidates,
    }


def build_trace_replay(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_algorithm = {
        algorithm: [
            row for row in observations if row["algorithm"] == algorithm
        ]
        for algorithm in ALGORITHMS
    }
    replay: Dict[str, Any] = {
        "scope": (
            "paired scheduler analysis over real API traces; estimates exclude "
            "thread-launch and ledger overhead"
        )
    }
    if by_algorithm["textgrad"]:
        replay["textgrad"] = _textgrad_trace_replay(by_algorithm["textgrad"])
    if by_algorithm["openevolve"]:
        replay["openevolve"] = _openevolve_trace_replay(
            by_algorithm["openevolve"]
        )
    return replay


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    require_api_environment(args.provider)
    result: Dict[str, Any] = {
        "experiment": "TextGrad and OpenEvolve parallel/async benchmark",
        "status": "running",
        "started_at": utc_now(),
        "config": {
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "api_timeout": args.api_timeout,
            "seed": args.seed,
            "algorithms": args.algorithms,
            "modes": args.modes,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "textgrad": {
                "batch_size": args.textgrad_batch_size,
                "validation_size": args.textgrad_val_size,
                "target_accuracy": args.textgrad_target_accuracy,
                "subset": args.textgrad_subset,
                "steps": 1,
            },
            "openevolve": {
                "candidates": args.openevolve_candidates,
                "trials": args.openevolve_trials,
                "objective_budget": args.openevolve_objective_budget,
                "min_score_gain": args.openevolve_min_score_gain,
                "candidate_timeout": args.openevolve_candidate_timeout,
                "archive_size": args.openevolve_archive_size,
                "generations": 1,
                "fixed_parent_within_generation": True,
            },
        },
        "method": {
            "provider_calls": "live API calls; no replay or synthetic sleep",
            "serial": "one model/evaluator trajectory at a time",
            "sync_parallel": "parallel work within each stage, with a barrier between stages",
            "async_pipeline": "each independent trajectory advances immediately to its next stage",
        },
        "observations": [],
    }
    write_json(args.output, result)
    observations: List[Dict[str, Any]] = []
    textgrad_data = None
    if "textgrad" in args.algorithms:
        train, val, _, digest = tg.load_bbh_splits(
            train_size=args.textgrad_batch_size,
            val_size=args.textgrad_val_size,
            test_size=1,
            subset=args.textgrad_subset,
        )
        textgrad_data = (train, val)
        result["dataset"] = {
            "textgrad_bbh_sha256": digest,
            "train_ids": [item.index for item in train],
            "validation_ids": [item.index for item in val],
        }

    for repeat in range(args.repeats):
        for algorithm_index, algorithm in enumerate(args.algorithms):
            for mode in _rotate(args.modes, repeat + algorithm_index):
                print(f"starting algorithm={algorithm} mode={mode} repeat={repeat + 1}")
                if algorithm == "textgrad":
                    train, val = textgrad_data
                    observation = run_textgrad_mode(
                        args,
                        mode=mode,
                        repeat=repeat,
                        train=train,
                        val=val,
                    )
                else:
                    observation = run_openevolve_mode(
                        args,
                        mode=mode,
                        repeat=repeat,
                    )
                observations.append(observation)
                result["observations"] = observations
                result["summary"] = summarize(observations)
                write_json(args.output, result)
                print(
                    f"completed algorithm={algorithm} mode={mode}: "
                    f"quality={observation['baseline_quality']:.4f}->"
                    f"{observation['final_quality']:.4f}, "
                    f"TTQ={observation['time_to_quality_s']}, "
                    f"wall={observation['wall_seconds']:.2f}s"
                )

    result.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "observations": observations,
            "summary": summarize(observations),
            "trace_replay": build_trace_replay(observations),
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    textgrad_calls = (
        2 * args.textgrad_val_size + 3 * args.textgrad_batch_size + 1
    )
    openevolve_calls = args.openevolve_candidates
    runs = args.repeats * len(args.modes)
    if args.dry_run:
        total = runs * sum(
            calls
            for algorithm, calls in (
                ("textgrad", textgrad_calls),
                ("openevolve", openevolve_calls),
            )
            if algorithm in args.algorithms
        )
        print(
            "algorithm parallel/async dry run: "
            f"algorithms={','.join(args.algorithms)}, modes={','.join(args.modes)}, "
            f"repeats={args.repeats}, upper-bound model calls={total} "
            "before provider retries"
        )
        return 0
    confirm_paid_run(args, "TextGrad + OpenEvolve parallel/async benchmark")
    result = run_benchmark(args)
    print(f"completed: observations={len(result['observations'])}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
