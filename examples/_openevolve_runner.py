"""Sandbox-side runner for the OpenEvolve example's candidate programs.

This module is executed by the system Python inside Bubblewrap. It deliberately
uses only the standard library and emits exactly one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import random
import resource
import time


def objective_value(x: float, y: float) -> float:
    return math.sin(x) * math.cos(y) + math.sin(x * y) + (x * x + y * y) / 20.0


class BudgetedObjective:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.calls = 0

    def __call__(self, x: float, y: float) -> float:
        if self.calls >= self.budget:
            raise RuntimeError(f"objective budget exceeded ({self.budget})")
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("objective coordinates must be finite")
        self.calls += 1
        return objective_value(x, y)


def set_resource_limits(cpu_seconds: int, nproc_limit: int) -> None:
    """Apply candidate limits after Bubblewrap starts, before importing its code."""
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, nproc_limit))
    except (ValueError, OSError):
        pass


def load_search(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load candidate module")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    search = getattr(module, "search_algorithm", None)
    if not callable(search):
        raise RuntimeError("candidate must define callable search_algorithm")
    return search


def run_trial(search, seed: int, budget: int, bound: float):
    objective = BudgetedObjective(budget)
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = search(objective, budget, random.Random(seed), (-bound, bound))
        if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
            raise TypeError("search_algorithm must return (x, y) or (x, y, value)")
        x, y = float(result[0]), float(result[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("returned coordinates must be finite")
        if abs(x) > bound or abs(y) > bound:
            raise ValueError(f"returned coordinates must stay in [-{bound}, {bound}]")
        # Never trust a candidate-provided value. The evaluator recomputes it.
        value = objective_value(x, y)
        return {
            "seed": seed,
            "success": True,
            "x": x,
            "y": y,
            "value": value,
            "objective_calls": objective.calls,
            "seconds": time.monotonic() - started,
        }
    except BaseException as exc:  # candidate failures become trial metrics
        return {
            "seed": seed,
            "success": False,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            "objective_calls": objective.calls,
            "seconds": time.monotonic() - started,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bound", type=float, default=5.0)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--nproc-limit", type=int, required=True)
    args = parser.parse_args()

    try:
        set_resource_limits(args.cpu_seconds, args.nproc_limit)
        search = load_search(args.candidate)
        trials = [
            run_trial(search, args.seed + index, args.budget, args.bound)
            for index in range(args.trials)
        ]
        payload = {"ok": True, "trials": trials}
    except BaseException as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}", "trials": []}
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
