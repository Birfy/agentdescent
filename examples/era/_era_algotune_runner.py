"""Sandbox-side runner for the ERA port's AlgoTune task.

Executed inside Bubblewrap or Seatbelt, exactly like ``_era_runner.py`` and
``_era_integration_runner.py`` next door, and with the same contract: one JSON
object on stdout, the candidate's own prints redirected into a buffer.

What is different here is that the *reference* runs inside the sandbox too. The
metric is a ratio of two timings, so the only honest place to take them is the
same process, moments apart, on the same problem, under the same CPU limit and
the same one-thread BLAS policy. A baseline measured once on the host and reused
would fold every scheduling artefact of the whole run into the score -- and the
score would then move when the machine got busy rather than when the program got
faster.

The order is fixed and it matters: **the reference is timed first**. A candidate
that mutated the problem, warmed a cache or fragmented the allocator could
otherwise change the number it is being compared against, and the comparison
would flatter it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import resource
import sys
import time
from typing import Any, Callable, Dict, List


def _load_support(name: str = "algotune_tasks"):
    """Load ``_algotune_tasks.py`` from beside this file, by path.

    ``python -I`` and a clearenv Bubblewrap profile mean there is no package on
    ``sys.path`` to import ``examples.era`` from -- the repository is mounted
    read-only at ``/`` but nothing has told the interpreter about it. The
    integrals runner loads its integrand catalogue the same way.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_algotune_tasks.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the AlgoTune support module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_resource_limits(cpu_seconds: int, nproc_limit: int,
                        address_space_mb: int) -> List[str]:
    """Apply candidate limits inside the sandbox, before importing its code.

    Identical in intent to the sibling runners': the names of the limits this
    platform refused come back rather than being pretended to hold.
    """
    address_space = address_space_mb * 1024 * 1024
    unavailable: List[str] = []
    for name, value in (
        ("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 1)),
        ("RLIMIT_AS", (address_space, address_space)),
        ("RLIMIT_FSIZE", (16 * 1024 * 1024, 16 * 1024 * 1024)),
        ("RLIMIT_NOFILE", (256, 256)),
        ("RLIMIT_NPROC", (nproc_limit, nproc_limit)),
    ):
        limit = getattr(resource, name, None)
        if limit is None:
            unavailable.append(name)
            continue
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            unavailable.append(name)
    return unavailable


def load_entrypoint(support, path: str) -> Callable[[Any], Any]:
    """Import the candidate and return its ``solve``.

    Module-level work -- a precomputed table, an FFT plan -- is *inside* the
    quiet block and outside every timed region, so a candidate pays for it once
    and is not credited with it per problem. That is the same deal a real
    library gets at import time.
    """
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        module = support.load_module(path, "candidate")
    entrypoint = getattr(module, "solve", None)
    if not callable(entrypoint):
        raise RuntimeError("candidate must define a callable solve(problem)")
    return entrypoint


def measure(support, task: Any, entrypoint: Callable[[Any], Any],
            problem: Any, *, repeats: int, slow_factor: float,
            deadline: float) -> Dict[str, Any]:
    """Time the reference, then the candidate, then check the candidate's answer.

    A candidate failure is *this problem's* failure, not the program's: it comes
    back as ``valid: False`` with the exception on it, and the rest of the shard
    still runs. Only a program that could not be imported, or that has no
    ``solve``, loses everything -- the same line the sibling runners draw
    between "the program is broken" and "the program is wrong".
    """
    row: Dict[str, Any] = {
        "baseline_ms": None, "candidate_ms": None, "valid": False,
        "runs": 0, "error": "",
    }
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        # Warm-up, discarded: the first call of anything numerical pays for
        # lazily imported submodules, a BLAS handshake and a cold allocator, and
        # charging that to whichever program happens to run first would be a
        # coin-flip worth several x on a millisecond-scale task.
        task.solve(copy.deepcopy(problem))
        baseline, _reference, _runs = support.best_seconds(
            task.solve, problem, repeats=repeats, deadline=deadline)
    row["baseline_ms"] = baseline * 1000.0

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            first_started = time.perf_counter()
            output = entrypoint(copy.deepcopy(problem))
            first = time.perf_counter() - first_started
            # Slower than `slow_factor` x the reference on its warm-up: measured
            # once and reported. Repeating it would spend the shard's whole
            # wall-clock proving a number we already have, and the alternative --
            # letting it overrun the timeout -- would record a correct-but-slow
            # program as one that failed to run, which is a different claim.
            if first > slow_factor * max(baseline, 1e-6):
                candidate, runs = first, 1
            else:
                candidate, output, runs = support.best_seconds(
                    entrypoint, problem, repeats=repeats, deadline=deadline)
        row["candidate_ms"] = candidate * 1000.0
        row["runs"] = runs
    except BaseException as exc:  # a failed problem is a scored failure
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return row

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            # `is_solution` gets its own copy: several AlgoTune checkers re-solve
            # the problem or normalise it in place, and a checker that mutated
            # the instance would make the *next* problem's timing a different
            # measurement.
            row["valid"] = bool(task.is_solution(copy.deepcopy(problem), output))
        if not row["valid"]:
            row["error"] = "is_solution rejected the output"
    except BaseException as exc:
        row["valid"] = False
        row["error"] = f"is_solution raised {type(exc).__name__}: {str(exc)[:200]}"
    return row


def evaluate(support, task: Any, entrypoint: Callable[[Any], Any],
             spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per problem in the shard, in the order the seeds were drawn."""
    results: List[Dict[str, Any]] = []
    n = int(spec["n"])
    repeats = int(spec.get("repeats") or 3)
    slow_factor = float(spec.get("slow_factor") or 20.0)
    seconds = float(spec.get("problem_seconds") or 60.0)
    for seed in spec["seeds"]:
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            problem = task.generate_problem(n, random_seed=int(seed))
        row = measure(support, task, entrypoint, problem, repeats=repeats,
                      slow_factor=slow_factor,
                      deadline=time.monotonic() + seconds)
        row["seed"] = int(seed)
        row["seconds"] = time.monotonic() - started
        results.append(row)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--task-source", required=True,
                        help="the AlgoTune task file this shard is drawn from")
    parser.add_argument("--spec", required=True,
                        help="JSON holding the task name, n, seeds and timing knobs")
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--nproc-limit", type=int, required=True)
    parser.add_argument("--address-space-mb", type=int, default=4096)
    args = parser.parse_args()

    started = time.monotonic()
    try:
        unavailable = set_resource_limits(
            args.cpu_seconds, args.nproc_limit, args.address_space_mb)
        support = _load_support()
        with open(args.spec, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            task = support.load_task(args.task_source, spec["task"])
        entrypoint = load_entrypoint(support, args.candidate)
        payload: Dict[str, Any] = {
            "ok": True,
            "results": evaluate(support, task, entrypoint, spec),
            "seconds": time.monotonic() - started,
            "limits_unavailable": unavailable,
        }
    except BaseException as exc:  # an unusable program, as distinct from a wrong one
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "seconds": time.monotonic() - started,
        }
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
