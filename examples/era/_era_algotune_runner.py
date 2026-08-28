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
import math
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


def _numeric_leaves(value: Any, out: List[float], budget: List[int]) -> None:
    """Flatten whatever a task returns into floats, up to a budget."""
    if budget[0] <= 0:
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _numeric_leaves(value[key], out, budget)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _numeric_leaves(item, out, budget)
        return
    try:
        import numpy as np
        array = np.asarray(value, dtype=float).ravel()
    except Exception:
        return
    take = array[: max(0, budget[0])]
    budget[0] -= take.size
    out.extend(float(x) for x in take)


def _structure_of(value: Any, depth: int = 0) -> str:
    """A compact description of the container an answer arrived in.

    `_numeric_leaves` flattens, so on its own it cannot tell a wrong answer from
    a right one in the wrong box -- and several tasks reject on exactly that.
    `affine_transform_2d`'s `is_solution` checks `proposed.shape != image.shape`
    before it compares any value, so a solver returning the correct 20000 pixels
    as a flat list is rejected with every number identical. Measured on that
    task, 13 of 29 rejections in one run were of that kind, and each was reported
    to the model as "off by 0.000e+00, 0x the tolerance" -- a message that says
    the answer is right.

    Only the first element of a sequence is described, so this stays O(depth)
    rather than O(size); a ragged container is described by its head, which is
    enough to see that two structures differ.
    """
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        inner = ", ".join(f"{key}: {_structure_of(value[key], depth + 1)}"
                          for key in sorted(value))
        return "{" + inner + "}"
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return f"array{tuple(value.shape)}"
    except BaseException:
        pass
    if isinstance(value, (list, tuple)):
        name = "list" if isinstance(value, list) else "tuple"
        if not value:
            return f"{name}[0]"
        return f"{name}[{len(value)} x {_structure_of(value[0], depth + 1)}]"
    return type(value).__name__


def _accuracy_note(reference: Any, candidate: Any, cap: int = 20_000) -> str:
    """How far off the rejected answer was, when that can be said at all.

    "is_solution rejected the output" is true and nearly useless: it cannot tell
    a program that is structurally wrong from one whose fixed-step integrator is
    a factor of three short on tolerance, and those need opposite next moves.
    The reference output is already in hand -- it was produced while timing the
    baseline and then thrown away -- so the distance is free to compute.

    Deliberately generic: flatten both sides to numbers and report the largest
    gap. A task whose answer is not numeric, or whose shapes disagree, gets the
    shape mismatch instead, which is itself the more useful message. Every
    failure path returns "" rather than raising: this runs inside the evaluator,
    and a diagnostic that can break an evaluation is worse than none.

    The gap is measured the way ``numpy.allclose`` measures it -- against
    ``atol + rtol * |reference|`` -- and *not* as a bare relative error. A bare
    one divides by the reference element, so wherever the answer is legitimately
    near zero it reports a number with no meaning. On ``kalman_filter`` it said
    "largest relative difference 5.033e+23, at element 2095 of 3197: reference
    1.19584887991e-23, yours 6.01848096735": the reference there is zero to
    within floating point, the candidate produced 6, and the honest way to say
    that is "off by 6", not "off by twenty-three orders of magnitude". Solution
    vectors full of near-zero entries -- a Kalman noise estimate, a sparse
    multiplier, anything with an active set -- are exactly where this fires, and
    the number it produced both overstated the error and hid where it was.

    The tolerances are ``numpy.allclose``'s defaults rather than the task's own,
    which this cannot see from here; the point is a sane scale, not a verdict.
    The verdict already came from ``is_solution``.
    """
    try:
        left: List[float] = []
        right: List[float] = []
        _numeric_leaves(reference, left, [cap])
        _numeric_leaves(candidate, right, [cap])
        shape_note = ""
        try:
            here, there = _structure_of(reference), _structure_of(candidate)
            if here != there:
                shape_note = (f" (the reference came back as {here}, yours as "
                              f"{there} -- the structure differs)")
        except BaseException:
            shape_note = ""
        if shape_note:
            return shape_note
        if not left or not right:
            return ""
        if len(left) != len(right):
            return (f" (the reference has {len(left)} numbers, yours has "
                    f"{len(right)} -- the shape or structure differs)")
        import numpy as np
        a = np.asarray(left)
        b = np.asarray(right)
        gap = np.abs(a - b)
        if not np.all(np.isfinite(b)):
            return " (your answer holds a non-finite value the reference does not)"
        # allclose's own yardstick: how many times the tolerance the gap is.
        tolerance = 1e-8 + 1e-5 * np.abs(a)
        worst = int(np.argmax(gap / tolerance))
        over = float(gap[worst] / tolerance[worst])
        if over <= 1.0:
            # Every number agrees, and `is_solution` still said no. Saying "0x
            # the tolerance" here reads as "your answer is right", which is the
            # one thing it is not; the model needs to look at dtype, ordering,
            # a key it did not set, or a check the task makes beyond closeness.
            return (" (every number matches the reference to within allclose, so"
                    " is_solution rejected it for something other than accuracy"
                    " -- check the dtype, the ordering, the keys you return, and"
                    " any condition the task checks besides closeness)")
        absolute = float(gap[worst])
        relative = (absolute / abs(float(a[worst]))
                    if a[worst] != 0.0 else float("inf"))
        note = (f" (worst element {worst} of {len(a)}: reference "
                f"{a[worst]:.12g}, yours {b[worst]:.12g} -- off by {absolute:.3e}")
        if math.isfinite(relative):
            note += f", a relative {relative:.3e}"
        return note + f", {over:.3g}x the tolerance allclose would allow)"
    except BaseException:
        return ""


def measure(support, task: Any, entrypoint: Callable[[Any], Any],
            problem: Any, *, warmup_problem: Any, repeats: int,
            slow_factor: float, deadline: float) -> Dict[str, Any]:
    """Time the reference, then the candidate, then check the candidate's answer.

    A candidate failure is *this problem's* failure, not the program's: it comes
    back as ``valid: False`` with the exception on it, and the rest of the shard
    still runs. Only a program that could not be imported, or that has no
    ``solve``, loses everything -- the same line the sibling runners draw
    between "the program is broken" and "the program is wrong".

    ``warmup_problem`` is a *different instance* of the same task, which is
    upstream's rule: ``warmup_idx = (idx - 1) % problem_count`` in
    ``AlgoTuner/utils/evaluator/main.py`` picks the previous record in the
    dataset, and its worker logs an error if the two ever coincide. Warming on a
    copy of the timed problem, which is what this did before, hands a free hit to
    any solver that memoises on the input -- and a solver that returns a cached
    answer in nanoseconds is indistinguishable from one that is genuinely fast.
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
        task.solve(copy.deepcopy(warmup_problem))
        baseline, reference, _runs = support.best_seconds(
            task.solve, problem, repeats=repeats, deadline=deadline)
    row["baseline_ms"] = baseline * 1000.0

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            # The candidate gets the same free warm-up the reference got two
            # statements ago, and it is not optional politeness.
            #
            # A `@numba.njit` function compiles on its first call. Without this
            # the compile landed inside the first *timed* call, which is also
            # the call the slow-check reads -- so an identical program scored
            # 0.052x when it compiled inside `solve` and 0.947x when it compiled
            # at import, an 18x swing that measures where the author put a line
            # rather than how fast the program is. Worse, the search learns from
            # it: a few of those and it concludes compiling makes things twenty
            # times slower, and steers away from the one lever that wins here.
            #
            # AlgoTune's own rule is that compilation is not charged
            # ("Compilation time of your init function will not count towards
            # your function's runtime"). Honouring it must not depend on the
            # model knowing to force compilation at import time.
            #
            # On the *warm-up* problem, for the reason above it: a candidate
            # warmed on the problem it is about to be timed on can answer the
            # timed call out of a dictionary.
            entrypoint(copy.deepcopy(warmup_problem))
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
            row["error"] = ("is_solution rejected the output"
                            + _accuracy_note(reference, output))
    except BaseException as exc:
        row["valid"] = False
        row["error"] = f"is_solution raised {type(exc).__name__}: {str(exc)[:200]}"
    return row


def profile(entrypoint: Callable[[Any], Any], problem: Any, *,
            top: int = 25) -> str:
    """Line-level cost of one call, in the shape AlgoTune's `profile` returns.

    AlgoTuner hands its model a `line_profiler` table on demand and this port
    had no equivalent, so a mutation was rewriting code it could not see the
    cost of. Same tool, same 25-line cut.

    Run **after** timing and never inside it: `line_profiler` traces every line,
    which inflates the call several-fold. It would land on whichever program was
    profiled and silently move the ratio the whole benchmark is.

    Returns "" on any failure. A profiler that can break an evaluation is worth
    less than no profiler.
    """
    try:
        from line_profiler import LineProfiler
    except Exception:
        return ""
    try:
        profiler = LineProfiler()
        wrapped = profiler(entrypoint)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            wrapped(copy.deepcopy(problem))
        buffer = io.StringIO()
        profiler.print_stats(buffer, output_unit=1e-3)
        rows = []
        header = []
        for line in buffer.getvalue().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Line #") or stripped.startswith("="):
                header.append(line)
                continue
            parts = stripped.split(None, 5)
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    rows.append((float(parts[2]), line))
                except ValueError:
                    continue
            elif not rows:
                header.append(line)
        if not rows:
            return ""
        keep = sorted(rows, key=lambda row: -row[0])[:top]
        kept = {id(row[1]) for row in keep}
        body = [line for _cost, line in rows if id(line) in kept]
        body.sort(key=lambda line: int(line.strip().split(None, 1)[0]))
        return "\n".join(header[-2:] + body)
    except BaseException:
        return ""


def evaluate(support, task: Any, entrypoint: Callable[[Any], Any],
             spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per problem in the shard, in the order the seeds were drawn.

    The shard's problems are generated up front so each can be timed against the
    *previous* one as its warm-up, which is upstream's ``warmup_idx = (idx - 1) %
    problem_count``. Generating inside the loop and warming on a copy of the
    timed problem, as this did before, is both a deviation and a hole: a solver
    that caches on its input gets the timed call for free.
    """
    results: List[Dict[str, Any]] = []
    n = int(spec["n"])
    repeats = int(spec.get("repeats") or 3)
    slow_factor = float(spec.get("slow_factor") or 10.0)
    seconds = float(spec.get("problem_seconds") or 60.0)
    want_profile = bool(spec.get("profile"))
    seeds = [int(seed) for seed in spec["seeds"]]
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        problems = [task.generate_problem(n, random_seed=seed) for seed in seeds]
    for index, seed in enumerate(seeds):
        started = time.monotonic()
        problem = problems[index]
        # Upstream's rule exactly. With a single-problem shard there is no
        # previous instance to reach for, so it degenerates to the timed one --
        # upstream's own fallback when `warmup_problem_instance is None`.
        warmup_problem = problems[(index - 1) % len(problems)]
        row = measure(support, task, entrypoint, problem,
                      warmup_problem=warmup_problem, repeats=repeats,
                      slow_factor=slow_factor,
                      deadline=time.monotonic() + seconds)
        row["seed"] = int(seed)
        row["seconds"] = time.monotonic() - started
        # One profile per shard, on the first problem, and only when asked: it
        # costs an extra traced call and every row of it is the same code.
        if want_profile and not results and row["candidate_ms"] is not None:
            row["profile"] = profile(entrypoint, problem)
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
