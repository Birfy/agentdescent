"""Sandbox-side runner for the ERA LLM-SRBench task.

Executed inside Bubblewrap or Seatbelt, exactly like ``_era_runner.py`` and
``_era_integration_runner.py`` next door, with the same contract: one JSON
object on stdout, the candidate's own prints redirected into a buffer, and one
failed problem never taking the shard down.

Two things differ from the sibling runners, and both come from what a
*symbolic-regression* candidate is being asked to do:

* **It imports numpy.** The other two are standard library only. This one hands
  the candidate float64 arrays and scores an expression against held-out ones,
  and re-implementing that on lists would be a second, slower copy of the thing
  every candidate here already depends on.
* **The held-out samples are opened after ``discover`` returns.** The candidate
  is given the training arrays and nothing else; the test and OOD arrays are
  read from the shard's ``.npz`` once its equation is in hand, and they are only
  ever touched by ``_era_srbench_expr``, which interprets a validated AST rather
  than running anything the candidate wrote. There is no point at which
  candidate code and test data are both live.

The per-problem wall-clock is enforced here rather than trusted to the
candidate, with ``SIGALRM``: a method that ignores its budget on one problem
would otherwise spend the whole shard's timeout there and every remaining
problem would score zero for its neighbour's fault.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import resource
import signal
import sys
import time
from typing import Any, Dict, List

import numpy as np


#: 2 GiB, matching the sibling runners: an LSR-Transform problem hands the
#: candidate 80 000 samples, and a design matrix over a few hundred basis
#: functions is an ordinary thing for it to build.
ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, not after: `@dataclass` resolves its own
    # module through `sys.modules[cls.__module__]`, and a module that is not
    # there yet fails with "'NoneType' object has no attribute '__dict__'" --
    # which reads as a broken candidate rather than a broken loader.
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def set_resource_limits(cpu_seconds: int, nproc_limit: int) -> List[str]:
    """Apply candidate limits inside the sandbox, before importing its code.

    Returns the names of the limits this platform refused rather than pretending
    it has them, as the sibling runners do.
    """
    unavailable: List[str] = []
    for name, value in (
        ("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 1)),
        ("RLIMIT_AS", (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES)),
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


class Deadline(RuntimeError):
    """Raised inside the candidate once a problem has spent its wall-clock."""


@contextlib.contextmanager
def time_limit(seconds: float):
    """Interrupt the candidate when its problem runs out of time.

    ``SIGALRM`` rather than a watchdog thread: the runner is single-threaded and
    the candidate is ordinary Python, so the handler runs at the next bytecode
    boundary and the exception surfaces inside ``discover``, where the
    per-problem ``except`` below turns it into a scored zero. A platform without
    ``SIGALRM`` falls back to the shard's own CPU limit -- and says so, rather
    than reporting a budget it did not enforce.
    """
    if not hasattr(signal, "setitimer") or seconds <= 0:
        yield False
        return

    def _fire(_signum, _frame):
        raise Deadline(f"per-problem time limit of {seconds:.1f}s reached")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def load_entrypoint(path: str):
    module = _load_module(path, "candidate")
    entrypoint = getattr(module, "discover", None)
    if not callable(entrypoint):
        raise RuntimeError("candidate must define callable discover(x, y, spec)")
    return entrypoint


def solve(entrypoint, expr_module, samples, problems: List[Dict[str, Any]], *,
          seconds: float) -> List[Dict[str, Any]]:
    """Run the candidate once per problem, and never let it take the shard down."""
    results: List[Dict[str, Any]] = []
    for position, problem in enumerate(problems):
        variables = list(problem["input_vars"])
        train_x = np.asarray(samples[f"p{position}_train_x"], dtype=np.float64)
        train_y = np.asarray(samples[f"p{position}_train_y"], dtype=np.float64)
        def evaluate(expression, data, _variables=tuple(variables)):
            """The scorer's own parser, handed to the candidate.

            A symbolic-regression method has to be able to evaluate the forms it
            is proposing, and the AST gate refuses ``eval``/``exec`` -- so
            without this the only way to write one would be to build closures
            that mirror the grammar and hope the mirror is exact. Handing over
            the real evaluator removes that whole class of near-miss: what the
            candidate scores a form with is the same code, with the same
            grammar, that will score its final answer. Raises ``ExpressionError``
            on anything that is not an equation, which is what the candidate
            needs to hear.
            """
            return expr_module.evaluate_expression(expression, list(_variables), data)

        spec = {
            "input_vars": list(variables),
            "output_var": problem["output_var"],
            "description": problem["description"],
            "seconds": seconds,
            "samples": int(train_x.shape[0]),
            "evaluate": evaluate,
        }
        started = time.monotonic()
        equation: Any = None
        error = ""
        try:
            with time_limit(seconds):
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    # Copies, so a candidate that writes into what it was handed
                    # cannot change what the next problem -- or the scoring --
                    # sees.
                    equation = entrypoint(train_x.copy(), train_y.copy(), dict(spec))
        except BaseException as exc:  # a failed problem is a scored zero
            equation = None
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        elapsed = time.monotonic() - started

        row: Dict[str, Any] = {
            "problem_id": problem["problem_id"],
            "equation": equation if isinstance(equation, str) else None,
            "seconds": elapsed,
            "error": error,
            "id": None,
            "ood": None,
        }
        if isinstance(equation, str) and not error:
            try:
                row["id"] = _score(expr_module, equation, variables, samples,
                                   position, "test")
                if f"p{position}_ood_x" in samples:
                    row["ood"] = _score(expr_module, equation, variables, samples,
                                        position, "ood")
            except BaseException as exc:
                row["id"] = None
                row["ood"] = None
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        elif not error:
            row["error"] = (
                f"discover returned {type(equation).__name__}, expected str")
        results.append(row)
    return results


def _score(expr_module, equation: str, variables: List[str], samples,
           position: int, split: str) -> Dict[str, Any]:
    x = np.asarray(samples[f"p{position}_{split}_x"], dtype=np.float64)
    y = np.asarray(samples[f"p{position}_{split}_y"], dtype=np.float64)
    prediction = expr_module.evaluate_expression(equation, variables, x)
    return expr_module.score_predictions(prediction, y)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--samples", required=True,
                        help="npz file holding the shard's sample arrays")
    parser.add_argument("--problems", required=True,
                        help="JSON file holding the shard's problem metadata")
    parser.add_argument("--problem-seconds", type=float, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--nproc-limit", type=int, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    try:
        unavailable = set_resource_limits(args.cpu_seconds, args.nproc_limit)
        expr_module = _load_module(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_era_srbench_expr.py"),
            "era_srbench_expr")
        with open(args.problems, "r", encoding="utf-8") as handle:
            problems = json.load(handle)
        samples = np.load(args.samples)
        entrypoint = load_entrypoint(args.candidate)
        if not hasattr(signal, "setitimer"):
            unavailable = list(unavailable) + ["SIGALRM"]
        payload = {
            "ok": True,
            "results": solve(entrypoint, expr_module, samples, problems,
                             seconds=args.problem_seconds),
            "seconds": time.monotonic() - started,
            "limits_unavailable": unavailable,
        }
    except BaseException as exc:  # an unusable program, as distinct from a wrong one
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "seconds": time.monotonic() - started,
        }
    print(json.dumps(payload, separators=(",", ":"), allow_nan=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
