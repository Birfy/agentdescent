"""Sandbox-side runner for the ERA numerical-integration task.

Executed inside Bubblewrap or Seatbelt, exactly like ``_era_runner.py`` next
door, and with the same contract: standard library only, and one JSON object on
stdout with the candidate's own prints redirected into a buffer.

Two things differ from the tabular runner, and both come from what a *quadrature*
candidate is being asked to do:

* **The candidate is handed a callable, not a file.** The integrand is rebuilt
  here from the problem's family and parameters through the shared catalogue, so
  the candidate cannot read the specification, only sample the function.
* **Failure is per problem.** A shard is a set of independent integrals; one that
  raises scores zero digits and the rest of the shard still counts. Only a
  candidate that fails to import, or that has no ``integrate``, loses everything
  -- which is the same line ``_era_runner.py`` draws between "the program is
  broken" and "the program is wrong".

The evaluation budget is enforced here rather than trusted to the candidate. It
is the honest half of the accuracy/cost trade the task is about: without it the
winning program is whichever one is allowed to spend the most function
evaluations, and the search would be measuring the time limit.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import resource
import sys
import time
from typing import Any, Callable, Dict, List


#: 2 GiB, matching `_era_runner.py`: a candidate that builds Gauss-Legendre
#: nodes with numpy for a few thousand points needs room, and a limit that kills
#: it would read as "the model wrote a bad program".
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

    Identical in intent to ``_era_runner.set_resource_limits``: returns the names
    of the limits this platform refused rather than pretending it has them.
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


class BudgetExceeded(RuntimeError):
    """Raised inside the integrand once a problem has spent its allowance."""


class Deadline(RuntimeError):
    """Raised inside the integrand once a problem has spent its wall-clock."""


class Counted:
    """The integrand, plus the two meters the task is scored against.

    Checked on evaluation rather than on a timer thread: a quadrature rule that
    is not calling the integrand is not making progress, and a candidate stuck
    in its own arithmetic is caught by ``RLIMIT_CPU`` instead.
    """

    def __init__(self, function: Callable[[float], float], budget: int, deadline: float) -> None:
        self._function = function
        self._budget = budget
        self._deadline = deadline
        self.calls = 0

    def __call__(self, x: float) -> float:
        self.calls += 1
        if self.calls > self._budget:
            raise BudgetExceeded(
                f"evaluation budget of {self._budget} calls exhausted")
        if not self.calls % 512 and time.monotonic() > self._deadline:
            raise Deadline("per-problem time limit reached")
        return self._function(x)


def load_entrypoint(path: str):
    module = _load_module(path, "candidate")
    entrypoint = getattr(module, "integrate", None)
    if not callable(entrypoint):
        raise RuntimeError("candidate must define callable integrate(f, a, b)")
    return entrypoint


def solve(entrypoint, catalogue, problems: List[Dict[str, Any]], *,
          budget: int, seconds: float) -> List[Dict[str, Any]]:
    """Run the candidate once per problem, and never let it take the shard down."""
    results: List[Dict[str, Any]] = []
    for payload in problems:
        problem = catalogue.Problem.from_dict(payload)
        counted = Counted(catalogue.integrand(problem), budget,
                          time.monotonic() + seconds)
        started = time.monotonic()
        estimate: Any = None
        error = ""
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                estimate = entrypoint(counted, problem.a, problem.b)
            value = float(estimate)
            if not math.isfinite(value):
                estimate, error = None, f"non-finite estimate: {value!r}"
            else:
                estimate = value
        except BaseException as exc:  # a failed problem is a scored zero
            estimate = None
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        results.append({
            "estimate": estimate,
            "calls": counted.calls,
            "seconds": time.monotonic() - started,
            "error": error,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--problems", required=True,
                        help="JSON file holding the shard's problem specifications")
    parser.add_argument("--eval-budget", type=int, required=True)
    parser.add_argument("--problem-seconds", type=float, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--nproc-limit", type=int, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    try:
        unavailable = set_resource_limits(args.cpu_seconds, args.nproc_limit)
        catalogue = _load_module(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "_era_integrals.py"),
            "era_integrals")
        with open(args.problems, "r", encoding="utf-8") as handle:
            problems = json.load(handle)
        entrypoint = load_entrypoint(args.candidate)
        payload = {
            "ok": True,
            "results": solve(entrypoint, catalogue, problems,
                             budget=args.eval_budget, seconds=args.problem_seconds),
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
