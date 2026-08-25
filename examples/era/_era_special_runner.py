"""Sandbox-side runner for the special-function precision tasks.

The same contract every ERA runner keeps -- standard library only, one JSON
object on stdout, the candidate's own prints redirected into a buffer, and a
failure scored per point rather than per program.

Generalised from :mod:`examples.era._era_hyp2f1_runner` in exactly one respect:
the entry point's name and its parameter list arrive on the command line
instead of being spelled ``hyp2f1(a, b, c, z)`` in the source. Everything else
is deliberately identical, because the two runners would otherwise be two
places to fix the same bug.

The reference values are *not* in the file this reads. A candidate that opened
the problem file would find the parameters it was already handed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import resource
import sys
import time
from typing import Any, Dict, List, Sequence


#: 2 GiB, matching the sibling runners: a candidate that precomputes a table of
#: Chebyshev or Gauss nodes with numpy needs room, and a limit that killed it
#: would read as "the model wrote a bad program".
ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024


def set_resource_limits(cpu_seconds: int, nproc_limit: int) -> List[str]:
    """Apply candidate limits inside the sandbox, before importing its code."""
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


def load_entrypoint(path: str, name: str, arity: int):
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load candidate module")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `@dataclass` resolves its own module through
    # `sys.modules[cls.__module__]`, and without this a candidate that declares
    # one dies with "'NoneType' object has no attribute '__dict__'".
    sys.modules["candidate"] = module
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    entrypoint = getattr(module, name, None)
    if not callable(entrypoint):
        raise RuntimeError(f"candidate must define callable {name} taking "
                           f"{arity} arguments")
    return entrypoint


def evaluate(entrypoint, points: List[Dict[str, Any]], params: Sequence[str],
             *, deadline: float) -> List[Dict[str, Any]]:
    """One call per point, and never let one point take the shard down.

    The estimate is returned as ``repr`` of a float rather than as a JSON
    number: the scorer compares it against a 25-digit decimal reference, and a
    value that had been through JSON's float formatting would be compared after
    a rounding this runner introduced.
    """
    results: List[Dict[str, Any]] = []
    for point in points:
        arguments = [float(point[name]) for name in params]
        started = time.monotonic()
        estimate = None
        error = ""
        if time.monotonic() > deadline:
            error = "TimeLimit: the shard's wall-clock budget was spent"
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    value = entrypoint(*arguments)
                value = float(value)
                if math.isfinite(value):
                    estimate = repr(value)
                else:
                    error = f"non-finite result: {value!r}"
            except BaseException as exc:  # a failed point is a scored zero
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
        results.append({
            "estimate": estimate,
            "seconds": time.monotonic() - started,
            "error": error,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--points", required=True,
                        help="JSON file holding the shard's points")
    parser.add_argument("--entrypoint", required=True,
                        help="name of the function the candidate must define")
    parser.add_argument("--params", required=True,
                        help="comma-separated parameter names, in call order")
    parser.add_argument("--shard-seconds", type=float, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--nproc-limit", type=int, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    params = [name for name in args.params.split(",") if name]
    try:
        unavailable = set_resource_limits(args.cpu_seconds, args.nproc_limit)
        with open(args.points, "r", encoding="utf-8") as handle:
            points = json.load(handle)
        entrypoint = load_entrypoint(args.candidate, args.entrypoint, len(params))
        # The clock starts after the import: a candidate that builds a node
        # table at import time is paying for it under RLIMIT_CPU already, and
        # charging it to the first point as well would score the same work twice.
        payload = {
            "ok": True,
            "results": evaluate(entrypoint, points, params,
                                deadline=time.monotonic() + args.shard_seconds),
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
