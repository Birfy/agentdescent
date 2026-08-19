"""Suite, sandboxed evaluator, and prompt for the Gauss hypergeometric task.

The runnable example is :mod:`examples.era.era_hypergeometric`; the stress set
and its references are the committed file ``data/hyp2f1_stress.json``, produced
once by ``tools/gen_hyp2f1_stress.py``.

What the task is
----------------
Evaluate ``2F1(a, b; c; z)`` in double precision, for real parameters drawn over
a wide range. It is a genuinely hard piece of scientific software and known to
be: the standard survey of the area (Pearson, Olver & Porter, *Numerical methods
for the computation of the confluent and Gauss hypergeometric functions*,
Numerical Algorithms 74:821-866, 2017) exists precisely because **no single
method is reliable across the parameter space** -- the Taylor series diverges,
each transformation has its own bad region, the recurrences are unstable in one
direction, and the useful ones cancel.

Why this is worth measuring
---------------------------
The baseline is not a strawman written for this benchmark. It is
``scipy.special.hyp2f1`` -- the function every scientist already calls, Cephes
underneath, in production for decades. On the declared distribution it loses
more than six digits on roughly a third of points, and the failures are
*algorithmic* rather than a float64 wall: applying a textbook Pfaff
transformation, decided from ``(a, b, c, z)`` alone with no access to the
answer, already recovers a large share of them.

So the two numbers a run reports mean something without a leaderboard: the
reference comes from an arbitrary-precision library that shares no code with
SciPy or with any candidate, and the thing being improved on is the state of
the practice rather than a placeholder.

What keeps it honest
--------------------
* The reference is **mpmath at 30 and 60 digits, kept only where the two
  agree to 25** -- see the generator. Nothing in the sandbox can reach it.
* The parameter distribution was **declared before anything was measured** and
  is recorded in the data file next to the values.
* Arbitrary-precision arithmetic is **off the allowlist** (`decimal`,
  `fractions`, `mpmath`). The deliverable is a float64 routine comparable to
  SciPy's; a candidate that reimplemented mpmath would be answering a different
  question, and would be scored against its own strategy rather than against
  the practice.
* The last four shards are never shown to the search.
"""

from __future__ import annotations

import decimal
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from examples.era._era_support import sandbox_wrapper, validate_source


RUNNER = Path(__file__).with_name("_era_hyp2f1_runner.py")
SUITE_PATH = Path(__file__).with_name("data") / "hyp2f1_stress.json"

#: The reference has 25 digits and float64 carries about 16, so a cap of 12 is
#: below both: the score can neither measure the reference's truncation nor be
#: farmed by chasing the last ulp on points that are already right. What it does
#: measure is how much of the parameter space a program gets *essentially*
#: right, which is where the whole difference between implementations lives.
DIGIT_CAP = 12.0

#: A point at or above this many correct digits is reported as solved.
SOLVED_DIGITS = 10.0

#: No `decimal`, no `fractions`, no `mpmath`: see the module docstring.
ALLOWED_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "numpy",
    "operator",
    "scipy",
    "statistics",
    "typing",
    "warnings",
}

#: The baseline, and the reason the task has a meaning: this is what a working
#: scientist writes. `hyp2f1` is aliased on import because the candidate's own
#: entry point has to take that name.
INITIAL_PROGRAM = '''"""Baseline: scipy.special.hyp2f1, the implementation everyone already calls."""
from scipy.special import hyp2f1 as _scipy_hyp2f1


def hyp2f1(a, b, c, z):
    return float(_scipy_hyp2f1(a, b, c, z))
'''

#: Wall-clock for one shard of points, inside `--candidate-timeout`. Generous:
#: a special-function call is microseconds -- the baseline does a 250-point
#: shard in 0.3 s and the best evolved program so far in 0.36 s -- and the point
#: of the limit is to stop a candidate that has decided to sum a million terms
#: per point, not to make speed part of the score.
SHARD_SECONDS = 45.0


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """The stress points, and the references that never leave this process."""

    shard_points: Tuple[Tuple[Dict[str, str], ...], ...]
    shard_truth: Tuple[Tuple[str, ...], ...]
    scoring_shards: int
    test_shards: int
    metadata: Dict[str, Any]

    def size(self, shard: int) -> int:
        return len(self.shard_points[shard])

    def test_range(self) -> Tuple[int, ...]:
        return tuple(range(self.scoring_shards,
                           self.scoring_shards + self.test_shards))


def load_suite(*, shards: int = 8, test_shards: int = 4,
               path: Path = SUITE_PATH) -> Suite:
    """Read the committed stress set. No draw, no network, no mpmath.

    The suite is a file rather than a seeded draw because its references cost
    arbitrary-precision arithmetic to produce and are the part a reader has to
    be able to audit. Regenerating it on every run would make the benchmark
    depend on which mpmath happened to be installed.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = len(payload["shards"])
    if shards + test_shards > available:
        raise ValueError(
            f"the stress set has {available} shards; {shards} + {test_shards} "
            f"were asked for")
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    points: List[Tuple[Dict[str, str], ...]] = []
    truth: List[Tuple[str, ...]] = []
    for shard in payload["shards"][: shards + test_shards]:
        points.append(tuple({key: row[key] for key in ("a", "b", "c", "z")}
                            for row in shard))
        truth.append(tuple(row["value"] for row in shard))
    metadata = {key: payload[key] for key in ("task", "reference", "distribution", "seed")
                if key in payload}
    return Suite(tuple(points), tuple(truth), shards, test_shards, metadata)


def suite_preview(suite: Suite) -> str:
    """What the model is told about the points: the distribution, not the values."""
    distribution = suite.metadata.get("distribution", {})
    lines = [
        f"Each problem set holds {suite.size(0)} points (a, b, c, z), drawn "
        "independently from a fixed distribution:",
        f"  a, b, c ~ {distribution.get('a', 'U(-30, 30)')}",
        f"  z       ~ {distribution.get('z', 'U(-40, 0.999)')}",
        "Points where c is at or near a non-positive integer, and points whose "
        "value is below 1e-250 in magnitude, were rejected when the set was "
        "drawn -- so every point you are given has a finite, well-defined value.",
        "The parameters are real and can be large, negative, or nearly equal to "
        "each other; z reaches far to the left of the unit disc.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def digits_of(estimate: Any, truth: str) -> float:
    """Correct significant digits, computed against the 25-digit reference.

    In :mod:`decimal` rather than in floats: the reference carries more digits
    than a float64 can hold, and subtracting it from the candidate's value in
    double precision would round away the very quantity being measured on the
    points that are nearly right.
    """
    if estimate is None:
        return 0.0
    with decimal.localcontext() as context:
        context.prec = 40
        try:
            value = decimal.Decimal(str(estimate))
            reference = decimal.Decimal(truth)
        except (decimal.InvalidOperation, ValueError, TypeError):
            return 0.0
        if not value.is_finite() or not reference.is_finite():
            return 0.0
        if reference == 0:  # pragma: no cover - the generator rejects these
            return 0.0
        error = abs(value - reference) / abs(reference)
        if error <= 0:
            return DIGIT_CAP
        return max(0.0, min(DIGIT_CAP, float(-error.log10())))


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "mean_digits": None,
        # `-inf` is upstream's failure sentinel; the node is appended anyway.
        "score": -math.inf,
        "solved": 0,
        "points": 0,
        "worst": [],
        "seconds": 0.0,
        "limits_unavailable": [],
        "error": error,
    }


def framework_score(metrics: Dict[str, Any]) -> float:
    """Mean digits on [0, 1], order-preserving with what the tree ranks on."""
    value = metrics.get("mean_digits")
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value) / DIGIT_CAP))


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def run_candidate(
    code: str,
    *,
    suite: Suite,
    shard: int,
    timeout: float,
    shard_seconds: float = SHARD_SECONDS,
    max_length: int = 20_000,
    nproc_limit: int = 64,
) -> Dict[str, Any]:
    """Execute one candidate against one shard, under the shared sandbox profile."""
    valid, reason = validate_source(
        code,
        max_length,
        entrypoint="hyp2f1",
        allowed_imports=ALLOWED_IMPORTS,
        literal_top_level=False,
    )
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix="era-hyp2f1-") as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        # Written into the scratch bind, which the profile makes visible; the
        # file carries the four parameters and nothing else.
        points = Path(scratch) / "points.json"
        points.write_text(json.dumps(list(suite.shard_points[shard])), encoding="utf-8")
        command, env = sandbox_wrapper(
            [
                str(RUNNER),
                str(candidate),
                "--points", str(points),
                "--shard-seconds", str(shard_seconds),
                "--cpu-seconds", str(max(2, int(math.ceil(timeout)))),
                "--nproc-limit", str(nproc_limit),
            ],
            scratch=Path(scratch).resolve(),
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=timeout + 10.0, env=env, cwd=scratch)
        except subprocess.TimeoutExpired:
            return {"ok": False,
                    "error": f"timeout after {timeout + 10.0:.0f}s",
                    "seconds": time.monotonic() - started}
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            tail = (completed.stderr or "").strip()[-300:]
            return {"ok": False,
                    "error": f"no runner output (rc={completed.returncode}): {tail}",
                    "seconds": time.monotonic() - started}
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"ok": False,
                    "error": f"unparseable runner output: {lines[-1][:200]}",
                    "seconds": time.monotonic() - started}


def evaluate_source(
    code: str,
    *,
    suite: Suite,
    shards: Sequence[int],
    timeout: float,
    shard_seconds: float = SHARD_SECONDS,
    max_length: int = 20_000,
    worst_reported: int = 5,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards, pooled over points.

    A point that raised, returned a non-finite value or ran out of the shard's
    wall-clock scores zero digits and the rest of the shard still counts. A
    *program* that could not be imported, or that has no ``hyp2f1``, fails the
    whole evaluation.
    """
    total = 0.0
    scored = 0
    solved = 0
    seconds = 0.0
    unavailable: List[str] = []
    detail: List[Dict[str, Any]] = []

    for shard in shards:
        payload = run_candidate(code, suite=suite, shard=shard, timeout=timeout,
                                shard_seconds=shard_seconds, max_length=max_length)
        seconds += float(payload.get("seconds") or 0.0)
        if not payload.get("ok"):
            error = str(payload.get("error") or "candidate failed")
            return False, _zero_metrics(error), error
        results = payload.get("results") or []
        points = suite.shard_points[shard]
        if len(results) != len(points):
            error = (f"runner returned {len(results)} results for "
                     f"{len(points)} points")
            return False, _zero_metrics(error), error
        unavailable = payload.get("limits_unavailable") or unavailable
        for index, (result, point, truth) in enumerate(
                zip(results, points, suite.shard_truth[shard])):
            digits = digits_of(result.get("estimate"), truth)
            total += digits
            scored += 1
            solved += int(digits >= SOLVED_DIGITS)
            detail.append({
                "shard": shard,
                "index": index,
                "a": float(point["a"]), "b": float(point["b"]),
                "c": float(point["c"]), "z": float(point["z"]),
                "digits": round(digits, 3),
                "error": str(result.get("error") or ""),
            })

    if not scored:
        return False, _zero_metrics("no points scored"), "no points scored"
    mean_digits = total / scored
    worst = sorted(detail, key=lambda row: (row["digits"], row["shard"], row["index"]))
    return (
        True,
        {
            "mean_digits": mean_digits,
            # FUTS maximises and more digits is better, so no sign flip.
            "score": mean_digits,
            "solved": solved,
            "points": scored,
            "worst": worst[:worst_reported],
            "seconds": seconds,
            "limits_unavailable": unavailable,
            "error": "",
        },
        "",
    )


# --------------------------------------------------------------------------
# The mutation prompt
# --------------------------------------------------------------------------

SYSTEM_PREAMBLE = """You are an expert in numerical special functions and a
Python programmer. Your task is to write a double-precision routine that
evaluates the Gauss hypergeometric function as accurately as possible.
Return ONLY the python code."""


def _failure_report(metrics: Dict[str, Any], limit: int = 5) -> str:
    """The worst points, with their parameters -- never with their values.

    Upstream's prompt shows a single score. A special-function routine fails
    *per region of the parameter space*, and a search told only its mean cannot
    tell a uniformly mediocre routine from one that is exact everywhere except
    where ``z`` is far to the left. The parameters are what a bug report would
    carry; the reference value is withheld, so the feedback cannot be turned
    into a table of answers -- and the test shards are different points anyway.
    """
    rows = metrics.get("worst") or []
    if not rows:
        return ""
    lines = [f"Worst points in the last evaluation (correct digits, out of "
             f"{DIGIT_CAP:.0f}):"]
    for row in rows[:limit]:
        note = f", raised {row['error']}" if row.get("error") else ""
        lines.append(
            f"  a={row['a']:.4f} b={row['b']:.4f} c={row['c']:.4f} "
            f"z={row['z']:.4f}: {row['digits']} digits{note}")
    return "\n".join(lines)


def mutation_prompt(
    parent: Any,
    *,
    preview: str,
    timeout: float = 60.0,
    shard_seconds: float = SHARD_SECONDS,
) -> str:
    """`PlaygroundGenerator.__call__`, re-pointed at special-function evaluation."""
    score = parent.metrics.get("mean_digits")
    shown = f"{float(score):.4f}" if score is not None else "failed to run"
    solved = parent.metrics.get("solved")
    points = parent.metrics.get("points")
    solved_line = (f"It reached {SOLVED_DIGITS:.0f}+ correct digits on {solved} of "
                   f"{points} points." if points else "")
    failures = _failure_report(parent.metrics)
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

Write a routine that evaluates the Gauss hypergeometric function 2F1(a, b; c; z)
for real a, b, c and real z, in double precision.

{preview}

The metric is the MEAN NUMBER OF CORRECT SIGNIFICANT DIGITS over the point set,
where a point scores min({DIGIT_CAP:.0f}, -log10(|estimate - exact| / |exact|))
against a reference computed independently at 25 digits, and a point that
raises, returns a non-finite value or overruns the time limit scores 0. Higher
is better; {DIGIT_CAP:.0f} is the maximum.

The previous solution scored: {shown}
{solved_line}
{failures}

Previous Solution Code:
```python
{parent.code}
```

Please generate a NEW, IMPROVED Python function named `hyp2f1` that:
1. Has the signature `hyp2f1(a, b, c, z)` and returns a single float.
2. Is correct across the whole declared range, not only near z = 0. The Taylor
   series in z does not converge for |z| > 1, and converges slowly and with
   cancellation well before that.
3. Decides what to do from (a, b, c, z) alone. You cannot see the exact value,
   so any choice between methods has to rest on a criterion you can evaluate --
   the size of the transformed argument, the size of the parameters, an estimate
   of the cancellation, a term-growth check on a series you are summing.
4. Never returns nan or inf for a point that has a finite value. Returning a
   poor estimate scores more than returning nothing.

You may call `scipy.special.hyp2f1` -- it is the baseline and it is very good in
much of the space. Beating it means knowing *where* it is not, and doing
something better there. `scipy.special` also has gamma, gammaln, poch, digamma
and the rest of the machinery the standard transformations need.

Your code must look like this:
```python
import math
import numpy as np
# ... other imports

def hyp2f1(a, b, c, z):
    # ... transform, select a method, sum, correct ...
    return value
```
Provide the full, runnable code including imports.

IMPORT CONSTRAINTS:
1. math, cmath, numpy and scipy only. `mpmath`, `decimal` and `fractions` are
   NOT available: the deliverable is a float64 routine, and arbitrary-precision
   arithmetic is a different problem.
2. There is no network and no filesystem. Do not read or write files, and do not
   set any thread count above 1.
3. {shard_seconds:.0f} seconds of wall-clock for the whole problem set,
   {timeout:.0f} seconds for the process. Per point that is far more than a
   special-function routine needs; it is there to stop an unbounded sum.
4. Deterministic methods only.
--- END PROMPT ---
"""
