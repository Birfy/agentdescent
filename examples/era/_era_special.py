"""Suite, sandboxed evaluator, and prompt for the special-function precision tasks.

The runnable example is :mod:`examples.era.era_special_precision`; the stress
sets are the committed files ``data/<target>_stress.json``, produced once by
``tools/gen_special_stress.py``.

Where the targets came from
---------------------------
Not from folklore. ``tools/scan_numeric_precision.py`` measures 48 NumPy and
SciPy float64 entry points against mpmath -- each one at 30 *and* 60 digits,
keeping only points where the two agree -- over parameter ranges declared
before anything was run. Most of what it measures is excellent: seven of the
eight NumPy probes return 16 correct digits, including ``sin`` and ``tan`` at
arguments up to 1e18, and 29 of the 40 SciPy entry points sit above 15. Two do
not, and they are this module's targets:

``scipy.special.pbdv`` -- the parabolic cylinder function D_v(x). Mean 11.67
correct digits, **12.2% of points with no correct digit at all**. The failures
are not last-ulp drift. At ``v=19.83, x=-29.28`` SciPy returns 4.81e100 where
the value is 2.46e80: wrong by twenty orders of magnitude. At ``v=17.02,
x=-14.61`` it returns -2.44e24 where the value is +6.01e15 -- wrong sign, wrong
size. The bad region is coherent: large positive order with negative argument,
where the recurrence SciPy uses is unstable in the direction it is run.

``scipy.special.hyperu`` -- the confluent hypergeometric function U(a, b, x).
Mean 14.36 digits, and 3% of points where SciPy returns **nan** -- not an
inaccurate value, no value at all. At ``a=-15.82, b=-1.30, x=23.10`` the
function equals 2.45e17, a perfectly ordinary well-conditioned number, and
SciPy declines to produce it. The bad region is concentrated on ``a < 0``.

Both are worth a search for the reason the 2F1 task is: the baseline is not a
strawman written for a benchmark but the function every scientist already
calls, and the failures are *algorithmic* -- a different method selection
recovers them -- rather than a float64 wall.

What keeps it honest
--------------------
* The reference is **mpmath at 30 and 60 digits, kept only where the two agree
  to 25**. Nothing in the sandbox can reach it.
* The parameter distribution was **declared before anything was measured** --
  it is the sweep's own declared range, not a region drawn around the failures
  the sweep found -- and is recorded in the data file next to the values.
* Arbitrary-precision arithmetic is **off the allowlist** (``decimal``,
  ``fractions``, ``mpmath``). The deliverable is a float64 routine comparable to
  SciPy's; a candidate that reimplemented mpmath would be answering a different
  question.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from examples.era._era_support import sandbox_wrapper, validate_source


RUNNER = Path(__file__).with_name("_era_special_runner.py")
DATA_DIR = Path(__file__).with_name("data")

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

#: Wall-clock for one shard of points, inside `--candidate-timeout`. Generous:
#: a special-function call is microseconds, and the point of the limit is to
#: stop a candidate that has decided to sum a million terms per point, not to
#: make speed part of the score.
SHARD_SECONDS = 45.0


# --------------------------------------------------------------------------
# The targets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One SciPy entry point, as the ERA search sees it.

    Only what is *not* already in the committed data file lives here: the
    baseline program and the mathematics the prompt has to state. Parameter
    names, the declared distribution and the reference's provenance are read
    from the file, so the prompt cannot drift away from the points.
    """

    key: str
    #: One line, for the run plan and the result file.
    title: str
    #: What a working scientist calls today, and what the root node runs.
    baseline: str
    initial_program: str
    #: The mathematics, stated for the model: definition and known structure.
    background: str
    #: Task-specific numbered guidance, appended to the shared instructions.
    guidance: Tuple[str, ...]


TARGETS: Tuple[Target, ...] = (
    Target(
        key="pbdv",
        title=("Parabolic cylinder D_v(x) in double precision, mean correct "
               "significant digits against a 25-digit reference"),
        baseline="scipy.special.pbdv",
        initial_program='''"""Baseline: scipy.special.pbdv, the implementation everyone already calls."""
from scipy.special import pbdv as _scipy_pbdv


def pbdv(v, x):
    return float(_scipy_pbdv(v, x)[0])
''',
        background="""D_v(x) is the parabolic cylinder function of Whittaker: the
solution of y'' + (v + 1/2 - x^2/4) y = 0 that decays like
exp(-x^2/4) x^v as x -> +inf. It is related to the confluent
hypergeometric functions by

    D_v(x) = 2^(v/2) exp(-x^2/4) * U(-v/2, 1/2, x^2/2)          (x > 0)

and, in the form that stays finite through x = 0, by the even/odd pair

    D_v(x) = 2^(v/2) sqrt(pi) exp(-x^2/4) * [
                 1F1(-v/2, 1/2, x^2/2) / Gamma((1-v)/2)
        - sqrt(2) x 1F1((1-v)/2, 3/2, x^2/2) / Gamma(-v/2) ]

For x < 0 the standard route is the connection formula

    D_v(-x) = cos(pi v) D_v(x) - (sqrt(2 pi) / Gamma(-v)) D_{-v-1}(x) * sin(pi v)

or equivalently a reflection through U. Note that D_v(-x) for large positive v
is *exponentially larger* than D_v(x), so the connection formula subtracts two
quantities of very different size -- which is where an implementation that
recurses in the wrong direction loses everything.""",
        guidance=(
            "The measured failure region of the baseline is large positive order "
            "with negative argument (roughly v >= 15, x < 0), where it returns "
            "values wrong by many orders of magnitude and sometimes with the "
            "wrong sign. That region is where the score is.",
            "`scipy.special.hyperu`, `scipy.special.hyp1f1`, `scipy.special.gamma`, "
            "`scipy.special.gammaln` and `scipy.special.poch` are all available and "
            "all far more accurate than `pbdv` is on that region. Composing them "
            "through one of the identities above is a legitimate and strong move.",
            "Values in this range run from 1e-250 to 1e250. Compute in logs and "
            "restore the sign and scale at the end wherever the direct product "
            "would overflow -- a routine that overflows to inf scores 0 on that "
            "point, and `gammaln` plus a sign is how that is avoided.",
        ),
    ),
    Target(
        key="hyperu",
        title=("Confluent hypergeometric U(a, b, x) in double precision, mean "
               "correct significant digits against a 25-digit reference"),
        baseline="scipy.special.hyperu",
        initial_program='''"""Baseline: scipy.special.hyperu, the implementation everyone already calls."""
from scipy.special import hyperu as _scipy_hyperu


def hyperu(a, b, x):
    return float(_scipy_hyperu(a, b, x))
''',
        background="""U(a, b, x) is Tricomi's confluent hypergeometric function: the
solution of x y'' + (b - x) y' - a y = 0 that behaves like x^-a as
x -> +inf. The standard representations are

    U(a, b, x) = Gamma(1-b)/Gamma(a+1-b) * 1F1(a, b, x)
               + Gamma(b-1)/Gamma(a) * x^(1-b) * 1F1(a+1-b, 2-b, x)

which fails when b is at or near an integer (both terms blow up and cancel),

    U(a, b, x) = x^-a * 2F0(a, a-b+1; ; -1/x)

the asymptotic series, good for large x and divergent for small,

    U(a, b, x) = x^(1-b) U(a-b+1, 2-b, x)                (Kummer transformation)

and the integral representation, valid for Re(a) > 0, x > 0:

    U(a, b, x) = 1/Gamma(a) * int_0^inf e^(-x t) t^(a-1) (1+t)^(b-a-1) dt

The recurrences in `a` are stable in the direction of increasing `a`.""",
        guidance=(
            "The measured failure of the baseline is not inaccuracy -- it returns "
            "`nan` on about 3% of this range, concentrated on a < 0, where the "
            "function is finite and well-conditioned. Every one of those points "
            "currently scores 0, so returning any reasonable value there is worth "
            "more than improving a point that is already right.",
            "For a < 0 the recurrence U(a-1, b, x) = (b - 2a - x) U(a, b, x) + "
            "a (a - b + 1) U(a+1, b, x), run downward from a value of `a` where "
            "the baseline is reliable, reaches the region the baseline refuses. "
            "Whether it is stable in that direction is worth checking rather than "
            "assuming.",
            "`scipy.special.hyp1f1` is accurate over essentially this whole range "
            "(the sweep measured 14.8 mean digits with no point below 11), so the "
            "Gamma-weighted 1F1 pair above is a usable route wherever b is not "
            "close to an integer -- and the Kummer transformation moves some cases "
            "that are close to one.",
        ),
    ),
)

TARGETS_BY_KEY = {target.key: target for target in TARGETS}


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """The stress points, and the references that never leave this process."""

    key: str
    entrypoint: str
    params: Tuple[str, ...]
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

    def signature(self) -> str:
        return f"{self.entrypoint}({', '.join(self.params)})"


def suite_path(key: str) -> Path:
    return DATA_DIR / f"{key}_stress.json"


def load_suite(key: str, *, shards: int = 8, test_shards: int = 4,
               path: Optional[Path] = None) -> Suite:
    """Read a committed stress set. No draw, no network, no mpmath.

    The suite is a file rather than a seeded draw because its references cost
    arbitrary-precision arithmetic to produce and are the part a reader has to
    be able to audit. Regenerating it on every run would make the benchmark
    depend on which mpmath happened to be installed.
    """
    payload = json.loads((path or suite_path(key)).read_text(encoding="utf-8"))
    params = tuple(payload["params"])
    available = len(payload["shards"])
    if shards + test_shards > available:
        raise ValueError(
            f"the {key} stress set has {available} shards; {shards} + "
            f"{test_shards} were asked for")
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    points: List[Tuple[Dict[str, str], ...]] = []
    truth: List[Tuple[str, ...]] = []
    for shard in payload["shards"][: shards + test_shards]:
        points.append(tuple({name: row[name] for name in params} for row in shard))
        truth.append(tuple(row["value"] for row in shard))
    metadata = {name: payload[name] for name
                in ("task", "target", "reference", "distribution", "seed",
                    "baseline_scan")
                if name in payload}
    return Suite(key, payload["entrypoint"], params, tuple(points), tuple(truth),
                 shards, test_shards, metadata)


def suite_preview(suite: Suite) -> str:
    """What the model is told about the points: the distribution, not the values."""
    distribution = suite.metadata.get("distribution", {})
    lines = [f"Each problem set holds {suite.size(0)} points "
             f"({', '.join(suite.params)}), drawn independently from a fixed "
             f"distribution:"]
    for name in suite.params:
        lines.append(f"  {name:<4}~ {distribution.get(name, 'unstated')}")
    rejected = distribution.get("rejected")
    if rejected:
        lines.append(f"Points rejected when the set was drawn: {rejected}. So "
                     "every point you are given has a finite, well-defined value "
                     "that a float64 can represent.")
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
        entrypoint=suite.entrypoint,
        allowed_imports=ALLOWED_IMPORTS,
        literal_top_level=False,
    )
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix=f"era-{suite.key}-") as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        # Written into the scratch bind, which the profile makes visible; the
        # file carries the parameters and nothing else.
        points = Path(scratch) / "points.json"
        points.write_text(json.dumps(list(suite.shard_points[shard])), encoding="utf-8")
        command, env = sandbox_wrapper(
            [
                str(RUNNER),
                str(candidate),
                "--points", str(points),
                "--entrypoint", suite.entrypoint,
                "--params", ",".join(suite.params),
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
    worst_reported: int = 6,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards, pooled over points.

    A point that raised, returned a non-finite value or ran out of the shard's
    wall-clock scores zero digits and the rest of the shard still counts. A
    *program* that could not be imported, or that has no entry point, fails the
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
            row: Dict[str, Any] = {"shard": shard, "index": index,
                                   "digits": round(digits, 3),
                                   "error": str(result.get("error") or "")}
            row.update({name: float(point[name]) for name in suite.params})
            detail.append(row)

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
evaluates the function below as accurately as possible.
Return ONLY the python code."""


def _failure_report(metrics: Dict[str, Any], params: Sequence[str],
                    limit: int = 6) -> str:
    """The worst points, with their parameters -- never with their values.

    Upstream's prompt shows a single score. A special-function routine fails
    *per region of the parameter space*, and a search told only its mean cannot
    tell a uniformly mediocre routine from one that is exact everywhere except
    where the order is large and the argument negative. The parameters are what
    a bug report would carry; the reference value is withheld, so the feedback
    cannot be turned into a table of answers -- and the test shards are
    different points anyway.
    """
    rows = metrics.get("worst") or []
    if not rows:
        return ""
    lines = [f"Worst points in the last evaluation (correct digits, out of "
             f"{DIGIT_CAP:.0f}):"]
    for row in rows[:limit]:
        note = f", raised {row['error']}" if row.get("error") else ""
        shown = " ".join(f"{name}={row[name]:.4f}" for name in params
                         if name in row)
        lines.append(f"  {shown}: {row['digits']} digits{note}")
    return "\n".join(lines)


def mutation_prompt(
    parent: Any,
    *,
    target: Target,
    suite: Suite,
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
    failures = _failure_report(parent.metrics, suite.params)
    signature = suite.signature()
    numbered = "\n".join(f"{index + 5}. {line}" for index, line
                         in enumerate(target.guidance))
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

Write a routine that evaluates {suite.metadata.get('task', target.title)}.

{target.background}

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

Please generate a NEW, IMPROVED Python function named `{suite.entrypoint}` that:
1. Has the signature `{signature}` and returns a single float.
2. Is correct across the whole declared range, not only where the baseline
   already is. A method that works in one region and diverges in another has to
   be guarded by a criterion, not applied everywhere.
3. Decides what to do from ({', '.join(suite.params)}) alone. You cannot see the
   exact value, so any choice between methods has to rest on a criterion you can
   evaluate -- the size of an argument, the size of the parameters, an estimate
   of the cancellation, a term-growth check on a series you are summing.
4. Never returns nan or inf for a point that has a finite value. Returning a
   poor estimate scores more than returning nothing.
{numbered}

You may call `{target.baseline}` -- it is the baseline and it is very good in
much of the space. Beating it means knowing *where* it is not, and doing
something better there. `scipy.special` also has gamma, gammaln, poch, digamma,
hyp1f1, hyperu and the rest of the machinery the standard transformations need.

Your code must look like this:
```python
import math
import numpy as np
# ... other imports

def {signature}:
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
