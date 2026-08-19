"""Suite, sandboxed evaluator, and prompt for the ERA numerical-integration task.

The public runnable example lives in :mod:`examples.era.era_hard_integrals`; the
integrand catalogue and its closed forms live in :mod:`examples.era._era_integrals`.
This module is the boundary between them: it writes the problem files a shard is
made of, runs a candidate against one under the *same* Bubblewrap/Seatbelt
profile the tabular task uses, and turns the estimates that come back into the
score FUTS ranks nodes by.

Why this task exists next to `era_empirical_software.py`
--------------------------------------------------------
The ERA paper's own task list ends with "numerical solution of integrals"
(arXiv:2509.06503, abstract), and it is the one task there whose ground truth is
*not* a leaderboard: an integral has a value, and a closed form settles what it
is. The Kaggle port measures whether a model can assemble a better regression
pipeline out of scikit-learn; this one measures whether it can write numerics --
the failure modes are endpoint singularities, non-decaying oscillation, sharp
peaks and cancellation, and no library call solves all nine families at once.

The score is **mean correct significant digits**, capped at 12 by the precision
of the references themselves, and every problem is evaluated under a hard limit
on calls to the integrand. That pairing is the task: unlimited evaluations turn
accuracy into a question of who is allowed to spend more, which is not a
question about method.

Upstream released `futs.py` and the Kaggle task; there is no integrals
implementation to port. The *search* is therefore the faithful part and the
nine-family suite is this repository's construction -- said here, in the port
notes, and beside every number the task produces.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentdescent.dataloader import cache_path

from examples.era._era_integrals import (
    DIFFICULTIES,
    DIGIT_CAP,
    FAMILIES,
    Problem,
    digits_of,
    draw_shard,
    exact,
)
from examples.era._era_support import sandbox_wrapper, validate_source


RUNNER = Path(__file__).with_name("_era_integration_runner.py")

#: Narrower than the tabular task's set, and for the opposite reason: pandas and
#: scikit-learn have nothing to offer a quadrature rule, while `cmath` does. The
#: sandbox is still the boundary -- scipy alone can spawn processes and read
#: files -- but an allowlist that admits only what the task can use keeps the
#: distance between "the gate passed it" and "the sandbox contained it" short.
ALLOWED_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "operator",
    "scipy",
    "statistics",
    "typing",
    "warnings",
}

#: The baseline node, and the one line of code every practitioner writes first.
#: `quad` is adaptive Gauss-Kronrod with a documented infinite-range transform,
#: it solves several of the nine families to machine precision, and it returns
#: confident nonsense on the rest -- which is exactly the shape a search needs:
#: a real method to beat rather than a strawman.
INITIAL_PROGRAM = '''"""Baseline: scipy.integrate.quad on its default settings."""
import warnings

from scipy.integrate import quad


def integrate(f, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value, _abserr = quad(f, a, b)
    return float(value)
'''

#: Calls to the integrand, per problem. Generous next to `quad`'s ~1000 and
#: mean enough that a uniform grid cannot buy the last digits by brute force.
EVAL_BUDGET = 200_000

#: Wall-clock per problem, inside the shard's own timeout. Nine of these have to
#: fit in `--candidate-timeout` with room for the interpreter to start.
PROBLEM_SECONDS = 5.0


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """The shards, their problem files, and the reference values they hide.

    Same shape as the tabular task's :class:`~examples.era._era_support.Splits`:
    what the candidate reads is on disk, what it is scored against never leaves
    this process. The last ``test_shards`` are never shown to the search.
    """

    root: Path
    seed: int
    shard_paths: Tuple[Path, ...]
    shard_problems: Tuple[Tuple[Problem, ...], ...]
    scoring_shards: int
    test_shards: int

    def truth(self, shard: int) -> Tuple[float, ...]:
        return tuple(exact(problem) for problem in self.shard_problems[shard])

    def size(self, shard: int) -> int:
        return len(self.shard_problems[shard])

    def test_range(self) -> Tuple[int, ...]:
        return tuple(range(self.scoring_shards,
                           self.scoring_shards + self.test_shards))


def prepare_suite(*, seed: int = 0, shards: int = 8, test_shards: int = 4) -> Suite:
    """Draw every shard and write the problem files exactly once per configuration.

    Written under the dataloader's cache rather than into a temporary directory:
    the files are inputs to a sandboxed process, they are reproducible from
    ``seed`` alone, and a rerun that re-drew them would not be the same
    benchmark.
    """
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    total = shards + test_shards
    problems = [tuple(draw_shard(seed, index)) for index in range(total)]
    serialised = [[problem.to_dict() for problem in shard] for shard in problems]

    # Keyed on the drawn problems themselves, not on `(seed, shards, families)`.
    # A change to a family's parameters leaves that tuple identical, the cached
    # files are then reused, and the sandbox integrates one problem while the
    # host scores it against another's reference -- silently. Deriving the path
    # from the content makes that state unreachable.
    fingerprint = hashlib.sha256(
        json.dumps(serialised, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    root = Path(cache_path("era-integrals", f"suite-{fingerprint}"))
    root.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for index, payload in enumerate(serialised):
        path = root / f"shard-{index:03d}.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        paths.append(path)
    return Suite(root, seed, tuple(paths), tuple(problems), shards, test_shards)


def suite_preview(suite: Suite) -> str:
    """What the model is told about the problems -- the difficulties, not the keys.

    The catalogue of *difficulties* is the task description an expert would be
    handed. The family of any individual problem is not in it, is not in the
    problem the candidate sees (it gets a callable and two limits), and is not in
    the feedback the search reports, because the shard order is shuffled per
    shard. So this describes what has to be handled without saying which case is
    which -- which is the difference between writing a quadrature rule and
    writing a lookup table.
    """
    lines = [
        f"Each problem set holds {len(FAMILIES)} definite integrals, one from each "
        f"of {len(FAMILIES)} difficulty classes, in a shuffled order you are not told.",
        "The classes, all of them present in every problem set:",
    ]
    for index, family in enumerate(FAMILIES, start=1):
        lines.append(f"  {index}. {DIFFICULTIES[family]}")
    lines.append(
        "Intervals are [0, 1], [0, inf) or (-inf, inf). Every integral converges "
        "and every one of them has a finite exact value.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "mean_digits": None,
        # `-inf` is upstream's failure sentinel, and the tree appends the node
        # anyway. Keeping the sentinel keeps node ordering identical to
        # `futs.search`.
        "score": -math.inf,
        "solved": 0,
        "problems": 0,
        "worst": [],
        "calls": 0,
        "seconds": 0.0,
        "limits_unavailable": [],
        "error": error,
    }


def run_candidate(
    code: str,
    *,
    suite: Suite,
    shard: int,
    timeout: float,
    eval_budget: int = EVAL_BUDGET,
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    nproc_limit: int = 64,
) -> Dict[str, Any]:
    """Execute one candidate against one shard and return the runner's payload."""
    valid, reason = validate_source(
        code,
        max_length,
        entrypoint="integrate",
        allowed_imports=ALLOWED_IMPORTS,
        literal_top_level=False,
    )
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix="era-quad-") as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        # The problem file is copied into the scratch bind rather than read from
        # wherever the suite lives. The Bubblewrap profile mounts a fresh tmpfs
        # over `/tmp`, so a suite under `/tmp` -- which is where a test fixture
        # or a host with `XDG_CACHE_HOME=/tmp` puts it -- is invisible inside the
        # sandbox, and the candidate would be blamed for a FileNotFoundError.
        problems = Path(scratch) / "problems.json"
        problems.write_bytes(suite.shard_paths[shard].read_bytes())
        command, env = sandbox_wrapper(
            [
                str(RUNNER),
                str(candidate),
                "--problems", str(problems),
                "--eval-budget", str(eval_budget),
                "--problem-seconds", str(problem_seconds),
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
    eval_budget: int = EVAL_BUDGET,
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    worst_reported: int = 5,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards, pooled over problems.

    Pooled and not averaged per shard, for the reason the tabular task pools its
    squared error: the shards are equal-sized draws from one distribution, and a
    mean of means would weight a shard rather than a problem.

    A problem that raised, exhausted its evaluation budget or returned ``nan``
    scores zero digits and the shard keeps going. A *program* that could not be
    imported, or that has no ``integrate``, fails the whole evaluation: there is
    nothing to give partial credit to.
    """
    total_digits = 0.0
    scored = 0
    solved = 0
    calls = 0
    seconds = 0.0
    unavailable: List[str] = []
    detail: List[Dict[str, Any]] = []

    for shard in shards:
        payload = run_candidate(
            code, suite=suite, shard=shard, timeout=timeout,
            eval_budget=eval_budget, problem_seconds=problem_seconds,
            max_length=max_length)
        seconds += float(payload.get("seconds") or 0.0)
        if not payload.get("ok"):
            error = str(payload.get("error") or "candidate failed")
            return False, _zero_metrics(error), error
        results = payload.get("results") or []
        problems = suite.shard_problems[shard]
        if len(results) != len(problems):
            error = (f"runner returned {len(results)} results for "
                     f"{len(problems)} problems")
            return False, _zero_metrics(error), error
        unavailable = payload.get("limits_unavailable") or unavailable
        for index, (result, problem) in enumerate(zip(results, problems)):
            digits = digits_of(result.get("estimate"), exact(problem))
            total_digits += digits
            scored += 1
            solved += int(digits >= 10.0)
            calls += int(result.get("calls") or 0)
            detail.append({
                "shard": shard,
                "index": index,
                "interval": problem.interval(),
                "digits": round(digits, 3),
                "calls": int(result.get("calls") or 0),
                "error": str(result.get("error") or ""),
            })

    if not scored:
        return False, _zero_metrics("no problems scored"), "no problems scored"
    mean_digits = total_digits / scored
    worst = sorted(detail, key=lambda row: (row["digits"], row["shard"], row["index"]))
    return (
        True,
        {
            "mean_digits": mean_digits,
            # FUTS maximises, and more digits is better, so the score is the
            # metric itself -- no sign flip, unlike the RMSE task.
            "score": mean_digits,
            "solved": solved,
            "problems": scored,
            "worst": worst[:worst_reported],
            "calls": calls,
            "seconds": seconds,
            "limits_unavailable": unavailable,
            "error": "",
        },
        "",
    )


def framework_score(metrics: Dict[str, Any]) -> float:
    """Map mean digits onto AgentDescent's [0, 1] reward, order-preserving.

    A plain rescale by the cap, because the metric is already bounded and
    linear in what the task values. As in the tabular port, the point is that
    the engine's acceptance gate and the tree's node ordering rank candidates
    identically -- a port where those disagreed would be selecting against its
    own acceptance rule.
    """
    value = metrics.get("mean_digits")
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value) / DIGIT_CAP))


# --------------------------------------------------------------------------
# The mutation prompt
# --------------------------------------------------------------------------

SYSTEM_PREAMBLE = """You are an expert numerical analyst and Python programmer.
Your task is to write Python code that evaluates definite integrals as
accurately as possible. Return ONLY the python code."""


def _failure_report(metrics: Dict[str, Any], limit: int = 5) -> str:
    """The worst problems, by interval and digits -- never by family.

    This is the port's one deliberate addition to upstream's prompt, which shows
    a single score and nothing else. A quadrature rule fails *per integrand*, and
    a search told only its mean cannot tell a rule that is uniformly mediocre
    from one that is exact on eight problems and wrong on the ninth. The interval
    and the evaluation count are what a failing report would carry anyway; the
    family name is withheld, so the feedback cannot become a dispatch table.
    """
    rows = metrics.get("worst") or []
    if not rows:
        return ""
    lines = ["Worst problems in the last evaluation (correct digits, out of "
             f"{DIGIT_CAP:.0f}):"]
    for row in rows[:limit]:
        note = f", raised {row['error']}" if row.get("error") else ""
        lines.append(f"  interval {row['interval']}: {row['digits']} digits "
                     f"after {row['calls']} evaluations{note}")
    return "\n".join(lines)


def mutation_prompt(
    parent: Any,
    *,
    preview: str,
    timeout: float = 60.0,
    eval_budget: int = EVAL_BUDGET,
    problem_seconds: float = PROBLEM_SECONDS,
) -> str:
    """`PlaygroundGenerator.__call__`, re-pointed at quadrature.

    Upstream's shape is kept exactly: system preamble, task, data preview,
    metric, the parent's score, the parent's code, a numbered contract, and a
    block of speed constraints. Only the contents are this task's.
    """
    score = parent.metrics.get("mean_digits")
    shown = f"{float(score):.4f}" if score is not None else "failed to run"
    solved = parent.metrics.get("solved")
    problems = parent.metrics.get("problems")
    solved_line = (f"It reached 10+ correct digits on {solved} of {problems} problems."
                   if problems else "")
    failures = _failure_report(parent.metrics)
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

Write a general-purpose numerical integrator for hard one-dimensional integrals.

{preview}

The metric is the MEAN NUMBER OF CORRECT SIGNIFICANT DIGITS over the problem
set, where a problem scores min(12, -log10(|estimate - exact| / |exact|)) and a
problem that raises, returns a non-finite value or overruns its budget scores 0.
Higher is better; 12 is the maximum.

The previous solution scored: {shown}
{solved_line}
{failures}

Previous Solution Code:
```python
{parent.code}
```

Please generate a NEW, IMPROVED Python function named `integrate` that:
1. Has the signature `integrate(f, a, b)` and returns a single float.
2. Treats `f` as a black box: it is a plain scalar function of one float that
   returns one float. It is NOT vectorised -- passing it a numpy array raises.
3. Handles `a = -inf` and `b = +inf`; those are ordinary cases here, not edge cases.
4. Never assumes `f` can be evaluated at an endpoint. `f` may return `inf` or
   raise there, and `f(x)` raises ValueError for any x outside [a, b].
5. Is one method that works on every class above, not a dispatch on the value of
   `a` and `b`.

Your code must look like this:
```python
import math
import numpy as np
# ... other imports

def integrate(f, a, b):
    # ... transform, subdivide, accelerate ...
    return value
```
Provide the full, runnable code including imports.

IMPORTANT CONSTRAINTS:
1. At most {eval_budget} calls to `f` per problem. Call number {eval_budget + 1}
   raises inside `f`; uncaught, that problem scores zero. Accuracy per
   evaluation is the point -- a finer uniform grid is not the answer.
2. {problem_seconds:.0f} seconds of wall-clock per problem, {timeout:.0f} seconds
   for the whole problem set. Overrunning raises inside `f` in the same way.
3. You may use math, cmath, numpy and scipy. There is no network and no
   filesystem; do not read or write files, and do not set any thread count
   above 1.
4. Deterministic methods only. A Monte-Carlo estimate cannot reach these digits.
--- END PROMPT ---
"""
