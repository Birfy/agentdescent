"""Suite, sandboxed evaluator, and prompt for the ERA port's AlgoTune task.

The public runnable example lives in :mod:`examples.era.era_algotune`; the
loader, the shim and the reference-to-program transform live in
:mod:`examples.era._algotune_tasks`. This module is the boundary between them:
it materialises the task file a shard is generated from, runs a candidate
against it under the *same* Bubblewrap/Seatbelt profile the other three ERA
tasks use, and turns the timings that come back into the score FUTS ranks nodes
by.

Why this task exists next to the other three
--------------------------------------------
The three ERA tasks already here optimise **accuracy**: lower RMSE, more correct
digits. AlgoTune (`arXiv:2507.15887 <https://arxiv.org/abs/2507.15887>`_,
`oripress/AlgoTune <https://github.com/oripress/AlgoTune>`_) optimises the other
axis, and holds accuracy fixed while doing it: a candidate must produce an
answer the task's own ``is_solution`` accepts, and is then scored on **how much
faster than the reference implementation it is**. Nothing about the search
changes -- same flat-PUCT tree, same aggregator, same sandbox, same governance
layer -- which is the point of :class:`~examples.era._era_domain.Domain`.

It is also the first ERA task here whose reference is a moving target in the
useful sense: the baseline is not a strawman someone wrote for the benchmark, it
is ``scipy.linalg.svd``, ``scipy.integrate.solve_ivp``, ``scipy.signal.upfirdn``
-- the call a working scientist already makes. A speedup over that is a claim
about the library, not about the benchmark.

What is faithful, and what is not
---------------------------------
Faithful to AlgoTune: the task files, their ``generate_problem`` /
``solve`` / ``is_solution`` triple, the published problem size per task (``n``
at a 100 ms reference time, read from upstream's own ``reports/generation.json``),
``speedup = baseline_time / solver_time`` per problem from the **minimum** of
repeated runs, the mean over problems, and the rule that a task whose solutions
are not all valid has no speedup at all.

Not faithful, and deliberately:

* **A candidate is a module-level ``solve(problem)``**, not AlgoTune's
  ``class Solver``. ERA's contract is a function per program and its gate checks
  for one; a class would be a second contract for one task.
* **The dataset is generated, not downloaded.** AlgoTune publishes 100 train and
  100 test instances per task as a HuggingFace dataset; this draws its problems
  from the same ``generate_problem`` at the same ``n``, seeded per shard, because
  a shard has to be a *disjoint* draw for the held-back split to mean anything.
* **Timing is per evaluation, not calibrated once.** The reference is re-timed
  in the same sandboxed process, moments before the candidate, on the same
  problem. That doubles the work and removes the machine from the ratio.
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

from agentdescent.dataloader import cache_path, fetch_text

from examples.era._algotune_tasks import UPSTREAM_COMMIT, derive_seed_program
from examples.era._era_support import sandbox_wrapper, validate_source


RUNNER = Path(__file__).with_name("_era_algotune_runner.py")

_RAW = f"https://raw.githubusercontent.com/oripress/AlgoTune/{UPSTREAM_COMMIT}"
TASK_URL = _RAW + "/AlgoTuneTasks/{task}/{task}.py"
DESCRIPTION_URL = _RAW + "/AlgoTuneTasks/{task}/description.txt"
#: Upstream's own record of the dataset it published: for each of the 154 tasks,
#: the problem size ``n`` at which the reference took ~100 ms on the machine
#: AlgoTune generated on, and the timings that established it. Read rather than
#: re-derived, so the problem sizes here are the benchmark's rather than this
#: host's -- a size calibrated on whatever machine happens to run the search
#: would make two runs of this port incomparable.
SIZES_URL = _RAW + "/reports/generation.json"

#: The 72 AlgoTune tasks this port can run, and why it is 72 rather than 154.
#:
#: Two filters, both mechanical and both checked by
#: ``tests/test_era_algotune.py`` rather than asserted here:
#:
#: 1. **The reference must import only numpy, scipy and the standard library.**
#:    AlgoTune's dependency list is 30 packages deep -- cvxpy, OR-Tools,
#:    networkx, torch, faiss, python-sat, dace. 81 tasks need one of them, and a
#:    port that silently skipped them at runtime would report a speedup over a
#:    benchmark it had quietly halved. They are named as excluded instead.
#: 2. **The reference must lift out of its class** (see
#:    :func:`~examples.era._algotune_tasks.derive_seed_program`), so the root
#:    node is a program rather than a bound method.
#:
#: ``lqr`` passes both and is still absent: its own ``is_solution`` does
#: ``float(xt.T @ Q @ xt + ...)`` on a 1x1 array, which NumPy has refused since
#: 1.25, so on any current NumPy the *reference implementation* is invalid by the
#: task's own oracle. That is upstream's defect, not this port's, and scoring a
#: search against an oracle that rejects its own baseline would measure nothing.
TASKS: Tuple[str, ...] = (
    "affine_transform_2d", "cholesky_factorization", "convex_hull",
    "convolve2d_full_fill", "convolve_1d", "correlate2d_full_fill",
    "correlate_1d", "cumulative_simpson_1d", "cumulative_simpson_multid",
    "dct_type_I_scipy_fftpack", "delaunay", "dijkstra_from_indices",
    "dst_type_II_scipy_fftpack", "eigenvalues_complex", "eigenvalues_real",
    "eigenvectors_complex", "eigenvectors_real", "elementwise_integration",
    "fft_cmplx_scipy_fftpack", "fft_convolution", "fft_real_scipy_fftpack",
    "generalized_eigenvalues_complex", "generalized_eigenvalues_real",
    "generalized_eigenvectors_complex", "generalized_eigenvectors_real",
    "graph_laplacian", "ks_test_2samp", "l0_pruning", "l1_pruning",
    "least_squares", "linear_system_solver", "lti_simulation",
    "lu_factorization", "matrix_exponential", "matrix_multiplication",
    "matrix_sqrt", "min_weight_assignment", "ode_brusselator",
    "ode_fitzhughnagumo", "ode_hires", "ode_hodgkinhuxley",
    "ode_lorenz96_nonchaotic", "ode_lotkavolterra", "ode_nbodyproblem",
    "ode_seirs", "ode_stiff_robertson", "ode_stiff_vanderpol", "odr",
    "outer_product", "pde_burgers1d", "pde_heat1d", "procrustes",
    "psd_cone_projection", "qr_factorization", "qz_factorization",
    "rbf_interpolation", "rotate_2d", "shift_2d", "shortest_path_dijkstra",
    "sparse_eigenvectors_complex", "sparse_lowest_eigenvalues_posdef",
    "sparse_lowest_eigenvectors_posdef", "stable_matching", "svd",
    "sylvester_solver", "toeplitz_solver", "two_eigenvalues_around_0",
    "unit_simplex_projection", "upfirdn1d", "voronoi_diagram",
    "wasserstein_dist", "zoom_2d",
)

#: What ``--tasks`` selects when nothing is named: eight tasks spanning the
#: categories AlgoTune groups by -- dense linear algebra, matrix functions,
#: signal processing, a stiff ODE, a sparse eigenproblem and computational
#: geometry -- and all of them cheap enough that a tree of a dozen nodes is
#: minutes rather than hours. Every other name in :data:`TASKS` is one flag away.
DEFAULT_TASKS: Tuple[str, ...] = (
    "svd",
    "matrix_exponential",
    "eigenvalues_real",
    "convolve_1d",
    "ode_stiff_vanderpol",
    "cholesky_factorization",
    "sparse_lowest_eigenvalues_posdef",
    "convex_hull",
)

#: Wide enough for the references and for a candidate that wants to rewrite one,
#: and no wider. ``logging`` and ``enum`` are here because upstream's own task
#: files import them; the rest is the numerical stack. As in the other three ERA
#: tasks, the sandbox rather than this set is the isolation boundary -- scipy
#: alone can spawn processes and read files.
#:
#: ``numba`` and ``cython`` are here because AlgoTune's own results say they have
#: to be. Across upstream's 2076 published solutions the two are 21% of
#: everything but **50% of the results at 100x or better** -- the reference on a
#: task like ``ode_seirs`` pays a Python callback per derivative evaluation, and
#: nothing written in NumPy can close that. An allowlist without them does not
#: make the benchmark harder, it deletes the half of it where the large wins
#: live, and a port that reported a geometric mean over what was left would be
#: reporting a different benchmark under AlgoTune's name.
#:
#: Both compile *inside* the sandbox: numba through LLVM in-process, Cython by
#: invoking ``gcc``, which the read-only bind of ``/`` makes reachable. The cost
#: lands on the warm-up call, which is discarded -- the same treatment AlgoTune
#: gives it, since its own solvers compile in ``Solver.__init__``.
ALLOWED_IMPORTS = {
    # A compiler directive rather than a module, and two of the task files open
    # with it. Left out, `prepare_suite` succeeds, the tree is built, and the
    # *root node* is then refused by the gate -- so the task dies with "the
    # initial ERA program failed to run" and the run reports one fewer task than
    # it was asked for. Found exactly that way, on
    # `sparse_lowest_eigenvalues_posdef`.
    "__future__",
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "dataclasses",
    "enum",
    "functools",
    "heapq",
    "itertools",
    "cython",
    "logging",
    "math",
    "numba",
    "numpy",
    "operator",
    "random",
    "scipy",
    "statistics",
    "string",
    "typing",
    "warnings",
}

#: Timed runs per program per problem, after one warm-up run that is discarded.
#: Three, because the metric is a *ratio of minima* and the minimum of three
#: runs already removes most of the scheduler noise a shared machine adds; more
#: buys precision the search cannot use, and every one of them is paid twice --
#: once for the reference and once for the candidate.
REPEATS = 3

#: Wall-clock a single problem may take, reference and candidate together.
PROBLEM_SECONDS = 60.0

#: How much slower than the reference a candidate may be before the remaining
#: timed runs are abandoned and its first run is reported as its time. A
#: candidate 200x slower is not a measurement worth repeating, and without this
#: it would instead overrun the shard timeout and be recorded as a program that
#: failed -- which is a different claim from "correct, and far too slow".
SLOW_FACTOR = 20.0

#: Address space per sandboxed evaluation. Larger than the 2 GiB the other ERA
#: tasks use because an AlgoTune problem at its published size can be hundreds of
#: megabytes on its own (``outer_product`` at n=10630 is a 904 MB result), and
#: every timed run is handed its own deep copy.
ADDRESS_SPACE_MB = 4096


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """One AlgoTune task, its problem sizes, and the shards drawn from it.

    Same shape as the other ERA tasks' :class:`~examples.era._era_support.Splits`
    and :class:`~examples.era._era_integration.Suite`: what the sandbox needs is
    on disk, the last ``test_shards`` are never shown to the search.

    A shard here is a set of **random seeds**, not a file of problems. AlgoTune's
    problems are numpy arrays, sparse matrices and graphs that no JSON survives
    intact, and ``generate_problem(n, random_seed)`` is deterministic -- so the
    seeds are what crosses the sandbox boundary and the problems are rebuilt
    inside it.
    """

    task: str
    source_path: Path
    description: str
    initial_program: str
    n: int
    published_n: int
    target_time_ms: int
    published_ms: Optional[float]
    problems: int
    scoring_shards: int
    test_shards: int
    seed: int
    #: Problems per **held-back** set, which the search never scores against and
    #: therefore never pays for. Separate from :attr:`problems` because the two
    #: numbers answer different questions and cost differently. The scoring sets
    #: are paid for on every rollout and again on every gate evaluation, so
    #: doubling them roughly doubles the run; the held-back sets are scored twice
    #: per task, at the end, and are the only thing the *reported* number rests
    #: on. AlgoTune reports over 100 test instances (`dataset.test_size`), and a
    #: figure taken over six is not comparable with one taken over a hundred
    #: however carefully each is measured -- so the reported split can be widened
    #: to match without making the search itself a hundred times more expensive.
    #:
    #: Defaults to :attr:`problems`, which is what every run before this field
    #: existed did.
    test_problems: int = 0

    def _count(self, shard: int) -> int:
        return (self.test_problems or self.problems
                if shard >= self.scoring_shards else self.problems)

    def seeds(self, shard: int) -> Tuple[int, ...]:
        """The problem seeds of one shard -- disjoint across shards by construction.

        The two splits are laid out in **separate runs of seeds**: the scoring
        shards fill the first ``scoring_shards * problems``, the held-back ones
        start after them. That is what lets ``--test-problems`` widen the
        reported measurement without moving a single seed the search will score
        against -- a run measured over a hundred problems is then a rerun of the
        search that was measured over six, not a different experiment. A single
        stride across both splits would have shifted every scoring shard the
        moment the reported split changed size.
        """
        origin = 1 + self.seed * 100_003
        if shard < self.scoring_shards:
            base = origin + shard * self.problems
        else:
            base = (origin + self.scoring_shards * self.problems
                    + (shard - self.scoring_shards) * self._count(shard))
        return tuple(base + index for index in range(self._count(shard)))

    def test_range(self) -> Tuple[int, ...]:
        return tuple(range(self.scoring_shards,
                           self.scoring_shards + self.test_shards))

    def size(self, shard: int = 0) -> int:
        return self._count(shard)


def published_sizes() -> Dict[str, Dict[str, Any]]:
    """Upstream's ``reports/generation.json``, cached on disk."""
    text = fetch_text(SIZES_URL, cache_subdir="era-algotune",
                      filename=f"generation-{UPSTREAM_COMMIT[:12]}.json")
    return json.loads(text)


def task_source(task: str) -> str:
    """The task file, cached, keyed by the pinned commit."""
    return fetch_text(TASK_URL.format(task=task), cache_subdir="era-algotune",
                      filename=f"{task}-{UPSTREAM_COMMIT[:12]}.py")


def task_description(task: str) -> str:
    """Upstream's ``description.txt`` -- the problem statement, as written."""
    return fetch_text(DESCRIPTION_URL.format(task=task),
                      cache_subdir="era-algotune",
                      filename=f"{task}-{UPSTREAM_COMMIT[:12]}.txt").strip()


def prepare_suite(
    task: str,
    *,
    seed: int = 0,
    shards: int = 4,
    test_shards: int = 2,
    problems: int = 2,
    test_problems: int = 0,
    size_scale: float = 1.0,
) -> Suite:
    """Fetch one task, derive its seed program, and fix the shards.

    Nothing is executed here. The task file is *parsed* to lift its reference out
    of its class, and the file itself is written under the dataloader's cache for
    the sandbox to import -- which is the only place any of this code runs.

    ``size_scale`` multiplies upstream's published ``n``. It is a wall-clock
    knob and a difficulty knob at once, so a scaled run says so in its result
    file and is not comparable to an unscaled one: at a tenth of the size a task
    can stop being memory-bound, and the ranking of two candidates can invert.
    """
    if task not in TASKS:
        raise ValueError(
            f"{task!r} is not one of the {len(TASKS)} AlgoTune tasks this port "
            f"can run (see examples.era._era_algotune.TASKS)")
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    if problems < 1:
        raise ValueError("need at least one problem per shard")
    if test_problems and test_problems < 1:
        raise ValueError("need at least one problem per held-back set")
    if not 0.0 < size_scale <= 1.0:
        raise ValueError("size_scale must be in (0, 1]")

    source = task_source(task)
    initial = derive_seed_program(source)
    sizes = published_sizes().get(task) or {}
    published_n = int(sizes.get("n") or 0)
    if published_n < 1:
        raise ValueError(f"upstream published no problem size for {task!r}")
    runs = [row for row in (sizes.get("baseline_runs") or {}).values()
            if isinstance(row, dict) and row.get("avg_min_ms")]
    published_ms = (sum(float(row["avg_min_ms"]) for row in runs) / len(runs)
                    if runs else None)

    fingerprint = hashlib.sha256(
        f"{UPSTREAM_COMMIT}|{task}|{source}".encode("utf-8")).hexdigest()[:12]
    root = Path(cache_path("era-algotune", f"task-{fingerprint}"))
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / f"{task}.py"
    if not source_path.exists():
        source_path.write_text(source, encoding="utf-8")

    return Suite(
        task=task,
        source_path=source_path,
        description=task_description(task),
        initial_program=initial,
        n=max(1, int(published_n * size_scale)),
        published_n=published_n,
        target_time_ms=int(sizes.get("target_time_ms") or 0),
        published_ms=published_ms,
        problems=problems,
        scoring_shards=shards,
        test_shards=test_shards,
        seed=seed,
        test_problems=test_problems,
    )


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "speedup": None,
        # `-inf` is upstream ERA's failure sentinel, and the tree appends the
        # node anyway. Keeping it keeps node ordering identical to `futs.search`.
        "score": -math.inf,
        "valid_problems": 0,
        "problems": 0,
        "baseline_ms": None,
        "candidate_ms": None,
        "slowest": [],
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
    repeats: int = REPEATS,
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    nproc_limit: int = 64,
) -> Dict[str, Any]:
    """Execute one candidate against one shard and return the runner's payload."""
    valid, reason = validate_source(
        code,
        max_length,
        entrypoint="solve",
        allowed_imports=ALLOWED_IMPORTS,
        # A precomputed table, a compiled regex, a cached plan: ordinary in a
        # program whose whole purpose is to be fast, and refusing them here
        # would reject exactly the candidates the task is looking for. The CPU
        # limit applies to module-level work like everything else.
        literal_top_level=False,
        # And a bare call as a statement, because that is how a JIT is warmed:
        # `_kernel(0.0, 0.0)` under an `@njit` def forces compilation at import,
        # where it is free, instead of inside the first timed call, where it
        # would be charged to the candidate.
        allow_top_level_calls=True,
    )
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix="era-algotune-") as scratch:
        root = Path(scratch)
        candidate = root / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        # Copied into the scratch bind rather than read where it was cached, for
        # the reason the integrals task copies its problem file: Bubblewrap
        # mounts a fresh tmpfs over `/tmp`, so a cache under `/tmp` is invisible
        # inside the sandbox and the candidate would be blamed for the
        # FileNotFoundError.
        task_file = root / "algotune_task.py"
        task_file.write_bytes(suite.source_path.read_bytes())
        spec = root / "spec.json"
        spec.write_text(json.dumps({
            "task": suite.task,
            "n": suite.n,
            "seeds": list(suite.seeds(shard)),
            "repeats": int(repeats),
            "problem_seconds": float(problem_seconds),
            "slow_factor": SLOW_FACTOR,
        }), encoding="utf-8")
        command, env = sandbox_wrapper(
            [
                str(RUNNER),
                str(candidate),
                "--task-source", str(task_file),
                "--spec", str(spec),
                "--cpu-seconds", str(max(2, int(math.ceil(timeout)))),
                "--nproc-limit", str(nproc_limit),
                "--address-space-mb", str(ADDRESS_SPACE_MB),
            ],
            scratch=root.resolve(),
            # Every compiler cache pointed at the scratch bind. The profile is
            # `--clearenv`, so without `HOME` Cython resolves its inline cache to
            # `/root/.cython` and dies on the read-only bind -- which reads as
            # "the candidate crashed" rather than "the sandbox gave it nowhere to
            # write". Numba compiles in-process and needs none of this, until a
            # candidate passes `cache=True`.
            extra_env={
                "HOME": str(root.resolve()),
                "XDG_CACHE_HOME": str(root.resolve()),
                "CYTHON_CACHE_DIR": str(root.resolve()),
                "NUMBA_CACHE_DIR": str(root.resolve()),
            },
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
    repeats: int = REPEATS,
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    slowest_reported: int = 3,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards: mean speedup, or nothing.

    **One invalid solution invalidates the whole evaluation.** That is AlgoTune's
    rule, not a choice made here -- ``aggregate_results`` sets ``mean_speedup``
    to ``None`` the moment a single instance fails ``is_solution`` -- and it is
    the rule that makes the benchmark about speed rather than about approximation.
    A program that is a thousand times faster on nine problems and wrong on the
    tenth has not sped anything up; it has changed the question.

    The failure still becomes a node, scoring ``-inf``, exactly as a program that
    would not import does. What the metrics carry back is *which* problem failed
    and why, so the next mutation prompt can say so.
    """
    speedups: List[float] = []
    baseline_total = 0.0
    candidate_total = 0.0
    scored = 0
    valid_problems = 0
    seconds = 0.0
    unavailable: List[str] = []
    detail: List[Dict[str, Any]] = []

    for shard in shards:
        payload = run_candidate(
            code, suite=suite, shard=shard, timeout=timeout, repeats=repeats,
            problem_seconds=problem_seconds, max_length=max_length)
        seconds += float(payload.get("seconds") or 0.0)
        if not payload.get("ok"):
            error = str(payload.get("error") or "candidate failed")
            metrics = _zero_metrics(error)
            metrics["seconds"] = seconds
            return False, metrics, error
        results = payload.get("results") or []
        expected = suite.size(shard)
        if len(results) != expected:
            error = f"runner returned {len(results)} results for {expected} problems"
            return False, _zero_metrics(error), error
        unavailable = payload.get("limits_unavailable") or unavailable
        for result in results:
            scored += 1
            baseline_ms = float(result.get("baseline_ms") or 0.0)
            candidate_ms = result.get("candidate_ms")
            row = {
                "seed": int(result.get("seed") or 0),
                "shard": shard,
                "baseline_ms": round(baseline_ms, 4),
                "candidate_ms": (round(float(candidate_ms), 4)
                                 if candidate_ms is not None else None),
                "speedup": None,
                "valid": bool(result.get("valid")),
                "error": str(result.get("error") or ""),
            }
            if row["valid"] and candidate_ms is not None and float(candidate_ms) > 0:
                speedup = baseline_ms / float(candidate_ms)
                row["speedup"] = round(speedup, 4)
                speedups.append(speedup)
                baseline_total += baseline_ms
                candidate_total += float(candidate_ms)
                valid_problems += 1
            detail.append(row)

    if not scored:
        return False, _zero_metrics("no problems scored"), "no problems scored"
    if valid_problems != scored:
        # Failures first in what the prompt gets to see. A report that led with
        # the three problems that *worked* would answer a question nobody asked:
        # this candidate scored nothing, and why is the only useful thing to say.
        failed_rows = [row for row in detail if row["speedup"] is None]
        error = (f"{scored - valid_problems}/{scored} problems were not solved "
                 f"correctly: "
                 f"{failed_rows[0]['error'] or 'is_solution rejected the output'}")
        metrics = _zero_metrics(error)
        metrics.update({"problems": scored, "valid_problems": valid_problems,
                        "slowest": failed_rows[:slowest_reported],
                        "seconds": seconds, "limits_unavailable": unavailable})
        return False, metrics, error

    # The mean of per-problem speedups, which is AlgoTune's `mean_speedup`. Not
    # the ratio of the summed times: that would let one slow problem dominate a
    # set the benchmark treats as equally weighted instances of one task.
    speedup = sum(speedups) / len(speedups)
    if not math.isfinite(speedup):
        return False, _zero_metrics("non-finite speedup"), "non-finite speedup"
    slowest = sorted(detail, key=lambda row: row["speedup"] or 0.0)
    return (
        True,
        {
            "speedup": speedup,
            # FUTS maximises and faster is better, so the score is the metric
            # itself -- no sign flip, unlike the RMSE task.
            "score": speedup,
            "valid_problems": valid_problems,
            "problems": scored,
            "baseline_ms": baseline_total / valid_problems,
            "candidate_ms": candidate_total / valid_problems,
            "slowest": slowest[:slowest_reported],
            "seconds": seconds,
            "limits_unavailable": unavailable,
            "error": "",
        },
        "",
    )


def framework_score(metrics: Dict[str, Any]) -> float:
    """Map a speedup onto AgentDescent's [0, 1] reward, order-preserving.

    ``s / (1 + s)`` is strictly increasing in ``s``, so it induces exactly the
    ranking the tree uses, and it has no ceiling to saturate against -- a 40x
    candidate still scores above a 20x one, where a rescale by some assumed
    maximum speedup would flatten both to 1.0 and make the acceptance gate blind
    precisely where the task gets interesting. The reference itself scores 0.5.
    """
    value = metrics.get("speedup")
    if value is None or not math.isfinite(float(value)):
        return 0.0
    speedup = max(0.0, float(value))
    return speedup / (1.0 + speedup)


# --------------------------------------------------------------------------
# The mutation prompt
# --------------------------------------------------------------------------

SYSTEM_PREAMBLE = """You are an expert in high-performance scientific Python.
Your task is to make a numerical routine as fast as possible without changing
what it computes. Return ONLY the python code."""


def _timing_report(metrics: Dict[str, Any], limit: int = 3) -> str:
    """Where the time went, per problem -- upstream ERA's prompt shows one score.

    The same addition the integrals task makes, for the same reason: a search
    told only its mean cannot tell a program that is uniformly 1.2x from one
    that is 5x on two problems and 0.3x on a third, and those need opposite next
    moves. The seed is shown so a failure is reproducible.
    """
    rows = metrics.get("slowest") or []
    if not rows:
        return ""
    lines = ["Per-problem timing in the last evaluation:"]
    for row in rows[:limit]:
        if row.get("speedup") is None:
            note = row.get("error") or "is_solution rejected the output"
            lines.append(f"  seed {row['seed']}: INVALID -- {note}")
        else:
            lines.append(
                f"  seed {row['seed']}: reference {row['baseline_ms']} ms, "
                f"yours {row['candidate_ms']} ms, speedup {row['speedup']}x")
    return "\n".join(lines)


def mutation_prompt(
    parent: Any,
    *,
    suite: Suite,
    timeout: float = 120.0,
    repeats: int = REPEATS,
) -> str:
    """`PlaygroundGenerator.__call__`, re-pointed at AlgoTune.

    Upstream ERA's shape is kept exactly -- system preamble, task, data preview,
    metric, the parent's score, the parent's code, a numbered contract, then a
    block of constraints -- and only the contents are this task's. The data
    preview is AlgoTune's own ``description.txt``, verbatim, because that is the
    problem statement its own agent is given.
    """
    speedup = parent.metrics.get("speedup")
    shown = (f"{float(speedup):.3f}x" if speedup is not None
             else "invalid or failed to run")
    error = parent.metrics.get("error") or ""
    failure_line = f"The previous solution's error was: {error}" if error else ""
    timing = _timing_report(parent.metrics)
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

Speed up the reference implementation of the AlgoTune task `{suite.task}`.

{suite.description}

Problems are generated by the task's own generator at n = {suite.n}, and your
output is checked by the task's own `is_solution`. The metric is SPEEDUP:
(reference time) / (your time), measured as the minimum of {repeats} runs of
each on the same problem, averaged over the problems in the set. Higher is
better; the reference itself scores 1.0.

A solution that `is_solution` rejects scores NOTHING AT ALL, no matter how fast
it is. Correctness first, then speed.

The previous solution scored: {shown}
{failure_line}
{timing}

Previous Solution Code:
```python
{parent.code}
```

Please generate a NEW, FASTER Python function named `solve` that:
1. Has the signature `solve(problem)` and returns the same thing the reference
   returns, in the same structure and dtype.
2. Is a module-level function, not a method and not a class.
3. Produces an answer the task's own `is_solution` accepts. **That checker is
   the contract**, and any method meeting it is in scope: a different algorithm,
   a different integration or quadrature scheme, a cheaper factorisation, a
   specialised LAPACK/BLAS driver, a compiled kernel, or skipping work the
   reference does that the answer does not need. What is out of scope is
   returning something you know is not the answer -- constants, zeros, a result
   cached from an earlier call, or output shaped to exploit a hole in the
   checker.
4. May do setup work at module level -- a precomputed table, a plan, a compiled
   kernel, a cached workspace -- if that is what makes it fast. Module-level
   work runs once, at import, and is not charged to your measured time.

WHERE THE LARGE WINS ACTUALLY COME FROM. The reference is a library call, so
beating it means finding what the library is paying for that this task does not
need. Two things dominate, and both are worth checking before tuning flags:

  (a) **An interpreted loop the library cannot vectorise.** A step-by-step
      integrator whose right-hand side is a Python callback, a scan with a
      carried dependency, an early-exit search. The interpreter, not the
      arithmetic, is the cost. An `@numba.njit` kernel over the same recurrence
      is routinely 100x or more here, and it is the single biggest lever
      available to you. If you see `solve_ivp`, `quad`, or any routine that
      calls back into Python per step, write the loop and compile it.
  (b) **Work the reference does that its caller does not need.** Input
      validation it can skip, a copy it can overwrite in place, a conversion of
      the result into Python objects that the checker never required.

Your code must look like this:
```python
import numpy as np
# ... other imports

def solve(problem):
    # ... the fast path ...
    return result
```
Provide the full, runnable code including imports.

IMPORTANT CONSTRAINTS:
1. You may use numpy, scipy, **numba**, **cython** and the standard library.
   Nothing else is installed -- no torch, no jax, no dace, no GPU.
   - `numba.njit` compiles in-process. Compilation happens on the first call,
     which is a discarded warm-up, so it is NOT charged to your time. Call your
     jitted function once at module level to force it.
   - `cython.inline("...C-like Python...", a=..., b=...)` compiles with gcc at
     module level, which is also outside the timed region. Give it typed locals
     and `boundscheck=False` to get anything out of it.
   - Compiling is worth it exactly where NumPy cannot express the loop: a
     step-by-step integrator whose right-hand side is a Python callback, a scan
     with a carried dependency, an early-exit search. It is not worth it in front
     of a BLAS or FFT call, which is already native code.
2. ONE CPU THREAD. The sandbox pins every BLAS thread pool to 1, so a solution
   that "parallelises" will be measured on one core and will not be faster.
   Do not set any thread count.
3. There is no network and no filesystem: do not read or write files.
4. {timeout:.0f} seconds of wall-clock for the whole problem set. A candidate
   more than {SLOW_FACTOR:.0f}x slower than the reference is measured once and
   reported as-is rather than repeated.
5. Do not import, read or reconstruct the task's own reference implementation to
   call it. Write the computation.
--- END PROMPT ---
"""
