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
Faithful to AlgoTune, checked against its own sources rather than its paper:

* the task files and their ``generate_problem`` / ``solve`` / ``is_solution``
  triple, and the published problem size per task (``n`` at a 100 ms reference
  time, read from upstream's ``reports/generation.json``);
* ``speedup = baseline_time / solver_time`` per instance, taken from the
  **minimum** of repeated runs (``main.py`` scores ``min_time_ms``);
* the **arithmetic mean** of the per-instance speedups as the task's score
  (``result_aggregator.py``), and the rule that a task whose solutions are not
  all valid has no speedup at all;
* the per-instance **10x** cut-off and the 60 s reference timeout
  (``benchmark.baseline_timeout``);
* the warm-up runs on the **previous instance**, not on the timed one
  (``warmup_idx = (idx - 1) % problem_count``). That is a correctness property
  as much as a fidelity one: warming on the timed problem hands a free hit to
  any solver that memoises on its input.
* the prompt is ``AlgoTuner/messages/initial_system_message.txt``, which names
  no acceleration technique. Neither does this.

**Where the measurement is made is a deliberate split.** Upstream scores with
``benchmark.eval_runs: 10`` over ``dataset.test_size: 100``; running that inside
a search would make one gate decision take minutes, so the search measures
cheaply -- it only has to *rank* candidates -- and the reported number is
re-taken at upstream's settings by :mod:`tools.algotune_rescore`.

Not faithful, and deliberately:

* **A candidate is a module-level ``solve(problem)``**, not AlgoTune's
  ``class Solver``. ERA's contract is a function per program and its gate checks
  for one; a class would be a second contract for one task.
* **The dataset is generated, not downloaded.** AlgoTune publishes 100 train and
  100 test instances per task as a HuggingFace dataset; this draws its problems
  from the same ``generate_problem`` at the same ``n``, seeded per shard, because
  a shard has to be a *disjoint* draw for the held-back split to mean anything.
* **One rewrite per node, not a ~100-turn command loop.** Upstream's agent gets
  ``edit`` / ``eval`` / ``profile`` / ``revert`` and a $1 budget. A flat tree
  search has no conversation to run those in, so what its prompt can carry is
  the *state* those commands would have produced, not the commands.
* **Timing is per evaluation, not calibrated once.** The reference is re-timed
  in the same sandboxed process, moments before the candidate. That doubles the
  work and removes the machine from the ratio.
* **Search hyper-parameters follow ERA, not AlgoTuner.** Upstream runs every
  model at ``temperature: 0.0``, which is right for one conversation and wrong
  for a tree: at zero, every draw from a given parent is the same draw and the
  search has nothing to explore with.
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

#: The 147 AlgoTune tasks this port can run, and why it is 147 rather than 154.
#:
#: This list used to be 72, and the gap was never the benchmark -- it was this
#: port's dependency set. AlgoTune pins its own 156 packages in
#: ``requirements.txt``; installing the ones its *references* actually need, and
#: allowing what its own 2595 published solvers actually import, takes the
#: runnable set from 72 to 147. What each one was the sole blocker for, measured
#: rather than guessed: ortools 13 tasks, networkx 9, sklearn 8, pysat 2, sympy
#: 2, POT 2, faiss 2, mpmath 1, hdbscan 1.
#:
#: Two filters remain, both mechanical and both checked by
#: ``tests/test_era_algotune.py`` rather than asserted here:
#:
#: 1. **The reference must import only what this port installs and allows.**
#: 2. **The reference must lift out of its class** (see
#:    :func:`~examples.era._algotune_tasks.derive_seed_program`), so the root
#:    node is a program rather than a bound method.
#:
#: Of the seven that are out: ``aes_gcm_encryption`` and ``chacha_encryption``
#: want ``os.urandom`` (see :data:`NOT_ALLOWED_ON_PURPOSE`), two do not lift out
#: of their class, ``lqr`` is in :data:`RUNTIME_EXCLUDED`, and two more still
#: need a package this port does not carry.
#:
#: Every dependency here has to be reachable *inside* the sandbox, which means
#: ``dist-packages`` rather than the user site. Three separate packages hit that
#: trap -- OSQP wants ``jinja2``, OR-Tools wants ``dateutil``, and ``jinja2``
#: itself had been installed into ``~/.local`` -- and each failed with a message
#: about the solver rather than about the bind, which reads as a broken task.
#:
#: Tasks that clear both mechanical filters and still cannot be scored, kept
#: here rather than quietly missing from :data:`TASKS` so a rebuild of that list
#: cannot silently re-admit them. It has happened: regenerating TASKS from
#: "derives and passes the gate" put ``lqr`` straight back, because what is
#: wrong with ``lqr`` only shows up when its checker runs.
RUNTIME_EXCLUDED: Tuple[str, ...] = (
    # Its own `is_solution` does `float(xt.T @ Q @ xt + ...)` on a 1x1 array,
    # which NumPy has refused since 1.25 -- so on any current NumPy the
    # *reference implementation* is invalid by the task's own oracle. Upstream's
    # defect, not this port's, and scoring a search against an oracle that
    # rejects its own baseline would measure nothing. Re-checked after cvxpy
    # went in, in case the import had been the only problem: it is not.
    "lqr",
)

TASKS: Tuple[str, ...] = (
    "affine_transform_2d", "aircraft_wing_design", "articulation_points",
    "base64_encoding", "battery_scheduling", "btsp",
    "capacitated_facility_location", "channel_capacity",
    "chebyshev_center", "cholesky_factorization", "clustering_outliers",
    "communicability", "convex_hull", "convolve2d_full_fill",
    "convolve_1d", "correlate2d_full_fill", "correlate_1d",
    "count_connected_components", "count_riemann_zeta_zeros",
    "cumulative_simpson_1d", "cumulative_simpson_multid",
    "cvar_projection", "cyclic_independent_set", "delaunay",
    "dijkstra_from_indices", "discrete_log",
    "dynamic_assortment_planning", "earth_movers_distance",
    "edge_expansion", "eigenvalues_complex", "eigenvalues_real",
    "eigenvectors_complex", "eigenvectors_real",
    "elementwise_integration", "feedback_controller_design",
    "fft_cmplx_scipy_fftpack", "fft_convolution",
    "fft_real_scipy_fftpack", "firls", "generalized_eigenvalues_complex",
    "generalized_eigenvalues_real", "generalized_eigenvectors_complex",
    "generalized_eigenvectors_real", "graph_coloring_assign",
    "graph_global_efficiency", "graph_isomorphism", "graph_laplacian",
    "group_lasso", "gzip_compression", "integer_factorization",
    "job_shop_scheduling", "kalman_filter", "kd_tree",
    "kernel_density_estimation", "kmeans", "ks_test_2samp", "l0_pruning",
    "l1_pruning", "lasso", "least_squares", "linear_system_solver",
    "lp_box", "lp_centering", "lp_mdp", "lti_simulation",
    "lu_factorization", "lyapunov_stability", "markowitz",
    "matrix_completion", "matrix_exponential",
    "matrix_exponential_sparse", "matrix_multiplication", "matrix_sqrt",
    "max_clique_cpsat", "max_common_subgraph", "max_flow_min_cost",
    "max_independent_set_cpsat", "max_weighted_independent_set",
    "min_dominating_set", "min_weight_assignment",
    "minimum_spanning_tree", "minimum_volume_ellipsoid",
    "multi_dim_knapsack", "nmf", "ode_brusselator", "ode_fitzhughnagumo",
    "ode_hires", "ode_hodgkinhuxley", "ode_lorenz96_nonchaotic",
    "ode_lotkavolterra", "ode_nbodyproblem", "ode_seirs",
    "ode_stiff_robertson", "ode_stiff_vanderpol", "odr",
    "optimal_advertising", "outer_product", "pagerank", "pca",
    "pde_burgers1d", "pde_heat1d", "polynomial_mixed", "polynomial_real",
    "power_control", "procrustes", "psd_cone_projection", "qp",
    "qr_factorization", "quantile_regression", "queens_with_obstacles",
    "queuing", "qz_factorization", "randomized_svd", "rbf_interpolation",
    "robust_kalman_filter", "robust_linear_program",
    "rocket_landing_optimization", "rotate_2d", "set_cover",
    "set_cover_conflicts", "sha256_hashing", "shift_2d",
    "shortest_path_dijkstra", "sinkhorn", "sparse_eigenvectors_complex",
    "sparse_lowest_eigenvalues_posdef",
    "sparse_lowest_eigenvectors_posdef", "sparse_pca",
    "spectral_clustering", "stable_matching", "svd", "svm",
    "sylvester_solver", "tensor_completion_3d", "toeplitz_solver", "tsp",
    "two_eigenvalues_around_0", "unit_simplex_projection", "upfirdn1d",
    "vector_quantization", "vectorized_newton", "vehicle_routing",
    "vertex_cover", "voronoi_diagram", "wasserstein_dist",
    "water_filling", "zoom_2d",
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
    # Two more stdlib modules upstream's task files import, found the same way
    # `__future__` was and costing the same thing: the reference does not pass
    # the gate, so the root node is refused and the task vanishes from the
    # runnable set without anything saying why. `contextlib` blocks
    # `polynomial_real`, `numbers` blocks `vectorized_newton`. A sweep of all
    # 154 task files for imports the allowlist rejects now says these were the
    # last two that plain stdlib was hiding.
    "contextlib",
    "numbers",
    "copy",
    # Nine of AlgoTune's convex-programming tasks -- and every task EvoMem
    # selected -- have a reference that is a cvxpy model. Without it those tasks
    # are not merely unrunnable, they are unspeakable: the reference does not
    # import, so `prepare_suite` fails before a tree exists. It buys eight tasks
    # here and it is the only entry in this set that is not either numpy/scipy
    # or something upstream's own task files import.
    "cvxpy",
    "dataclasses",
    "enum",
    "functools",
    "heapq",
    "itertools",
    "cython",
    # `jax==0.5.3` is in AlgoTune's own requirements.txt, and 54 of upstream's
    # 2595 published solvers import it -- nine of them on
    # `fft_cmplx_scipy_fftpack` alone. Leaving it out does not make the benchmark
    # harder, it makes it a different one, which is the argument that put numba
    # and cython here. It is also how OpenEvolve's published AlgoTune example
    # reaches 321x on `polynomial_real`, so a comparison against that number
    # without it would be measuring the allowlist.
    #
    # Checked inside the sandbox under the limits the runner really sets
    # (4 GB address space, 64 processes, --unshare-net, --clearenv): jax imports
    # in 0.4s, a jitted 1500x1500 SVD compiles and runs in 0.67s, and
    # `jnp.roots(400)` takes 0.37s. XLA does not need the network and does not
    # blow the address space.
    "jax",
    "logging",
    "math",
    "numba",
    "numpy",
    "operator",
    "random",
    "scipy",
    "statistics",
    "string",
    # `polynomial_real`'s reference limits BLAS threads around its own root-find,
    # which is the one thing standing between this port and all eight tasks
    # OpenEvolve's published AlgoTune example runs. A pure-Python package on
    # AlgoTune's own dependency list, and it has to reach `dist-packages` rather
    # than the user site or the sandbox bind cannot see it.
    "threadpoolctl",
    "typing",
    "warnings",
    # ---------------------------------------------------------------------
    # The rest of AlgoTune's own dependency list, read off `requirements.txt`
    # and ranked by what it unlocks rather than guessed at. Each of these is
    # the *sole* blocker for tasks this port could otherwise not name:
    #
    #   ortools        13 tasks    networkx        9 tasks
    #   sklearn         8 tasks    pysat           2 tasks
    #   sympy           2 tasks    ot (POT)        2 tasks
    #   faiss           2 tasks    mpmath          1 task
    #
    # and they are what upstream's own solvers reach for: 182 of its 2595
    # published solutions import ortools, 81 sklearn, 46 networkx, 50 faiss.
    # Every one verified to import *and solve* inside the sandbox under the
    # runner's real limits -- OR-Tools needed `dateutil` in `dist-packages`
    # rather than the user site, the same trap OSQP hit with `jinja2`.
    "faiss",
    "hdbscan",
    "mpmath",
    "networkx",
    "ortools",
    "ot",
    "pysat",
    "sklearn",
    "sympy",
    # cvxpy's solver backends, so a candidate can call one directly instead of
    # paying for cvxpy's modelling layer -- which is exactly the move this
    # benchmark rewards, and which 58 of upstream's solvers make.
    "ecos",
    "highspy",
    "osqp",
    # Used by upstream's crypto and hashing tasks and already installed here.
    "cryptography",
    # Stdlib upstream's own solvers import, free and previously just missing.
    # `ast` is here because a reference parses source with it, and parsing is
    # not executing -- `compile`, `eval` and `exec` are forbidden names and stay
    # forbidden.
    "ast",
    "base64",
    "gzip",
    "hashlib",
    "hmac",
    "json",
    "time",
    "zlib",
}

#: Deliberately *not* allowed, though upstream's references use them.
#:
#: ``os`` and ``sys``. Two crypto tasks -- ``aes_gcm_encryption`` and
#: ``chacha_encryption`` -- call ``os.urandom`` to make a key, and that is all
#: they want. But the gate's forbidden-name check is by *name* (``eval``,
#: ``exec``, ``getattr``), so it cannot see ``os.system`` or ``os.popen``:
#: allowing the module to get ``urandom`` allows shelling out with it. The
#: sandbox is the real boundary and would hold, but the gate exists so that the
#: common accidents fail in-process with a readable message, and widening it for
#: two tasks whose speedup is hardware-bound anyway is a bad trade. ``torch`` is
#: absent for a different reason: it is the sole blocker for nothing.
NOT_ALLOWED_ON_PURPOSE = ("os", "sys", "torch")

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
#:
#: Ten because that is AlgoTune's own per-instance rule ("your function can run
#: for at most 10x the reference runtime for that instance"), and a limit this
#: port sets higher would let a candidate be counted that upstream's harness
#: would have cut off.
SLOW_FACTOR = 10.0

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
    want_profile: bool = False,
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
        # And `try: from x import y / except: y = None`, which is how an optional
        # dependency is bound and how `polynomial_real`'s own reference binds
        # `threadpool_limits`. Refused, the reference does not pass its own gate.
        allow_top_level_try=True,
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
            "profile": bool(want_profile),
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
                    "error": _died(completed.returncode, tail, timeout),
                    "seconds": time.monotonic() - started}
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"ok": False,
                    "error": f"unparseable runner output: {lines[-1][:200]}",
                    "seconds": time.monotonic() - started}


#: What the kernel did to a candidate that never printed anything, in words.
#: A compiled fixed-step loop that picks its step badly does not fail, it runs
#: away -- and until this existed the search was told "no runner output
#: (rc=152)", which names the signal in the one encoding nothing reads. The
#: distinction matters to the next prompt: a program killed for CPU should
#: shrink its work, one killed for memory should stop allocating, and one that
#: died some other way should be debugged.
_SIGNALS = {
    9: ("SIGKILL", "the sandbox killed it -- usually the memory limit"),
    11: ("SIGSEGV", "it crashed in native code"),
    24: ("SIGXCPU", "it exceeded the CPU limit"),
    25: ("SIGXFSZ", "it tried to write a file larger than the limit"),
}


def _died(returncode: int, tail: str, timeout: float) -> str:
    """Turn an exit status into something the next mutation prompt can act on."""
    signal_number = -returncode if returncode < 0 else (
        returncode - 128 if returncode > 128 else 0)
    named = _SIGNALS.get(signal_number)
    if named:
        name, why = named
        hint = ""
        if signal_number == 24:
            hint = (f" -- the whole problem set, including compiling, has "
                    f"{timeout:.0f} CPU-seconds. A fixed-step method that "
                    f"chose too small a step will do this")
        return f"killed by {name}: {why}{hint}. {tail}".strip()
    return f"no runner output (rc={returncode}): {tail}"


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
    want_profile: bool = False,
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

    profile_text = ""
    for index, shard in enumerate(shards):
        payload = run_candidate(
            code, suite=suite, shard=shard, timeout=timeout, repeats=repeats,
            problem_seconds=problem_seconds, max_length=max_length,
            # Profiled once per evaluation, on the first shard. Every shard would
            # produce the same table for the same code at a real cost.
            want_profile=want_profile and index == 0)
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
            profile_text = profile_text or str(result.get("profile") or "")
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
            "profile": profile_text,
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


def _eval_block(metrics: Dict[str, Any]) -> str:
    """AlgoTuner's own post-`eval` summary, in its own words and order.

    Upstream's `MessageWriter.format_evaluation_result_from_raw` prints exactly
    this shape after every evaluation, and it is the only quantitative feedback
    its agent gets between edits. Matched line for line so the two systems'
    models are reading the same report.

    One deviation, and it is a correction rather than a liberty: the counts are
    printed beside the percentages. Upstream evaluates 100 instances, where
    "Invalid Solutions: 50%" is a rate; here a scoring set is a handful, where
    the same string would be one failure out of two dressed up as a statistic.
    """
    total = int(metrics.get("problems") or 0)
    valid = int(metrics.get("valid_problems") or 0)
    speedup = metrics.get("speedup")
    shown = f"{float(speedup):.3f}x" if speedup is not None else "N/A"
    if not total:
        return f"Speedup: {shown}\n  (Speedup = Baseline Time / Your Time; Higher is better)"
    invalid = total - valid
    return (
        f"Speedup: {shown}\n"
        f"  (Speedup = Baseline Time / Your Time; Higher is better)\n"
        f"\n"
        f"  Valid Solutions: {100.0 * valid / total:.0f}% ({valid}/{total})\n"
        f"  Invalid Solutions: {100.0 * invalid / total:.0f}% ({invalid}/{total})\n"
        f"  Timeouts: 0% (0/{total})"
    )


def _invalid_examples(metrics: Dict[str, Any], limit: int = 3) -> str:
    """Upstream's `Invalid Example #i` block, carrying this port's own detail.

    AlgoTuner shows up to three rejected instances with the source context from
    `is_solution`. That context is only available to it because its checker
    raised; a checker that simply returns False leaves nothing to print. This
    reports the distance instead -- which upstream has no equivalent of, and
    which is the one thing that separates "structurally wrong" from "a factor of
    three short on tolerance".
    """
    rows = [row for row in (metrics.get("slowest") or []) if row.get("speedup") is None]
    if not rows:
        return ""
    lines = ["Snapshot not saved - invalid solutions present", ""]
    for index, row in enumerate(rows[:limit], start=1):
        lines.append(f"Invalid Example #{index}:")
        lines.append("Error in 'is_solution':")
        lines.append(f"  {row.get('error') or 'is_solution returned False'}")
        lines.append("")
    return "\n".join(lines)


def _timing_report(metrics: Dict[str, Any], limit: int = 3) -> str:
    """Per-problem timings, which upstream's summary does not carry.

    Kept because the mean alone cannot separate a program that is uniformly
    1.2x from one that is 5x on two problems and 0.3x on a third, and those need
    opposite next moves. The seed is shown so a failure is reproducible.
    """
    rows = [row for row in (metrics.get("slowest") or []) if row.get("speedup") is not None]
    if not rows:
        return ""
    lines = ["Per-problem timing:"]
    for row in rows[:limit]:
        lines.append(f"  seed {row['seed']}: reference {row['baseline_ms']} ms, "
                     f"yours {row['candidate_ms']} ms, speedup {row['speedup']}x")
    return "\n".join(lines)


def _profile_block(metrics: Dict[str, Any]) -> str:
    """The `line_profiler` table, which is upstream's `profile` command."""
    text = str(metrics.get("profile") or "").strip()
    if not text:
        return ""
    return ("Line-level profile of your previous solve (25 most expensive lines, "
            "milliseconds):\n" + text)


#: The package list, in upstream's own shape. `base_interface.py` substitutes the
#: ` - placeholder` line in its system message with one ` - {name}` bullet per
#: installed extra and stops there -- no gloss on what any of them is or when to
#: reach for it. This reproduces that, which means a bare name has to carry the
#: whole hint, exactly as it does upstream.
PACKAGES = """ - numpy
 - scipy
 - numba
 - cython
 - cvxpy
 - jax
 - scikit-learn
 - networkx
 - ortools
 - sympy
 - mpmath
 - POT
 - python-sat
 - faiss
 - hdbscan"""

#: The same list with each name labelled -- ``--packages labelled``, and a
#: deliberate deviation from upstream recorded in every result file that uses it.
#:
#: It says what each library *is* and how it is invoked. It does not say when to
#: reach for one, what to point it at, or which of them tends to win: that was
#: :data:`ACCELERATION_TIPS`, it was worth 3/8 draws reaching for a compiler
#: against 0/8, and it is gone because a search told which lever to pull is not
#: measuring what AlgoTuner measured.
#:
#: The gap this closes is narrower and more defensible. Upstream's bare list is
#: fine for an agent with ~100 turns to discover that `numba` is a compiler by
#: trying it; one rewrite per node has no such turn to spend. Measured on
#: ``polynomial_real``, where the compiled direction is worth 70x-138x upstream:
#: ten draws under the bare list produced seven programs that re-wrapped
#: ``np.roots``, one numba attempt that failed to compile, and no jax at all.
LABELLED_PACKAGES = """ - numpy
 - scipy
 - numba -- a just-in-time compiler. `@numba.njit` on a function compiles it to
   native code on its first call.
 - cython -- an ahead-of-time compiler. `cython.inline("<C-like Python>", a=...)`
   compiles a snippet at import and returns the compiled callable.
 - jax -- an array library with a tracing JIT. `@jax.jit` compiles a function
   through XLA; `jax.numpy` mirrors numpy's API.
 - cvxpy -- convex modelling, with the ECOS, OSQP, HiGHS and SCS solvers behind
   it, each also importable directly.
 - scikit-learn, networkx, ortools, sympy, mpmath, POT, python-sat, faiss,
   hdbscan -- machine learning, graphs, constraint and integer programming,
   symbolic and arbitrary-precision arithmetic, optimal transport, SAT,
   similarity search, clustering."""

#: ``bare`` is upstream's own list and the default; ``labelled`` is the above.
PACKAGE_STYLES = ("bare", "labelled")

#: Upstream's `AlgoTuner/messages/initial_system_message.txt`, verbatim except
#: where this port's contract genuinely differs. Two things differ and both are
#: structural rather than editorial:
#:
#: 1. **A module-level `solve(problem)` rather than a `Solver` class.** ERA's
#:    contract is one function per program and its gate checks for one.
#: 2. **One rewrite per turn rather than the `edit`/`eval`/`revert` command
#:    loop.** A flat tree search has no conversation to run commands in, so the
#:    command block, the budget messaging and the linter/`.pyx` notes that only
#:    describe that loop are dropped rather than paraphrased.
#:
#: Everything else is upstream's wording: the 10x rule, "Compilation time of your
#: init function will not count towards your function's runtime", "Be creative
#: and optimize your approach!", and the GOALS paragraph.
#:
#: **It says nothing about how to make code fast, and that is the point.**
#: Upstream's agent is told the setting and the rules and left to it -- it
#: reaches for numba on 24% of tasks with no encouragement at all, over ~100
#: turns of watching its own edits fail. An earlier version of this file carried
#: a block naming four acceleration techniques, and a number measured against
#: that prompt is not comparable with upstream's. It is gone.
SETTING = """SETTING:
You're an autonomous programmer tasked with solving a specific problem. You are
to write a single Python function, and you will be evaluated based on the
best-performing piece of code you produce.
Apart from the default Python packages, you have access to the following
additional packages:
{packages}

YOUR TASK:
Your objective is to define a module-level function in `solver.py`:
```
def solve(problem):
    \"\"\"Your implementation goes here.\"\"\"
    ...
```

IMPORTANT: Compilation time of your init function will not count towards your
function's runtime.

This `solve` function will be the entrypoint called by the evaluation harness.
Strive to align your implementation as closely as possible with the desired
performance criteria. For each instance, your function can run for at most 10x
the reference runtime for that instance. Strive to have your implementation run
as fast as possible, while returning the same output as the reference function
(for the same given input). Be creative and optimize your approach!

**TIPS:**
Do not put an if __name__ == "__main__": block in your code, as it will not be
ran (only the solve function will).

**GOALS:**
Your primary objective is to optimize the `solve` function to run as as fast as
possible, while returning the optimal solution.
You will receive better scores the quicker your solution runs, and you will be
penalized for exceeding the time limit or returning non-optimal solutions."""


def mutation_prompt(
    parent: Any,
    *,
    suite: Suite,
    timeout: float = 300.0,
    repeats: int = REPEATS,
    packages: str = "bare",
) -> str:
    """One rewrite of the parent program, in AlgoTuner's own framing.

    Upstream's agent sees: its system message, the task description, and after
    every `eval` a speedup summary, the invalid examples, and -- on demand -- a
    line profile. This assembles the same things around the parent's code, which
    is the closest a one-rewrite-per-node search can get to that loop without
    becoming a different algorithm.

    It names no acceleration technique, because upstream names none. There was a
    ``--prompt guided`` here that did, and it worked -- placed against the
    evidence rather than at the top of the prompt it took draws that reached for
    a compiler from 0/8 to 3/8, and on ``ode_stiff_vanderpol`` it is what
    produced 2785x. It is still gone: a search told which levers to pull is not
    measuring the same thing as an agent that had to find them, and this port
    exists to be compared with upstream.

    ``packages="labelled"`` is the one step back towards that, and a smaller one:
    :data:`LABELLED_PACKAGES` says what each library *is*, not when to use it.
    It is off by default and recorded in the result file when it is on, because
    it is still a deviation -- upstream ships a bare list of names.
    """
    if packages not in PACKAGE_STYLES:
        raise ValueError(f"packages must be one of {PACKAGE_STYLES}")
    blocks = [
        SETTING.format(packages=PACKAGES if packages == "bare" else LABELLED_PACKAGES),
        "**TASK DESCRIPTION:**\n" + suite.description,
        (f"Problems are generated by the task's own generator at n = {suite.n}, "
         f"and your output is checked by the task's own `is_solution`. Timing is "
         f"the minimum of {repeats} runs after a discarded warm-up, averaged over "
         f"the problem set. {timeout:.0f} seconds of CPU for the whole set, "
         f"including any compilation."),
    ]
    blocks.append("**EVALUATION OF YOUR PREVIOUS CODE:**\n" + _eval_block(parent.metrics))
    for optional in (_invalid_examples(parent.metrics),
                     _timing_report(parent.metrics),
                     _profile_block(parent.metrics)):
        if optional:
            blocks.append(optional)
    blocks.append("**YOUR PREVIOUS CODE:**\n```python\n" + parent.code.rstrip()
                  + "\n```")
    blocks.append(
        "Write the full contents of `solver.py` -- a complete, runnable module "
        "defining `solve(problem)`, with its imports. Return ONLY the python "
        "code, in a single fenced block. Do not read or reconstruct the task's "
        "reference implementation to call it; write the computation.")
    return "\n\n".join(blocks) + "\n"


def repair_prompt(parent: Any, code: str, error: str, attempt: int,
                  *, suite: Suite) -> str:
    """Hand a failed draw back with the failure attached, and ask for a fix.

    This is the message AlgoTuner's agent gets for free by being a conversation:
    it edits, the harness answers, it edits again. A tree whose generator sees
    one program and one number has no equivalent, so a failed expansion becomes
    a node scoring `-inf` that nothing will ever select again -- and on this
    benchmark the direction that wins is the same direction whose first attempt
    usually fails to compile, misses the tolerance, or runs away. Retrying is
    not politeness towards the model; it is the difference between exploring
    that direction and abandoning it.

    Deliberately short, and deliberately not a re-run of the whole mutation
    prompt: the task, the contract and the tips have already been read once in
    this exchange. What is new is the program that failed and why.
    """
    return f"""That program does not work. Here it is, and here is what happened:

```python
{code.rstrip()}
```

{error}

Fix it. Keep the approach -- if it was going to be faster, it is still worth
having; the problem is the defect, not the idea. Common causes, in the order
they usually bite:

  - a numba `@njit` function using something nopython mode does not support
    (a Python list where an array is wanted, a closure over a changing value,
    an unsupported numpy overload);
  - a fixed-step method whose step is too large for the tolerance, or too small
    and so never finishes;
  - a result whose dtype, shape or container differs from what the reference
    returns;
  - an array modified in place that something later still needed.

If the approach cannot be made to work, say so by returning a different one.
Return ONLY the full contents of `solver.py`, in a single fenced block.
Attempt {attempt + 1}.
"""
