"""Suite, sandboxed evaluator, and prompt for the ERA LLM-SRBench task.

The public runnable example lives in :mod:`examples.era.era_llm_srbench`; the
answer format and the metrics live in :mod:`examples.era._era_srbench_expr`.
This module is the boundary between them: it downloads the benchmark, deals its
problems into shards, runs a candidate against one shard under the *same*
Bubblewrap/Seatbelt profile the other three ERA tasks use, and turns the
equations that come back into the score FUTS ranks nodes by.

What a candidate is here
------------------------
Not an equation -- a **method for finding equations**. The tree search writes
one program, and that program is asked to discover the governing equation of
every problem in a shard, each from its own training samples and its own
one-paragraph description. So the thing being optimised is a symbolic-regression
procedure, in the same sense that the integrals task optimises a quadrature rule
rather than the value of any particular integral.

That is a *different experiment* from the one LLM-SRBench's own leaderboard
runs. There, an LLM proposes hypotheses for one problem at a time, with the
data in its context, and a per-problem budget of samples. Here the model never
sees a data point: it writes the searcher, the searcher runs sandboxed against
all of them, and the score comes back. The benchmark, the splits and the metrics
are the benchmark's; the protocol is ERA's. A number produced here belongs
beside the paper's tables with that said out loud, which
``docs/algo-era.md`` does and every result file repeats.

Where the data comes from
-------------------------
The benchmark's own release (``nnheui/llm-srbench``) is a **gated** HuggingFace
dataset: the parquet metadata and the 239 MB ``lsr_bench_data.hdf5`` of samples
both 401 without a token, so a port that reached for them would work only for
whoever had already clicked through the gate. ``pkuHaowei/llm-srbench`` is an
ungated re-upload of the same benchmark with the samples inlined into parquet,
and that is what this module fetches. It is a third party's copy, so it is
checked rather than trusted: ``tests/test_era_srbench.py`` re-evaluates every
ground-truth expression that is evaluable as written against the samples shipped
beside it, which pins the data to the equations the paper published.

Two things in that copy's metadata do not survive the check, and neither one
touches scoring, which is numeric throughout:

* the 36 ``chem_react`` ground truths carry a mangled parameter (a constant
  followed by ``_z``/``_w``/``_s``), so they do not parse;
* the 44 ``phys_osc`` ground truths are symbolic templates -- ``F0``, ``beta``,
  ``omega0`` appear with no values attached -- so they do not evaluate.

The remaining 49 synthetic problems and all 111 transformed Feynman problems do
evaluate, and they reproduce their own samples. Ground-truth expressions are
carried into the result file as metadata for exactly that reason: they are what
a reader checks the benchmark with, not what a run is scored against.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agentdescent.dataloader import cache_path

from examples.era._era_srbench_expr import DIGIT_CAP, TOLERANCE, aggregate
from examples.era._era_support import sandbox_wrapper, validate_source


RUNNER = Path(__file__).with_name("_era_srbench_runner.py")
EXPR_MODULE = Path(__file__).with_name("_era_srbench_expr.py")

#: The ungated mirror, pinned to a revision rather than to `main`, so a rerun
#: fetches the same bytes. See the module docstring for why the benchmark's own
#: repository is not the source here.
MIRROR_REPO = "pkuHaowei/llm-srbench"
MIRROR_REVISION = "f4101e201952bb35cab8332b6dcf337b6df02aa8"
MIRROR_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}")

#: LLM-SRBench's own paper: arXiv:2504.10415 (ICML 2025).
BENCHMARK_PAPER = "arXiv:2504.10415"

#: Each subset, the mirror's files for it, and the problem count the paper
#: states. The counts are asserted at load time: a mirror that silently changed
#: shape would otherwise be scored as if it were the benchmark.
SUBSETS: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "lsr_synth_bio_pop_growth": (
        ("lsr_synth_bio_pop_growth/train-00000-of-00001.parquet",), 24),
    "lsr_synth_chem_react": (
        ("lsr_synth_chem_react/train-00000-of-00001.parquet",), 36),
    "lsr_synth_matsci": (
        ("lsr_synth_matsci/train-00000-of-00001.parquet",), 25),
    "lsr_synth_phys_osc": (
        ("lsr_synth_phys_osc/train-00000-of-00001.parquet",), 44),
    "lsr_transform": (
        ("lsr_transform/train-00000-of-00002.parquet",
         "lsr_transform/train-00001-of-00002.parquet"), 111),
}

#: The two categories the paper draws, and the union.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "lsr_synth": ("lsr_synth_bio_pop_growth", "lsr_synth_chem_react",
                  "lsr_synth_matsci", "lsr_synth_phys_osc"),
    "lsr_transform": ("lsr_transform",),
    "all": tuple(SUBSETS),
}

#: Short names for the result file and the plan line.
SUBSET_LABELS = {
    "lsr_synth_bio_pop_growth": "biology (population growth)",
    "lsr_synth_chem_react": "chemistry (reaction kinetics)",
    "lsr_synth_matsci": "material science",
    "lsr_synth_phys_osc": "physics (nonlinear oscillators)",
    "lsr_transform": "transformed Feynman equations",
}

#: What a candidate may import. Wider than the integrals task's set because a
#: symbolic-regression method legitimately wants optimisers, linear algebra and
#: a seeded RNG; still narrow enough that an import of `os` or `subprocess`
#: fails in-process with a readable message rather than against the sandbox.
ALLOWED_IMPORTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "dataclasses",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "operator",
    "random",
    "re",
    "scipy",
    "statistics",
    # The prompt tells a candidate to budget its own search per problem, so it
    # has to be able to read a clock; refusing `time` would be asking for a
    # deadline to be respected and then hiding the deadline.
    "time",
    "typing",
    "warnings",
}

#: Wall-clock a candidate gets for one problem, enforced inside the runner.
PROBLEM_SECONDS = 10.0

#: Training rows handed to a candidate, 0 for all of them. LSR-Transform ships
#: 80 000 rows per problem and LSR-Synth 4 000; a cap makes the two comparable
#: in cost without changing which problems are being solved. Recorded either way.
TRAIN_POINTS = 0


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SrProblem:
    """One LLM-SRBench problem, as the search sees it."""

    problem_id: str
    subset: str
    input_vars: Tuple[str, ...]
    output_var: str
    description: str
    gt_expression: str
    train_rows: int
    test_rows: int
    ood_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "subset": self.subset,
            "input_vars": list(self.input_vars),
            "output_var": self.output_var,
            "description": self.description,
            "gt_expression": self.gt_expression,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "ood_rows": self.ood_rows,
        }


def _download(path: str, timeout: float = 900.0) -> Path:
    """Fetch one mirror file into the dataloader's cache, streaming.

    :func:`agentdescent.dataloader.fetch_bytes` is the house helper and is used
    everywhere else, but it returns the whole body: the two LSR-Transform shards
    are 172 MB and 181 MB, and holding either in memory to write it straight
    back out is a cost with nothing to buy it.
    """
    target = Path(cache_path("llm-srbench", _cache_name(path)))
    if target.exists() and target.stat().st_size > 0:
        return target
    url = MIRROR_URL.format(repo=MIRROR_REPO, revision=MIRROR_REVISION, path=path)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, \
            open(partial, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, target)
    return target


def _cache_name(path: str) -> str:
    """``lsr_transform/train-00001-of-00002.parquet`` -> ``lsr_transform-00001.parquet``.

    A one-file subset drops the index, so the cache reads as one file per subset
    where there is one.
    """
    subset, _, filename = path.partition("/")
    parts = Path(filename).stem.split("-")  # train-00000-of-00002
    if len(parts) != 4:
        return f"{subset}-{filename}"
    index, total = parts[1], parts[3]
    if total == "00001":
        return f"{subset}.parquet"
    return f"{subset}-{index}.parquet"


def _read_subset(subset: str) -> List[Tuple[SrProblem, Dict[str, np.ndarray]]]:
    """Every problem in one subset, with its samples, from the cached parquet."""
    try:
        import pyarrow.parquet as pq  # lazy: only a real run needs the data
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "reading the LLM-SRBench parquet needs pyarrow "
            "(`pip install pyarrow`)") from exc

    files, expected = SUBSETS[subset]
    loaded: List[Tuple[SrProblem, Dict[str, np.ndarray]]] = []
    for path in files:
        handle = pq.ParquetFile(_download(path))
        for group in range(handle.num_row_groups):
            # Row group at a time: a whole LSR-Transform shard materialised as
            # Python lists is several gigabytes.
            for row in handle.read_row_group(group).to_pylist():
                loaded.append(_problem_from_row(row, subset))
    if len(loaded) != expected:
        raise RuntimeError(
            f"{subset}: mirror holds {len(loaded)} problems, the benchmark has "
            f"{expected}")
    loaded.sort(key=lambda pair: pair[0].problem_id)
    return loaded


def _as_matrix(value: Any) -> np.ndarray:
    array = np.asarray([list(row) for row in value], dtype=np.float64)
    return array.reshape(array.shape[0], -1)


def _problem_from_row(row: Dict[str, Any],
                      subset: str) -> Tuple[SrProblem, Dict[str, np.ndarray]]:
    train_x = _as_matrix(row["train_input"])
    train_y = _as_matrix(row["train_output"])[:, 0]
    test_x = _as_matrix(row["test_input"])
    test_y = _as_matrix(row["test_output"])[:, 0]
    samples = {"train_x": train_x, "train_y": train_y,
               "test_x": test_x, "test_y": test_y}
    ood_rows = 0
    if row.get("ood_input") is not None and len(row["ood_input"]):
        samples["ood_x"] = _as_matrix(row["ood_input"])
        samples["ood_y"] = _as_matrix(row["ood_output"])[:, 0]
        ood_rows = int(samples["ood_x"].shape[0])
    problem = SrProblem(
        problem_id=str(row["instance_id"]),
        subset=subset,
        input_vars=tuple(str(name) for name in row["input_vars"]),
        output_var=str(row["output_vars"][0]),
        description=str(row["description"]),
        gt_expression=str(row["gt_expression"]),
        train_rows=int(train_x.shape[0]),
        test_rows=int(test_x.shape[0]),
        ood_rows=ood_rows,
    )
    if train_x.shape[1] != len(problem.input_vars):
        raise RuntimeError(
            f"{problem.problem_id}: {train_x.shape[1]} input columns for "
            f"{len(problem.input_vars)} named variables")
    return problem, samples


# --------------------------------------------------------------------------
# The suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """The shards, their sample files, and the problems each one holds.

    Same shape as the sibling tasks' suites: what the candidate reads is on
    disk, and the last ``test_shards`` are never shown to the search.
    """

    root: Path
    seed: int
    subsets: Tuple[str, ...]
    shard_paths: Tuple[Path, ...]
    shard_meta: Tuple[Path, ...]
    shard_problems: Tuple[Tuple[SrProblem, ...], ...]
    scoring_shards: int
    test_shards: int
    train_points: int = TRAIN_POINTS

    def size(self, shard: int) -> int:
        return len(self.shard_problems[shard])

    def problems(self) -> Tuple[SrProblem, ...]:
        return tuple(p for shard in self.shard_problems for p in shard)

    def test_range(self) -> Tuple[int, ...]:
        return tuple(range(self.scoring_shards,
                           self.scoring_shards + self.test_shards))

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for problem in self.problems():
            counts[problem.subset] = counts.get(problem.subset, 0) + 1
        return counts


def resolve_subsets(name: str) -> Tuple[str, ...]:
    """``lsr_synth`` / ``lsr_transform`` / ``all`` / one subset's own name."""
    if name in GROUPS:
        return GROUPS[name]
    if name in SUBSETS:
        return (name,)
    raise ValueError(
        f"unknown dataset {name!r}; choose from "
        f"{', '.join(sorted(set(GROUPS) | set(SUBSETS)))}")


def _deal(problems: Sequence[SrProblem], shards: int,
          seed: int) -> List[List[SrProblem]]:
    """Shuffle within each domain, then deal round-robin *across* the domains.

    A contiguous split would put all 36 chemistry problems in one shard, and a
    node's score would then depend on which shards the verifier happened to draw
    rather than on the program. Shuffling the whole list and dealing it would fix
    that only on average -- with four domains and eight shards, a draw that hands
    one shard no physics at all is perfectly ordinary.

    Dealing each domain's own shuffled order onto a single rotating cursor makes
    the property exact: every shard holds each domain to within one problem, and
    the shard sizes differ by at most one. That is what makes one shard's score a
    usable estimate of another's, which the whole train/held-out/test split rests
    on.
    """
    by_subset: Dict[str, List[SrProblem]] = {}
    for problem in problems:
        by_subset.setdefault(problem.subset, []).append(problem)
    rng = np.random.default_rng(seed)
    dealt: List[List[SrProblem]] = [[] for _ in range(shards)]
    cursor = 0
    for subset in sorted(by_subset):
        group = sorted(by_subset[subset], key=lambda p: p.problem_id)
        rng.shuffle(group)  # type: ignore[arg-type]
        for problem in group:
            dealt[cursor % shards].append(problem)
            cursor += 1
    return dealt


def _stratified_cap(problems: Sequence[SrProblem], limit: int,
                    seed: int) -> List[SrProblem]:
    """Take ``limit`` problems, proportionally across subsets, deterministically."""
    by_subset: Dict[str, List[SrProblem]] = {}
    for problem in problems:
        by_subset.setdefault(problem.subset, []).append(problem)
    rng = np.random.default_rng(seed + 1)
    for group in by_subset.values():
        group.sort(key=lambda p: p.problem_id)
        rng.shuffle(group)  # type: ignore[arg-type]
    chosen: List[SrProblem] = []
    cursors = {name: 0 for name in by_subset}
    while len(chosen) < limit:
        progressed = False
        for name in sorted(by_subset):
            if len(chosen) >= limit:
                break
            cursor = cursors[name]
            if cursor < len(by_subset[name]):
                chosen.append(by_subset[name][cursor])
                cursors[name] = cursor + 1
                progressed = True
        if not progressed:
            break
    chosen.sort(key=lambda p: p.problem_id)
    return chosen


def prepare_suite(
    *,
    seed: int = 0,
    shards: int = 6,
    test_shards: int = 2,
    dataset: str = "lsr_synth",
    problems: int = 0,
    train_points: int = TRAIN_POINTS,
) -> Suite:
    """Download the benchmark, choose the problems, and write the shard files.

    Written under the dataloader's cache rather than a temporary directory, and
    keyed on a fingerprint of the problems themselves: the files are inputs to a
    sandboxed process, they are reproducible from the configuration alone, and a
    rerun that re-dealt them would not be the same benchmark.
    """
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    names = resolve_subsets(dataset)
    catalogue: List[Tuple[SrProblem, Dict[str, np.ndarray]]] = []
    for subset in names:
        catalogue.extend(_read_subset(subset))
    samples = {problem.problem_id: arrays for problem, arrays in catalogue}
    chosen = [problem for problem, _ in catalogue]
    if problems:
        chosen = _stratified_cap(chosen, problems, seed)
    total = shards + test_shards
    if len(chosen) < total:
        raise ValueError(
            f"{len(chosen)} problems cannot fill {total} shards")
    dealt = _deal(chosen, total, seed)

    fingerprint = hashlib.sha256(json.dumps(
        [[p.problem_id for p in shard] for shard in dealt]
        + [[dataset, seed, int(train_points)]],
        sort_keys=True).encode("utf-8")).hexdigest()[:12]
    root = Path(cache_path("llm-srbench", f"suite-{fingerprint}"))
    root.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    metas: List[Path] = []
    for index, shard in enumerate(dealt):
        data_path = root / f"shard-{index:03d}.npz"
        meta_path = root / f"shard-{index:03d}.json"
        if not data_path.exists() or not meta_path.exists():
            payload: Dict[str, np.ndarray] = {}
            for position, problem in enumerate(shard):
                arrays = samples[problem.problem_id]
                for key, array in arrays.items():
                    if key == "train_x" and train_points:
                        array = array[:train_points]
                    elif key == "train_y" and train_points:
                        array = array[:train_points]
                    payload[f"p{position}_{key}"] = array
            np.savez(data_path, **payload)
            meta_path.write_text(
                json.dumps([problem.to_dict() for problem in shard], indent=1) + "\n",
                encoding="utf-8")
        paths.append(data_path)
        metas.append(meta_path)
    return Suite(root, seed, names, tuple(paths), tuple(metas),
                 tuple(tuple(shard) for shard in dealt), shards, test_shards,
                 train_points)


def suite_preview(suite: Suite) -> str:
    """What the model is told about the benchmark -- never a problem's answer.

    The domains, the shape of a problem and the size of the splits are the
    briefing a scientist would get. The ground-truth expressions are not in it,
    are not in what the candidate is handed, and are not in the feedback the
    search reports: a program is told which *problem ids* it did worst on and
    what their variables are, which is what a failing report would carry anyway.
    """
    counts = suite.counts()
    lines = [
        f"The benchmark is LLM-SRBench ({BENCHMARK_PAPER}). Each problem set "
        f"holds {suite.size(0)} problems drawn from these domains:",
    ]
    for subset in sorted(counts):
        lines.append(f"  - {SUBSET_LABELS.get(subset, subset)}: "
                     f"{counts[subset]} problems in this run")
    example = suite.shard_problems[0][0]
    lines.append(
        "A problem gives you a table of samples and a one-paragraph description "
        "naming the output and every input, for example:")
    lines.append("    " + example.description.replace("\n", "\n    "))
    lines.append(
        "Every problem has a closed-form ground truth built from +, -, *, /, "
        "powers and elementary functions (sin, cos, exp, log, sqrt, abs, tanh). "
        "The synthetic domains hold ODE right-hand sides with an added novel "
        "term; the transformed problems are Feynman equations rearranged into "
        "unfamiliar forms.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The seed program
# --------------------------------------------------------------------------

#: The baseline node: sparse regression over a fixed nonlinear library, which is
#: what a practitioner reaches for before reaching for an LLM (SINDy, Brunton et
#: al. 2016, is this with a domain-chosen library). It is a real method -- it
#: recovers several of the synthetic right-hand sides outright -- and it has the
#: property a root node needs: its ceiling is its library, so the search has
#: somewhere to go that is not tuning.
INITIAL_PROGRAM = '''"""Baseline: sequentially thresholded least squares over a fixed library."""
import numpy as np


def _terms(x, names):
    """Candidate basis functions, as (values, source text) pairs."""
    rows = x.shape[0]
    columns = [(np.ones(rows), "1")]
    for index, name in enumerate(names):
        v = x[:, index]
        columns.append((v, name))
        columns.append((v * v, f"{name}**2"))
        columns.append((v ** 3, f"{name}**3"))
        columns.append((np.sin(v), f"sin({name})"))
        columns.append((np.cos(v), f"cos({name})"))
        if np.all(v > 1e-9):
            columns.append((np.log(v), f"log({name})"))
            columns.append((np.sqrt(v), f"sqrt({name})"))
        if np.min(np.abs(v)) > 1e-6:
            columns.append((1.0 / v, f"1/({name})"))
        if np.max(np.abs(v)) < 30.0:
            columns.append((np.exp(-v), f"exp(-({name}))"))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            columns.append((x[:, i] * x[:, j], f"{names[i]}*{names[j]}"))
    return columns


def _threshold(design, y, cut, sweeps=8):
    coefficients, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(sweeps):
        small = np.abs(coefficients) < cut
        if small.all() or not small.any():
            break
        coefficients[small] = 0.0
        keep = ~small
        fitted, _r, _k, _s = np.linalg.lstsq(design[:, keep], y, rcond=None)
        coefficients = np.zeros_like(coefficients)
        coefficients[keep] = fitted
    return coefficients


def discover(x, y, spec):
    names = list(spec["input_vars"])
    columns = _terms(x, names)
    design = np.column_stack([values for values, _text in columns])
    labels = [text for _values, text in columns]
    finite = np.all(np.isfinite(design), axis=1) & np.isfinite(y)
    design, target = design[finite], y[finite]
    scale = np.maximum(np.abs(design).max(axis=0), 1e-12)
    design = design / scale

    best, best_cost = None, np.inf
    spread = float(np.mean((target - target.mean()) ** 2)) or 1.0
    for cut in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.2):
        coefficients = _threshold(design, target, cut)
        residual = target - design @ coefficients
        used = int(np.count_nonzero(coefficients))
        cost = float(np.mean(residual ** 2)) / spread * (1.0 + 0.02 * used)
        if np.isfinite(cost) and cost < best_cost:
            best, best_cost = coefficients / scale, cost

    if best is None:
        return repr(float(np.mean(y)))
    parts = [f"({float(value)!r})*{label}"
             for value, label in zip(best, labels) if value != 0.0]
    equation = " + ".join(parts) if parts else repr(float(np.mean(y)))

    # `spec["evaluate"]` is the grader's own parser. Checking the answer with it
    # before returning it means a form this method cannot express is caught here,
    # where there is still something to fall back to, rather than scoring zero.
    try:
        check = spec["evaluate"](equation, x[:32])
        if not np.all(np.isfinite(check)):
            raise ValueError("equation is not finite on its own training data")
    except Exception:
        equation = repr(float(np.mean(y)))
    return equation
'''

INITIAL_SUMMARY = "sequentially thresholded least squares over a fixed library"


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "mean_digits": None,
        # `-inf` is upstream's failure sentinel, and the tree appends the node
        # anyway. Keeping it keeps node ordering identical to `futs.search`.
        "score": -math.inf,
        "median_nmse": None,
        "acc_0.1": None,
        "solved": 0,
        "problems": 0,
        "worst": [],
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
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    nproc_limit: int = 64,
) -> Dict[str, Any]:
    """Execute one candidate against one shard and return the runner's payload."""
    valid, reason = validate_source(
        code,
        max_length,
        entrypoint="discover",
        allowed_imports=ALLOWED_IMPORTS,
        literal_top_level=False,
    )
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix="era-srbench-") as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        # Copied into the scratch bind rather than read where the suite lives:
        # the Bubblewrap profile mounts a fresh tmpfs over `/tmp`, so a suite
        # under `/tmp` -- a test fixture, or a host with `XDG_CACHE_HOME=/tmp` --
        # is invisible inside the sandbox and the candidate gets blamed for a
        # FileNotFoundError.
        data = Path(scratch) / "shard.npz"
        meta = Path(scratch) / "shard.json"
        data.write_bytes(suite.shard_paths[shard].read_bytes())
        meta.write_bytes(suite.shard_meta[shard].read_bytes())
        command, env = sandbox_wrapper(
            [
                str(RUNNER),
                str(candidate),
                "--samples", str(data),
                "--problems", str(meta),
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
                timeout=timeout + 30.0, env=env, cwd=scratch)
        except subprocess.TimeoutExpired:
            return {"ok": False,
                    "error": f"timeout after {timeout + 30.0:.0f}s",
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
    problem_seconds: float = PROBLEM_SECONDS,
    max_length: int = 20_000,
    worst_reported: int = 5,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards, pooled over problems.

    A problem whose equation was rejected, raised, or ran out of time scores
    zero and the shard keeps going. A *program* that could not be imported, or
    that has no ``discover``, fails the whole evaluation: there is nothing to
    give partial credit to. That is the same line the sibling tasks draw between
    "the program is broken" and "the program is wrong".
    """
    rows: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    seconds = 0.0
    unavailable: List[str] = []

    for shard in shards:
        payload = run_candidate(
            code, suite=suite, shard=shard, timeout=timeout,
            problem_seconds=problem_seconds, max_length=max_length)
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
        for result, problem in zip(results, problems):
            rows.append(result)
            digits = (result["id"]["digits"] if result.get("id") is not None
                      else 0.0)
            detail.append({
                "shard": shard,
                "problem_id": problem.problem_id,
                "subset": problem.subset,
                "variables": list(problem.input_vars),
                "digits": round(float(digits), 3),
                "equation": (result.get("equation") or "")[:240],
                "error": str(result.get("error") or ""),
                "seconds": round(float(result.get("seconds") or 0.0), 2),
            })

    if not rows:
        return False, _zero_metrics("no problems scored"), "no problems scored"
    pooled = aggregate(rows)
    if pooled["problems"] == 0:
        # Every problem failed. That is a legitimate node, not a broken run:
        # it scores the floor and the tree keeps it as something to improve on.
        pooled["mean_digits"] = 0.0
    worst = sorted(detail, key=lambda row: (row["digits"], row["problem_id"]))
    metrics: Dict[str, Any] = dict(pooled)
    metrics.update({
        # FUTS maximises and more digits is better, so the score is the metric
        # itself -- no sign flip, unlike the RMSE task.
        "score": float(pooled["mean_digits"]),
        "worst": worst[:worst_reported],
        "per_problem": detail,
        "seconds": seconds,
        "limits_unavailable": unavailable,
        "error": "",
    })
    return True, metrics, ""


def framework_score(metrics: Dict[str, Any]) -> float:
    """Map mean digits onto AgentDescent's [0, 1] reward, order-preserving.

    A plain rescale by the cap, as in the integrals task: the engine's
    acceptance gate and the tree's node ordering then rank candidates
    identically, and a port where those disagreed would be selecting against
    its own acceptance rule.
    """
    value = metrics.get("mean_digits")
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value) / DIGIT_CAP))


# --------------------------------------------------------------------------
# The mutation prompt
# --------------------------------------------------------------------------

SYSTEM_PREAMBLE = """You are an expert in symbolic regression and scientific
machine learning, and an expert Python programmer. Your task is to write Python
code that discovers closed-form equations from data. Return ONLY the python
code."""


def _failure_report(metrics: Dict[str, Any], limit: int = 5) -> str:
    """The worst problems, by id and variables -- never by ground truth.

    Upstream's prompt shows a single score and nothing else. A
    symbolic-regression method fails *per problem*, and a search told only its
    mean cannot tell a method that is mediocre everywhere from one that is exact
    on ten problems and hopeless on two. What the ground truth was stays out:
    the feedback has to be usable as method design, not as an answer key.
    """
    rows = metrics.get("worst") or []
    if not rows:
        return ""
    lines = [f"Worst problems in the last evaluation (score out of {DIGIT_CAP:.0f}):"]
    for row in rows[:limit]:
        note = f", {row['error']}" if row.get("error") else ""
        equation = row.get("equation") or ""
        shown = f" -> `{equation[:110]}`" if equation else ""
        lines.append(
            f"  {row['problem_id']} ({', '.join(row['variables'])}): "
            f"{row['digits']} digits in {row['seconds']}s{note}{shown}")
    return "\n".join(lines)


def mutation_prompt(
    parent: Any,
    *,
    preview: str,
    timeout: float = 300.0,
    problem_seconds: float = PROBLEM_SECONDS,
    functions: Sequence[str] = (),
) -> str:
    """`PlaygroundGenerator.__call__`, re-pointed at equation discovery.

    Upstream's shape is kept exactly: system preamble, task, data preview,
    metric, the parent's score, the parent's code, a numbered contract, and a
    block of constraints. Only the contents are this task's.
    """
    score = parent.metrics.get("mean_digits")
    shown = f"{float(score):.4f}" if score is not None else "failed to run"
    accuracy = parent.metrics.get("acc_0.1")
    median = parent.metrics.get("median_nmse")
    stats = []
    if accuracy is not None:
        stats.append(f"Acc(0.1) = {100.0 * float(accuracy):.1f}% of problems")
    if median is not None and math.isfinite(float(median)):
        stats.append(f"median NMSE = {float(median):.3e}")
    stats_line = ("It also scored: " + ", ".join(stats) + ".") if stats else ""
    failures = _failure_report(parent.metrics)
    allowed = ", ".join(functions)
    imports = ", ".join(sorted(ALLOWED_IMPORTS))
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

Write a general-purpose symbolic-regression method: a program that is handed one
scientific dataset at a time and returns the closed-form equation behind it.

{preview}

The metric per problem is min({DIGIT_CAP:.0f}, -log10(NMSE)) on held-out test
samples the program never sees, where NMSE is the mean squared error of your
equation divided by the variance of the target. A problem whose equation is
rejected, raises, produces a non-finite value anywhere on the test set, or
overruns its time scores 0. The reported score is the mean over the problem set;
higher is better, {DIGIT_CAP:.0f} is the maximum. Acc(0.1) -- the share of
problems whose worst relative error on the test set is under 10% -- is reported
beside it.

The previous solution scored: {shown}
{stats_line}
{failures}

Previous Solution Code:
```python
{parent.code}
```

Please generate a NEW, IMPROVED Python function named `discover` that:
1. Has the signature `discover(x, y, spec)` and returns a STRING.
2. `x` is a float64 numpy array of shape (n_samples, n_inputs), `y` is a float64
   numpy array of shape (n_samples,). `spec` is a dict with `input_vars` (the
   column names of `x`, in order), `output_var`, `description` (the problem's own
   natural-language statement, which names the science), `seconds`, and
   `evaluate`.
3. Returns the discovered equation as an expression in the names in
   `spec["input_vars"]` -- for example `"1.7*P - 0.02*P**2 + sin(t)"`. It is
   parsed, not executed: the only things allowed in it are numeric constants,
   those variable names, `pi`, `e`, the operators + - * / **, and these
   functions: {allowed}. Anything else -- a comparison, an index, an unknown
   name -- is rejected and that problem scores zero.
4. Scores its own candidate forms with `spec["evaluate"](expression, data)`,
   which is the grader's own parser: it takes an expression string and a
   (n, n_inputs) array in `input_vars` order, returns a length-n float64 array,
   and raises ValueError on anything the grammar in (3) does not accept. Use it
   rather than building the expression twice -- what it accepts is exactly what
   your answer is allowed to be. `eval`, `exec` and `compile` are rejected
   before your program ever runs.
5. Fits the numeric constants of whatever form it proposes. An equation with the
   right shape and untuned constants scores near zero here.
6. Is one method that works across every domain above, not a dispatch on the
   problem id. Reading `spec["description"]` to bias the search towards forms
   that are plausible for that science is exactly the point; hard-coding an
   answer for a particular problem is not.

Your code must look like this:
```python
import numpy as np
# ... other imports

def discover(x, y, spec):
    # ... propose candidate forms, fit their constants, keep the best ...
    values = spec["evaluate"]("1.7*P - 0.02*P**2", x)   # score a form
    return "..."
```
Provide the full, runnable code including imports.

IMPORTANT CONSTRAINTS:
1. {problem_seconds:.0f} seconds of wall-clock per problem, {timeout:.0f} seconds
   for the whole problem set. A problem that overruns is interrupted and scores
   zero, so budget your search and keep the best answer found so far.
2. These are the only imports a static gate accepts, and a refused program
   scores nothing at all: {imports}. `eval`, `exec`, `compile` and `sympy` are
   refused too -- build candidate models as closures, or evaluate them with
   `spec["evaluate"]`. There is no network and no filesystem; do not read or
   write files, and do not set any thread count above 1.
3. Seed every random number generator you use, from a constant. Two runs of this
   program on the same data must return the same equation.
4. Guard against overfitting: a long equation that fits the training samples and
   misses the held-out ones scores worse than a short one that generalises. Hold
   part of the training data back for your own selection.
--- END PROMPT ---
"""
