"""Data, sandbox, evaluator, and prompt helpers for the ERA port.

The public runnable example lives in :mod:`examples.era.era_empirical_software`.
Keeping the generated-code boundary here makes the AST gate and the sandbox
runner independently testable without a second copy of the search loop.

Upstream reference: `google-research/era` at commit
`b836730b5c000526af95116b1d0e2c60c8cf0a10`, `implementation/playground_s3e1.py`
plus `implementation/futs.py`.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from agentdescent.dataloader import cache_path, fetch_text


UPSTREAM_COMMIT = "b836730b5c000526af95116b1d0e2c60c8cf0a10"
DATA_URL = (
    "https://raw.githubusercontent.com/google-research/era/"
    f"{UPSTREAM_COMMIT}/implementation/data/playground-series-s3e1/train.csv"
)
#: Kaggle Playground Series S3E1 -- a synthetic California-housing regression.
#: The target column and the 80/20 head/tail split below are upstream's
#: `prepare_data()`, not this port's choices.
TARGET_COLUMN = "MedHouseVal"
TRAIN_FRACTION = 0.8
RUNNER = Path(__file__).with_name("_era_runner.py")

#: Wide enough for the benchmark, and no wider. Upstream ships no gate at all --
#: `sandbox.py` is an abstract class that raises -- so every name here is this
#: port's decision, and the trade is explicit: admitting scikit-learn admits a
#: package that can spawn processes and read files, which is why the sandbox
#: profile rather than this set is the actual boundary.
ALLOWED_IMPORTS = {
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "numpy",
    "pandas",
    "random",
    "scipy",
    "sklearn",
    "statistics",
    "typing",
    "warnings",
}
FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}

#: Upstream's `initial_code` from `run_experiment`, verbatim.
INITIAL_PROGRAM = '''import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_and_predict(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop('MedHouseVal', axis=1)
    y = train['MedHouseVal']

    model = LinearRegression()
    model.fit(X, y)

    return model.predict(test)
'''


@dataclass
class Program:
    """One node's payload: the genome plus where it came from and how it did."""

    program_id: str
    iteration: int
    parent_id: Optional[str]
    code: str
    change_summary: str
    metrics: Dict[str, Any]
    valid: bool
    error: str = ""


def program_id(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# The AST gate
# --------------------------------------------------------------------------


def validate_source(
    source: str,
    max_length: int = 20_000,
    *,
    entrypoint: str = "train_and_predict",
    allowed_imports: Optional[Set[str]] = None,
    literal_top_level: bool = True,
) -> Tuple[bool, str]:
    """Reject what the sandbox should never have to contain in the first place.

    Deliberately *not* as strict as the OpenEvolve port's gate: that one allows
    six standard-library modules and this one has to allow scikit-learn, so it
    cannot claim to be the isolation boundary. What it does buy is that the
    common accidents -- a candidate that shells out, opens a file by hand, or
    reaches for a dunder -- fail in-process with a readable message instead of
    dying against a sandbox profile.

    The three keyword arguments are what a second task on this evaluator needs
    and nothing more. ``entrypoint`` names the function the candidate must
    define; ``allowed_imports`` narrows or widens the import set; and
    ``literal_top_level=False`` admits computed module-level constants -- a
    Gauss-Legendre node table built once at import is ordinary numerics, and
    refusing it would reject good programs for a rule whose real work (no
    imports outside the set, no dunders, no ``exec``) is done elsewhere. The
    cost is bounded by the sandbox: module-level work runs under the same CPU
    limit as everything else the candidate does.
    """
    allowed = ALLOWED_IMPORTS if allowed_imports is None else allowed_imports
    if not source.strip():
        return False, "empty source"
    if len(source) > max_length:
        return False, f"source length {len(source)} exceeds {max_length}"
    if "\x00" in source:
        return False, "source contains a NUL byte"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} at line {exc.lineno}"

    if not any(
        isinstance(node, ast.FunctionDef) and node.name == entrypoint
        for node in tree.body
    ):
        return False, f"missing {entrypoint} function"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    return False, f"import {alias.name!r} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in allowed:
                return False, f"import from {node.module!r} is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return False, f"name {node.id!r} is not allowed"

    allowed_top_level = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
    )
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            return False, f"top-level {type(node).__name__} is not allowed"
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            return False, "only a module docstring may be a top-level expression"
        if literal_top_level and isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
                return False, "top-level assignments must be literal constants"
    return True, ""


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Splits:
    """Upstream's 80/20 split, with the 20% cut into scoring shards."""

    root: Path
    train_path: Path
    shard_paths: Tuple[Path, ...]
    shard_truth: Tuple[Tuple[float, ...], ...]
    train_rows: int
    header: Tuple[str, ...]

    def rows(self, shard: int) -> int:
        return len(self.shard_truth[shard])

    def sizes(self) -> Tuple[int, int, int]:
        return self.train_rows, len(self.shard_paths), sum(
            len(truth) for truth in self.shard_truth)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def prepare_splits(
    *,
    shards: int = 8,
    test_shards: int = 4,
    train_rows: int = 0,
    url: str = DATA_URL,
) -> Splits:
    """Materialise the CSVs a candidate reads, exactly once per configuration.

    Upstream's `prepare_data()` takes the head 80% as the training file and the
    tail 20% as the scoring file, target dropped. This keeps that split and then
    cuts the 20% into contiguous equal shards: the first ``shards`` of them are
    what the search can score, the last ``test_shards`` are never shown to it.
    Upstream reports one RMSE over the whole tail, which is the split it also
    optimised against -- a number this port could not report honestly.
    """
    if shards < 2 or test_shards < 1:
        raise ValueError("need at least two scoring shards and one test shard")
    text = fetch_text(url, cache_subdir="era", filename="s3e1-train.csv")
    reader = csv.reader(io.StringIO(text))
    header = tuple(next(reader))
    if TARGET_COLUMN not in header:
        raise ValueError(f"{TARGET_COLUMN!r} missing from the upstream header")
    rows = [row for row in reader if row]

    cut = int(TRAIN_FRACTION * len(rows))
    train, holdout = rows[:cut], rows[cut:]
    if train_rows:
        train = train[:train_rows]

    total_shards = shards + test_shards
    per_shard = len(holdout) // total_shards
    if per_shard < 20:
        raise ValueError(
            f"{total_shards} shards of {len(holdout)} held-out rows leaves "
            f"{per_shard} rows each, too few for a stable RMSE")

    fingerprint = hashlib.sha256(
        f"{url}|{len(train)}|{shards}|{test_shards}|{per_shard}".encode("utf-8")
    ).hexdigest()[:12]
    root = Path(cache_path("era", f"splits-{fingerprint}"))
    root.mkdir(parents=True, exist_ok=True)

    train_path = root / "local_train.csv"
    if not train_path.exists():
        _write_csv(train_path, header, train)

    target_index = header.index(TARGET_COLUMN)
    feature_header = tuple(name for name in header if name != TARGET_COLUMN)
    shard_paths: List[Path] = []
    shard_truth: List[Tuple[float, ...]] = []
    for index in range(total_shards):
        block = holdout[index * per_shard : (index + 1) * per_shard]
        path = root / f"shard-{index:03d}.csv"
        if not path.exists():
            _write_csv(
                path,
                feature_header,
                [
                    [cell for position, cell in enumerate(row) if position != target_index]
                    for row in block
                ],
            )
        shard_paths.append(path)
        shard_truth.append(tuple(float(row[target_index]) for row in block))

    return Splits(
        root=root,
        train_path=train_path,
        shard_paths=tuple(shard_paths),
        shard_truth=tuple(shard_truth),
        train_rows=len(train),
        header=header,
    )


def data_preview(splits: Splits, rows: int = 5) -> str:
    """Upstream's `get_data_head()` -- `df.head().to_markdown()`, without pandas."""
    with splits.train_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        head = [next(reader) for _ in range(rows)]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in head)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Candidate isolation
# --------------------------------------------------------------------------


#: The Seatbelt analogue of the Bubblewrap profile below. Both give the two
#: guarantees that matter for running model-written code against a scientific
#: stack: **no network**, and **no writing outside a scratch directory**. Reads
#: stay open on both, which is a weaker boundary than the OpenEvolve port's and
#: is stated rather than glossed -- a candidate that must `import sklearn` needs
#: to read site-packages, so "read nothing" was never on the table here.
_SEATBELT_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write-data (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/zero"))
(allow file-write* (subpath "{scratch}"))
"""

#: Threads are capped at one because `RLIMIT_CPU` counts CPU seconds across all
#: of them: an OpenBLAS that helpfully starts eight threads burns a 60-second
#: budget in eight wall-clock seconds, and the candidate is then killed for
#: being fast. Upstream sets no thread policy and has no CPU limit to protect.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "JOBLIB_MULTIPROCESSING": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def sandbox_backend() -> Optional[str]:
    """Which isolation backend :func:`sandbox_command` would pick, or None."""
    if shutil.which("bwrap"):
        return "Bubblewrap"
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
        return "Seatbelt (sandbox-exec)"
    return None


def _python_roots() -> List[str]:
    """Directories a candidate's `import sklearn` has to be able to read.

    OpenEvolve's Bubblewrap profile binds `/usr` and `/lib` and runs
    `/usr/bin/python3`, which works when the candidate imports only `math`. Here
    the interpreter is this process's own -- a virtualenv, a conda prefix, a
    Homebrew cellar -- and its packages live wherever the installer put them.
    """
    roots = {sys.prefix, sys.base_prefix, str(Path(sys.executable).resolve().parent)}
    for key in ("purelib", "platlib", "stdlib", "platstdlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(path)
    return sorted(root for root in roots if root and Path(root).exists())


def sandbox_command(
    candidate: Path,
    *,
    train: Path,
    test: Path,
    rows: int,
    timeout: float,
    nproc_limit: int,
) -> Tuple[List[str], Dict[str, str]]:
    """The tabular task's runner, under this platform's isolation backend."""
    return sandbox_wrapper(
        [
            str(RUNNER),
            str(candidate),
            "--train", str(train),
            "--test", str(test),
            "--rows", str(rows),
            "--cpu-seconds", str(max(2, int(math.ceil(timeout)))),
            "--nproc-limit", str(nproc_limit),
        ],
        scratch=candidate.parent.resolve(),
    )


def sandbox_wrapper(
    runner_args: Sequence[str],
    *,
    scratch: Path,
) -> Tuple[List[str], Dict[str, str]]:
    """The isolation backend this platform has, plus the environment to run it in.

    Neither backend is optional: a candidate here is model-written Python that
    imports scikit-learn and gets executed, and running it unconfined because
    the sandbox is missing would be the wrong way to make an example portable.
    Upstream ships `Sandbox.run` as a `NotImplementedError` with the comment
    "Must provide a sandbox for executing untrusted code" -- this is that.

    Split from :func:`sandbox_command` so a second task on this evaluator gets
    the *same* profile rather than a copy of it. The profile is what
    ``test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block``
    checks against the kernel, and a copy would be a second thing to check.
    ``runner_args`` is everything after ``python -I``: the runner script and its
    arguments, which is the only part a task owns.
    """
    backend = sandbox_backend()
    env = dict(_THREAD_ENV)
    env["TMPDIR"] = str(scratch)
    env["PATH"] = "/usr/bin:/bin"

    if backend == "Bubblewrap":
        command = [
            shutil.which("bwrap") or "bwrap",
            # A read-only root rather than OpenEvolve's handful of binds: the
            # candidate's imports are spread across the interpreter prefix and
            # site-packages, and enumerating those would be a guess that fails
            # differently on every host. Writes are still denied everywhere but
            # the scratch bind, and the network is still unshared.
            "--ro-bind", "/", "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--bind", str(scratch), str(scratch),
            "--unshare-net",
            "--die-with-parent",
            "--clearenv",
        ]
        for name, value in env.items():
            command.extend(("--setenv", name, value))
        for root in _python_roots():
            command.extend(("--ro-bind-try", root, root))
        command.extend((sys.executable, "-I", *runner_args))
        return command, env

    if backend is not None:
        profile = scratch / "sandbox.sb"
        # `.resolve()` matters: `tempfile` hands back `/var/folders/...`, `/var`
        # is a symlink to `/private/var`, and Seatbelt matches the resolved path.
        profile.write_text(_SEATBELT_PROFILE.format(scratch=str(scratch)), encoding="utf-8")
        command = ["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-I", *runner_args]
        return command, {**os.environ, **env}

    raise RuntimeError(
        "no candidate isolation available: install Bubblewrap (bwrap) on Linux, "
        "or run on macOS where sandbox-exec ships with the system")


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "rmse": None,
        "score": -math.inf,
        "rows_scored": 0,
        "shards": 0,
        "seconds": 0.0,
        "limits_unavailable": [],
        "error": error,
    }


def rmse(truth: Sequence[float], predictions: Sequence[float]) -> float:
    """Upstream's `np.sqrt(mean_squared_error(y_true, predictions))`."""
    if len(truth) != len(predictions):
        raise ValueError("prediction/truth length mismatch")
    total = sum((float(t) - float(p)) ** 2 for t, p in zip(truth, predictions))
    return math.sqrt(total / len(truth))


def run_candidate(
    code: str,
    *,
    splits: Splits,
    shard: int,
    timeout: float,
    max_length: int = 20_000,
    nproc_limit: int = 64,
) -> Dict[str, Any]:
    """Execute one candidate against one shard and return its raw payload."""
    valid, reason = validate_source(code, max_length=max_length)
    if not valid:
        return {"ok": False, "error": f"gate: {reason}", "seconds": 0.0}
    with tempfile.TemporaryDirectory(prefix="era-candidate-") as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        command, env = sandbox_command(
            candidate,
            train=splits.train_path,
            test=splits.shard_paths[shard],
            rows=splits.rows(shard),
            timeout=timeout,
            nproc_limit=nproc_limit,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout + 10.0,
                env=env,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"timeout after {timeout + 10.0:.0f}s",
                "seconds": time.monotonic() - started,
            }
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            tail = (completed.stderr or "").strip()[-300:]
            return {
                "ok": False,
                "error": f"no runner output (rc={completed.returncode}): {tail}",
                "seconds": time.monotonic() - started,
            }
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": f"unparseable runner output: {lines[-1][:200]}",
                "seconds": time.monotonic() - started,
            }


def evaluate_source(
    code: str,
    *,
    splits: Splits,
    shards: Sequence[int],
    timeout: float,
    max_length: int = 20_000,
) -> Tuple[bool, Dict[str, Any], str]:
    """Score a candidate over one or more shards, pooled into a single RMSE.

    Pooled rather than averaged per shard: the shards are equal-sized contiguous
    blocks of one split, so pooling reproduces the number upstream would have
    computed over the same rows, and averaging would not.
    """
    squared = 0.0
    scored = 0
    seconds = 0.0
    unavailable: List[str] = []
    for shard in shards:
        payload = run_candidate(
            code, splits=splits, shard=shard, timeout=timeout, max_length=max_length)
        seconds += float(payload.get("seconds") or 0.0)
        if not payload.get("ok"):
            error = str(payload.get("error") or "candidate failed")
            return False, _zero_metrics(error), error
        truth = splits.shard_truth[shard]
        predictions = payload["predictions"]
        try:
            squared += sum(
                (float(t) - float(p)) ** 2 for t, p in zip(truth, predictions))
        except (TypeError, ValueError) as exc:
            error = f"non-numeric prediction: {exc}"
            return False, _zero_metrics(error), error
        scored += len(truth)
        unavailable = payload.get("limits_unavailable") or unavailable
    if not scored:
        return False, _zero_metrics("no shards scored"), "no shards scored"
    value = math.sqrt(squared / scored)
    if not math.isfinite(value):
        return False, _zero_metrics("non-finite RMSE"), "non-finite RMSE"
    return (
        True,
        {
            "rmse": value,
            # Upstream returns `-score` because FUTS maximises. Keeping its sign
            # convention keeps the node ordering identical to `futs.search`.
            "score": -value,
            "rows_scored": scored,
            "shards": len(list(shards)),
            "seconds": seconds,
            "limits_unavailable": unavailable,
            "error": "",
        },
        "",
    )


def framework_score(metrics: Dict[str, Any]) -> float:
    """Map an RMSE onto AgentDescent's [0, 1] reward, order-preserving.

    ``1 / (1 + rmse)`` is strictly decreasing in RMSE, so it induces exactly the
    ranking ``-rmse`` does. That equality is the point: the tree ranks nodes on
    upstream's own score and the engine gates on this one, and a port where
    those two disagreed would be selecting against its own acceptance rule.
    """
    value = metrics.get("rmse")
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return 1.0 / (1.0 + float(value))


# --------------------------------------------------------------------------
# The mutation prompt
# --------------------------------------------------------------------------


_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_program(reply: str) -> Tuple[str, str]:
    """Upstream's markdown stripping, plus the docstring as a change summary.

    `GeminiLLM.draw_sample` removes ```python fences with three regexes and
    returns whatever is left. This takes the *longest* fenced block when there
    are several -- a model that explains itself in a second snippet should not
    have the explanation compiled -- and falls back to upstream's behaviour of
    treating the whole reply as code when there is no fence at all.
    """
    blocks = _FENCE.findall(reply)
    code = max(blocks, key=len).strip() if blocks else reply.strip()
    summary = ""
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return code, summary
    doc = ast.get_docstring(parsed)
    if doc:
        summary = doc.strip().splitlines()[0][:200]
    return code, summary


#: Characters that cannot occur in Python source once strings and comments are
#: removed. Their presence is proof the reply was damaged in transit rather than
#: written badly -- no sampler emits `$` in the middle of an identifier.
_IMPOSSIBLE = re.compile(r"[^\x20-\x7e\n\t\r]|[$?`]")
_STRINGS_AND_COMMENTS = re.compile(
    r"(\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*'|#[^\n]*)", re.DOTALL)


def reply_is_intact(reply: str) -> Tuple[bool, str]:
    """Did this reply arrive as Python at all, or did the channel damage it?

    Two checks, both of which a *badly written* program passes: the extracted
    code parses, and it holds no character Python source cannot hold. A model
    that writes a wrong algorithm, an unbounded loop or a call that raises
    passes both and goes on to be a node scoring `-inf`, which is what upstream
    requires. What fails here is a reply that came back with bytes spliced into
    the middle of tokens.

    Measured on one hosted GLM-5.2 endpoint (Anthropic-shaped): replies of a few
    thousand characters came back damaged the majority of the time, with
    fragments like ``return val9.3192`` and ``c_orig,0$ zG$C$F1_orig``. The rate
    was identical through the Anthropic SDK, through the SDK's streaming API and
    through a hand-rolled ``urllib`` request, while 25 fetches of a
    similarly-sized file over the same proxy returned one identical hash -- so
    the damage is at the endpoint, not in any client this repository controls.
    """
    if not reply.strip():
        return False, "empty reply"
    code, _ = extract_program(reply)
    if not code.strip():
        return False, "no code in the reply"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"reply did not parse: {exc.msg} at line {exc.lineno}"
    found = sorted(set(_IMPOSSIBLE.findall(_STRINGS_AND_COMMENTS.sub("", code))))
    if found:
        return False, f"reply holds characters Python source cannot: {found}"
    return True, ""


def with_intact_replies(
    completion: Callable[[str], str],
    *,
    attempts: int = 4,
    counter: Optional[Dict[str, int]] = None,
) -> Callable[[str], str]:
    """Redraw a reply the channel damaged. **Not** a retry of a failed program.

    The distinction is the whole justification. Upstream's loop turns a program
    that fails into a node scoring `-inf`, and this port keeps that: every
    program that arrives intact is executed and scored, however bad it is.
    What this retries is a reply that never was a program -- and treating those
    as failed candidates would silently attribute an endpoint defect to the
    model, which on the endpoint measured above meant most of the search budget.

    After ``attempts`` draws the last reply is returned unchanged, so a run
    against a channel that is *always* damaged still terminates, and still
    records those expansions as the failed nodes they are. ``counter`` collects
    ``drawn`` / ``damaged`` so a run can report the tax it paid.
    """

    def complete(prompt: str) -> str:
        reply = ""
        for attempt in range(attempts):
            reply = completion(prompt)
            if counter is not None:
                counter["drawn"] = counter.get("drawn", 0) + 1
            intact, reason = reply_is_intact(reply)
            if intact:
                return reply
            if counter is not None:
                counter["damaged"] = counter.get("damaged", 0) + 1
            if attempt == attempts - 1:
                warnings.warn(
                    f"{attempts} replies in a row arrived damaged ({reason}); "
                    f"scoring the last one as the failed program it is. This is "
                    f"an endpoint defect, not a model result -- see "
                    f"examples.era._era_support.reply_is_intact.",
                    RuntimeWarning, stacklevel=2)
        return reply

    return complete


#: Upstream's `GeminiLLM.draw_sample` wraps every prompt in this preamble.
SYSTEM_PREAMBLE = """You are an expert Data Scientist and Python programmer.
Your task is to write Python code to solve a machine learning problem.
Return ONLY the python code."""


def mutation_prompt(
    parent: Program,
    *,
    preview: str,
    description: str = "Improve the regression model for the California Housing dataset.",
    timeout: float = 60.0,
) -> str:
    """`PlaygroundGenerator.__call__`, with upstream's wording preserved.

    The score is shown as a positive RMSE because upstream shows it that way
    (`rmse = -parent_score`), and the four numbered constraints are upstream's
    -- including the ban on xgboost/lightgbm, which is what keeps the search
    about *method* rather than about which gradient-boosting wheel is installed.
    """
    score = parent.metrics.get("rmse")
    shown = f"{float(score):.5f}" if score is not None else "failed to run"
    return f"""{SYSTEM_PREAMBLE}

--- BEGIN PROMPT ---

{description}

Here is a preview of the training data:
{preview}

The goal is to predict '{TARGET_COLUMN}'. The metric is RMSE (Root Mean Squared Error).
Lower is better.

The previous solution had a score (RMSE) of: {shown}
Previous Solution Code:
```python
{parent.code}
```

Please generate a NEW, IMPROVED Python function named `train_and_predict` that:
1. Accepts `train_path` and `test_path` as strings.
2. Trains a regression model.
3. Returns the predictions for the test set as a numpy array or list.
4. You can use pandas, numpy, scikit-learn.

IMPORTANT: DO NOT use `xgboost` or `lightgbm`.

Your code must look like this:
```python
import pandas as pd
import numpy as np
# ... other imports

def train_and_predict(train_path, test_path):
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # ... Feature Engineering ...
    # ... Training ...

    # Predict
    predictions = ...
    return predictions
```
Provide the full, runnable code including imports.

IMPORTANT CONSTRAINTS FOR SPEED:
1. DO NOT use GridSearchCV or RandomizedSearchCV.
2. If using RandomForest or Boosting, set `n_estimators` to maximum 50.
3. Keep the model lightweight (execution time limit is {timeout:.0f} seconds).
4. The sandbox has no network access and one CPU thread; do not download data
   or set `n_jobs` above 1.
--- END PROMPT ---
"""
