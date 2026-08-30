"""Release catalogue, Docker workspace, verifier and prompts for SWE-bench Science.

The runnable entry point is :mod:`examples.era.era_swe_science`; the search it
runs is `era_empirical_software.py`, unchanged. This module is the boundary
between that search and someone else's benchmark: what a task *is*, how a
candidate is scored, and what a coding agent is shown before it edits.

What a node carries
-------------------
Every other ERA task in this repository evolves **a program**, and a mutation is
one model call that rewrites it. This one evolves **a patch to a repository**:
the benchmark's tasks are repository-scale ("repair an inconsistent
transition-state rotor workflow" across a 3 MB scientific package), and no
single reply rewrites those. So the mutation operator here is a **coding agent
session** -- Claude Code, Codex, or any command-line agent -- run inside a git
checkout of the task, and the node's payload is the artifact the benchmark
itself collects: ``git diff --cached --binary <baseline>``, byte for byte what
`tasks/task_NNN/pre_artifacts.sh` writes to ``/logs/artifacts/model.patch``.

Three containers per expansion, and what each is for
----------------------------------------------------
* The **environment image** holds the baseline source, the public fixtures and
  the dependencies. A copy of its task directory becomes the agent's workspace,
  bind-mounted back into a container of the same image so the agent can *run*
  what it edits (``run-in-env``, and ``python``/``pytest`` shims on its PATH).
  That container has no network, which is the release's own
  ``[agent] network_mode = "no-network"``.
* The **verifier image** holds the held-out tests and the grader. Scoring
  applies the candidate patch inside it and runs
  :mod:`examples.era._era_swe_science_runner`, which is the release's own
  grader with a node-id selection added.

The agent process itself runs on the host and reaches its model provider, the
way Pier's own ``--agent claude-code`` does. Nothing the *task* runs has
network access.

The held-back split, and why the score is not the benchmark's
-------------------------------------------------------------
The benchmark's own reward is binary: the public reproduction exits 0 **and**
every private test passes. That is the number this port reports. It is not a
number a tree can search on -- it is 0 for every node until it is 1, and PUCT's
exploitation term would rank a patch that fixes four of five checks exactly
level with one that deleted the module.

So the tree ranks on a **pass rate**, and the private suite is split the way
every other ERA task splits its data: a fraction is **held back** and the search
never sees it, in any shard, in any prompt. What the search may see is the
public reproduction and the visible tests; what is reported is (a) the
release's own grader, run unmodified over the whole suite, and (b) whether the
held-back tests moved with the visible ones.

**The protocol is ERA's, not the benchmark's**, and that is the thing to keep
straight before putting a number from here beside a SWE-bench Science
leaderboard row. There, an agent gets one attempt at a task and no feedback
from any verifier. Here a tree of agent sessions is scored between attempts on
part of the private suite, and the best node is kept. Same tasks, same images,
same grader, different experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agentdescent.dataloader import cache_path, fetch_text

from examples.era._era_support import Program


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------

#: The HuggingFace dataset repository, and the revision every number here was
#: produced against. Pinned rather than tracked: `main` moving would silently
#: change which images a rerun pulls.
RELEASE_REPO = "OpenMOSS-Team/SWE-bench-Science"
RELEASE_REVISION = "d8bdbcb4ecb2b565686382459c815d2b6291fd31"
RELEASE_GITHUB = "https://github.com/OpenMOSS/SWE-bench-Science"
_RESOLVE = f"https://huggingface.co/datasets/{RELEASE_REPO}/resolve/{RELEASE_REVISION}"
MANIFEST_URL = f"{_RESOLVE}/manifests/tasks.jsonl"
CACHE_SUBDIR = "swe_bench_science"

#: 119 tasks, ids `001`..`119`; 96 of them carry no license gate and are the
#: release's own default selection (`selections/default-96.json`, which
#: `tests/test_era_swe_science.py` pins against this rule).
RELEASE_TASKS = 119
UNRESTRICTED_TASKS = 96

#: Uniform across all 119 published bundles, so derived rather than fetched --
#: and pinned by `tests/test_era_swe_science.py` so a release that changed one
#: of them fails a test instead of running the wrong directory.
WORKDIR_TEMPLATE = "/app/task_{task_id}"
PRIVATE_TESTS_ROOT = "/tests/private_tests"
RELEASE_AGENT_TIMEOUT = 5400.0
RELEASE_VERIFIER_TIMEOUT = 1800.0
MODEL_PATCH_ARTIFACT = "/logs/artifacts/model.patch"

#: A spanning default: six python tasks from six scientific domains, small
#: enough that a first run finishes. `--tasks unrestricted` is the release's own
#: 96, `--tasks all` is every one of the 119 and needs the license opt-in.
DEFAULT_TASKS = ("001", "002", "022", "029", "034", "045")

RUNNER = Path(__file__).with_name("_era_swe_science_runner.py")
SENTINEL = "___ERA_SWE_SCIENCE_RESULT___"

#: What an evaluation writes into `Program.metrics["report"]`, and therefore
#: what a child's prompt can quote back. Bounded because the whole metrics dict
#: is carried in the result file.
REPORT_CHARS = 3000


class BenchmarkError(RuntimeError):
    """The release, Docker, or an image is not usable -- not a failed candidate."""


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


def docker_backend() -> Optional[str]:
    """``docker`` if a daemon answers, else ``None``.

    The equivalent of `_era_support.sandbox_backend` for this task: every other
    ERA task isolates a candidate with Bubblewrap or Seatbelt, and this one runs
    someone else's published images, so Docker is not an implementation detail
    that could be swapped -- it is what the benchmark is distributed as.
    """
    if shutil.which("docker") is None:
        return None
    try:
        proc = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return "docker" if proc.returncode == 0 else None


def _docker(args: Sequence[str], *, timeout: float = 600.0,
            check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-600:]
        raise BenchmarkError(f"docker {' '.join(args[:2])} failed: {detail}")
    return proc


def image_present(ref: str) -> bool:
    proc = _docker(["image", "inspect", ref], timeout=120, check=False)
    return proc.returncode == 0


def pull_image(ref: str, *, timeout: float = 3600.0) -> bool:
    """Fetch a pinned image unless it is already local. Returns True if pulled."""
    if image_present(ref):
        return False
    _docker(["pull", "--platform", "linux/amd64", ref], timeout=timeout)
    return True


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def _release_file(relative: str, *, release: Optional[Path], timeout: float) -> str:
    """One release file, from a local download if there is one, else from HF."""
    if release is not None:
        path = Path(release) / relative
        if not path.is_file():
            raise BenchmarkError(
                f"{path} is missing -- --release must point at a download of "
                f"{RELEASE_REPO} (see its README's `hf download` line)")
        return path.read_text(encoding="utf-8")
    return fetch_text(f"{_RESOLVE}/{relative}", cache_subdir=CACHE_SUBDIR,
                      filename=relative.replace("/", "_"), timeout=timeout)


def load_manifest(*, release: Optional[Path] = None,
                  timeout: float = 120.0) -> Dict[str, Dict[str, Any]]:
    """`manifests/tasks.jsonl`, the release's own machine-readable catalogue."""
    text = _release_file("manifests/tasks.jsonl", release=release, timeout=timeout)
    rows: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["release_id"])] = row
    if not rows:
        raise BenchmarkError("the release manifest is empty")
    return rows


def unrestricted(manifest: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    """The release's default selection, by its own rule.

    `selections/default-96.json` and ``license_gate == "none"`` name the same 96
    tasks; deriving it keeps one source of truth, and
    `tests/test_era_swe_science.py` holds the two together.
    """
    return tuple(sorted(task_id for task_id, row in manifest.items()
                        if row.get("license_gate", "none") == "none"))


def parse_selection(selection: str, manifest: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    """``001,005-007`` / ``default`` / ``unrestricted`` / ``all``.

    Ranges are the release's own `scripts/materialize.py --task-id` syntax, so a
    selection copied from its README means the same thing here.
    """
    text = (selection or "").strip()
    if not text or text == "default":
        return DEFAULT_TASKS
    if text == "unrestricted":
        return unrestricted(manifest)
    if text == "all":
        return tuple(sorted(manifest))
    ids: List[str] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, _, high = piece.partition("-")
            try:
                span = range(int(low), int(high) + 1)
            except ValueError:
                raise SystemExit(f"--tasks: {piece!r} is not a task id or range")
            ids.extend(f"{n:03d}" for n in span)
        else:
            ids.append(f"{int(piece):03d}" if piece.isdigit() else piece)
    unknown = [task_id for task_id in ids if task_id not in manifest]
    if unknown:
        raise SystemExit(
            f"unknown SWE-bench Science task id(s): {', '.join(unknown)}. The "
            f"release has {len(manifest)}, `001`..`{max(manifest)}`.")
    seen: Dict[str, None] = {}
    for task_id in ids:
        seen.setdefault(task_id, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """One benchmark task, and the split the search is allowed to see."""

    task_id: str
    title: str
    domain: str
    language: str
    repository_url: str
    base_commit: str
    source_license: str
    license_gate: str
    environment_image: str
    verifier_image: str
    instruction: str
    workdir: str
    #: Every private node id the verifier collects, in pytest's own order.
    tests: Tuple[str, ...]
    #: The ones the search never sees, in any shard or prompt.
    held_back: Tuple[str, ...]
    #: One entry per shard: the scoring shards first, then the held-back ones,
    #: so a shard index means the same thing here as everywhere else in the port.
    shard_tests: Tuple[Tuple[str, ...], ...]
    scoring_shards: int
    held_back_shards: int
    seed: int = 0

    @property
    def visible(self) -> Tuple[str, ...]:
        return tuple(t for t in self.tests if t not in set(self.held_back))

    @property
    def restricted(self) -> bool:
        return self.license_gate != "none"

    def test_range(self) -> Tuple[int, ...]:
        """The shard indices the search never sees -- `Domain.test_shards`."""
        return tuple(range(self.scoring_shards,
                           self.scoring_shards + self.held_back_shards))

    def tests_for(self, shards: Sequence[int]) -> Tuple[str, ...]:
        chosen: List[str] = []
        for index in shards:
            if 0 <= int(index) < len(self.shard_tests):
                for node in self.shard_tests[int(index)]:
                    if node not in chosen:
                        chosen.append(node)
        return tuple(chosen)

    def data_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "domain": self.domain,
            "language": self.language,
            "repository": self.repository_url,
            "base_commit": self.base_commit,
            "source_license": self.source_license,
            "license_gate": self.license_gate,
            "environment_image": self.environment_image,
            "verifier_image": self.verifier_image,
            "private_tests": len(self.tests),
            "visible_tests": len(self.visible),
            "held_back_tests": len(self.held_back),
            "scoring_shards": self.scoring_shards,
            "held_back_shards": self.held_back_shards,
            "shard_rule": self.shard_rule,
            "seed": self.seed,
        }

    @property
    def shard_rule(self) -> str:
        """``subset`` or ``replicate`` -- which rule `build_shards` applied."""
        scoring = self.shard_tests[:self.scoring_shards]
        if len(scoring) < 2:
            return "subset"
        return "replicate" if scoring[0] == scoring[1] else "subset"


def split_tests(tests: Sequence[str], *, held_back_frac: float,
                seed: int) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Visible / held-back, drawn once per task from a seeded shuffle.

    At least one test stays visible -- a task whose whole suite was held back
    would give the tree a constant score and the search nothing to do -- and a
    task with a single private test holds nothing back, which the run says out
    loud rather than reporting a held-back figure it could not have measured.
    """
    ordered = list(tests)
    if len(ordered) < 2 or held_back_frac <= 0.0:
        return tuple(ordered), ()
    count = max(1, int(round(len(ordered) * float(held_back_frac))))
    count = min(count, len(ordered) - 1)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    held_back = set(shuffled[:count])
    return (tuple(t for t in ordered if t not in held_back),
            tuple(t for t in ordered if t in held_back))


#: Tests per shard below which subsetting is destroying signal rather than
#: sampling. `build_shards` replicates instead of partitioning under it.
MIN_TESTS_PER_SHARD = 2


def build_shards(visible: Sequence[str], held_back: Sequence[str], *,
                 scoring_shards: int, held_back_shards: int,
                 min_per_shard: int = MIN_TESTS_PER_SHARD
                 ) -> Tuple[Tuple[str, ...], ...]:
    """Cut the tests into the shards the engine wants -- or replicate them.

    ``evolve()`` refuses to split fewer than four rollout tasks into train and
    held-out, and most of this benchmark's private suites hold three to five
    tests. Round-robining those over four shards is not sampling, it is
    shredding: a node's score would come from whichever single test the engine's
    held-out split happened to land on, and a patch that fixed one of the others
    would score no better than one that fixed nothing.

    So a shard is a **subset** only when the suite is big enough for the subsets
    to mean something (``min_per_shard`` each), and a **replicate** of the whole
    visible set otherwise. Evaluations here are deterministic and memoised per
    run, so replicated shards cost one container between them rather than one
    each -- the cost of the choice is nothing, and the alternative was a
    per-node score with a standard deviation the size of its range.

    Which rule applied is recorded per task: `Suite.data_summary()` reports
    `shard_rule`, so a result file says whether its shards were subsets.
    """
    def spread(nodes: Sequence[str], count: int) -> Tuple[Tuple[str, ...], ...]:
        if count <= 0:
            return ()
        if not nodes:
            return tuple(() for _ in range(count))
        if len(nodes) < count * max(1, min_per_shard):
            return tuple(tuple(nodes) for _ in range(count))
        return tuple(
            tuple(nodes[j] for j in range(len(nodes)) if j % count == index)
            for index in range(count))

    return spread(list(visible), scoring_shards) + spread(list(held_back),
                                                          held_back_shards)


def discover_tests(*, verifier_image: str, workdir: str, task_id: str,
                   timeout: float = RELEASE_VERIFIER_TIMEOUT,
                   cache: bool = True) -> Tuple[str, ...]:
    """Ask the verifier's own pytest what the private suite contains.

    The release says the private test filenames are "intentionally
    unconstrained", so they are read out of the image rather than guessed, and
    cached against the image digest -- a different digest is a different suite.
    """
    digest = verifier_image.rsplit("@", 1)[-1].replace(":", "-")[:32]
    path = Path(cache_path(CACHE_SUBDIR, f"tests-{task_id}-{digest}.json"))
    if cache and path.is_file():
        try:
            return tuple(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    payload = _run_in_verifier(
        verifier_image=verifier_image,
        config={"mode": "collect", "workdir": workdir, "task_id": task_id,
                "tests_root": PRIVATE_TESTS_ROOT, "private_timeout": timeout},
        patch="", timeout=timeout + 120)
    tests = tuple(payload.get("tests") or ())
    if not tests:
        raise BenchmarkError(
            f"task {task_id}: the verifier collected no private tests "
            f"(pytest exited {payload.get('return_code')})")
    if cache:
        path.write_text(json.dumps(list(tests), indent=2), encoding="utf-8")
    return tests


def prepare_suite(task_id: str, *, manifest: Dict[str, Dict[str, Any]],
                  shards: int = 6, test_shards: int = 2,
                  held_back_frac: float = 0.25, seed: int = 0,
                  release: Optional[Path] = None, pull: bool = True,
                  timeout: float = 120.0) -> Suite:
    """Everything one task needs before its tree can be seeded.

    Fetches the task's instruction, makes sure both pinned images are local, and
    asks the verifier what its private suite holds. This is the boundary a
    `--dry-run` must not cross.
    """
    row = manifest.get(task_id)
    if row is None:
        raise BenchmarkError(f"task {task_id} is not in the release manifest")
    instruction = _release_file(f"tasks/task_{task_id}/instruction.md",
                                release=release, timeout=timeout)
    workdir = WORKDIR_TEMPLATE.format(task_id=task_id)
    if pull:
        pull_image(row["environment_image"])
        pull_image(row["verifier_image"])
    tests = discover_tests(verifier_image=row["verifier_image"], workdir=workdir,
                           task_id=f"task_{task_id}")
    visible, held_back = split_tests(tests, held_back_frac=held_back_frac, seed=seed)
    held_back_shards = test_shards if held_back else 0
    return Suite(
        task_id=task_id,
        title=row.get("title", ""),
        domain=row.get("domain", ""),
        language=row.get("language", ""),
        repository_url=row.get("repository_url", ""),
        base_commit=row.get("base_commit", ""),
        source_license=row.get("source_license", ""),
        license_gate=row.get("license_gate", "none"),
        environment_image=row["environment_image"],
        verifier_image=row["verifier_image"],
        instruction=instruction.strip(),
        workdir=workdir,
        tests=tuple(tests),
        held_back=held_back,
        shard_tests=build_shards(visible, held_back, scoring_shards=shards,
                                 held_back_shards=held_back_shards),
        scoring_shards=shards,
        held_back_shards=held_back_shards,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Scoring: the release's grader, in the release's verifier image
# ---------------------------------------------------------------------------


def _run_in_verifier(*, verifier_image: str, config: Dict[str, Any], patch: str,
                     timeout: float) -> Dict[str, Any]:
    """One verifier container: patch in, the runner's JSON payload out."""
    job = Path(tempfile.mkdtemp(prefix="era-swe-eval-"))
    try:
        (job / "model.patch").write_text(_normalise_patch(patch), encoding="utf-8")
        shutil.copy2(RUNNER, job / "runner.py")
        payload = dict(config)
        payload.setdefault("patch", "/era/model.patch")
        payload.setdefault("result", "/era/result.json")
        (job / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        proc = _docker(
            ["run", "--rm", "--network", "none", "--platform", "linux/amd64",
             "-v", f"{job}:/era", "-w", config["workdir"], verifier_image,
             "python", "/era/runner.py", "/era/config.json"],
            timeout=timeout, check=False)
        text = proc.stdout or ""
        marker = text.rfind(SENTINEL)
        if marker < 0:
            local = job / "result.json"
            if local.is_file():
                return json.loads(local.read_text(encoding="utf-8"))
            detail = ((proc.stderr or "") + text).strip()[-600:]
            raise BenchmarkError(
                f"the verifier produced no result (exit {proc.returncode}): {detail}")
        return json.loads(text[marker + len(SENTINEL):].splitlines()[0])
    finally:
        shutil.rmtree(job, ignore_errors=True)


def _normalise_patch(patch: str) -> str:
    """`git apply` needs the trailing newline the ledger's `.strip()` removes."""
    text = patch or ""
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def patch_id(patch: str) -> str:
    return hashlib.sha256(_normalise_patch(patch).encode("utf-8")).hexdigest()[:16]


_PYTEST_SECTION = re.compile(r"^=+ (.+?) =+$")


def pytest_failures(output: str) -> str:
    """The failure sections of a pytest run, without its warnings summary.

    The evaluator runs pytest with the release grader's own flags -- changing
    them could change what passes -- so the noise is dropped here instead. On
    a scientific stack the warnings summary is routinely longer than the
    traceback, and truncating a raw tail hands the child a page of
    ``PyparsingDeprecationWarning`` where the assertion should be.
    """
    if not output:
        return ""
    keep: List[str] = []
    active = False
    for line in output.splitlines():
        header = _PYTEST_SECTION.match(line.strip())
        if header:
            name = header.group(1).lower()
            active = ("failure" in name or "error" in name
                      or "short test summary" in name)
            if active:
                keep.append(line)
            continue
        if active:
            keep.append(line)
    return "\n".join(keep).strip() or output.strip()


def _failure_report(payload: Dict[str, Any]) -> str:
    """What a child's prompt is allowed to quote: only what this run scored."""
    lines: List[str] = []
    public = payload.get("public") or {}
    if public.get("return_code") is not None and not public.get("passed"):
        lines.append("public reproduction (reproduce.py) exited "
                     f"{public.get('return_code')}:")
        lines.append((public.get("output") or "").strip()[-1200:])
    private = payload.get("private") or {}
    if private.get("failed"):
        lines.append("visible private checks:")
        lines.append(pytest_failures(private.get("output") or "")[-2400:])
    return "\n".join(lines)[-REPORT_CHARS:]


def evaluate_patch(patch: str, *, suite: Suite, shards: Sequence[int],
                   timeout: float = RELEASE_VERIFIER_TIMEOUT,
                   public: bool = True, max_patch_bytes: int = 400_000,
                   cache: Optional[Dict[str, Any]] = None
                   ) -> Tuple[bool, Dict[str, Any], str]:
    """Score one candidate patch on the tests those shards name.

    Returns the `(valid, metrics, error)` triple a
    :class:`~examples.era._era_domain.Domain` owes the search. ``valid`` is
    False -- and the node's score `-inf`, upstream's sentinel -- only when the
    candidate could not be *evaluated*: a patch that does not apply, an
    oversized one, a container that died. A patch that applies and fails every
    check is a valid node with a score of zero, which is the thing PUCT needs to
    be able to rank.
    """
    selection = suite.tests_for(shards)
    key = (suite.verifier_image, patch_id(patch), tuple(sorted(selection)), bool(public))
    if cache is not None:
        # No lock: a dict get/set is atomic under the GIL, evaluations are
        # deterministic, and the worst a race can cost is one container run
        # that two threads both paid for.
        hit = cache.get(key)
        if hit is not None:
            valid, metrics, error = hit
            return valid, dict(metrics), error

    if len(_normalise_patch(patch).encode("utf-8")) > max_patch_bytes:
        result = (False, _zero_metrics("patch exceeds --max-patch-bytes"),
                  "patch exceeds --max-patch-bytes")
    else:
        started = time.monotonic()
        try:
            payload = _run_in_verifier(
                verifier_image=suite.verifier_image,
                config={"workdir": suite.workdir, "task_id": f"task_{suite.task_id}",
                        "tests": list(selection), "public": public,
                        "private_timeout": timeout, "public_timeout": timeout,
                        # Generous, because the warnings summary is dropped
                        # host-side rather than in the container: what matters
                        # is that the traceback survives the truncation.
                        "tail_chars": 24_000},
                patch=patch, timeout=timeout + 300)
        except (BenchmarkError, subprocess.SubprocessError, OSError, ValueError) as exc:
            result = (False, _zero_metrics(str(exc)), f"{type(exc).__name__}: {exc}")
        else:
            result = _metrics_from(payload, selection, public,
                                   round(time.monotonic() - started, 3))
    if cache is not None:
        cache[key] = (result[0], dict(result[1]), result[2])
    return result


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {"score": float("-inf"), "pass_rate": 0.0, "passed": 0, "collected": 0,
            "public": 0, "checks": 0, "subset_resolved": 0, "seconds": 0.0,
            "error": error, "report": ""}


def _metrics_from(payload: Dict[str, Any], selection: Sequence[str], public: bool,
                  seconds: float) -> Tuple[bool, Dict[str, Any], str]:
    if not payload.get("applied", True):
        error = payload.get("apply_error") or "the patch did not apply"
        return False, _zero_metrics(error), error
    private = payload.get("private") or {}
    collected = int(private.get("collected", 0))
    passed = int(private.get("passed", 0))
    public_passed = int((payload.get("public") or {}).get("passed", 0))
    if selection and collected == 0:
        error = ("the verifier collected none of the selected tests: "
                 + (private.get("output") or "").strip()[-400:])
        return False, _zero_metrics(error), error
    # One number over every check this evaluation ran, the public reproduction
    # included -- it is the benchmark's own first condition, and a patch that
    # breaks it has broken the task however many private tests still pass.
    checks = collected + (1 if public else 0)
    passes = passed + (public_passed if public else 0)
    rate = (passes / checks) if checks else 0.0
    metrics: Dict[str, Any] = {
        "score": rate,
        "pass_rate": rate,
        "passed": passed,
        "collected": collected,
        "public": public_passed if public else None,
        "checks": checks,
        # The benchmark's rule, applied to the tests this evaluation could see.
        # It is the benchmark's *reward* only when the selection is the whole
        # private suite, which is why `grade()` exists and this is not it.
        "subset_resolved": int(bool(checks) and passes == checks),
        "seconds": seconds,
        "error": "",
        "report": _failure_report(payload),
    }
    return True, metrics, ""


def grade(patch: str, *, suite: Suite, timeout: float = RELEASE_VERIFIER_TIMEOUT,
          release: Optional[Path] = None) -> Dict[str, Any]:
    """The release's own grader, unmodified, over the whole private suite.

    This is the number that belongs beside a SWE-bench Science result:
    ``tasks/task_NNN/tests/grader.py`` as published, its binary reward, its
    counts. The search never sees it -- it is read once per task for the root
    patch and once for the best node.

    **The grader comes from the task bundle, not from the image.** The release
    mounts it: `tests/docker-compose.yaml` binds `./grader.py` over
    `/tests/grader.py`, and `tests/Dockerfile` copies it in with the comment
    "Keep the runtime entrypoint in sync with the task bundle." That is not
    ceremony. Some published verifier images carry a *stale* grader -- task
    034's runs pytest on `/tests/private_tests/test_task_034.py`, a file that
    does not exist beside the `test_stochastic_orientation.py` that does, so it
    collects nothing, exits 4 and scores every submission 0 however correct.
    Running the image's copy would have reported a floor of zero as a result.
    `tests/test_era_swe_science.py::test_the_bundle_grader_is_the_one_that_runs`
    pins the mount against that task.
    """
    job = Path(tempfile.mkdtemp(prefix="era-swe-grade-"))
    try:
        (job / "model.patch").write_text(_normalise_patch(patch), encoding="utf-8")
        (job / "grader.py").write_text(
            _release_file(f"tasks/task_{suite.task_id}/tests/grader.py",
                          release=release, timeout=120.0),
            encoding="utf-8")
        script = (
            f'set -e; cd {suite.workdir}; '
            'if [ -s /era/model.patch ]; then git apply --binary -p1 /era/model.patch; fi; '
            'python /era/grader.py'
        )
        proc = _docker(
            ["run", "--rm", "--network", "none", "--platform", "linux/amd64",
             "-v", f"{job}:/era", "-w", suite.workdir, suite.verifier_image,
             "sh", "-lc", script],
            timeout=timeout + 300, check=False)
        text = (proc.stdout or "").strip().splitlines()
        for line in reversed(text):
            line = line.strip()
            if line.startswith("{") and '"reward"' in line:
                return json.loads(line)
        return {"reward": 0, "error": "the grader produced no summary",
                "exit": proc.returncode,
                "output": ((proc.stderr or "") + (proc.stdout or "")).strip()[-800:]}
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        return {"reward": 0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(job, ignore_errors=True)


def framework_score(metrics: Dict[str, Any]) -> float:
    """The tree's score as AgentDescent's [0, 1] reward -- the same number.

    ``pass_rate`` is already in [0, 1] and already oriented so that larger is
    better, so the tree ranks nodes on exactly what the engine's gate accepts
    on. A port where those two disagreed would be selecting against its own
    acceptance rule.
    """
    value = metrics.get("pass_rate")
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if rate != rate or rate in (float("inf"), float("-inf")):    # NaN / -inf
        return 0.0
    return max(0.0, min(1.0, rate))


# ---------------------------------------------------------------------------
# The agent's workspace
# ---------------------------------------------------------------------------


_RUN_IN_ENV = """#!/bin/sh
# Run a command inside this task's prepared, offline scientific environment.
exec docker exec -w {workdir} -e PYTHONPATH={workdir}:{workdir}/source \\
    {container} sh -lc "$*"
"""

_FORWARD = """#!/bin/sh
exec docker exec -w {workdir} -e PYTHONPATH={workdir}:{workdir}/source \\
    {container} {program} "$@"
"""


@dataclass
class Workspace:
    """A git checkout of the task, and a container that can run it."""

    suite: Suite
    root: Path
    work: Path
    bin: Path
    container: str

    def env(self) -> Dict[str, str]:
        return {"PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"}

    def baseline(self) -> str:
        """The single root commit `pre_artifacts.sh` insists on."""
        roots = self._git("rev-list", "--max-parents=0", "--reverse", "HEAD").split()
        if len(roots) != 1:
            raise BenchmarkError(
                "task repository must have exactly one baseline root commit")
        return roots[0]

    def diff(self) -> str:
        """``model.patch``: the artifact the release's own hook collects."""
        baseline = self.baseline()
        self._git("add", "-A")
        return self._git("diff", "--cached", "--binary", baseline)

    def apply(self, patch: str) -> None:
        text = _normalise_patch(patch)
        if not text.strip():
            return
        target = self.root / "parent.patch"
        target.write_text(text, encoding="utf-8")
        self._git("apply", "--binary", "-p1", str(target))

    def _git(self, *args: str) -> str:
        # `-c` rather than `git config --global`: the release's own hook marks
        # the checkout safe inside a throwaway container, and doing the same on
        # the host would write to the user's real gitconfig for a directory
        # that is deleted when the expansion ends.
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={self.work}", *args],
            cwd=self.work, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise BenchmarkError(
                f"git {args[0]} failed in the workspace: "
                f"{(proc.stderr or proc.stdout).strip()[-400:]}")
        return proc.stdout


@contextmanager
def open_workspace(suite: Suite, patch: str = "", *, root: Optional[Path] = None,
                   keep: bool = False, timeout: float = 600.0):
    """A checkout of the task with ``patch`` applied, plus a container to run it.

    The checkout is copied out of the environment image and bind-mounted back
    into a container of that same image, so the agent edits ordinary files on
    the host while ``run-in-env`` executes them against the task's own installed
    dependencies. That container has **no network**, which is the release's own
    ``[agent] network_mode``; the agent process itself runs on the host and
    reaches its provider, exactly as Pier's own agent harnesses do.
    """
    base = Path(root) if root else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f"era-swe-{suite.task_id}-", dir=str(base)))
    work = run_root / "work"
    binaries = run_root / "bin"
    work.mkdir()
    binaries.mkdir()
    container = ""
    try:
        staging = _docker(["create", "--network", "none", "--platform", "linux/amd64",
                           suite.environment_image, "sleep", "infinity"],
                          timeout=timeout).stdout.strip()
        try:
            _docker(["cp", f"{staging}:{suite.workdir}/.", str(work)], timeout=timeout)
        finally:
            _docker(["rm", "-f", staging], timeout=timeout, check=False)
        container = _docker(
            ["run", "-d", "--network", "none", "--platform", "linux/amd64",
             "-v", f"{work}:{suite.workdir}", "-w", suite.workdir,
             suite.environment_image, "sleep", "infinity"], timeout=timeout
        ).stdout.strip()
        for name, template, program in (
            ("run-in-env", _RUN_IN_ENV, ""),
            ("python", _FORWARD, "python"),
            ("python3", _FORWARD, "python3"),
            ("pytest", _FORWARD, "pytest"),
        ):
            script = binaries / name
            script.write_text(
                template.format(workdir=suite.workdir, container=container,
                                program=program), encoding="utf-8")
            script.chmod(0o755)
        space = Workspace(suite, run_root, work, binaries, container)
        space.apply(patch)
        yield space
    finally:
        if container:
            _docker(["rm", "-f", container], timeout=timeout, check=False)
        if not keep:
            shutil.rmtree(run_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# The mutation: an agent session, and the reply it turns into
# ---------------------------------------------------------------------------

PATCH_BEGIN = "===ERA-PATCH-BEGIN==="
PATCH_END = "===ERA-PATCH-END==="


def wrap_reply(message: str, patch: str) -> str:
    """The reply shape this port's `extract` reads.

    A sentinel rather than a markdown fence: a diff of a markdown file carries
    fences of its own, and `_era_support.extract_program`'s "longest fenced
    block" rule would take the wrong one.
    """
    return f"{message.strip()}\n{PATCH_BEGIN}\n{patch}\n{PATCH_END}\n"


def extract_patch(reply: str) -> Tuple[str, str]:
    """`(patch, change summary)`, this port's `extract_program`.

    The summary is the agent's own first line rather than a parsed docstring --
    a patch has none -- and is what the node table and a child's prompt show.
    """
    start = reply.find(PATCH_BEGIN)
    end = reply.rfind(PATCH_END)
    patch = "" if start < 0 or end < start else reply[start + len(PATCH_BEGIN):end]
    message = reply[:start] if start >= 0 else reply
    summary = ""
    for line in message.strip().splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.startswith(("PROMISE:", "[agent")):
            summary = line[:200]
            break
    return patch.strip("\n"), summary


def _numbered(text: str, limit: int = 2000) -> str:
    text = (text or "").strip()
    return text[-limit:] if text else ""


def mutation_prompt(program: Program, *, suite: Suite, agent_timeout: float,
                    ask_promise: bool = False) -> str:
    """What the coding agent is told before it edits the checkout.

    The task's own `instruction.md` goes in verbatim -- it is the benchmark's
    prompt and rewriting it would make this a different benchmark. What this
    port adds around it is the ERA loop: where the agent is, how to run the
    task's code, what the previous attempt in this branch of the tree scored,
    and the fact that tests it will never see are what decides the result.
    """
    metrics = program.metrics or {}
    lines = [
        "You are one expansion of a tree search over patches to a scientific "
        "software repository. Your job is to leave the checkout you are in "
        "closer to passing a held-out programmatic verifier.",
        "",
        "## The task",
        "",
        suite.instruction,
        "",
        "## Where you are",
        "",
        f"The working directory is a git checkout of the task at "
        f"{suite.workdir} inside its own container. The dependencies are NOT "
        f"installed on this host: run everything inside the task's prepared, "
        f"offline environment with",
        "",
        "    run-in-env '<shell command>'",
        "",
        "`python`, `python3` and `pytest` on your PATH already forward into it, "
        "so `python reproduce.py` runs the public reproduction. That "
        "environment has no network access and neither should your fix.",
        "",
    ]
    if program.iteration:
        lines += [
            "## What this branch already did",
            "",
            f"This checkout already carries the patch from tree node "
            f"{program.iteration}"
            + (f' ("{program.change_summary}")' if program.change_summary else "")
            + ". Build on it, or revert part of it if it was the wrong idea -- "
              "the diff you leave behind is judged whole, not as a delta.",
            "",
        ]
    else:
        lines += ["## What this branch already did",
                  "",
                  "Nothing: this is the unmodified baseline.", ""]

    passed, checks = metrics.get("passed"), metrics.get("checks")
    if checks:
        lines += [
            "## How it scored",
            "",
            f"{(passed or 0) + (metrics.get('public') or 0)} of {checks} visible "
            f"checks passed (the public reproduction plus "
            f"{metrics.get('collected', 0)} of the private tests this search is "
            f"allowed to see).",
            "",
        ]
        report = _numbered(metrics.get("report", ""), REPORT_CHARS)
        if report:
            lines += ["What failed:", "", "```", report, "```", ""]
    elif metrics.get("error"):
        lines += ["## How it scored", "",
                  f"It could not be evaluated: {metrics['error'][:600]}", ""]

    lines += [
        "## Rules",
        "",
        "* Fix the real defect in the task's own source. The verifier holds "
        "tests you cannot see and cannot read, so a change that special-cases "
        "the public reproduction, the fixtures, or a value you observed will "
        "score worse, not better.",
        "* Do not edit `reproduce.py`, the fixtures, or any generated report.",
        "* Do not add dependencies or reach the network; the evaluation "
        "environment is offline.",
        "* Leave your work as edited files in the checkout. Do not commit and "
        "do not run `git checkout`/`git stash` -- the diff of the working tree "
        "against the baseline commit is what gets scored.",
        "* Delete every scratch file you create before you finish. Everything "
        "left in the checkout, tracked or not, goes into that diff.",
        f"* You have about {int(agent_timeout)}s of wall clock.",
        "",
        "## Finish with",
        "",
        "One line naming the change you made.",
    ]
    if ask_promise:
        lines += [
            "",
            "Then a final line",
            "",
            "    PROMISE: <n>",
            "",
            "rating 1-10 how far this direction can go *once refined* -- not how "
            "finished this attempt is. 10 means it should make every hidden "
            "check pass after tuning.",
        ]
    return "\n".join(lines)


def envelope(program: Program, *, suite: Suite, agent_timeout: float,
             ask_promise: bool = False) -> str:
    """`Domain.prompt` for this task: the parent patch plus the agent's prompt.

    The search hands a mutation the parent *program*; here the parent program is
    a patch, and the mutation needs it as a checkout rather than as text. So the
    "prompt" this domain produces carries both, and the mutation opens the one
    and reads the other. Upstream's shape is unchanged -- the parent program
    goes to the mutation verbatim -- only its destination is.
    """
    return json.dumps({
        "patch": program.code,
        "text": mutation_prompt(program, suite=suite, agent_timeout=agent_timeout,
                                ask_promise=ask_promise),
    })


def make_agent_mutation(
    suite: Suite,
    *,
    launch: Callable[[str, Dict[str, str]], Callable[[str], str]],
    run_root: Optional[Path] = None,
    keep: bool = False,
    counter: Optional[Dict[str, int]] = None,
    on_event: Optional[Callable[[str], None]] = None,
) -> Callable[[str], str]:
    """The `Completion` the ERA search calls to expand a node.

    ``launch(workspace, env)`` returns the callable that actually runs an agent
    there -- `cli_agent(...)` bound to that directory in the entry point. The
    seam is here rather than a hard-coded ``claude`` so that the port can run
    any command-line agent, and so the tests can drive the whole loop without
    one.

    An agent that fails or runs out of time still produces a node: whatever it
    had written is diffed and scored, which is the release's own rule --
    `pre_artifacts.sh` writes the artifact "including for a clean or timed-out
    agent run".
    """
    tally = counter if counter is not None else {}

    def mutate(prompt: str) -> str:
        payload = json.loads(prompt)
        parent_patch, text = payload.get("patch", ""), payload["text"]
        tally["sessions"] = tally.get("sessions", 0) + 1
        with open_workspace(suite, parent_patch, root=run_root, keep=keep) as space:
            message = ""
            try:
                message = launch(str(space.work), space.env())(text)
            except Exception as exc:              # the agent's own failure
                tally["failed"] = tally.get("failed", 0) + 1
                message = f"[agent failed: {type(exc).__name__}: {exc}]"
                if on_event is not None:
                    on_event(f"agent failed: {type(exc).__name__}: {exc}")
            patch = space.diff()
        if not patch.strip():
            tally["empty"] = tally.get("empty", 0) + 1
        return wrap_reply(message, patch)

    return mutate


def make_completion_mutation(
    suite: Suite,
    complete: Callable[[str], str],
    *,
    run_root: Optional[Path] = None,
    listing_files: int = 400,
    counter: Optional[Dict[str, int]] = None,
) -> Callable[[str], str]:
    """The control arm: one model call, no tools, asked for a unified diff.

    It exists to be beaten. The other ERA tasks mutate a single file that fits
    in a prompt, and the obvious question about this one is whether a repository
    really needs an agent -- so the port ships the answer as a runnable arm
    rather than as a claim. The model gets the instruction and the repository's
    file list, and its reply is applied to the same workspace with `git apply`;
    a diff that does not apply leaves the workspace unchanged and the node
    scores the parent's patch, which is what "no useful proposal" looks like.
    """
    tally = counter if counter is not None else {}

    def mutate(prompt: str) -> str:
        payload = json.loads(prompt)
        parent_patch, text = payload.get("patch", ""), payload["text"]
        tally["sessions"] = tally.get("sessions", 0) + 1
        with open_workspace(suite, parent_patch, root=run_root) as space:
            listing = sorted(
                str(path.relative_to(space.work))
                for path in space.work.rglob("*")
                if path.is_file() and ".git/" not in str(path.relative_to(space.work))
            )[:listing_files]
            ask = (text + "\n\n## The repository\n\n```\n" + "\n".join(listing)
                   + "\n```\n\nYou have no tools: reply with a single unified "
                     "diff against this checkout, in one ```diff fenced block, "
                     "applying with `git apply -p1`.")
            message = complete(ask)
            block = re.search(r"```(?:diff|patch)?\n(.*?)```", message, re.DOTALL)
            if block:
                try:
                    space.apply(block.group(1))
                except BenchmarkError as exc:
                    tally["unapplied"] = tally.get("unapplied", 0) + 1
                    message += f"\n[the proposed diff did not apply: {exc}]"
            else:
                tally["no_diff"] = tally.get("no_diff", 0) + 1
            patch = space.diff()
        return wrap_reply(message, patch)

    return mutate
