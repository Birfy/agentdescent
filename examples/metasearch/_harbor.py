"""A Harbor task (SWE-bench-Science, Terminal-Bench-Science) as an ERA ``Domain``.

Stage 2 of the meta-evolution plan (`docs/design-meta-evolution.md`, §4.3): the
search rule evolved on a cheap inner domain is *validated* by running the same
ERA tree search on tasks from the 2026 science benchmarks it never saw. Both
benchmarks ship in Harbor's task layout::

    task/
    ├── instruction.md
    ├── task.toml             # [task] name, [verifier] timeout_sec, [agent] timeout_sec, ...
    ├── environment/          # Dockerfile, or [environment].docker_image in task.toml
    ├── solution/solve.sh     # the oracle
    └── tests/test.sh         # copied to /tests; writes /logs/verifier/reward.json|txt

ERA's search is indifferent to what it searches over; its ``Domain`` is four
things, and for a Harbor task they are:

* **the program is a patch.** The root is the empty patch against the task's
  baseline; a child is the unified diff an agent leaves behind after working
  on the parent's workspace. ``program_id`` is the patch's hash, as for code.
* **a shard is a verifier metric.** ``tests/test.sh`` may write
  ``reward.json`` with several metrics, and those are the only test-level
  granularity Harbor exposes. The *scoring* metrics are what the search reads;
  the *held-back* ones are reported once at the end, which is the same split
  discipline ERA keeps on its data shards. A task that writes one number has
  nothing to hold back, and then the reported figure is the scoring figure --
  said in the run plan rather than hidden.
* **the evaluator is the verifier**, run once per distinct patch and cached:
  Harbor's verifier is deterministic by construction, and every shard of one
  patch is a column of the same run.
* **the prompt is the instruction**, the parent patch, and the scoring
  metrics the parent reached. The reply is a patch in a bare ``` fence.

Two runners. :class:`LocalRunner` works on a **host checkout** of the task's
source tree -- materialise, ``git apply``, run the agent there, ``git diff``,
run ``tests/test.sh`` with ``/logs/verifier`` redirected -- and is what the
offline test suite exercises end to end with a real ``git`` and a real
``test.sh``. It is honest for tasks whose environment is a repository and an
interpreter, which is many of SWE-bench-Science's and few of Terminal-Bench-
Science's. :class:`DockerRunner` verifies inside the task's own image, which is
what Harbor does; it needs a Docker daemon, cannot run an agent *inside* the
container (that is Harbor's own ``harbor run --agent`` and is not reimplemented
here), and is the stated boundary.

Not wrapped in ``with_intact_replies``: that guard rejects characters Python
source cannot hold, and a patch holds all of them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from agentdescent.agents import Completion, WorkspaceAgent

from examples.era._era_domain import Domain

try:  # 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - 3.9/3.10
    _toml = None


__all__ = ["HarborTask", "load_task", "parse_reward", "flatten_metrics", "LocalRunner", "DockerRunner",
           "harbor_domain", "harbor_completion", "whole_file_patch", "PARENT_PATCH_BEGIN", "PARENT_PATCH_END"]


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarborTask:
    root: Path
    name: str
    instruction: str
    docker_image: Optional[str]
    dockerfile: Optional[Path]
    tests_dir: Path
    verifier_timeout: float
    agent_timeout: float
    #: The container working directory the patch is relative to.
    workdir: str = "/app"
    #: ``[verifier.environment].docker_image`` -- set when the benchmark ships a
    #: *separate* verifier image, which SWE-bench-Science does for every task.
    verifier_image: Optional[str] = None
    #: ``[verifier].environment_mode == "separate"``: the grader and the private
    #: tests live in ``verifier_image``, which carries its own copy of the task
    #: source. The public ``tests/`` directory is then only Harbor's required
    #: entrypoint stub, and mounting it over ``/tests`` would hide the real
    #: grader -- so in this mode it is not mounted.
    verifier_separate: bool = False


def _parse_toml(text: str) -> Dict[str, Any]:
    if _toml is not None:
        return _toml.loads(text)
    # A four-line fallback for the fields this module reads: section headers and
    # `key = value` with strings, numbers and booleans. Arrays are ignored.
    out: Dict[str, Any] = {}
    section = out
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = out
            for part in line[1:-1].split("."):
                section = section.setdefault(part.strip(), {})
            continue
        if "=" in line:
            key, value = (s.strip() for s in line.split("=", 1))
            if value.startswith('"') and value.endswith('"'):
                section[key] = value[1:-1]
            elif value in ("true", "false"):
                section[key] = value == "true"
            else:
                try:
                    section[key] = float(value) if "." in value else int(value)
                except ValueError:
                    section[key] = value
    return out


def load_task(root: os.PathLike, *, workdir: str = "/app") -> HarborTask:
    """Read a Harbor task directory; the fields this module needs, no more."""
    root = Path(root)
    toml_path, instruction_path = root / "task.toml", root / "instruction.md"
    if not toml_path.exists() or not instruction_path.exists():
        raise FileNotFoundError(f"{root} is not a Harbor task: task.toml and instruction.md required")
    config = _parse_toml(toml_path.read_text(encoding="utf-8"))
    env = config.get("environment", {}) or {}
    dockerfile = root / "environment" / "Dockerfile"
    tests_dir = root / "tests"
    if not tests_dir.exists():
        raise FileNotFoundError(f"{root} has no tests/ directory")
    return HarborTask(
        root=root,
        name=str((config.get("task") or {}).get("name") or root.name),
        instruction=instruction_path.read_text(encoding="utf-8"),
        docker_image=env.get("docker_image"),
        dockerfile=dockerfile if dockerfile.exists() else None,
        tests_dir=tests_dir,
        verifier_timeout=float((config.get("verifier") or {}).get("timeout_sec", 120)),
        agent_timeout=float((config.get("agent") or {}).get("timeout_sec", 1800)),
        workdir=str(env.get("workdir", workdir)),
        verifier_image=((config.get("verifier") or {}).get("environment") or {}).get("docker_image"),
        verifier_separate=(config.get("verifier") or {}).get("environment_mode") == "separate",
    )


def flatten_metrics(payload: Any, prefix: str = "") -> Dict[str, float]:
    """Every numeric leaf of a ``reward.json`` payload, under a dotted name.

    Harbor only requires the top-level ``reward``, and SWE-bench-Science's
    grader writes that as ``int(public_ok and private_ok)`` -- a *binary*
    number, the coarsest reward there is. The partial credit its own grader
    computes (``{"private": {"passed": 1, "collected": 3}}``) is one level
    down, so reading only the top level throws away the one graded signal on
    the benchmark that can rank two failing patches. Flattening keeps both:
    ``reward`` and ``private.passed`` are then both selectable metrics.
    """
    out: Dict[str, float] = {}
    for key, value in (payload or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=f"{name}."))
    return out


def parse_reward(logs_dir: os.PathLike) -> Dict[str, float]:
    """``reward.json`` (several metrics) if present, else ``reward.txt`` as ``{"reward": x}``."""
    logs = Path(logs_dir)
    as_json = logs / "reward.json"
    if as_json.exists():
        payload = json.loads(as_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            out = flatten_metrics(payload)
            if out:
                return out
        raise ValueError(f"{as_json} holds no numeric metrics")
    as_txt = logs / "reward.txt"
    if as_txt.exists():
        return {"reward": float(as_txt.read_text(encoding="utf-8").strip() or "0")}
    raise FileNotFoundError(f"no reward.json or reward.txt under {logs}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


class Runner(Protocol):
    """Where a patch is applied and the verifier runs."""

    def verify(self, task: HarborTask, patch: str) -> Dict[str, float]: ...


def _apply_patch(workspace: Path, patch: str) -> None:
    if not patch.strip():
        return
    # A patch that came through a code fence has been `strip()`ped, and a
    # unified diff whose last line lacks its newline is "malformed" to both
    # `git apply` and `patch`. The newline is part of the format, not content.
    if not patch.endswith("\n"):
        patch += "\n"
    for command in (["git", "apply", "--whitespace=nowarn", "-"], ["patch", "-p1", "-s"]):
        proc = subprocess.run(command, cwd=workspace, input=patch, text=True,
                              capture_output=True)
        if proc.returncode == 0:
            return
        error = (proc.stderr or proc.stdout).strip()
    raise RuntimeError(f"patch did not apply: {error[:400]}")


class LocalRunner:
    """Apply, run and verify on a host checkout of the task's source tree.

    ``source`` is the directory the patch is relative to (what ``/app`` holds in
    the container). ``test_command`` runs from the workspace with ``LOGS`` set
    to a fresh directory; Harbor's ``test.sh`` writes to ``/logs/verifier``, so
    a host run passes ``VERIFIER_LOGS`` too and the default command rewrites the
    absolute path. ``git`` is required, for ``apply`` and ``diff``.
    """

    def __init__(self, source: os.PathLike, *, test_command: Optional[Sequence[str]] = None,
                 timeout: Optional[float] = None, keep: bool = False) -> None:
        self.source = Path(source)
        self.test_command = list(test_command) if test_command else None
        self.timeout = timeout
        self.keep = keep

    def materialize(self, task: HarborTask, patch: str) -> Path:
        """A fresh workspace: the source tree, a git baseline, the patch applied."""
        workspace = Path(tempfile.mkdtemp(prefix="harbor-ws-"))
        shutil.copytree(self.source, workspace, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git"))
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        subprocess.run(["git", "-c", "user.name=harbor", "-c", "user.email=harbor@local",
                        "add", "-A"], cwd=workspace, check=True)
        subprocess.run(["git", "-c", "user.name=harbor", "-c", "user.email=harbor@local",
                        "commit", "-q", "-m", "baseline", "--allow-empty"],
                       cwd=workspace, check=True)
        _apply_patch(workspace, patch)
        return workspace

    @staticmethod
    def diff(workspace: Path) -> str:
        """Everything the agent changed, as a patch against the baseline commit."""
        subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
        proc = subprocess.run(["git", "diff", "--cached", "--binary"], cwd=workspace,
                              capture_output=True, text=True, check=True)
        return proc.stdout

    def verify(self, task: HarborTask, patch: str) -> Dict[str, float]:
        workspace = self.materialize(task, patch)
        try:
            logs = workspace / ".harbor-logs" / "verifier"
            logs.mkdir(parents=True)
            command = self.test_command or [
                "bash", "-c",
                # Harbor's script writes to /logs/verifier; on a host run that
                # path is rewritten to the workspace's own logs directory.
                'sed "s#/logs/verifier#$VERIFIER_LOGS#g" "$TESTS_DIR/test.sh" | bash',
            ]
            env = {**os.environ, "VERIFIER_LOGS": str(logs), "LOGS": str(logs),
                   "TESTS_DIR": str(task.tests_dir)}
            proc = subprocess.run(command, cwd=workspace, env=env, capture_output=True,
                                  text=True, timeout=self.timeout or task.verifier_timeout)
            try:
                return parse_reward(logs)
            except (FileNotFoundError, ValueError) as error:
                raise RuntimeError(
                    f"verifier wrote no reward (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout)[-400:]}") from error
        finally:
            if not self.keep:
                shutil.rmtree(workspace, ignore_errors=True)


class DockerRunner:
    """Verify inside the task's own image, as Harbor does. Needs a Docker daemon.

    The image is built from ``environment/Dockerfile`` (tagged by task) or
    pulled by name. The patch goes in on stdin, ``tests/`` is mounted read-only
    at ``/tests``, and the reward is read back from ``/logs/verifier`` in the
    container. The agent phase is not here: running an agent *inside* the
    container is ``harbor run --agent`` and is not reimplemented.
    """

    def __init__(self, *, docker: str = "docker", build: bool = True) -> None:
        self.docker = docker
        self.build = build
        self._images: Dict[str, str] = {}
        self._lock = threading.Lock()

    def image_for(self, task: HarborTask) -> str:
        """The image the *verifier* runs in: its own where the task ships one."""
        with self._lock:
            if task.name in self._images:
                return self._images[task.name]
            if task.verifier_image:
                image = task.verifier_image
            elif task.docker_image:
                image = task.docker_image
            elif task.dockerfile and self.build:
                image = "harbor-task-" + re.sub(r"[^a-z0-9]+", "-", task.name.lower()).strip("-")
                subprocess.run([self.docker, "build", "-q", "-t", image,
                                str(task.dockerfile.parent)], check=True, capture_output=True)
            else:
                raise RuntimeError(f"{task.name}: no docker_image and no environment/Dockerfile")
            self._images[task.name] = image
            return image

    def verify(self, task: HarborTask, patch: str) -> Dict[str, float]:
        image = self.image_for(task)
        mounts: List[str] = []
        if task.verifier_separate:
            # Harbor's separate-verifier contract: the image ships its own
            # ``tests/`` *and* its own ``test.sh``, and that script begins
            # ``git reset --hard; git clean -ffdqx`` before applying the
            # agent's patch from ``/logs/artifacts/model.patch``. Applying the
            # patch to the workdir ourselves would therefore be silently
            # undone -- every candidate would score the baseline and the
            # search would look flat for a reason that is not the search.
            # So hand the patch over where the verifier looks for it, and let
            # its own script apply it.
            apply_step = ""
            patch_mount = "/logs/artifacts/model.patch"
            # Published images can carry a *stale* grader. Several
            # SWE-bench-Science verifier images run
            # `pytest /tests/private_tests/test_task_NNN.py` while the private
            # test they ship is named for its subject, so pytest exits 4 with
            # "file or directory not found", `private` reports 0 of 0, and
            # `reward` is 0 for any patch whatsoever. The dataset's own
            # `tests/grader.py` is the fixed version -- it runs the whole
            # directory -- so where the task ships one, mount it over the
            # image's. Measured on task_030: `private 0/0` becomes `5/10`.
            grader = task.tests_dir / "grader.py"
            if grader.is_file():
                mounts += ["-v", f"{grader.resolve()}:/tests/grader.py:ro"]
        else:
            apply_step = (
                "if [ -s /tmp/candidate.patch ]; then "
                "(git apply --whitespace=nowarn /tmp/candidate.patch "
                "|| patch -p1 -s < /tmp/candidate.patch); fi; ")
            patch_mount = "/tmp/candidate.patch"
            mounts += ["-v", f"{task.tests_dir.resolve()}:/tests:ro"]
        script = (
            f"set -e; cd {task.workdir}; mkdir -p /logs/verifier; " + apply_step +
            "bash /tests/test.sh >/logs/verifier/stdout.txt 2>&1 || true; "
            "if [ -f /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; "
            "else printf '{\"reward\": %s}' \"$(cat /logs/verifier/reward.txt 2>/dev/null || echo 0)\"; fi"
        )
        with tempfile.TemporaryDirectory(prefix="harbor-patch-") as tmp:
            patch_path = Path(tmp) / "candidate.patch"
            patch_path.write_text(patch, encoding="utf-8")
            proc = subprocess.run(
                [self.docker, "run", "--rm", "--network", "none", *mounts,
                 "-v", f"{patch_path.resolve()}:{patch_mount}:ro",
                 image, "bash", "-c", script],
                capture_output=True, text=True, timeout=task.verifier_timeout + 60)
        if proc.returncode != 0:
            raise RuntimeError(f"verifier container failed: {(proc.stderr or proc.stdout)[-400:]}")
        return flatten_metrics(_trailing_json(proc.stdout))


    def export_baseline(self, task: HarborTask, dest: os.PathLike) -> Path:
        """Copy the task's baseline checkout out of its image, into ``dest``.

        The benchmark is agentic: an agent edits files in a workspace and
        Harbor turns what it left behind into a patch. Asking a bare
        completion for a *unified diff* instead measures diff syntax, which is
        not the science under test -- measured on this task, 3 of 3 sampled
        patches were rejected by the verifier's own ``git apply`` before a
        single test ran, and a patch that fails to apply is scored exactly like
        one that applies and fails everything.

        With the baseline on disk, :func:`whole_file_patch` builds the diff
        with ``git`` from whole files, so it applies by construction and the
        model is never asked to count context lines.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        image = self.image_for(task)
        cid = subprocess.run([self.docker, "create", image], check=True,
                             capture_output=True, text=True).stdout.strip()
        try:
            subprocess.run([self.docker, "cp", f"{cid}:{task.workdir}/.", str(dest)],
                           check=True, capture_output=True)
        finally:
            subprocess.run([self.docker, "rm", "-f", cid], capture_output=True)
        return dest


def whole_file_patch(workspace: os.PathLike, files: Mapping[str, str]) -> str:
    """A diff against the baseline, from whole file contents. Applies by construction.

    ``files`` maps a path relative to the workspace to its complete new text.
    The workspace is restored to the baseline first, so each call is
    independent of the last.
    """
    ws = Path(workspace)
    # `git checkout -- .` restores the working tree from the *index*, so the
    # previous call's `git add -A` would survive it and leak into this patch.
    # Reset the index too, and clean ignored files (`-x`): the task's own
    # reproduction writes gitignored outputs.
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fdqx"], cwd=ws, check=True, capture_output=True)
    for rel, text in files.items():
        target = (ws / rel).resolve()
        if not str(target).startswith(str(ws.resolve()) + os.sep):
            raise ValueError(f"{rel!r} escapes the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, capture_output=True)
    return subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd=ws,
                          check=True, capture_output=True, text=True).stdout


def _trailing_json(text: str) -> Dict[str, Any]:
    """The JSON object at the end of ``text``, however many lines it spans.

    ``reward.json`` is written with ``indent=2`` by at least one benchmark, so
    the last *line* of the verifier's output is ``}``, not the payload.
    """
    stripped = text.rstrip()
    end = stripped.rfind("}")
    if end < 0:
        raise ValueError(f"verifier printed no JSON object: {stripped[-200:]!r}")
    for start in range(end, -1, -1):
        if stripped[start] == "{":
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"verifier printed no parseable JSON: {stripped[-200:]!r}")


# ---------------------------------------------------------------------------
# The domain
# ---------------------------------------------------------------------------


PARENT_PATCH_BEGIN = "<<<PARENT PATCH>>>"
PARENT_PATCH_END = "<<<END PARENT PATCH>>>"


def _patch_id(patch: str) -> str:
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()[:16]


def harbor_domain(task: HarborTask, runner: Runner, *, scoring: Sequence[str],
                  held_back: Sequence[str] = (), shards: Optional[int] = None) -> Domain:
    """One Harbor task in the four terms the ERA search needs.

    ``scoring`` and ``held_back`` name verifier metrics (``reward.json`` keys, or
    ``"reward"`` for a ``reward.txt`` task). Shard ``i`` reads
    ``scoring[i % len(scoring)]`` -- a task with one metric still gets the
    ``shards`` rollout tasks ``evolve()`` needs to split train from held-out,
    and the repeats cost nothing because a patch is verified once and cached.
    ``domain.test_shards`` are the held-back metrics; empty when nothing is
    held back, in which case the reported figure is the scoring figure.
    """
    scoring, held_back = list(scoring), list(held_back)
    if not scoring:
        raise ValueError("harbor_domain needs at least one scoring metric")
    if set(scoring) & set(held_back):
        raise ValueError(f"metrics cannot be both scoring and held back: {set(scoring) & set(held_back)}")
    n_shards = max(len(scoring), shards or 0)
    names = [scoring[i % len(scoring)] for i in range(n_shards)] + held_back
    test_shards = tuple(range(n_shards, n_shards + len(held_back)))
    cache: Dict[str, Tuple[bool, Dict[str, float], str]] = {}
    lock = threading.Lock()

    def verified(patch: str) -> Tuple[bool, Dict[str, float], str]:
        key = _patch_id(patch)
        with lock:
            if key in cache:
                return cache[key]
        try:
            metrics, error, valid = runner.verify(task, patch), "", True
        except Exception as exc:  # noqa: BLE001 - the candidate's failure, as a node
            metrics, error, valid = {}, f"{type(exc).__name__}: {exc}", False
        with lock:
            cache[key] = (valid, metrics, error)
        return cache[key]

    def evaluate(patch: str, shard_ids: Sequence[int]) -> Tuple[bool, Dict[str, Any], str]:
        valid, metrics, error = verified(patch)
        if not valid:
            return False, {"score": float("-inf"), "pass_rate": None, "metrics": {}}, error
        wanted = [names[int(i)] for i in shard_ids]
        missing = [m for m in wanted if m not in metrics]
        if missing:
            return False, {"score": float("-inf"), "pass_rate": None, "metrics": metrics}, \
                f"verifier reported no metric {missing}; it wrote {sorted(metrics)}"
        score = sum(metrics[m] for m in wanted) / len(wanted)
        return True, {"score": score, "pass_rate": score,
                      "metrics": {m: metrics[m] for m in wanted}}, ""

    def reward(metrics: Mapping[str, Any]) -> float:
        score = metrics.get("score")
        if score is None or score != score or score in (float("inf"), float("-inf")):
            return 0.0
        return min(1.0, max(0.0, float(score)))

    def prompt(parent: Any) -> str:
        code = getattr(parent, "code", "") or ""
        reached = (getattr(parent, "metrics", {}) or {}).get("metrics", {})
        return (
            f"# Task: {task.name}\n\n{task.instruction.strip()}\n\n"
            "## Where the search is\n"
            + ("The current attempt is the unmodified baseline.\n" if not code.strip()
               else "The current attempt is this patch against the baseline:\n"
                    f"{PARENT_PATCH_BEGIN}\n{code}\n{PARENT_PATCH_END}\n")
            + f"Scoring metrics it reached: {json.dumps(reached)}\n\n"
            "## What to return\n"
            "Work on the task in the workspace, then return the COMPLETE patch "
            "against the unmodified baseline (not against the current attempt) as "
            "a unified diff, in one fence with no language tag (```). Nothing else."
        )

    def task_prompt(index: int) -> str:
        return f"Verify the patch on scoring metric {names[index]!r} of {task.name}."

    return Domain(
        name=f"Harbor/{task.name}, mean of scoring metrics {scoring}",
        entrypoint="patch",
        metric_key="pass_rate",
        metric_better="higher",
        initial_program="",
        initial_summary="the unmodified baseline (empty patch)",
        evaluate=evaluate,
        reward=reward,
        prompt=prompt,
        task_prompt=task_prompt,
        test_shards=test_shards,
        data_summary={"task": task.name, "scoring": scoring, "held_back": held_back,
                      "shards": n_shards, "workdir": task.workdir,
                      "runner": type(runner).__name__},
    )


def harbor_completion(task: HarborTask, runner: LocalRunner, agent: WorkspaceAgent,
                      *, keep_workspaces: bool = False) -> Completion:
    """An agent that acts in a workspace, behind ERA's ``prompt -> text`` contract.

    The prompt carries the parent patch between :data:`PARENT_PATCH_BEGIN` and
    :data:`PARENT_PATCH_END`; this materialises a workspace with that patch
    applied, runs the agent there with the same prompt, and returns whatever it
    changed as a patch in a bare fence -- so the model never formats a diff.
    """
    begin, end = re.escape(PARENT_PATCH_BEGIN), re.escape(PARENT_PATCH_END)
    marker = re.compile(f"{begin}\n(.*?)\n{end}", re.DOTALL)

    def complete(prompt: str) -> str:
        match = marker.search(prompt)
        parent = match.group(1) if match else ""
        workspace = runner.materialize(task, parent)
        try:
            agent.in_workspace(str(workspace))(prompt)
            patch = runner.diff(workspace)
        finally:
            if not keep_workspaces:
                shutil.rmtree(workspace, ignore_errors=True)
        return f"```\n{patch}\n```"

    return complete
