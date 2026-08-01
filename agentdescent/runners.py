"""Runners: let a **real agent** use the tree that is being evolved.

:data:`~agentdescent.evolution.Run` is ``(rendered, task) -> output``, and for a
text artifact that is enough -- the instruction goes in the prompt. A directory
cannot: a skill directory is only a skill directory if the agent can *read the
files*, which means the candidate has to exist on disk, in the layout the agent
expects, for the duration of one rollout.

That is what a runner does:

    parse the rendered tree -> materialise it into a fresh workspace
    -> overlay the pristine frozen files -> stage the task's fixtures
    -> run the agent bound to that directory -> clean up

**One workspace per call, always.** ``evolve(max_concurrency=N)`` runs workers in
threads and ``EvolvingArtifact.score`` opens its own pool of ``eval_concurrency``
threads, so a shared directory would let two candidates overwrite each other --
producing scores that look perfectly ordinary and are simply wrong.

The overlay is the other half of ``FileTree(frozen=...)``. Filtering proposals
stops a reflector from *editing* the test suite; it does nothing about candidate
code that rewrites ``conftest.py`` at run time. Writing the pristine copies back
over the candidate, after it has been materialised, is what makes "frozen"
a property of every rollout rather than a property of well-behaved proposals.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Dict, Mapping, Optional, Sequence

from .agents import AgentError, Completion, WorkspaceAgent
from .evolution import Task
from .filetree import TreeError, materialize, parse_tree

__all__ = [
    "LAYOUTS",
    "TEST_FAILURE_MARKER",
    "code_runner",
    "layout_prefix",
    "tree_runner",
]

#: Where the tree is written inside the workspace. ``{name}`` is the artifact's
#: directory name. Add your own by passing ``layout=`` a literal prefix string.
LAYOUTS: Dict[str, str] = {
    "claude_skill": ".claude/skills/{name}",   # project-scoped Claude Code skill
    "claude_agent": ".claude/agents",          # project-scoped subagent definitions
    "skill_library": ".claude/skills",         # a directory OF skills, one dir each
    "root": "",                                # the tree *is* the working directory
}

#: Prefix of the output a runner produces when the frozen test suite fails. The
#: rollout is not an error -- "the candidate broke the build" is exactly the
#: signal a reflector needs -- so it is reported in-band and scored 0.
TEST_FAILURE_MARKER = "AGENTDESCENT_GATE_FAILED"

_WS_PREFIX = "agentdescent-ws-"
_WS_MAX_AGE = 6 * 3600.0

_DEFAULT_PROMPT = (
    "{prompt}\n\n"
    "(The files under {tree_dir} in this directory are available to you; read "
    "them and follow them. Reply with only the final answer.)"
)


def layout_prefix(layout: str, name: str) -> str:
    """Resolve a layout name (or a literal prefix) to a path inside the workspace."""
    template = LAYOUTS.get(layout, layout)
    return template.format(name=name) if "{name}" in template else template


def _reap_stale_workspaces(max_age: float = _WS_MAX_AGE) -> int:
    """Collect workspaces left behind by a killed process.

    ``evolve()`` reaps its own scratch *ledgers* the same way
    (``_reap_stale_scratch_repos``) and for the same reason: ``atexit`` does not
    run on SIGKILL or an OOM kill, and a long sweep otherwise leaves one
    directory per rollout in ``$TMPDIR``. Best-effort and silent -- reclaiming
    disk must never fail a run."""
    root, now, n = tempfile.gettempdir(), time.time(), 0
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(_WS_PREFIX):
            continue
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) < max_age:
                continue
            shutil.rmtree(path, ignore_errors=True)
            n += 1
        except OSError:
            pass
    return n


def _stage(rendered: str, task: Task, *, prefix: str,
           overlay: Optional[Mapping[str, str]],
           fixtures: Optional[Callable[[Task], Mapping[str, str]]],
           workspace_root: Optional[str]) -> str:
    """Build one workspace for one rollout and return its path."""
    tree = parse_tree(rendered)
    ws = tempfile.mkdtemp(prefix=_WS_PREFIX, dir=workspace_root)
    try:
        materialize(tree, ws, prefix=prefix)
        if overlay:
            # after the candidate, so the candidate cannot win the race.
            materialize(overlay, ws, prefix=prefix)
        staged = fixtures(task) if fixtures else (task.meta or {}).get("fixtures")
        if staged:
            materialize(staged, ws)
    except Exception:
        shutil.rmtree(ws, ignore_errors=True)
        raise
    return ws


def tree_runner(agent: Completion, *, layout: str = "claude_skill",
                name: str = "artifact",
                prompt_template: str = _DEFAULT_PROMPT,
                overlay: Optional[Mapping[str, str]] = None,
                fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
                answer_file: Optional[str] = None,
                keep_failed: bool = False,
                workspace_root: Optional[str] = None) -> Callable[[str, Task], str]:
    """Build a ``run(rendered, task)`` that gives ``agent`` the evolving directory.

    ``agent`` should be a :class:`~agentdescent.agents.WorkspaceAgent`
    (``claude_code()``, ``codex()``, ``cli_agent([...])``, ``openhands()``); a
    plain completion still works but never sees the files, which makes the whole
    exercise meaningless -- so that case raises rather than quietly scoring the
    model's prior knowledge.

    ``prompt_template`` may use ``{prompt}`` (the task), ``{tree_dir}`` (where the
    tree was written, relative to the workspace) and ``{name}``.
    ``answer_file`` reads the answer from a file the agent is told to write,
    instead of stdout -- useful for agents whose stdout carries chatter.
    """
    if not isinstance(agent, WorkspaceAgent):
        raise TypeError(
            f"tree_runner needs a WorkspaceAgent so the evolving directory can be "
            f"put in front of it; {type(agent).__name__} is a plain Completion and "
            "would only ever see the prompt. Use claude_code(), codex(), "
            "cli_agent([...]) or openhands().")
    prefix = layout_prefix(layout, name)
    _reap_stale_workspaces()

    def run(rendered: str, task: Task) -> str:
        ws = _stage(rendered, task, prefix=prefix, overlay=overlay,
                    fixtures=fixtures, workspace_root=workspace_root)
        ok = False
        try:
            prompt = prompt_template.format(prompt=task.prompt, name=name,
                                            tree_dir=prefix or ".")
            out = agent.in_workspace(ws)(prompt)
            if answer_file:
                path = os.path.join(ws, answer_file)
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        out = fh.read()
            ok = True
            return (out or "").strip()
        finally:
            if ok or not keep_failed:
                shutil.rmtree(ws, ignore_errors=True)

    return run


# ---------------------------------------------------------------------------
# Evolving code: the same staging, plus a gate that actually runs it
# ---------------------------------------------------------------------------

#: Environment handed to candidate code. Deliberately tiny: the process is
#: model-authored, and the ambient environment of a research run holds API keys.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "TEMP", "TMP")


def _child_env(ws: str, extra: Optional[Mapping[str, str]]) -> Dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    # HOME points at the workspace, not the real one: candidate code should not
    # be able to read ~/.claude or ~/.aws just because it can read files.
    env["HOME"] = ws
    env["TMPDIR"] = ws
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def _sh(cmd: Sequence[str], ws: str, timeout: float,
        env: Mapping[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), cwd=ws, capture_output=True, text=True,
                          timeout=timeout, env=dict(env))


def code_runner(entrypoint: Sequence[str], *, layout: str = "root",
                name: str = "agent",
                setup_cmd: Optional[Sequence[str]] = None,
                test_cmd: Optional[Sequence[str]] = None,
                overlay: Optional[Mapping[str, str]] = None,
                fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
                timeout: float = 120.0,
                env: Optional[Mapping[str, str]] = None,
                workspace_root: Optional[str] = None) -> Callable[[str, Task], str]:
    """Run **candidate code** on a task: materialise, gate, execute.

    ``entrypoint`` is argv; the task prompt is appended as the last argument and
    stdout is the output. ``setup_cmd`` runs first (e.g. ``["pip", "install",
    "-e", "."]``), ``test_cmd`` is the correctness gate (e.g. ``["pytest", "-q"]``)
    -- and because ``test_cmd`` is invoked *from outside the tree* over an overlay
    of pristine frozen files, a candidate cannot pass by rewriting its own tests.

    A failing gate is **not** an exception: the output becomes
    ``TEST_FAILURE_MARKER`` plus the captured stderr, which scores 0 and gives the
    reflector something to fix. Only infrastructure faults raise.

    This is process isolation, **not a sandbox**: a trimmed environment, ``HOME``
    inside the workspace, and a hard timeout. Model-authored code still runs with
    your user's permissions. Use a container for anything you would not run by
    hand.
    """
    prefix = layout_prefix(layout, name)
    _reap_stale_workspaces()

    def run(rendered: str, task: Task) -> str:
        ws = _stage(rendered, task, prefix=prefix, overlay=overlay,
                    fixtures=fixtures, workspace_root=workspace_root)
        child = _child_env(ws, env)
        try:
            for label, cmd in (("setup", setup_cmd), ("tests", test_cmd)):
                if not cmd:
                    continue
                try:
                    proc = _sh(cmd, ws, timeout, child)
                except subprocess.TimeoutExpired:
                    return (f"{TEST_FAILURE_MARKER} ({label} timed out after "
                            f"{timeout:g}s)")
                except FileNotFoundError as e:
                    raise AgentError(f"{cmd[0]!r} is not installed or not on PATH") from e
                if proc.returncode != 0:
                    detail = (proc.stdout or "") + (proc.stderr or "")
                    return f"{TEST_FAILURE_MARKER} ({label}):\n{detail.strip()[:2000]}"
            try:
                proc = _sh([*entrypoint, task.prompt], ws, timeout, child)
            except subprocess.TimeoutExpired:
                return f"{TEST_FAILURE_MARKER} (entrypoint timed out after {timeout:g}s)"
            except FileNotFoundError as e:
                raise AgentError(
                    f"{entrypoint[0]!r} is not installed or not on PATH") from e
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()[:2000]
                return f"{TEST_FAILURE_MARKER} (entrypoint exited {proc.returncode}):\n{detail}"
            return (proc.stdout or "").strip()
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    return run
