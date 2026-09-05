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
import signal
import subprocess
import tempfile
import time
import warnings
from typing import (
    TYPE_CHECKING, Callable, Dict, Mapping, Optional, Sequence, Tuple,
)

from .agents import AgentError, Completion, WorkspaceAgent
from .evolution import Task
from .filetree import TreeError, materialize, parse_tree

if TYPE_CHECKING:                                   # pragma: no cover
    from .policies import SandboxSpec
    from .sandbox import SandboxPool

__all__ = [
    "LAYOUTS",
    "TEST_FAILURE_MARKER",
    "code_runner",
    "layout_prefix",
    "tree_runner",
    "gated_reward",
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

#: How many workspaces a runner that was given no pool will allow at once.
#: Matches `evolve`'s default `eval_concurrency`, which is the larger of the two
#: concurrencies that reach a runner today -- so the default ceiling does not
#: bind on any configuration that worked before.
_DEFAULT_MAX_SANDBOXES = 8

_DEFAULT_PROMPT = (
    "{prompt}\n\n"
    "(The files under {tree_dir} in this directory are available to you; read "
    "them and follow them. Reply with only the final answer.)"
)


def layout_prefix(layout: str, name: str) -> str:
    """Resolve a layout name (or a literal prefix) to a path inside the workspace."""
    template = LAYOUTS.get(layout, layout)
    return template.format(name=name) if "{name}" in template else template


def _pool_for(pool: Optional["SandboxPool"],
              workspace_root: Optional[str]) -> "Tuple[SandboxPool, SandboxSpec]":
    """The pool this runner will lease from, and the spec it asks for.

    A runner given no pool makes its own, so the common single-`evolve()` case
    needs no wiring. Sharing one across runners is what puts rollouts and
    evaluations under a single ceiling, which is the point of having one."""
    from .policies import SandboxSpec
    from .sandbox import SandboxPool, WorkspaceProvider

    spec = SandboxSpec(workspace_root=workspace_root)
    if pool is not None:
        return pool, spec
    provider = WorkspaceProvider()
    provider.reap()                 # collect what earlier killed processes left
    return SandboxPool(provider, max_sandboxes=_DEFAULT_MAX_SANDBOXES), spec


def _preserve(ws: str) -> Optional[str]:
    """Copy a workspace out before the pool reclaims it (`keep_failed`).

    Holding the sandbox itself would be the easy implementation and the wrong
    one: the ceiling exists to bound what is on the machine, and a debugging
    flag must not be able to raise it one failed rollout at a time."""
    try:
        dest = tempfile.mkdtemp(prefix="agentdescent-failed-")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(ws, dest, symlinks=True)
        return dest
    except OSError:
        return None


def _stage_into(ws: str, rendered: str, task: Task, *, prefix: str,
                overlay: Optional[Mapping[str, str]],
                fixtures: Optional[Callable[[Task], Mapping[str, str]]]) -> None:
    """Lay one rollout's files out inside an already-acquired workspace.

    Splitting this from *making* the directory is what lets the pool own the
    lifetime: staging can fail, and when it does the sandbox still has to go
    back through the same release path as a successful rollout."""
    tree = parse_tree(rendered)
    materialize(tree, ws, prefix=prefix)
    if overlay:
        # after the candidate, so the candidate cannot win the race.
        materialize(overlay, ws, prefix=prefix)
    staged = fixtures(task) if fixtures else (task.meta or {}).get("fixtures")
    if staged:
        materialize(staged, ws)


def tree_runner(agent: Completion, *, layout: str = "claude_skill",
                name: str = "artifact",
                prompt_template: str = _DEFAULT_PROMPT,
                overlay: Optional[Mapping[str, str]] = None,
                fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
                answer_file: Optional[str] = None,
                keep_failed: bool = False,
                workspace_root: Optional[str] = None,
                sandbox_pool: Optional["SandboxPool"] = None) -> Callable[[str, Task], str]:
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
    pool, spec = _pool_for(sandbox_pool, workspace_root)

    def run(rendered: str, task: Task) -> str:
        with pool.lease(spec) as sandbox:
            ws, ok = sandbox.root, False
            try:
                _stage_into(ws, rendered, task, prefix=prefix, overlay=overlay,
                            fixtures=fixtures)
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
                if keep_failed and not ok:
                    kept = _preserve(ws)
                    if kept:
                        warnings.warn(f"kept the failed workspace at {kept}",
                                      RuntimeWarning, stacklevel=2)

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
        env: Mapping[str, str],
        sandbox: Optional[object] = None) -> subprocess.CompletedProcess:
    """Run one gate command, and on timeout kill **everything it started**.

    ``subprocess.run(timeout=)`` kills the process it launched and nothing else.
    That is fine for a leaf command and wrong for every command here: ``pytest``
    forks, ``pip install`` forks, an agent CLI forks. On a timeout the direct
    child dies, the run continues, and the grandchildren keep going -- still
    burning CPU, still holding open a workspace the ``finally`` has already
    deleted. Eight workers hitting a timeout leave eight of them behind, and
    nothing in the run reports it.

    So the child gets its own process group (``start_new_session``) and the
    timeout path signals the **group**. ``SIGTERM`` first, because a test runner
    given the chance will remove its own temporary files; ``SIGKILL`` for
    whatever is left.

    POSIX-only: Windows has no process groups in this sense, so there the
    behaviour is what it always was -- documented rather than silently different.
    """
    # A sandbox that runs the command elsewhere says so by offering a prefix.
    # The local one does not, so this is `[] + cmd` and the default path is
    # exactly what it was.
    prefix = list(getattr(sandbox, "exec_prefix", list)() or ())
    if prefix:
        # The prefix runs on the *host* -- it is the engine's own CLI, and it
        # needs the host's environment to find its socket. The trimmed
        # environment is for the candidate, which is on the far side of the
        # prefix, so the sandbox injects it there instead. Handing the engine a
        # `HOME` inside the workspace makes it look for its socket in a directory
        # that is about to be deleted.
        env, cwd = os.environ, None
    else:
        cwd = ws
    posix = os.name != "nt"
    proc = subprocess.Popen(prefix + list(cmd), cwd=cwd, env=dict(env), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=posix)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Killing the local process group does nothing to a process running
        # inside a container -- the exec dies and its child keeps going. A
        # sandbox that can stop its own contents is asked to.
        stop = getattr(sandbox, "kill", None)
        if prefix and callable(stop):
            stop()
        _kill_group(proc, posix)
        # Drain after killing: the pipes may hold output that explains the hang,
        # and not reading them can leave the child blocked on a full pipe.
        try:
            out, err = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:                    # pragma: no cover
            out, err = "", ""
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _kill_group(proc: "subprocess.Popen", posix: bool) -> None:
    """Terminate the process group ``proc`` leads, then make sure it is gone."""
    if not posix:
        proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:                      # already reaped
        return
    for sig, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:                  # the group is gone; nothing left to do
            return
        if grace:
            deadline = time.time() + grace
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.02)


def code_runner(entrypoint: Sequence[str], *, layout: str = "root",
                name: str = "agent",
                setup_cmd: Optional[Sequence[str]] = None,
                test_cmd: Optional[Sequence[str]] = None,
                overlay: Optional[Mapping[str, str]] = None,
                fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
                timeout: float = 120.0,
                env: Optional[Mapping[str, str]] = None,
                workspace_root: Optional[str] = None,
                sandbox_pool: Optional["SandboxPool"] = None) -> Callable[[str, Task], str]:
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
    pool, spec = _pool_for(sandbox_pool, workspace_root)

    def run(rendered: str, task: Task) -> str:
        with pool.lease(spec) as sandbox:
            ws = sandbox.root
            _stage_into(ws, rendered, task, prefix=prefix, overlay=overlay,
                        fixtures=fixtures)
            return _gate_and_run(ws, task, sandbox)

    def _gate_and_run(ws: str, task: Task, sandbox=None) -> str:
        """Gate first, then the entrypoint. A failing gate speaks in-band."""
        child = _child_env(ws, env)
        for label, cmd in (("setup", setup_cmd), ("tests", test_cmd)):
            if not cmd:
                continue
            try:
                proc = _sh(cmd, ws, timeout, child, sandbox)
            except subprocess.TimeoutExpired:
                return (f"{TEST_FAILURE_MARKER} ({label} timed out after "
                        f"{timeout:g}s)")
            except FileNotFoundError as e:
                raise AgentError(f"{cmd[0]!r} is not installed or not on PATH") from e
            if proc.returncode != 0:
                detail = (proc.stdout or "") + (proc.stderr or "")
                return f"{TEST_FAILURE_MARKER} ({label}):\n{detail.strip()[:2000]}"
        try:
            proc = _sh([*entrypoint, task.prompt], ws, timeout, child, sandbox)
        except subprocess.TimeoutExpired:
            return f"{TEST_FAILURE_MARKER} (entrypoint timed out after {timeout:g}s)"
        except FileNotFoundError as e:
            raise AgentError(
                f"{entrypoint[0]!r} is not installed or not on PATH") from e
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:2000]
            return f"{TEST_FAILURE_MARKER} (entrypoint exited {proc.returncode}):\n{detail}"
        return (proc.stdout or "").strip()

    return run


def gated_reward(reward: Callable[[Task, str], float]) -> Callable[[Task, str], float]:
    """``reward``, with a failed :func:`code_runner` gate scoring 0.

    The gate speaks in-band -- a failing ``test_cmd`` makes the rollout's output
    ``TEST_FAILURE_MARKER`` plus the captured text, so the reflector can read
    what broke. Scoring that as an ordinary answer would let a candidate "pass"
    by echoing the marker; this is the wrapper every ``code_runner`` run needs
    around its reward, and the reason ``evolve()`` is handed this rather than
    ``reward`` directly."""
    def gated(task: Task, output: str) -> float:
        if isinstance(output, str) and output.startswith(TEST_FAILURE_MARKER):
            return 0.0
        return reward(task, output)
    return gated
