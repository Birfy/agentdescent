"""Upstream's agent roles as Claude Code sessions, and the machinery they share.

Every EvoGit agent is a session with tools. ``EvoGit.Agent`` is "an agent session loop
template that manages a single agent session, handling tool loops, timeouts, and
graceful recovery", and ``agent/tools.ex`` hands out two sets by ``agent_type``:

* ``:read_write`` -- Architect, Manager, Executor -- gets ``file_read``,
  ``file_create``, ``file_write``, ``file_edit``, ``make_dir``, ``context_read`` /
  ``context_write`` / ``context_edit``, ``run_bash``, ``ripgrep``, ``glob``,
  ``list_dir``, ``search_context``, ``search_history`` and the skill tools.
* ``:read`` -- ContextExtractor, Investigator -- gets the read half, *plus*
  ``context_write`` / ``context_edit`` and ``run_bash``, because a read-only agent
  still has to write ``CONTEXT.md`` in its own repository and still has to be able to
  run the tests.

This port had every one of them as a single completion returning JSON. That is a real
simplification with two measured costs, and they are the same for each role:

* **It is the shape a reasoning model is slowest at.** One reply carries the whole
  answer, so the model thinks once, at length. On a coding-plan endpoint a single
  architect node took 148-264 s and ~15 000 output tokens, where the same domain on an
  endpoint that emitted ~800 tokens per call designed 23 nodes in under three minutes.
* **A truncated reply is not a short answer, it is no answer.** One run reported
  ``architect designed 0 nodes, 1 replies unusable`` and then grew a whole phase 2
  against no tree at all, because the JSON stopped mid-word.

A session has neither problem: it works in small turns, and its answer is a **file**,
so a file that exists is an answer that parses. What it also has, and the completion
never did, is the thing that makes these roles what upstream says they are -- a Manager
that "validates results" can *read the files the child wrote* instead of a diff
someone rendered for it, and a ContextExtractor whose whole job is "read code, write
the tree" can actually read the code.

One deliberate difference from upstream throughout: **the recursion stays in the
driver.** Upstream's agents spawn subagents and recurse inside the session; here one
session does one node's worth of work and the phase or the delegation policy drives the
next step. That keeps the level-parallelism, keeps each session small enough to watch,
and keeps the spatial contract checkable per node.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.filetree import materialize

from .._common import cli_env

from ._sandbox import PROVIDER_FILES, LocalSandbox, Workspace

__all__ = ["ARTIFACT_DIRS", "ARTIFACT_FILES", "AgentSession",
           "HOST_SESSION_VARS", "READ_ONLY_TOOLS",
           "READ_WRITE_TOOLS", "SCRATCH_DIR", "available_tools",
           "isolation_flags", "run_cli", "session_env", "session_home"]


#: How the host tells a CLI it spawns who it is. Inherited, every session in a run
#: *is* the host: one fly run's 52 episodes each wrote a transcript named with the
#: host's own session id, and their `TodoWrite` state -- keyed by that id -- landed in
#: the host's task list, two hundred entries of "Implement src/brain package".
HOST_SESSION_VARS = (
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_REMOTE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_DIAGNOSTICS_FILE",
    "CLAUDE_PID",
    # Not identity, but the same kind of leak: the reasoning cap the *host* session is
    # running under is not the one this run chose, and inherited it silently overrides
    # `--thinking-tokens`. Dropped here so the variable is present only when a role
    # puts it there.
    "MAX_THINKING_TOKENS",
)

_HOME_LOCK = threading.Lock()
_HOME: Optional[str] = None


def session_home() -> str:
    """One CLI state directory for this process's sessions, and not the host's.

    `CLAUDE_CONFIG_DIR` moves everything the CLI keeps per user -- transcripts, todos,
    synced skills, settings -- out of `~/.claude`. Without it a run writes into the
    state of whatever session launched it; one left 685 project directories behind.

    It is **not** deleted with the workspace, deliberately. The per-session transcripts
    are the only record of what an episode actually did, and reading 52 of them is how
    the wall was found to be what ends an episode. They are in the system temp
    directory, so they are transient without being gone before they can be read.
    """
    global _HOME
    with _HOME_LOCK:
        if _HOME is None:
            _HOME = tempfile.mkdtemp(prefix="genesis-home-")
        return _HOME


def session_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment a session runs in: the host's credentials, not its identity."""
    env = dict(cli_env() if base is None else base)
    for name in HOST_SESSION_VARS:
        env.pop(name, None)
    env["CLAUDE_CONFIG_DIR"] = session_home()
    return env


def isolation_flags(env: Mapping[str, str]) -> List[str]:
    """`--bare --strict-mcp-config` when the run brings its own key, else nothing.

    A session launched from inside a Claude Code session is handed that session's
    *equipment*, and equipment is priced per turn whether or not it is reachable.
    Measured against the same endpoint with the same one-line prompt: **31 850 input
    tokens plain, 31 099 with this port's own `--allowedTools`/`--disallowedTools`,
    1 317 bare.** The middle number is the mechanism: permission flags say what may be
    *called*, and every other schema is sent anyway. From the transcript's own
    `prompt_snapshot`, the system prompt is 5 720 characters and the tool schemas are
    178 742, of which `Artifact` alone is 64 168 -- beside a 13.5 KB skill listing, an
    agent listing, and the user's email address. Latency followed, 6.7 s to 2.4 s.

    Bare mode reads credentials strictly from `ANTHROPIC_API_KEY` -- never OAuth, never
    the keychain -- so it is used only when the run has one. A session billed to the
    local CLI's own sign-in would make no API call at all with it (measured:
    `duration_api_ms: 0`), which is a worse failure than a long prompt.

    What it drops is what a sandboxed executor has no use for: hooks, LSP, plugin sync,
    commit attribution, auto-memory, and `CLAUDE.md` auto-discovery -- the artifact
    carries `CONTEXT.md` records the brief names, not a `CLAUDE.md`. The fences are
    unaffected: a bare session still honours `.claude/settings.local.json`, verified by
    asking one to append to a denied path and watching it refuse.
    """
    if not env.get("ANTHROPIC_API_KEY"):
        return []
    return ["--bare", "--strict-mcp-config"]


def available_tools(tools: Sequence[str], flags: Sequence[str]) -> List[str]:
    """The requested tools, minus the one bare mode does not carry.

    **`--bare` does not expose `Write`**, and neither `--tools` nor `--allowedTools`
    brings it back: a bare session given `--tools Read,Write,Glob,Grep` answers "I only
    have a file Read tool available". Asking for it anyway is asking for a tool the
    session will be told does not exist -- one executor spent a turn discovering that
    and then wrote every file through `cat > f << EOF` instead.

    `Edit` is there and it creates a file that does not exist, which is how all seven
    phase-1 records in one run were written by architect sessions that have no shell at
    all. So under bare the write path is Edit, and the tool list should say so.
    """
    if "--bare" not in flags:
        return list(tools)
    out = [t for t in tools if t != "Write"]
    if out != list(tools) and "Edit" not in out:
        out.append("Edit")            # a role that could write must still be able to
    return out


#: Where a session that has to *answer* rather than edit puts its answer.
#:
#: A Manager decides and a reviewer judges; neither produces a file the repository
#: keeps. Upstream those answers are tool calls -- ``subagent_manager``, an accept or
#: reject -- and this port has no tool bus to carry them, so the session writes the
#: answer to a scratch path and the caller reads it back. The path is outside the
#: project by construction, so nothing here can be mistaken for work: `_read_tree`
#: skips it, and the workspace is deleted either way.
SCRATCH_DIR = ".genesis"

#: Directories a session *makes* rather than writes, skipped when the worktree is read
#: back. `.git`, `.claude` and `__pycache__` were here from the start; `.pytest_cache`
#: was not, and an episode is told to run the suite it is judged by, so every one of
#: them makes one.
#:
#: Measured, in a run four hours in: `src/arena/.pytest_cache/` reached the accepted
#: state, was materialised into every later workspace, and a manager session at
#: `src/arena` then spent seven turns reading `CACHEDIR.TAG`, `README.md` and
#: `v/cache/lastfailed` before writing its plan to
#: `src/arena/.pytest_cache/.genesis/plan.md`. :meth:`AgentSession.run` reads back the
#: path the caller named and nothing else, so that plan was never read: the episode
#: cost a session and returned nothing. One of them had nested the directory twice.
#:
#: A virtual environment is the same failure with three orders of magnitude more files,
#: which is why `venv` is here without having been seen yet.
ARTIFACT_DIRS = frozenset({
    ".git", ".claude", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".hypothesis", ".tox", ".nox", "node_modules", ".venv", "venv", "htmlcov",
    ".genesis",
})

#: The same idea for files that are output rather than work.
ARTIFACT_FILES = frozenset({".coverage", "coverage.xml", ".DS_Store"})


#: What upstream's `:read` agents get, mapped onto the CLI's tool names. The one job a
#: read-only agent does write -- `CONTEXT.md` -- is granted per role rather than here.
#:
#: Read this as a *request*, not a fence. `--allowedTools` is the CLI's auto-approve
#: list, and under `--permission-mode acceptEdits` a session reaches for whatever
#: built-in tool it likes: measured over one run, the manager used `Edit` 41 times and
#: the reviewer 6, and neither had `Edit` in its allowed tools. The fences that do hold
#: are `--disallowedTools` -- no role session in that run ran a shell or reached the
#: network -- and :meth:`AgentSession.run`, which returns the paths the caller named and
#: silently drops everything else the session touched.
READ_ONLY_TOOLS = ("Read", "Glob", "Grep")

#: What upstream's `:read_write` agents get, minus the shell. Bash is granted per role,
#: and only to the executor: the point of a session rather than a completion is that
#: the *implementer* can run the suite it is judged by. A design or review session that
#: can run things is one that starts implementing.
READ_WRITE_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep")


def run_cli(command: Sequence[str], *, cwd: str, env: Mapping[str, str],
            timeout: float) -> Tuple[bytes, int, bool]:
    """Run one session to completion or to the wall; ``(stdout, code, timed_out)``.

    The **process group** is the point, and `subprocess.run(timeout=)` does not have
    one: it signals the CLI and nothing else, so a shell the session started outlives
    the episode that started it. One run left a `python3 _cli.py serve` listening with
    its working directory already deleted, because its parent was the only process that
    got the signal. Here the session leads its own group and the group is killed on the
    way out -- after the wall, and after a clean finish too, because a session that
    leaves a server running has left it running either way.

    `timed_out` is returned rather than folded into the exit status because those are
    different events with different fixes: a session that hit the wall was still
    working, and one that died at 26 seconds was not.
    """
    proc = subprocess.Popen(list(command), cwd=cwd, env=dict(env),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    timed_out = False
    try:
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            try:
                out, _ = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:      # a pipe a grandchild still holds
                out = b""
    finally:
        _kill_group(proc)
    return out or b"", proc.returncode or 0, timed_out


def _kill_group(proc: "subprocess.Popen") -> None:
    """SIGKILL everything the session started, itself included. Never raises."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, PermissionError):
        pass


def _strays(workspace: str, state: Mapping[str, str],
            read: Sequence[str]) -> List[str]:
    """Paths the session changed that were neither its answer nor already there.

    Called before the workspace is deleted, because that is the only moment it can
    be. The scratch directory is not a stray -- it is where a session that has to
    answer rather than edit puts its answer -- and neither is anything in `read`.
    """
    skip = set(read) | set(PROVIDER_FILES)
    out: List[str] = []
    for base, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in ARTIFACT_DIRS]
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, workspace).replace(os.sep, "/")
            if rel in skip or name in ARTIFACT_FILES:
                continue
            try:
                with open(full, encoding="utf-8") as handle:
                    body = handle.read()
            except (UnicodeDecodeError, OSError):
                continue
            if state.get(rel) != body:
                out.append(rel)
    return sorted(out)


class AgentSession:
    """One headless Claude Code session in a throwaway copy of the repository.

    The shared half of every role below: put the state on disk, run the session with
    the tools that role is entitled to, and read back the paths the caller asked for.
    What the session *decided* is whatever is in those paths; what it says in its reply
    is not read, because a reply is the thing that truncates.
    """

    def __init__(self, *, tools: Sequence[str] = READ_WRITE_TOOLS,
                 frozen: Sequence[str] = (), binary: str = "claude",
                 model: str = "", max_turns: int = 40, timeout: float = 900.0,
                 root: Optional[str] = None, thinking_tokens: int = 0,
                 allow_bash: bool = False, sandbox=None):
        #: Where a session runs and what it can see from there -- see `._sandbox`. A
        #: plain directory unless the run hands over a real one, because a default that
        #: went looking for a container engine would make every unit test wait for one.
        self.sandbox = sandbox if sandbox is not None else LocalSandbox(root or "")
        self._tools = tuple(tools)
        self._frozen = tuple(frozen)
        self._binary = binary
        self._model = model
        self._max_turns = max(1, int(max_turns))
        self._timeout = timeout
        self._root = root
        #: A per-turn reasoning cap, or 0 to leave the CLI's own. Upstream carries
        #: reasoning strength per model profile (`reasoning_effort`, beside
        #: `max_tokens` and `concurrency`) rather than fixing it, and on a coding-plan
        #: endpoint this was 497s against 203-242s for the same design, same record.
        #: It bounds reasoning; it does not disable it, which is a different and worse
        #: lever -- that changes what the model produces, not only how long it takes.
        self._thinking_tokens = max(0, int(thinking_tokens))
        self._allow_bash = allow_bash
        #: Sessions run, and what came back. `failed` is the total; `timeouts` is
        #: the half of it that was still working when the wall arrived, which is the
        #: only one of the two that `--session-timeout` can do anything about.
        self.sessions = 0
        self.failed = 0
        self.timeouts = 0
        self.turns = 0
        #: Paths the last session changed that it was not asked for -- set by
        #: :meth:`run` before the workspace is deleted, because afterwards there is
        #: nothing left to look at. A role that cares (an architect, which is meant
        #: to write one record and no code) counts them.
        self.changed: List[str] = []

    def run(self, state: Mapping[str, str], prompt: str, *,
            read: Sequence[str]) -> Dict[str, str]:
        """Run one session over `state`; return the `read` paths that came back.

        A path the session did not write is simply absent from the result. A session
        reported as failed is still read: the turn limit in particular lands *after*
        the work, so the file is what decides, not the exit status.
        """
        space = self.sandbox.open("genesis-sess-")
        try:
            materialize(dict(state), space.path)
            self._write_settings(space.path)
            self.sessions += 1
            if not self._invoke(space, prompt):
                self.failed += 1
            out: Dict[str, str] = {}
            for rel in read:
                try:
                    with open(os.path.join(space.path, rel), encoding="utf-8") as fh:
                        body = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if body.strip():
                    out[rel] = body
            self.changed = _strays(space.path, state, read)
            return out
        finally:
            space.close()

    def summary(self) -> str:
        return (f"sessions={self.sessions} failed={self.failed} "
                f"timeout={self.timeouts} turns={self.turns}")

    # -- internals ---------------------------------------------------------

    def _write_settings(self, workspace: str) -> None:
        """Deny the frozen globs by name, inside the sandbox as well as outside it."""
        deny: List[str] = []
        for pattern in self._frozen:
            deny += [f"Write({pattern})", f"Edit({pattern})",
                     f"NotebookEdit({pattern})"]
        settings = {"permissions": {"deny": deny, "additionalDirectories": []}}
        target = os.path.join(workspace, ".claude")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "settings.local.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(settings, handle)

    def _command(self, prompt: str, env: Optional[Mapping[str, str]] = None,
                 ) -> List[str]:
        tools = list(self._tools)
        if self._allow_bash and "Bash" not in tools:
            tools.append("Bash")
        denied = ["WebFetch", "WebSearch", "Task"]
        if not self._allow_bash:
            denied.insert(0, "Bash")
        flags = isolation_flags(env or {})
        tools = available_tools(tools, flags)
        command = [self._binary, "-p", prompt,
                   "--output-format", "json",
                   "--permission-mode", "acceptEdits",
                   "--max-turns", str(self._max_turns),
                   "--allowedTools", ",".join(tools),
                   "--disallowedTools", ",".join(denied)] + flags
        if self._model:
            command += ["--model", self._model]
        return command

    def _invoke(self, space: Workspace, prompt: str) -> bool:
        env = session_env()
        if self._thinking_tokens:
            env["MAX_THINKING_TOKENS"] = str(self._thinking_tokens)
        try:
            out, code, timed_out = run_cli(space.command(self._command(prompt, env), env),
                                           cwd=space.path, env=env,
                                           timeout=self._timeout)
        except Exception:  # noqa: BLE001 - a dead session costs its turn
            return False
        if timed_out:
            # Killing the process group kills the `exec`, not what it is exec'ing into:
            # inside a container that is a different process tree on the same machine.
            space.kill()
            # The wall, and it is worth telling apart: there is no report to read,
            # so `turns` does not count what this session did. The file it may have
            # written by then is still read back, which is why `run` reads either way.
            self.timeouts += 1
            return False
        try:
            report = json.loads(out.decode("utf-8", "replace") or "{}")
            self.turns += int(report.get("num_turns") or 0)
            return not report.get("is_error", code != 0)
        except Exception:  # noqa: BLE001 - no JSON is not a reason to lose the work
            return code == 0
