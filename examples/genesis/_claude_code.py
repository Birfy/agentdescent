"""An episode as a Claude Code session, which is what an episode is upstream.

The largest distance between this port and the system it ports has a number on it. The
paper is explicit that "the agent can execute **multiple model-tool turns** during one
supervised episode", and the runs are configured at **2 048 root turns and 128 turns
per non-root episode**, with file, shell and test tools inside each one. Every actor
here was a *single completion* with no tools: the manager is asked once where to
delegate, the executor is asked once for whole files, and everything either of them
might have looked up has to be pushed into the brief because neither can go and look.

This closes that. An executor episode becomes one headless Claude Code invocation in a
worktree of its own, and the work it did is recovered by diffing that worktree. What
changes is not the contract -- the spatial contract, the parent's verdict, the merge and
the frozen restore are all unchanged, and they are enforced on the edits that come back
exactly as before -- but what happens *inside* one episode: it can read the failing
test, open the module next door, try something, run the suite, and fix what it broke,
which is precisely the loop a single completion cannot have.

**Permissions, and why they are set the way they are.** Three fences, none of which
replaces the others:

1. *The session's own*: the network tools are off and `Bash` is off by default, both
   through `--disallowedTools`, which is the half of the CLI's tool flags that is
   actually a fence -- `--allowedTools` only auto-approves, and a session reaches for
   built-in tools that are not on it. A settings file denies the frozen globs by name,
   so the suite and the specification cannot be edited even by accident -- a system
   that can edit its own tests has no tests.
2. *The sandbox*: the session runs with its cwd inside a throwaway worktree holding a
   copy of the accepted version, never the real repository. Nothing it does survives
   except through the diff.
3. *The port's own contract*, unchanged and last: an edit outside the node's subtree is
   not an error, it is an upward **request** (`agents/executor.ex`), and a write to a
   frozen path is dropped and counted. Both were already enforced on every edit and
   still are -- this module hands them edits and nothing else.

**What the sandbox is not.** It bounds what *survives*, not what can be *seen*. A
session that has a shell can read anything the process can read, and four of twelve
episodes in one run ran `find /` and read a previous run's output from `/tmp` -- one of
them opened the very `_cli.py` that answered the acceptance failure it had been asked
to reproduce. The brief now says not to, which is a rule and not a wall: the blind
property here is enforced by the prompt and by what is *in* the worktree, not by the
operating system. A run that needs it enforced needs a container, and a run on a machine
with an earlier run's output still on it should clean that up first.

The session is billed to whatever credentials the local `claude` CLI is signed in with,
which is *not* the `--model` endpoint the rest of the run uses. That is a deliberate
restriction: it is opt-in with `--executor claude-code`, and the header says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from agentdescent.filetree import match_any, materialize

from .._common import cli_env

from ._delegation import Brief, Edit
from ._sandbox import PROVIDER_FILES, LocalSandbox, Workspace
from ._session import (ARTIFACT_DIRS, ARTIFACT_FILES, available_tools,
                       lean_agent_flags,
                       isolation_flags, run_cli, session_env)
from ._spatial import SITUATED_EDIT_PROTOCOL  # noqa: F401  (documented sibling)
from ._world import normalise, owns

__all__ = ["CLAUDE_CODE_BRIEF", "ClaudeCodeExecutor", "claude_code_available"]

#: What the session is told. Deliberately short: unlike a single completion, this agent
#: can read the tree it is standing in, so the brief carries the objective, the
#: evidence and the fences -- not the contents of files it can open itself.
CLAUDE_CODE_BRIEF = """You are an executor agent in a recursive software world. You are \
situated at the repository path `{path}` inside this checkout, and you may write ONLY \
files under it.

OBJECTIVE
{objective}

{failure}

Rules, in order of importance:

- Write only under `{path}`. If the change you need belongs somewhere else, do not make \
it: say so in your final message and the agent responsible for that path will be asked.
- These paths are the contract and are read-only: {frozen}. The suite is what you are \
judged by; editing it is not a way to pass it.
- Everything you need is in this checkout. Do not read outside it -- no `find /`, no \
other directory on this machine. The acceptance suite is not here and never will be, \
and another copy of this project that you find elsewhere is some other run's work, not \
this repository's state: reading it tells you nothing true and copying it is not the \
task.
- The tests that already pass have to keep passing. Run them.
- Make the change and stop. Do not refactor beyond the objective.

You are in a scratch copy of the project, so nothing you do here touches anything real. \
When you are done, finish. What you changed is read from the working tree."""


def claude_code_available(binary: str = "claude") -> bool:
    """Is the Claude Code CLI on the path and runnable?"""
    if shutil.which(binary) is None:
        return False
    try:
        out = subprocess.run([binary, "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=30, env=cli_env())
        return out.returncode == 0
    except Exception:  # noqa: BLE001 - anything at all means "no"
        return False


class ClaudeCodeExecutor:
    """``executor(brief) -> [Edit]``, where the episode is a Claude Code session.

    Drop-in for :func:`examples.genesis._suite.llm_executor`: same signature, same
    return type, and the edits it produces go through the same spatial contract.
    """

    #: Upstream's own budget, and the two numbers are not the same: a root agent gets
    #: up to **2,048** model-tool turns and a child **128**. A root is running the whole
    #: objective and a child one node of it, so one ceiling for both either starves the
    #: root or hands every leaf a session it has no use for.
    ROOT_TURNS = 2048
    CHILD_TURNS = 128

    #: What actually ends an episode, measured. Across 52 sessions of one formation
    #: run, 27 ran to this wall and 25 of those were killed mid-tool-call, 17 more had
    #: the endpoint drop the stream before it, and **none** reached :attr:`CHILD_TURNS`
    #: -- the busiest made 97 of its 128. The turn budget is upstream's and it is not
    #: the binding constraint; this is, so it is a number the run gets to choose rather
    #: than one buried in a default.
    TIMEOUT = 900.0

    def __init__(self, *, frozen: Sequence[str] = (), binary: str = "claude",
                 model: str = "", max_turns: int = 0, root_turns: int = 0,
                 timeout: float = 0.0, thinking_tokens: int = 0,
                 allow_bash: bool = True, failure: str = "",
                 root: Optional[str] = None, sandbox=None):
        #: Where an episode runs and what it can see -- see `._sandbox`. The executor is
        #: the role this matters most for: it is the only one with a shell.
        self.sandbox = sandbox if sandbox is not None else LocalSandbox(root or "")
        self._frozen = tuple(frozen)
        self._binary = binary
        self._model = model
        self._max_turns = max_turns or self.CHILD_TURNS
        self._root_turns = root_turns or max_turns or self.ROOT_TURNS
        self._timeout = float(timeout or self.TIMEOUT)
        #: The same per-turn reasoning cap the other roles carry. It reaches the CLI
        #: as an environment variable, and an executor left without it while every
        #: other role had one is an executor thinking at a different length than the
        #: run asked for -- which is exactly how it runs out of wall.
        self._thinking_tokens = max(0, int(thinking_tokens))
        self._allow_bash = allow_bash
        self._failure = failure
        self._root = root
        #: Sessions run, and what came back. `failed` is the total; `timeouts` is the
        #: part of it that was still working when the wall came, and `edits` is
        #: counted for those too -- a session that ran out of time wrote what it wrote.
        #: Siblings run concurrently, so every counter below is incremented
        #: from more than one thread. `x += 1` is a read and a write with a
        #: bytecode boundary between them; without this the numbers this port
        #: reports would quietly undercount exactly when the run is busiest.
        self._tally = threading.Lock()
        self.sessions = 0
        self.failed = 0
        self.timeouts = 0
        self.edits = 0
        self.requests = 0
        self.turns = 0

    def __call__(self, brief: Brief) -> Sequence[Edit]:
        owner = normalise(brief.world.path)
        space = self.sandbox.open("genesis-cc-")
        try:
            before = dict(brief.state)
            materialize(before, space.path)
            self._write_settings(space.path)
            with self._tally:
                self.sessions += 1
            ok = self._run(space, owner, brief)
            if not ok:
                with self._tally:
                    self.failed += 1
            after = _read_tree(space.path)
            edits = _diff(before, after, owner)
            with self._tally:
                self.edits += sum(1 for e in edits if owns(owner, e.path))
            with self._tally:
                self.requests += sum(1 for e in edits if not owns(owner, e.path))
            return edits
        finally:
            space.close()

    # -- internals ---------------------------------------------------------

    def _write_settings(self, workspace: str) -> None:
        """Deny the frozen globs by name, inside the sandbox as well as outside it."""
        deny: List[str] = []
        for pattern in self._frozen:
            deny += [f"Write({pattern})", f"Edit({pattern})",
                     f"NotebookEdit({pattern})"]
        settings = {"permissions": {"deny": deny,
                                    "additionalDirectories": []}}
        path = os.path.join(workspace, ".claude")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "settings.local.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(settings, handle)

    def _command(self, prompt: str, turns: int,
                 env: Optional[Mapping[str, str]] = None) -> List[str]:
        # The fourth fence, and it faces the other way: the first three keep the
        # session out of the repository, this one keeps the host out of the session.
        flags = isolation_flags(env or {})
        tools = available_tools(["Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"],
                                flags)
        if self._allow_bash:
            # The point of a session rather than a completion is that it can run the
            # suite it is judged by. Bash is the only way to do that.
            tools.append("Bash")
        command = [self._binary, "-p", prompt,
                   "--output-format", "json",
                   "--permission-mode", "acceptEdits",
                   "--max-turns", str(turns),
                   "--allowedTools", ",".join(tools),
                   "--disallowedTools", "WebFetch,WebSearch,Task"] + flags
        # Declaring the toolset is what keeps the other thirty-seven schemas out of a
        # signed-in run; `--allowedTools` above only auto-approves. No-ops under
        # `--bare`. See `_session.lean_agent_flags`.
        command += lean_agent_flags(tools, flags)
        if self._model:
            command += ["--model", self._model]
        return command

    def _run(self, space: Workspace, owner: str, brief: Brief) -> bool:
        prompt = CLAUDE_CODE_BRIEF.format(
            path=owner or ".", objective=brief.objective,
            frozen=", ".join(self._frozen) or "(none)",
            failure=self._failure.format(prompt=getattr(brief.task, "prompt", ""),
                                         output=(brief.output or "")[:400],
                                         reward=brief.reward) if self._failure else "")
        # `session_env` carries the credentials and leaves the identity: `cli_env`
        # underneath it drops the variables by which a managed session tells the CLI
        # to use the *host's* provider and ignore the environment -- inherited, they
        # make every session here 401 -- and `session_env` then drops the ones that
        # say *which* session this is, and points the CLI's state somewhere private.
        env = session_env()
        if self._thinking_tokens:
            env["MAX_THINKING_TOKENS"] = str(self._thinking_tokens)
        try:
            turns = self._root_turns if brief.depth == 0 else self._max_turns
            out, code, timed_out = run_cli(
                space.command(self._command(prompt, turns, env), env),
                cwd=space.path, env=env, timeout=self._timeout)
        except Exception:  # noqa: BLE001 - a dead session costs its episode
            return False
        if timed_out:
            # The process group that gets killed is the `exec`'s, and inside a container
            # the session is a different tree on the same machine: stop it explicitly.
            space.kill()
            # There is no report after the wall, so this episode's turns are not in
            # `turns` -- which is why the counter read 309 for a run whose transcripts
            # hold 2 607. The diff is taken either way: the caller reads the worktree
            # after this returns, and half-finished work is still work.
            with self._tally:
                self.timeouts += 1
            return False
        try:
            report = json.loads(out.decode("utf-8", "replace") or "{}")
            with self._tally:
                self.turns += int(report.get("num_turns") or 0)
            return not report.get("is_error", code != 0)
        except Exception:  # noqa: BLE001 - no JSON is not a reason to lose the diff
            return code == 0

    def summary(self) -> str:
        return (f"sessions={self.sessions} failed={self.failed} "
                f"timeout={self.timeouts} turns={self.turns} "
                f"edits={self.edits} requests={self.requests}")


def _read_tree(workspace: str) -> Dict[str, str]:
    """The worktree as a state dict, skipping what is not the project."""
    out: Dict[str, str] = {}
    for base, dirs, files in os.walk(workspace):
        # What the session *made* is not what it wrote: an episode is told to run the
        # suite, a suite run leaves `.pytest_cache`, and without this that lands in the
        # accepted version and is materialised into every later workspace -- where the
        # next agent reads it as if it were the project. See `_session.ARTIFACT_DIRS`.
        dirs[:] = [d for d in dirs if d not in ARTIFACT_DIRS]
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, workspace).replace(os.sep, "/")
            if rel in PROVIDER_FILES or name in ARTIFACT_FILES:
                continue              # the sandbox's own bookkeeping, not the node's work
            try:
                with open(full, encoding="utf-8") as handle:
                    out[rel] = handle.read()
            except (UnicodeDecodeError, OSError):
                continue                      # a binary or unreadable file is not work
    return out


def _diff(before: Mapping[str, str], after: Mapping[str, str], owner: str) -> List[Edit]:
    """What the session changed, as edits owned by the node it was situated at.

    Deletions included: a session that removed a file did work, and the contract
    decides whether that work was its to do.
    """
    edits: List[Edit] = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        edits.append(Edit(owner=owner, path=path, content=new))
    return edits
