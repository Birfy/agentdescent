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

1. *The session's own*: writes are allowed only through `Write`/`Edit`, `Bash` is off
   by default, and the network tools are off outright. A settings file denies the
   frozen globs by name, so the suite and the specification cannot be edited even by
   accident -- a system that can edit its own tests has no tests.
2. *The sandbox*: the session runs with its cwd inside a throwaway worktree holding a
   copy of the accepted version, never the real repository. Nothing it does survives
   except through the diff.
3. *The port's own contract*, unchanged and last: an edit outside the node's subtree is
   not an error, it is an upward **request** (`agents/executor.ex`), and a write to a
   frozen path is dropped and counted. Both were already enforced on every edit and
   still are -- this module hands them edits and nothing else.

The session is billed to whatever credentials the local `claude` CLI is signed in with,
which is *not* the `--model` endpoint the rest of the run uses. That is a deliberate
restriction: it is opt-in with `--executor claude-code`, and the header says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from agentdescent.filetree import match_any, materialize

from .._common import cli_env

from ._delegation import Brief, Edit
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

    def __init__(self, *, frozen: Sequence[str] = (), binary: str = "claude",
                 model: str = "", max_turns: int = 0, root_turns: int = 0,
                 timeout: float = 900.0,
                 allow_bash: bool = True, failure: str = "",
                 root: Optional[str] = None):
        self._frozen = tuple(frozen)
        self._binary = binary
        self._model = model
        self._max_turns = max_turns or self.CHILD_TURNS
        self._root_turns = root_turns or max_turns or self.ROOT_TURNS
        self._timeout = timeout
        self._allow_bash = allow_bash
        self._failure = failure
        self._root = root
        #: Sessions run, and what came back.
        self.sessions = 0
        self.failed = 0
        self.edits = 0
        self.requests = 0
        self.turns = 0

    def __call__(self, brief: Brief) -> Sequence[Edit]:
        owner = normalise(brief.world.path)
        workspace = tempfile.mkdtemp(prefix="genesis-cc-", dir=self._root)
        try:
            before = dict(brief.state)
            materialize(before, workspace)
            self._write_settings(workspace)
            self.sessions += 1
            ok = self._run(workspace, owner, brief)
            if not ok:
                self.failed += 1
            after = _read_tree(workspace)
            edits = _diff(before, after, owner)
            self.edits += sum(1 for e in edits if owns(owner, e.path))
            self.requests += sum(1 for e in edits if not owns(owner, e.path))
            return edits
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

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

    def _command(self, prompt: str, turns: int) -> List[str]:
        tools = ["Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"]
        if self._allow_bash:
            # The point of a session rather than a completion is that it can run the
            # suite it is judged by. Bash is the only way to do that.
            tools.append("Bash")
        command = [self._binary, "-p", prompt,
                   "--output-format", "json",
                   "--permission-mode", "acceptEdits",
                   "--max-turns", str(turns),
                   "--allowedTools", ",".join(tools),
                   "--disallowedTools", "WebFetch,WebSearch,Task"]
        if self._model:
            command += ["--model", self._model]
        return command

    def _run(self, workspace: str, owner: str, brief: Brief) -> bool:
        prompt = CLAUDE_CODE_BRIEF.format(
            path=owner or ".", objective=brief.objective,
            frozen=", ".join(self._frozen) or "(none)",
            failure=self._failure.format(prompt=getattr(brief.task, "prompt", ""),
                                         output=(brief.output or "")[:400],
                                         reward=brief.reward) if self._failure else "")
        try:
            turns = self._root_turns if brief.depth == 0 else self._max_turns
            # `cli_env` is a no-op unless this run points ANTHROPIC_BASE_URL at a
            # third-party endpoint. When it does, it drops the variables by which a
            # managed session tells the CLI to use the *host's* provider and ignore
            # the environment -- inherited, they make every session here 401.
            out = subprocess.run(self._command(prompt, turns), cwd=workspace,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=self._timeout, env=cli_env())
        except Exception:  # noqa: BLE001 - a dead session costs its episode
            return False
        try:
            report = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
            self.turns += int(report.get("num_turns") or 0)
            return not report.get("is_error", out.returncode != 0)
        except Exception:  # noqa: BLE001 - no JSON is not a reason to lose the diff
            return out.returncode == 0

    def summary(self) -> str:
        return (f"sessions={self.sessions} failed={self.failed} "
                f"turns={self.turns} edits={self.edits} requests={self.requests}")


def _read_tree(workspace: str) -> Dict[str, str]:
    """The worktree as a state dict, skipping what is not the project."""
    out: Dict[str, str] = {}
    for base, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", ".claude", "__pycache__")]
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, workspace).replace(os.sep, "/")
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
