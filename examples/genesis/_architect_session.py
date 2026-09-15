"""Phase 1's architect as a Claude Code session, the way upstream runs it.

Upstream's Architect is not a question with an answer -- it is an agent session with
tools. ``agents/architect.ex`` is ``use EvoGit.Agent``, whose moduledoc calls itself
"an agent session loop template that manages a single agent session, handling tool
loops, timeouts, and graceful recovery", and its ``agent_type`` is ``:read_write``,
which in ``agent/tools.ex`` means the full set: ``file_read``, ``file_create``,
``file_write``, ``file_edit``, ``make_dir``, ``context_read``/``context_write``/
``context_edit``, ``run_bash``, ``ripgrep``, ``glob``, ``list_dir``. **`CONTEXT.md` is
written with a tool.** Nothing is returned as a structured reply, and there is nothing
to parse.

This port had the architect as one completion per node returning
``{"record": ..., "children": [...]}``. That is a real simplification and it cost
twice:

* **It is the shape a reasoning model is slowest at.** One reply has to hold a whole
  record and a whole child list, so the model thinks once, at length, and emits once,
  at length. Measured against a coding-plan endpoint: ~15 000 output tokens and 148-264
  s for a single node, where the same domain on an endpoint that emitted ~800 tokens
  per call designed 23 nodes in under three minutes. A session spends the same design
  on many small turns instead.
* **A truncated reply is an empty phase 1.** One run reported
  ``architect designed 0 nodes, 1 replies unusable`` and then grew an entire phase 2
  against no tree, because the JSON stopped mid-word. A session has no JSON: the record
  is a file, and a file that is written is a file that parses.

What is *not* upstream here, and is a deliberate choice: upstream's architect spawns
``subagent_architect`` for each child and the recursion happens inside the session.
Here one session designs **one node**, the port reads the routing table it wrote, and
the next level is driven by the phase, a level at a time. That keeps phase 1's existing
level-parallelism, keeps every session small enough to watch, and keeps the spatial
contract checkable per node -- at the cost of the recursion living in the driver rather
than in the agent. :class:`~examples.genesis._claude_code.ClaudeCodeExecutor` already
works this way for implementation episodes, and this is the same trade applied to
design.

The other deliberate difference: **no Bash.** The executor gets a shell because the
point of a session rather than a completion is that it can run the suite it is judged
by. An architect designs and does not implement, so there is nothing for it to run, and
a shell is how a design session turns into an implementation session by accident.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.filetree import materialize

from .._common import cli_env
from ._world import CONTEXT_FILE, normalise, parse_route_sizes, parse_routes

__all__ = ["ARCHITECT_SESSION_BRIEF", "ArchitectSession"]


#: What the session is told. The design rules themselves stay in
#: :data:`examples.genesis._architect.ARCHITECT_PROMPT` -- they are the same rules
#: whichever way the record is produced, and duplicating them is how the two paths
#: would drift apart. This wrapper says only what a *session* has to be told: which
#: file to write, and that writing it is the whole job.
ARCHITECT_SESSION_BRIEF = """{rules}

--- HOW TO DELIVER THIS ---

You are in a scratch copy of the repository, and you have file tools. Do not reply \
with the record: **write it to `{record_path}`** and stop. What you wrote is read from \
the working tree, so a reply is not read at all.

Write that one file. Create no directories, no source files, no stubs, and no other \
`CONTEXT.md` -- a child's record belongs to the agent situated there, and anything you \
write outside `{record_path}` is discarded.

Your routing table carries each child's size, so the line is:

    - `./{path_prefix}<name>/` (<N> files) -> <what it handles>

When you have written the file, you are done."""


class ArchitectSession:
    """``session(state, path, objective) -> (record, children)``.

    Drop-in for the completion the architect phase otherwise calls: same inputs, same
    return shape, so :class:`~examples.genesis._architect.ArchitectPhase` does not know
    which one it has. Failure is the same shape too -- ``(None, [])`` -- and is counted
    by the phase as an unusable reply, because from the tree's point of view a session
    that wrote nothing and a reply that did not parse are the same event.
    """

    #: A design turn is small -- read the contract, read the parent records, write one
    #: file -- so this is a bound on going wrong rather than a budget to spend. The
    #: implementation executor's 2048/128 are upstream's numbers for *implementation*
    #: sessions; there is no upstream number for a single-node design session, because
    #: upstream's architect session designs a whole subtree.
    TURNS = 40

    def __init__(self, *, frozen: Sequence[str] = (), binary: str = "claude",
                 model: str = "", max_turns: int = 0, timeout: float = 900.0,
                 root: Optional[str] = None, thinking_tokens: int = 0):
        self._frozen = tuple(frozen)
        self._binary = binary
        self._model = model
        self._max_turns = max_turns or self.TURNS
        self._timeout = timeout
        self._root = root
        #: A per-turn reasoning cap for the session, or 0 to leave the CLI's own.
        #: Upstream treats reasoning strength as a per-model setting rather than a
        #: constant -- a model profile carries `reasoning_effort` beside `max_tokens`
        #: and `concurrency` (`config/schema/definitions.ex`) -- and on a coding-plan
        #: endpoint this is the difference between a design turn that thinks for
        #: seconds and one that thinks for minutes. It bounds thinking; it does not
        #: disable it, which is a different and worse lever: turning reasoning off
        #: changes what the model produces, not only how long it takes.
        self._thinking_tokens = max(0, int(thinking_tokens))
        #: Sessions run, and what came back.
        self.sessions = 0
        self.failed = 0
        self.turns = 0
        #: Files a session wrote that were not its own record, discarded. An architect
        #: that starts writing code is the failure mode this counts.
        self.strays = 0

    def __call__(self, state: Mapping[str, str], path: str, rules: str,
                 ) -> Tuple[Optional[str], List[Dict[str, object]]]:
        path = normalise(path)
        record_path = f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE
        prefix = f"{path}/" if path else ""
        workspace = tempfile.mkdtemp(prefix="genesis-arch-", dir=self._root)
        try:
            before = dict(state)
            materialize(before, workspace)
            self._write_settings(workspace)
            self.sessions += 1
            prompt = ARCHITECT_SESSION_BRIEF.format(
                rules=rules, record_path=record_path, path_prefix=prefix)
            if not self._run(workspace, prompt):
                self.failed += 1
            record = self._read(workspace, record_path, before)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        if record is None:
            return None, []
        return record, _children(record, prefix)

    # -- internals ---------------------------------------------------------

    def _write_settings(self, workspace: str) -> None:
        """Deny the frozen globs by name, as the implementation executor does."""
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

    def _command(self, prompt: str) -> List[str]:
        # No Bash, no TodoWrite: this session reads and writes one file. Glob and Grep
        # stay because a node is designed against what is already in the tree.
        command = [self._binary, "-p", prompt,
                   "--output-format", "json",
                   "--permission-mode", "acceptEdits",
                   "--max-turns", str(self._max_turns),
                   "--allowedTools", "Read,Write,Edit,Glob,Grep",
                   "--disallowedTools", "Bash,WebFetch,WebSearch,Task"]
        if self._model:
            command += ["--model", self._model]
        return command

    def _run(self, workspace: str, prompt: str) -> bool:
        try:
            env = cli_env()
            if self._thinking_tokens:
                env["MAX_THINKING_TOKENS"] = str(self._thinking_tokens)
            out = subprocess.run(self._command(prompt), cwd=workspace,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=self._timeout, env=env)
        except Exception:  # noqa: BLE001 - a dead session costs its node
            return False
        try:
            report = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
            self.turns += int(report.get("num_turns") or 0)
            return not report.get("is_error", out.returncode != 0)
        except Exception:  # noqa: BLE001 - no JSON is not a reason to lose a record
            return out.returncode == 0

    def _read(self, workspace: str, record_path: str,
              before: Mapping[str, str]) -> Optional[str]:
        """The record the session wrote, or `None`; everything else it wrote is a stray.

        A session reported as failed may still have written the file before it died --
        the turn limit, in particular, lands *after* the work -- so the tree is read
        either way and the file is what decides.
        """
        try:
            with open(os.path.join(workspace, record_path), encoding="utf-8") as handle:
                record = handle.read()
        except OSError:
            record = ""
        self.strays += _strays(workspace, record_path, before)
        record = record.strip()
        return (record + "\n") if record else None

    def summary(self) -> str:
        return (f"sessions={self.sessions} failed={self.failed} "
                f"turns={self.turns} strays={self.strays}")


def _children(record: str, prefix: str) -> List[Dict[str, object]]:
    """The routing table, as the child dicts the architect phase folds.

    Same shape the JSON path produces, so the phase's guards -- the file-count
    threshold, the module-shadowing refusal, the ancestor-repeat check -- apply
    unchanged. A bare name is relative to this node; a path with a separator is taken
    as written, so a route out of the subtree stays a violation the caller counts
    rather than a relocation this function performs silently.
    """
    sizes = parse_route_sizes(record)
    out: List[Dict[str, object]] = []
    for child, handles in parse_routes(record):
        # The size is keyed by the path *as the record wrote it*, so it is read
        # before the bare name is resolved against this node -- `web/` is `web` in
        # the sizes map and `src/web` in the tree, and looking it up after the
        # rewrite silently reported every bare-named child as undeclared.
        declared = int(sizes.get(normalise(child), 0))
        if prefix and "/" not in child:
            child = prefix + child
        out.append({"path": child,
                    "objective": handles or f"implement {child}",
                    "files": declared or int(sizes.get(normalise(child), 0))})
    return out


def _strays(workspace: str, record_path: str, before: Mapping[str, str]) -> int:
    """How many files the session wrote that were not its own record."""
    count = 0
    for base, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", ".claude", "__pycache__")]
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), workspace)
            rel = rel.replace(os.sep, "/")
            if rel == record_path:
                continue
            try:
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    body = handle.read()
            except (UnicodeDecodeError, OSError):
                continue
            if before.get(rel) != body:
                count += 1
    return count
