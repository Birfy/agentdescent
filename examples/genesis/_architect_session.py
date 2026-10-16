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

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ._session import READ_WRITE_TOOLS, AgentSession
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
                 root: Optional[str] = None, thinking_tokens: int = 0,
                 sandbox=None):
        # `:read_write` upstream, minus the shell -- see `_session.READ_WRITE_TOOLS`
        # for why the executor gets Bash and a design session does not.
        self.session = AgentSession(
            tools=READ_WRITE_TOOLS, frozen=frozen, binary=binary, model=model,
            max_turns=max_turns or self.TURNS, timeout=timeout, root=root,
            thinking_tokens=thinking_tokens, sandbox=sandbox)
        #: Files a session wrote that were not its own record, discarded. An architect
        #: that starts writing code is the failure mode this counts. The session runs
        #: in a throwaway copy, so a stray costs nothing but is worth seeing.
        self.strays = 0

    # The phase reads these off the designer; keeping them as properties means one
    # set of counters rather than two that can disagree.
    @property
    def sessions(self) -> int:
        return self.session.sessions

    @property
    def failed(self) -> int:
        return self.session.failed

    @property
    def turns(self) -> int:
        return self.session.turns

    def __call__(self, state: Mapping[str, str], path: str, rules: str,
                 ) -> Tuple[Optional[str], List[Dict[str, object]]]:
        path = normalise(path)
        record_path = f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE
        prefix = f"{path}/" if path else ""
        prompt = ARCHITECT_SESSION_BRIEF.format(
            rules=rules, record_path=record_path, path_prefix=prefix)
        got = self.session.run(state, prompt, read=[record_path])
        self.strays += len(self.session.changed)
        body = (got.get(record_path) or "").strip()
        if not body:
            return None, []
        record = body + "\n"
        return record, _children(record, prefix)

    def summary(self) -> str:
        return f"{self.session.summary()} strays={self.strays}"


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
