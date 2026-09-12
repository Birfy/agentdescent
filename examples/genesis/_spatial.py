"""The spatial contract as a :class:`~agentdescent.strategies.Strategy`.

Genesis's central invariant is one sentence, and both projects wrote it
independently::

    every agent is only supposed to edit files belonging to its own path
        -- genesis, apps/evo_git/lib/evo_git/agents/manager.ex

    each worker owns a section, so edits are conflict-free *by construction*
        -- agentdescent/parallel.py, TensorParallel

The engine's version of it is tensor parallelism, and TP cannot be used here for
a reason that is structural rather than incidental: its ownership map is built
once, before round 0, from a declared key space, and a key that is not in the map
belongs to no section -- so **every newly created file is a section violation**
(``evolution.py:2739``; ``FileTree.keys`` says so in its own docstring). Genesis's
formation setting is a repository with no implementation in it. Creating files is
not an edge case there, it is the entire run.

So the contract is enforced one layer lower, where creation is free: a strategy's
``to_diff`` already receives the ``author`` and the proposal, and it is the only
place a proposal becomes a diff. An edit carries the path of the agent that made
it, and an edit outside that agent's subtree is dropped and **counted**. Nothing
in the engine changes, and a file appearing for the first time is ordinary.

What that costs, stated rather than hidden: TP's guarantee is checked by the
engine against a map it owns, and this one is checked by example code against a
claim the proposal makes about itself. It binds a cooperative proposer -- which
is what the upstream system has too, since there the boundary is the sandbox's
write scope, not the agent's honesty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.evolvable import Diff, stable_hash
from agentdescent.filetree import (DEFAULT_MAX_FILE_BYTES, TreeError, canonical,
                                   match_any, safe_relpath)

from ._world import WorldLog, normalise, owns

__all__ = ["SITUATED_EDIT_PROTOCOL", "SpatialContract", "parse_situated_edits"]


#: The proposal protocol. `FileTree`'s, plus the one field that makes an edit
#: *situated*: the path of the finite-lived agent that produced it. Whole-file
#: replacement for `FileTree`'s reason -- a model-written patch that does not
#: apply fails silently, a whole file either parses or does not.
SITUATED_EDIT_PROTOCOL = """Reply with ONE block in exactly this format and nothing else:

<EDITS>
{{"rationale": "<one sentence>",
 "edits": [{{"owner": "<the path you were situated at>",
            "path": "<relative path, inside that subtree>",
            "content": "<the COMPLETE new file>"}}]}}
</EDITS>

Rules:
- `content` is the whole file, never a patch or an excerpt.
- `owner` is your own path. `path` MUST lie inside it -- an edit outside your
  subtree belongs to the agent responsible for that path, not to you.
- Only these paths are editable: {editable}
- Never edit: {frozen}
- To delete a file use {{"owner": "...", "path": "...", "delete": true}}."""


def parse_situated_edits(proposal: str) -> List[Dict[str, Any]]:
    """Parse a reply into ``[{owner, path, content|None}]``.

    Lenient about the wrapper and strict about the payload, for
    :func:`~agentdescent.treestrategy.parse_edits`' reason: a proposer that
    ignores the protocol is a quality problem the run should absorb and count,
    not a crash. An item with no ``owner`` is attributed to the root, which is
    the only attribution that cannot silently widen anyone's authority.
    """
    raw = _payload(proposal)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - malformed proposer output, not a bug
        return []
    if not isinstance(data, dict):
        return []
    items = data["edits"] if isinstance(data.get("edits"), list) else (
        [data] if "path" in data else [])
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or "path" not in item:
            continue
        try:
            path = safe_relpath(str(item["path"]))
        except TreeError:
            continue                        # a hostile or malformed path: drop it
        entry: Dict[str, Any] = {"owner": normalise(str(item.get("owner", ""))),
                                 "path": path}
        if item.get("delete") is True:
            entry["content"] = None
        elif isinstance(item.get("content"), str):
            entry["content"] = item["content"]
        else:
            continue
        out.append(entry)
    return out


def _payload(proposal: str) -> Optional[str]:
    """The JSON object inside an ``<EDITS>`` block, a fence, or bare text."""
    if not isinstance(proposal, str) or not proposal.strip():
        return None
    lowered = proposal.lower()
    start_tag, end_tag = lowered.find("<edits>"), lowered.rfind("</edits>")
    if start_tag >= 0 and end_tag > start_tag:
        proposal = proposal[start_tag + len("<edits>"):end_tag]
    return _first_json_object(proposal)


def _first_json_object(text: str) -> Optional[str]:
    """The first balanced ``{...}`` run, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


@dataclass
class SpatialContract:
    """A directory, one key per file path, with authority scoped by subtree.

    Deliberately does **not** implement ``keys()``. A strategy that declares one
    is offering tensor parallelism a key space to partition, and this artifact
    does not have a fixed one -- it grows. Withholding it makes ``evolve()``
    *refuse* TP with a message naming the reason, which is the behaviour worth
    having; declaring a partial one would let TP run and discard every creation.
    """

    initial_files: Mapping[str, str] = field(default_factory=dict)
    editable: Sequence[str] = ("**",)
    #: Human-supplied validation, never the agents'. The paper's compiler run
    #: uses c-testsuite / LLVM / Csmith exactly this way, and it is also L0 in
    #: this repository's sense: a system that can edit its own evaluator has no
    #: evaluator. Enforced here for *proposals* and, where the frozen files are
    #: what score the candidate, by the runner's pristine overlay.
    frozen: Sequence[str] = ()
    max_files_per_diff: int = 4
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    #: Where dropped edits are counted. Optional so the strategy can be used on
    #: its own in a test without building a log first.
    log: Optional[WorldLog] = None

    def __post_init__(self) -> None:
        self.initial_files = {safe_relpath(k): v for k, v in dict(self.initial_files).items()}

    # -- the Strategy protocol ---------------------------------------------

    def initial(self) -> Dict[str, str]:
        return dict(self.initial_files)

    def render(self, state: Mapping[str, str]) -> str:
        return canonical(state)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        ops: Dict[str, Optional[str]] = {}
        for edit in parse_situated_edits(proposal):
            owner, path, content = edit["owner"], edit["path"], edit["content"]
            if not owns(owner, path):
                # The contract. Counted, never silent -- see WorldLog.
                if self.log is not None:
                    self.log.note_contract_violation()
                continue
            if not self.writable(path):
                continue
            if content is None:
                if path in state:
                    ops[path] = None        # delete
                continue
            if len(content) > self.max_file_bytes:
                continue                    # would be rejected as oversized anyway
            if state.get(path) != content:
                ops[path] = content
        if not ops or len(ops) > self.max_files_per_diff:
            return None
        fingerprint = stable_hash(tuple(sorted((k, v) for k, v in ops.items())))
        return Diff(diff_id=f"{author}:{fingerprint & 0xFFFFFFFF:08x}:{base_version}",
                    target=target, ops=dict(ops), author=author)

    # -- helpers -----------------------------------------------------------

    def writable(self, path: str) -> bool:
        """May the loop write this path? ``frozen`` beats ``editable``."""
        return match_any(path, self.editable) and not match_any(path, self.frozen)

    def frozen_files(self, source: Mapping[str, str]) -> Dict[str, str]:
        """The pristine content of every frozen path, for the runner's overlay."""
        return {p: c for p, c in source.items() if match_any(p, self.frozen)}
