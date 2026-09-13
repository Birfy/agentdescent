"""Phase 1: an agent writes the `CONTEXT.md` tree, before any code exists.

Upstream a formation run is **two root agents in sequence** (``runtime/genesis.ex``)::

    # --- Phase 1: Architecture (Architect as root agent) ---
    # --- Phase 2: Implementation (Manager as second root agent) ---

and the handoff into Phase 2 says what Phase 1 produced: "The architecture, directory
structure, CONTEXT.md routing tables, and public APIs are already in place (**created
by an Architect agent**)."

This port had neither. A domain shipped its `CONTEXT.md` records hand-written, which is
Phase 1 done by a human and leaves only Phase 2 to measure; `--cold-start` removed them
and left nothing in their place, so managers opened nodes as they went and recorded a
line of routing table each time. Neither is what upstream does, and the difference is
not cosmetic: upstream's tree is *designed* -- "the Routing Table is your primary
delegation tool... they are the map that makes recursive delegation work" -- before one
line of implementation is written against it.

So: one architect per node, recursive, depth-bounded, writing records and nothing else.
It inherits the chain as every other agent does, it may only name children inside its
own subtree (``LocalWorld.delegate`` refuses otherwise), and what it produces is the
initial artifact the implementation phase then grows. Code is not its job -- upstream's
architect creates API *stubs* through an executor, and here the public surface is
written in Phase 2 by the agent accountable for that node.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ._world import (CONTEXT_FILE, LocalWorld, ROUTING_HEADING,
                     STANDARD_SECTIONS, looks_like_file, normalise,
                     parse_routing)

__all__ = ["ARCHITECT_PROMPT", "ArchitectPhase"]

ARCHITECT_PROMPT = """You are an architect agent in a recursive software world, \
situated at the repository path `{path}`. You design; you do not implement.

{context}

THE OBJECTIVE
{objective}

Write the `CONTEXT.md` for **your own node** and name the child directories it should \
route to. Nothing else: no code, no stubs, no files other than this record.

The four sections are required, in this order:

- `## Intent` -- what this directory is for, in two or three sentences.
- `## API Surface` -- the files this node owns and what each exposes. Signatures where \
the contract above gives them.
- `## Constraints` -- rules for code in this directory. Omit what the parent already says.
- `## Routing Table` -- one line per child, exactly `- ./{path_prefix}<name>/ -> <what it handles>`.

Keep it short. An agent that arrives here inherits the whole chain from the root down, \
so add one layer of specificity and do not repeat the parent. The routing table is not \
documentation: it is the map a manager delegates by, so every entry has to be specific \
enough to route work without investigating the subtree.

**A node is a directory, never a file.** `src/frontend` is a node; \
`src/frontend/lexer.py` is a file belonging to the agent situated at `src/frontend`. \
Name directories in the routing table and files in the API Surface.

Decompose only where the work genuinely separates. A node with one child is a node that \
should not exist; a leaf is a node whose files one agent can write.

Reply with ONE JSON object and nothing else:
{{"record": "<the whole CONTEXT.md, markdown>",
  "children": [{{"path": "{path_prefix}<name>", "objective": "<one sentence>"}}]}}

An empty `children` list means this node is a leaf and its files are written here."""


class ArchitectPhase:
    """Run Phase 1 and return the `CONTEXT.md` tree it designed.

    ``max_depth`` bounds the recursion the way the implementation phase is bounded;
    ``max_nodes`` is a budget, because an architect that decomposes forever is a
    failure mode with a cost attached.
    """

    def __init__(self, complete, *, contracts: Sequence[str] = (),
                 max_depth: int = 3, max_nodes: int = 12, root_path: str = ""):
        self._complete = complete
        self._contracts = tuple(contracts)
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._root = normalise(root_path)
        #: Nodes it designed, and the deepest it went.
        self.nodes: List[str] = []
        self.depth = 0
        self.refused = 0
        self.mistaken_nodes = 0
        self.unparsed = 0

    def design(self, given: Mapping[str, str], objective: str) -> Dict[str, str]:
        """``given`` plus one ``CONTEXT.md`` per node the architect decided on."""
        state = dict(given)
        queue: List[Tuple[str, str, int]] = [(self._root, objective, 0)]
        while queue and len(self.nodes) < self._max_nodes:
            path, node_objective, depth = queue.pop(0)
            record, children = self._ask(state, path, node_objective, depth)
            if record is None:
                continue
            key = f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE
            state[key] = record
            self.nodes.append(path)
            self.depth = max(self.depth, depth)
            if depth + 1 > self._max_depth:
                continue
            world = LocalWorld(version=0, path=path)
            for child in children:
                if looks_like_file(child["path"]):
                    # One architect designed `observables/observables.py` as a node and
                    # hung two children under it. The delegation policy refuses a path
                    # that is an existing file; a designed tree contains nothing that
                    # exists yet, so the shape is all there is to go on.
                    self.mistaken_nodes += 1
                    continue
                try:
                    world.delegate(child["path"])       # the spatial contract, early
                except ValueError:
                    self.refused += 1                   # named outside its own subtree
                    continue
                queue.append((normalise(child["path"]), child["objective"], depth + 1))
        return state

    def summary(self) -> str:
        return (f"designed {len(self.nodes)} nodes, deepest {self.depth}"
                + (f", {self.refused} outside their subtree" if self.refused else "")
                + (f", {self.mistaken_nodes} file paths refused as nodes"
                   if self.mistaken_nodes else "")
                + (f", {self.unparsed} replies unusable" if self.unparsed else ""))

    # -- internals ---------------------------------------------------------

    def _ask(self, state: Mapping[str, str], path: str, objective: str,
             depth: int) -> Tuple[Optional[str], List[Dict[str, str]]]:
        world = LocalWorld(version=0, path=path, readonly=self._contracts)
        prefix = f"{path}/" if path else ""
        try:
            reply = self._complete(ARCHITECT_PROMPT.format(
                path=path or "./", path_prefix=prefix, objective=objective,
                context=world.situate(state, contracts=self._contracts))) or ""
        except Exception:  # noqa: BLE001 - one dead call costs one node
            self.unparsed += 1
            return None, []
        record, children = _parse(reply, prefix)
        if record is None:
            self.unparsed += 1
        return record, children


def _parse(reply: str, prefix: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """``(record, children)``, or ``(None, [])`` when the reply is not usable.

    A record without the four sections is still written: the sections are what the
    architect is *told* to produce, and refusing a record that names its children but
    forgets a heading would throw away the routing table with it.
    """
    start = reply.find("{")
    if start < 0:
        return None, []
    try:
        data = json.loads(reply[start:reply.rfind("}") + 1])
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return None, []
    if not isinstance(data, Mapping):
        return None, []
    record = data.get("record")
    if not isinstance(record, str) or not record.strip():
        return None, []
    record = record.strip() + "\n"
    children = []
    for item in data.get("children") or ():
        if isinstance(item, Mapping) and item.get("path"):
            child = normalise(str(item["path"]))
            # A bare name is relative to this node -- "core" at `src` is `src/core`.
            # A path with a separator in it is repository-relative and is taken as
            # written: rewriting it into the subtree would turn a violation of the
            # spatial contract into a silent relocation, and the caller refuses and
            # counts it instead.
            if prefix and "/" not in child:
                child = prefix + child
            children.append({"path": child,
                             "objective": str(item.get("objective", "")).strip()
                             or f"implement {child}"})
    # The record is the map the run delegates by, so a child the architect named and
    # did not route to would be unreachable. Add the entry it forgot.
    if children and ROUTING_HEADING.lower() not in record.lower():
        record += ("\n" + ROUTING_HEADING + "\n"
                   + "\n".join(f"- `./{c['path']}/` -> {c['objective']}"
                               for c in children) + "\n")
    return record, children


def missing_sections(record: str) -> List[str]:
    """Which of upstream's four standard sections a record does not have."""
    lowered = record.lower()
    return [s for s in STANDARD_SECTIONS if s.lower() not in lowered]
