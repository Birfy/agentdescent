"""Mode A: the Context Tree extracted from a repository that already has code.

`Genesis.run` has two modes and this port had one of them::

    if mode == :new do
      run_new_codebase(...)      # Mode B: Architect designs the tree, then Manager
    else
      run_existing_codebase(...) # Mode A
    end

and Mode A's root agent is not the Manager -- it is the **ContextExtractor**
(``runtime/genesis.ex:50``). That is the whole entry point: a repository with code and
no ``CONTEXT.md`` cannot be worked on by recursive delegation at all, because the
routing table is the map, so the first thing Genesis does to such a repository is read
it and write the tree over it.

The difference from :mod:`examples.genesis._architect` is where the children come from.
An architect **invents** them -- nothing exists yet, and the decomposition is the
product. An extractor **finds** them: the directories are already there, and inventing
one would be describing a repository that does not exist. So this walks the tree it is
given, bottom-relevant parts first, and what it produces is description rather than
design.

Upstream's extractor may also decide a directory is unimportant and stop
("If `src/` is unimportant or if `src/CONTEXT.md` already fulfils your objective, call
`complete_task` immediately"), which is why a node here may return an empty record and
be skipped rather than forced to say something.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ._world import (CONTEXT_FILE, KNOWN_ISSUES, LocalWorld, ROUTING_HEADING,
                     child_paths, normalise, owns, shadowed_by_module)

__all__ = ["EXTRACTOR_PROMPT", "ExtractPhase"]

EXTRACTOR_PROMPT = """You are a context-extraction agent, situated at the repository \
path `{path}`. You describe what is here; you do not change it and you do not design it.

{context}

The files at this node, and the subdirectories under it:

{listing}

And this is what the files at this node contain. Read them: every name you put in the \
API Surface has to come from here, not from the file's name.

{sources}

Write this node's `CONTEXT.md`. The sections that apply, in this order -- omit one \
rather than pad it:

- `## Intent` -- what this directory is for. Two or three sentences.
- `## API Surface` -- the files here and what each exposes. Real names, read off the \
code, with signatures where they are short.
- `## Constraints` -- rules a later agent must follow here, if the code implies any.
- `## Known Issues` / `## Notes for Agents` -- anything that would otherwise be \
re-discovered: a dependency between two files, a generated file, a gap in the tests.
- `## Routing Table` -- one line per **existing** subdirectory, `- ./<path>/ -> <what \
it handles>`. The subdirectories are listed above. Do not invent one, and do not leave \
one out: this table is the map a manager delegates by, and a directory missing from it \
is a directory nobody can be sent to.

Describe the current state, not the history, and not what you would have written \
instead. If this directory holds nothing worth recording, reply with an empty record.

Reply with ONE JSON object and nothing else:
{{"record": "<the whole CONTEXT.md, markdown, or \\"\\" to skip this node>"}}"""


class ExtractPhase:
    """Walk a repository and write one ``CONTEXT.md`` per directory that has code."""

    def __init__(self, complete, *, contracts: Sequence[str] = (), max_depth: int = 4,
                 max_nodes: int = 24, root_path: str = "", skip: Sequence[str] = (),
                 file_chars: int = 6_000, max_files: int = 8):
        self._complete = complete
        self._contracts = tuple(contracts)
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._root = normalise(root_path)
        #: Directories never described -- the frozen contract, and anything hidden.
        self._skip = tuple(skip)
        self._file_chars = file_chars
        self._max_files = max_files
        self.nodes: List[str] = []
        self.depth = 0
        self.skipped: List[str] = []
        self.unparsed = 0
        #: Directories described that an existing sibling module makes unimportable.
        self.shadowed = 0

    def extract(self, state: Mapping[str, str], objective: str) -> Dict[str, str]:
        """``state`` plus a record per node, parents before children as upstream reads."""
        out = dict(state)
        queue: List[Tuple[str, int]] = [(self._root, 0)]
        while queue and len(self.nodes) < self._max_nodes:
            path, depth = queue.pop(0)
            children = [c for c in child_paths(path, list(out))
                        if not self._ignored(c, out)]
            record = self._ask(out, path, objective, children)
            if record:
                key = f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE
                out[key] = record
                self.nodes.append(path)
                self.depth = max(self.depth, depth)
            else:
                self.skipped.append(path or "./")
            if depth + 1 <= self._max_depth:
                queue += [(child, depth + 1) for child in children]
        return out

    def summary(self) -> str:
        return (f"described {len(self.nodes)} nodes, deepest {self.depth}"
                + (f", {len(self.skipped)} skipped" if self.skipped else "")
                + (f", {self.shadowed} shadowed by a sibling module"
                   if self.shadowed else "")
                + (f", {self.unparsed} replies unusable" if self.unparsed else ""))

    # -- internals ---------------------------------------------------------

    def _ignored(self, path: str, state: Mapping[str, str]) -> bool:
        if any(part.startswith(".") for part in path.split("/")):
            return True
        world = LocalWorld(version=0, path="", readonly=self._contracts)
        return (any(path == normalise(s) or owns(normalise(s), path) for s in self._skip)
                or world._all_readonly(state, path))

    def _ask(self, state: Mapping[str, str], path: str, objective: str,
             children: Sequence[str]) -> Optional[str]:
        world = LocalWorld(version=0, path=path, readonly=self._contracts)
        here = sorted(k for k in state if owns(path, k) and "/" not in
                      (k[len(path) + 1:] if path else k))
        listing = ("files: " + (", ".join(here) or "(none)")
                   + "\nsubdirectories: " + (", ".join(f"./{c}/" for c in children)
                                             or "(none)"))
        # The extractor describes code, so it has to be shown the code. Upstream it
        # reads what it needs with a tool; an agent here has none, so the node's own
        # files travel in the prompt -- and the first run without them invented an API
        # surface off the file names, swapping what two modules do.
        sources = "\n\n".join(
            f"# {key}\n```\n{_clip(state[key], self._file_chars)}\n```"
            for key in here[:self._max_files]) or "(this node owns no file)"
        try:
            reply = self._complete(EXTRACTOR_PROMPT.format(
                path=path or "./", objective=objective, listing=listing,
                sources=sources,
                context=world.situate(state, contracts=self._contracts))) or ""
        except Exception:  # noqa: BLE001 - one dead call costs one node
            self.unparsed += 1
            return None
        record = _record_of(reply)
        if record is None:
            self.unparsed += 1
            return None
        shadowed = [c for c in children if shadowed_by_module(state, c)]
        if record and shadowed:
            # The guard in `_is_node` stops this being *created*; a repository handed
            # to mode A may already contain one, and then describing the directory
            # without saying so sends every later agent to write code no import can
            # reach. Measured: `src/observe/rdf/` beside `src/observe/rdf.py`, three
            # runs and 30 000 rollouts that never got the repository off 0.812.
            record = record.rstrip("\n") + "\n\n" + KNOWN_ISSUES + "\n" + "\n".join(
                f"- `{c}/` is shadowed by the module `{c}.py` beside it: both are "
                f"`{c.replace('/', '.')}` to Python and the module wins, so nothing "
                f"written in that directory can be imported. Forwarding from "
                f"`{c}.py` into it cannot work either -- the name resolves back to "
                f"the forwarding file. One of the two has to go."
                for c in shadowed) + "\n"
            self.shadowed += len(shadowed)
        if record and children and ROUTING_HEADING.lower() not in record.lower():
            # A directory missing from its parent's table is a directory nobody can be
            # sent to, and the extractor is the only agent that knows it is there.
            record += ("\n" + ROUTING_HEADING + "\n"
                       + "\n".join(f"- `./{c}/` -> (undescribed)" for c in children)
                       + "\n")
        return record


def _clip(body: str, limit: int) -> str:
    return body if len(body) <= limit else body[:limit] + "\n... [Content Truncated] ..."


def _record_of(reply: str) -> Optional[str]:
    start = reply.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(reply[start:reply.rfind("}") + 1])
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return None
    if not isinstance(data, Mapping) or not isinstance(data.get("record"), str):
        return None
    record = data["record"].strip()
    return record + "\n" if record else ""
