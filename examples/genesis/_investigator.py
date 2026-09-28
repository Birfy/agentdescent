"""The investigator: a read-only delegation whose findings persist into the record.

A manager hands it a question and it answers from the node's code; the answer is written
into the node's ``CONTEXT.md`` so the next agent inherits it instead of re-discovering
it. Ports the ``investigator`` role (``agents/investigator.ex``), whose one permitted
write is a ``CONTEXT.md``.

The gap it fills is concrete. ``_MANAGER_PROMPT`` carries the ``CONTEXT.md`` chain and a
listing of the files at the node; it does **not** carry the files' contents, because this
port's agents have no read tool. So a manager that must decide *from the code* is
deciding blind, and the investigator is the one actor allowed to read a node's code and
leave a note about it.

The role's fan-out to ``subagent_investigator`` is deliberately not ported -- a manager
with read tools of its own is what that is for, and giving the manager read tools is the
port's session work. What is left is the part nothing else here does.

The findings go under a ``## Findings`` heading of the node's ``CONTEXT.md``, through the
same :func:`~examples.genesis._world.under_heading` the rest of the port uses: findings
belong in a named section, because the next agent reads the section, not the whole file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence

from agentdescent.filetree import match_any

from ._world import CONTEXT_FILE, child_paths, directly_at, under_heading

if TYPE_CHECKING:                       # `_delegation` imports this module
    from ._delegation import Brief, Edit

__all__ = ["FINDINGS_HEADING", "INVESTIGATOR_PROMPT", "Investigator"]

#: The section a finding is recorded under. ``KNOWN_ISSUES`` is for problems, this is
#: for knowledge, and a finding can be either.
FINDINGS_HEADING = "## Findings"


INVESTIGATOR_PROMPT = """You are a read-only investigator in a recursive software \
world, situated at the repository path `{path}`. You answer one question about what is \
here. You do not change code and you do not design anything.

{context}

The question you were sent to answer:

{objective}

The files at this node, in full:

{sources}

Answer only from what is in front of you, and say so when the answer is "nothing here" \
rather than guessing. Then record what a future agent would otherwise have to \
re-discover: a dependency between two files, a name that resolves somewhere unexpected, \
a gotcha, a test gap, an external interface's real shape, a fact that contradicts the \
record above. One markdown list item each, each one sentence.

Every finding you do not record is a finding the next agent will have to re-discover. \
If nothing here is worth persisting, record nothing -- an empty record is the honest \
answer, not a failure.

Reply with ONE JSON object and nothing else:
{{"findings": "<markdown list items, or \\"\\" for nothing worth recording>",
 "answer": "<one sentence: the answer to the question, for the record>"}}"""


class Investigator:
    """``(brief) -> [Edit]`` -- one read-only investigation of one node.

    Installs on :class:`~examples.genesis._delegation.RecursiveDelegation` as
    ``investigator=``, and runs for a delegation the manager marked read-only (see
    ``llm_manager``). ``None`` there is the old behaviour: every delegation is work.

    The one write it may make is a ``CONTEXT.md`` **at the node it was sent to**. A
    delegation that returned source edits would not be an investigation, and a write to
    an ancestor's record is the ancestor's own business -- both are refused and counted
    rather than silently applied.
    """

    def __init__(self, complete, *, contracts: Sequence[str] = (),
                 frozen: Sequence[str] = (), per_file_chars: int = 4_000,
                 max_chars: int = 12_000, max_files: int = 12):
        self._complete = complete
        self._contracts = tuple(contracts)
        self._frozen = tuple(frozen)
        self._per_file_chars = per_file_chars
        self._max_chars = max_chars
        self._max_files = max_files
        #: Calls made, findings recorded, the two ways one was refused, and the two
        #: "nothing happened" cases that must not look alike.
        self.calls = 0
        self.recorded = 0
        self.empty = 0
        self.unparsed = 0
        self.refused = 0

    def __call__(self, brief: "Brief") -> List["Edit"]:
        from ._delegation import Edit        # circular at import time, not at call time

        node = brief.world.path
        sources = _sources(brief.state, node, self._per_file_chars,
                           self._max_chars, self._max_files)
        if sources is None:
            # Nothing at the node to read: there is nothing to investigate, and a
            # call spent on an empty directory is a call spent on nothing.
            self.empty += 1
            return []
        self.calls += 1
        try:
            reply = self._complete(INVESTIGATOR_PROMPT.format(
                path=node or "./", context=brief.context, objective=brief.objective,
                sources=sources)) or ""
        except Exception:  # noqa: BLE001 - a dead call costs a finding, not the run
            self.unparsed += 1
            return []
        findings = _findings_of(reply)
        if findings is None:
            self.unparsed += 1
            return []
        if not findings.strip():
            self.empty += 1
            return []

        key = f"{node}/{CONTEXT_FILE}" if node else CONTEXT_FILE
        if self._frozen and match_any(key, self._frozen):
            self.refused += 1
            return []
        body = under_heading(brief.state.get(key, f"# {node or './'}\n"),
                             FINDINGS_HEADING,
                             "\n".join(f"- {line.strip().lstrip('- ').strip()}"
                                       for line in findings.splitlines()
                                       if line.strip()))
        self.recorded += 1
        # `context` when the record was already there and this adds to it, `record`
        # when it is the write that brings the node's record into existence -- the
        # same distinction the rest of the port makes, for the same reason.
        return [Edit(owner=node, path=key, content=body,
                     kind="context" if key in brief.state else "record")]

    def stats(self) -> str:
        return (f"investigations={self.calls} findings={self.recorded} "
                f"empty={self.empty} refused={self.refused} "
                f"unparsed={self.unparsed}")


def _sources(state: Mapping[str, str], node: str, per_file: int, budget: int,
             max_files: int) -> Optional[str]:
    """The node's own files, in full; ``None`` when the node owns none.

    Its own files, not its subtree's: an investigator's scope is the node it stands on,
    and a deeper level is what sending it deeper is for. Children are listed by name so
    the reader knows the tree continues.
    """
    here = sorted(k for k in state if directly_at(node, k))
    if not here:
        return None
    out: List[str] = []
    for key in here[:max_files]:
        body = state[key]
        if per_file and len(body) > per_file:
            body = body[:per_file] + "\n... [truncated] ..."
        out.append(f"# {key}\n```\n{body}\n```")
    children = child_paths(node, list(state))
    if children:
        out.append("subdirectories in the repository here: "
                   + ", ".join(f"{c}/" for c in children))
    text = "\n\n".join(out)
    return text if len(text) <= budget else text[:budget] + "\n... [truncated] ..."


def _findings_of(reply: str) -> Optional[str]:
    """``findings`` from a reply, or ``None`` when the reply is not a JSON object."""
    start = reply.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(reply[start:reply.rfind("}") + 1])
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return None
    if not isinstance(data, Mapping) or "findings" not in data:
        return None
    findings = data["findings"]
    return findings if isinstance(findings, str) else None
