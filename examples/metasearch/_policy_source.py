"""The evolvable surface of a tree search, as this example names it.

Everything here lives in :mod:`agentdescent.meta` now -- the gate, the seed
rule, the wrapper -- and this module keeps the example's own names for them so
its tests and README read as one thing:

* :data:`SEED_SOURCE` is :data:`agentdescent.meta.PRIORITY_SEED`;
* :func:`compile_priority` is the gate;
* :class:`EvolvedSelection` is :class:`agentdescent.meta.PrioritySelection`;
* :func:`SearchPolicySlot` builds the spec, :func:`agentdescent.meta.priority_selection`.
"""

from __future__ import annotations

from agentdescent.meta import (PRIORITY_SEED, PrioritySelection, SourceSlot,
                               compile_priority, priority_selection)

__all__ = ["ARGS", "FUNCTION", "SEED_SOURCE", "EvolvedSelection",
           "SearchPolicySlot", "compile_priority", "policy_source"]

FUNCTION = "priority"
ARGS = ("rank", "visits", "total", "prior", "depth", "n_nodes")
SEED_SOURCE = PRIORITY_SEED
EvolvedSelection = PrioritySelection


def policy_source(text: str) -> str:
    """The gate as a validator: fence-tolerant, raises ``ValueError``."""
    spec = priority_selection()
    return spec._validate(text if "```" not in (text or "") else _strip(text))


def _strip(text: str) -> str:
    chunks = text.split("```")
    for index in range(1, len(chunks), 2):
        block = chunks[index].strip()
        if block.lower().startswith("python"):
            return block[6:].lstrip("\r\n ")
    return chunks[1].strip()


def SearchPolicySlot(initial_value: str = SEED_SOURCE) -> SourceSlot:
    """The artifact: one slot holding the priority function's source."""
    return priority_selection(initial_value)
