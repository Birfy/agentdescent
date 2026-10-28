"""A provenance graph for the artifacts a self-improvement loop creates.

The archive knows *what* was adopted; the decision log knows *why*; the ledger
knows *where*. None of them answers the causal question a reviewer asks first:
**which ancestor introduced the behaviour we are looking at now?**

This module is the graph between those stores. A node is one artifact version
with its parents, run, slot and score; an edge is “this artifact was derived
from that one”. Merges have several parents, so this is a DAG, not a tree. The
graph is deliberately in-memory and side-effect free; callers build it from
whatever records they already persist.

    graph = build_lineage(records)
    graph.ancestors("run-8")
    graph.descendants("run-2")
    graph.path("run-1", "run-8")
    report = lineage_report(graph)

A cycle is a bug in the producer, not a feature of the graph: it is reported
with the members and the node that closed it, and ``topological_order`` refuses
to invent an order. The DFS and BFS walks are iterative, so a long-running
self-improvement lineage (thousands of generations) does not hit Python's
recursion limit.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "LineageNode",
    "LineageGraph",
    "LineageReport",
    "build_lineage",
    "lineage_report",
]


@dataclass(frozen=True)
class LineageNode:
    """One artifact version in the graph.

    ``metadata`` is excluded from equality and hashing, so a node is hashable
    while its metadata stays a plain dictionary. Nodes themselves are immutable;
    the metadata mapping is not deep-frozen.
    """

    artifact_id: str
    parent_ids: Tuple[str, ...] = ()
    run_id: str = ""
    score: Optional[float] = None
    slot: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict,
                                        hash=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.artifact_id):
            raise ValueError("LineageNode.artifact_id must be non-empty")
        if any(parent == self.artifact_id for parent in self.parent_ids):
            raise ValueError(f"node {self.artifact_id!r} cannot be its own parent")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError(f"node {self.artifact_id!r} repeats a parent")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("LineageNode.score must be finite")


class LineageGraph:
    """A directed acyclic graph from parent artifacts to descendants."""

    def __init__(self) -> None:
        self._nodes: Dict[str, LineageNode] = {}
        # Parent -> known children. Built in add(), including children whose
        # parent has not been added yet; that keeps children() O(children) and
        # lets a later parent insertion see the already-registered edge.
        self._children: Dict[str, Set[str]] = {}

    def add(self, node: LineageNode) -> None:
        if node.artifact_id in self._nodes:
            raise ValueError(f"duplicate artifact id {node.artifact_id!r}")
        self._nodes[node.artifact_id] = node
        for parent in node.parent_ids:
            self._children.setdefault(parent, set()).add(node.artifact_id)

    def add_record(self, artifact_id: str, *, parent_ids: Sequence[str] = (),
                   run_id: str = "", score: Optional[float] = None,
                   slot: str = "", metadata: Optional[Mapping[str, Any]] = None) -> LineageNode:
        node = LineageNode(artifact_id=str(artifact_id),
                           parent_ids=tuple(str(parent) for parent in parent_ids),
                           run_id=run_id, score=score, slot=slot,
                           metadata=dict(metadata or {}))
        self.add(node)
        return node

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._nodes

    def __getitem__(self, artifact_id: str) -> LineageNode:
        return self._nodes[artifact_id]

    @property
    def nodes(self) -> Tuple[LineageNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    def children(self, artifact_id: str) -> Tuple[str, ...]:
        self._require(artifact_id)
        return tuple(sorted(self._children.get(artifact_id, ())))

    def parents(self, artifact_id: str) -> Tuple[str, ...]:
        return tuple(sorted(self._require(artifact_id).parent_ids))

    def known_parents(self, artifact_id: str) -> Tuple[str, ...]:
        return tuple(parent for parent in self.parents(artifact_id)
                     if parent in self._nodes)

    def external_parents(self, artifact_id: str) -> Tuple[str, ...]:
        """Parents named by this node but absent from this graph."""
        return tuple(parent for parent in self.parents(artifact_id)
                     if parent not in self._nodes)

    def externals(self) -> Tuple[str, ...]:
        """Every external parent id, sorted and unique."""
        found: Set[str] = set()
        for node in self._nodes.values():
            found.update(self.external_parents(node.artifact_id))
        return tuple(sorted(found))

    def roots(self) -> Tuple[str, ...]:
        return tuple(sorted(
            node.artifact_id for node in self._nodes.values()
            if not self.known_parents(node.artifact_id)))

    def leaves(self) -> Tuple[str, ...]:
        return tuple(sorted(
            node.artifact_id for node in self._nodes.values()
            if not self._children.get(node.artifact_id)))

    def ancestors(self, artifact_id: str, *, depth: Optional[int] = None) -> Tuple[str, ...]:
        """Transitive parents, nearest first; ``depth`` bounds the walk.

        External (unknown) parents are included as leaves of the walk.
        """
        self._require(artifact_id)
        return self._walk(artifact_id, self._any_parents, depth)

    def descendants(self, artifact_id: str, *, depth: Optional[int] = None) -> Tuple[str, ...]:
        """Transitive known children, nearest first; ``depth`` bounds the walk."""
        self._require(artifact_id)
        return self._walk(artifact_id, self._any_children, depth)

    def _any_parents(self, artifact_id: str) -> Tuple[str, ...]:
        node = self._nodes.get(artifact_id)
        return tuple(sorted(node.parent_ids)) if node is not None else ()

    def _any_children(self, artifact_id: str) -> Tuple[str, ...]:
        return tuple(sorted(self._children.get(artifact_id, ())))

    def _walk(self, start: str, step, depth: Optional[int]) -> Tuple[str, ...]:
        if depth is not None and depth < 0:
            raise ValueError("depth must be >= 0")
        seen: Set[str] = set()
        order: List[str] = []
        queue = deque([(start, 0)])
        while queue:
            current, distance = queue.popleft()
            if depth is not None and distance >= depth:
                continue
            for neighbour in step(current):
                if neighbour in seen or neighbour == start:
                    continue
                seen.add(neighbour)
                order.append(neighbour)
                queue.append((neighbour, distance + 1))
        return tuple(order)

    def path(self, source: str, target: str) -> Optional[Tuple[str, ...]]:
        """Shortest parent-to-child path, inclusive, or ``None``."""
        self._require(source)
        self._require(target)
        if source == target:
            return (source,)
        queue = deque([(source, (source,))])
        seen = {source}
        while queue:
            current, trail = queue.popleft()
            for child in self.children(current):
                if child == target:
                    return trail + (child,)
                if child not in seen:
                    seen.add(child)
                    queue.append((child, trail + (child,)))
        return None

    def impact(self, artifact_id: str) -> int:
        """Number of known descendants, the size of the change's blast radius."""
        return len(self.descendants(artifact_id))

    def cycles(self) -> Tuple[Tuple[str, ...], ...]:
        """Every back-edge cycle found by a deterministic iterative DFS."""
        found: List[Tuple[str, ...]] = []
        state: Dict[str, int] = {key: 0 for key in self._nodes}
        for start in sorted(self._nodes):
            if state[start] != 0:
                continue
            state[start] = 1
            path = [start]
            stack: List[Tuple[str, Any]] = [(start, iter(sorted(self._children.get(start, ()))))]
            while stack:
                node, iterator = stack[-1]
                try:
                    child = next(iterator)
                except StopIteration:
                    stack.pop()
                    path.pop()
                    state[node] = 2
                    continue
                if child not in state:
                    continue                 # external child (not a node)
                if state[child] == 0:
                    state[child] = 1
                    path.append(child)
                    stack.append((child, iter(sorted(self._children.get(child, ())))))
                elif state[child] == 1:
                    index = path.index(child)
                    cycle = tuple(path[index:] + [child])
                    if cycle not in found:
                        found.append(cycle)
        return tuple(found)

    def _topological_order_unchecked(self) -> Tuple[str, ...]:
        indegree = {node.artifact_id: len(self.known_parents(node.artifact_id))
                    for node in self._nodes.values()}
        ready = [key for key, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: List[str] = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for child in self.children(current):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        return tuple(order)

    def topological_order(self) -> Tuple[str, ...]:
        """Parents before children; raises on a cycle."""
        cycles = self.cycles()
        if cycles:
            raise ValueError(f"lineage has a cycle: {cycles[0]}")
        return self._topological_order_unchecked()

    def _depth_from_order(self, order: Sequence[str]) -> int:
        distance = {node: 0 for node in order}
        for node in order:
            for child in self.children(node):
                distance[child] = max(distance[child], distance[node] + 1)
        return max(distance.values(), default=0)

    def depth(self) -> int:
        """Longest root-to-leaf path length; ``-1`` when the graph has a cycle."""
        if self.cycles():
            return -1
        return self._depth_from_order(self._topological_order_unchecked())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "artifact_id": node.artifact_id,
                    "parent_ids": list(node.parent_ids),
                    "run_id": node.run_id,
                    "score": node.score,
                    "slot": node.slot,
                    "metadata": dict(node.metadata),
                }
                for node in self.nodes
            ]
        }

    def _require(self, artifact_id: str) -> LineageNode:
        try:
            return self._nodes[artifact_id]
        except KeyError:
            raise KeyError(f"unknown artifact {artifact_id!r}") from None


def build_lineage(records: Iterable[Mapping[str, Any]]) -> LineageGraph:
    """Build a graph from dict records with the node fields as keys.

    Unknown parents are allowed (they may live in another store); they are
    ignored for root detection and counted by ``LineageReport.external``.
    A single string parent is treated as one parent, not iterated character by
    character. Duplicate ids raise.
    """
    graph = LineageGraph()
    for record in records:
        if "artifact_id" not in record:
            raise ValueError("lineage record is missing 'artifact_id'")
        artifact_id = str(record["artifact_id"])
        if "parent_ids" in record:
            raw = record["parent_ids"]
        elif "parents" in record:
            raw = record["parents"]
        else:
            raw = ()
        if isinstance(raw, str):
            parents = (raw,)
        elif raw is None:
            parents = ()
        else:
            parents = tuple(str(parent) for parent in raw)
        graph.add_record(artifact_id, parent_ids=parents,
                         run_id=str(record.get("run_id", "")),
                         score=record.get("score"),
                         slot=str(record.get("slot", "")),
                         metadata=record.get("metadata") or {})
    return graph


@dataclass(frozen=True)
class LineageReport:
    """Shape and health of a lineage graph."""

    nodes: int
    edges: int
    roots: Tuple[str, ...]
    leaves: Tuple[str, ...]
    depth: int
    max_impact: int
    max_impact_id: str
    cycles: Tuple[Tuple[str, ...], ...]
    external: int = 0
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return not self.cycles


def lineage_report(graph: LineageGraph) -> LineageReport:
    """Summarise the graph; a cycle turns ``healthy`` false and ``depth`` -1."""
    cycles = graph.cycles()
    if cycles:
        depth = -1
    else:
        depth = graph._depth_from_order(graph._topological_order_unchecked())
    impacts = {node.artifact_id: graph.impact(node.artifact_id)
               for node in graph.nodes}
    max_impact_id = max(impacts, key=lambda key: (impacts[key], key), default="")
    edges = sum(len(graph._children.get(node.artifact_id, ()))
                for node in graph.nodes)
    external = sum(len(graph.external_parents(node.artifact_id))
                   for node in graph.nodes)
    detail = (f"{len(graph)} nodes, {edges} edges, depth={depth}, "
              f"cycles={len(cycles)}, external={external}")
    return LineageReport(nodes=len(graph), edges=edges, roots=graph.roots(),
                         leaves=graph.leaves(), depth=depth,
                         max_impact=impacts.get(max_impact_id, 0),
                         max_impact_id=max_impact_id, cycles=cycles,
                         external=external, detail=detail)
