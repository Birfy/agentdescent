"""Trajectory tree search: the evolution loop becomes a learned MCTS over states.

LATS (Language Agent Tree Search) / AlphaCodium run MCTS over agent
trajectories; the parallel engine here explores rollouts in *batches*, with no
learned preference for which branch of the state tree deserves the next
rollout. This module is the tree-search dimension: a :class:`SearchTree` over
artifact states (rendered values), expanded by a branch callable, valued by a
critic (:class:`~agentdescent.meta.MetaReward` -- an oracle or the
self-rewarding judge), and searched by UCB -- the same bandit the tree policies
in :mod:`agentdescent.selection` run, applied one level up. ``search_evolve``
spends a fixed budget on the most promising branches and returns the best state
and the tree that found it; the report shows the value estimates the search
used, so the search's own reasoning is inspectable against final truth.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evolution import EvolutionResult
from .meta import MetaOutcome, MetaReward, SlotSpec, auc

__all__ = ["SearchTree", "SearchReport", "search_evolve"]


@dataclass
class _Node:
    visits: int = 0
    value: float = 0.0
    children: List[str] = field(default_factory=list)


class SearchTree:
    """A state tree over rendered artifact values, searched by UCB.

    ``root`` is the starting value; ``expand(rendered, children)`` adds
    branches; ``update(rendered, value)`` back-propagates a visit; ``select``
    walks from the root choosing the child with the highest UCB score (mean
    value plus an exploration bonus), so the budget concentrates on the
    branches the critic has rated most promising.
    """

    def __init__(self, root: str, *, ucb_c: float = 1.4) -> None:
        self.ucb_c = ucb_c
        self._nodes: Dict[str, _Node] = {root: _Node()}
        self.root = root

    def select(self) -> str:
        """The leaf the next expansion should explore, by UCB from the root.
        Cycle-safe: a child that is already an ancestor on this descent stops
        the walk instead of spinning."""
        current = self.root
        visited = {current}
        while self._nodes[current].children:
            total = self._nodes[current].visits
            log_total = math.log(total + 1.0)
            children = self._nodes[current].children
            best, best_score = children[0], float("-inf")
            for child in children:
                node = self._nodes[child]
                mean = node.value / node.visits if node.visits else 0.0
                bonus = self.ucb_c * math.sqrt(log_total / (node.visits + 1e-9))
                score = mean + bonus
                if score > best_score:
                    best, best_score = child, score
            if best in visited:
                break                       # a cycle: stop here, don't spin
            visited.add(best)
            current = best
        return current

    def expand(self, rendered: str, children: Sequence[str]) -> None:
        node = self._nodes[rendered]
        for child in children:
            if child == rendered:
                continue                       # a self-loop would make select spin
            if child not in self._nodes:
                self._nodes[child] = _Node()
            if child not in node.children:
                node.children.append(child)

    def update(self, rendered: str, value: float, *, path: Optional[List[str]] = None) -> None:
        """Back-propagate one visit. When ``path`` is given (the chain from the
        root to ``rendered``), every ancestor is updated too; otherwise only
        the node itself."""
        targets = path or [rendered]
        for name in targets:
            node = self._nodes.get(name)
            if node is None:
                continue
            node.visits += 1
            node.value += value

    def best(self) -> str:
        """The rendered state with the highest mean value among visited nodes."""
        best, best_mean = self.root, float("-inf")
        for name, node in self._nodes.items():
            mean = node.value / node.visits if node.visits else 0.0
            if mean > best_mean:
                best, best_mean = name, mean
        return best

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)


@dataclass(frozen=True)
class SearchReport:
    """What a :func:`search_evolve` found: the best state, the budget spent,
    and the value estimates the search used (best seen vs the tree's choice)."""

    best: str
    value_estimate: float
    verified_value: float
    budget: int
    nodes: int
    depth: int


def search_evolve(
    problem: Callable[[Any, int], MetaOutcome],
    spec: SlotSpec,
    seed_value: str,
    branch: Callable[[str, int], Sequence[str]],
    critic: Optional[MetaReward] = None,
    *,
    budget: int = 40,
    seeds: Sequence[int] = (0, 1),
    ucb_c: float = 1.4,
) -> Tuple[str, SearchReport]:
    """Search the state tree for the best value under a fixed budget.

    From ``seed_value``, each iteration selects a leaf by UCB, branches it into
    child renderings via ``branch``, scores each child with ``critic`` (``auc``
    by default) over ``seeds``, and back-propagates. A leaf whose ``branch``
    yields nothing new is itself marked visited (a dead end must not keep the
    budget). After ``budget`` expansions the **best evaluated state** is
    returned -- never a worse internal node whose mean was diluted by
    back-propagation -- with the report's ``value_estimate`` (the search's own
    belief) and ``verified_value`` (a fresh critic score of the best state)
    side by side, so the search's reasoning is inspectable against final truth.
    """
    score = critic or auc
    tree = SearchTree(seed_value, ucb_c=ucb_c)

    def value_of(rendered: str) -> float:
        compiled = spec.compile(rendered)
        return statistics.fmean(float(score(problem(compiled, s))) for s in seeds)

    best_seen, best_seen_value = seed_value, value_of(seed_value)
    for _ in range(budget):
        leaf = tree.select()
        path = _path_to(tree, leaf)
        children = list(branch(leaf, _))
        tree.expand(leaf, children)
        if not children:
            tree.update(leaf, value_of(leaf), path=path)     # dead end: count it
        for child in children:
            v = value_of(child)
            if v > best_seen_value:
                best_seen, best_seen_value = child, v
            tree.update(child, v, path=path + [child])

    best = best_seen
    verified = value_of(best)
    depth = _depth(tree, best)
    node = tree._nodes[best]
    estimate = node.value / node.visits if node.visits else 0.0
    report = SearchReport(
        best=best, value_estimate=estimate, verified_value=verified,
        budget=budget, nodes=tree.n_nodes, depth=depth)
    return best, report


def _path_to(tree: SearchTree, target: str) -> List[str]:
    """The chain from the root to ``target``, via depth-first search.
    Cycle-safe: a visited set stops a back-referencing branch from recursing
    forever."""
    def walk(name: str, seen: set) -> Optional[List[str]]:
        if name == target:
            return [name]
        seen.add(name)
        for child in tree._nodes[name].children:
            if child in seen:
                continue
            sub = walk(child, seen)
            if sub is not None:
                return [name] + sub
        return None
    out = walk(tree.root, set())
    return out or [tree.root]


def _depth(tree: SearchTree, name: str) -> int:
    return len(_path_to(tree, name)) - 1
