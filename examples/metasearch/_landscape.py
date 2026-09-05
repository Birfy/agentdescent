"""A synthetic program-search landscape: the cheap inner domain.

The outer loop needs to run a *whole tree search* per rollout and again per
held-out task at every gate, so its inner problem has to be cheap, seeded, and
free of any model. This is that problem, in the shape the real ones have:

* a **node** is a hidden vector ``x``; the root starts far from an optimum;
* an **expansion** is a random local move from the parent, ``x + step * g``;
* a **score** is ``exp(-|x - x*|^2 / d)`` in ``(0, 1]``, optionally made rugged
  by a cosine term so that a greedy rule can be misled;
* a **dead end** happens with probability ``p_dead``: the child is invalid and
  scores ``-inf``, as a program that fails to run does in ERA.

Two families are defined. ``SOURCE`` is what the policy is evolved on;
``TARGET`` is higher-dimensional, ruggeder and deadlier, and is never seen by
the outer loop -- it is the *validation* set for whether an evolved rule is a
better search rule or a fit to one landscape.

The tree that runs the search is :class:`~examples.era.era_empirical_software.EraTree`
itself, not a copy: the visit reservation, back-propagation and node bookkeeping
an evolved rule is measured under here are the ones it will meet on AlgoTune or
a Harbor task. Only the payload is synthetic.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from agentdescent.selection import SelectionPolicy

from examples.era._era_support import Program
from examples.era.era_empirical_software import EraTree


@dataclass(frozen=True)
class Family:
    name: str
    dim: int
    step: float
    p_dead: float
    ruggedness: float
    #: How far from the optimum the root is placed, per coordinate.
    offset: float


SOURCE = Family("source", dim=4, step=0.45, p_dead=0.25, ruggedness=0.0, offset=1.2)
TARGET = Family("target", dim=6, step=0.35, p_dead=0.45, ruggedness=0.35, offset=1.4)
FAMILIES: Dict[str, Family] = {f.name: f for f in (SOURCE, TARGET)}


def score(family: Family, x: Sequence[float], optimum: Sequence[float]) -> float:
    dist = sum((a - b) ** 2 for a, b in zip(x, optimum)) / family.dim
    base = math.exp(-dist)
    if family.ruggedness:
        base *= 1.0 - family.ruggedness * (0.5 + 0.5 * math.cos(3.0 * sum(x)))
    return base


@dataclass
class SearchTrace:
    """What one inner search did, for the reward and for the reflector."""

    family: str
    seed: int
    budget: int
    curve: List[float]            # best-so-far score after each expansion
    expanded_depths: List[int]
    dead_ends: int
    root_score: float
    best_score: float
    nodes: int

    @property
    def auc(self) -> float:
        """Mean best-so-far over the budget, in ``[0, 1]``.

        The final best alone barely separates selection rules at a fixed
        budget; *how fast* the best rises is what a selection rule controls.
        """
        return sum(self.curve) / len(self.curve) if self.curve else 0.0

    def to_json(self) -> str:
        return json.dumps({
            "family": self.family, "seed": self.seed, "budget": self.budget,
            "auc": self.auc, "root_score": self.root_score,
            "best_score": self.best_score, "nodes": self.nodes,
            "dead_ends": self.dead_ends, "expanded_depths": self.expanded_depths,
            "curve": [round(v, 4) for v in self.curve],
        }, separators=(",", ":"))


def search(policy: Optional[SelectionPolicy], family: Family, seed: int,
           budget: int = 24) -> SearchTrace:
    """Run one flat tree search over the landscape with ``policy`` choosing parents."""
    rng = random.Random(seed * 100_003 + family.dim)
    optimum = [rng.uniform(-1.0, 1.0) for _ in range(family.dim)]
    root_x = [o + family.offset * (1 if rng.random() < 0.5 else -1) for o in optimum]
    tree = EraTree(candidate_limit=budget, metric_key="score", policy=policy)
    vectors: Dict[int, List[float]] = {}
    root_score = score(family, root_x, optimum)
    root = tree.seed(Program("root", 0, None, json.dumps(root_x), "", {"score": root_score}, True),
                     root_score)
    vectors[root.index] = root_x
    curve: List[float] = []
    depths: List[int] = []
    dead = 0
    best = root_score
    while True:
        selection = tree.select_parent()
        if selection is None:
            break
        iteration, parent = selection
        depth, cursor = 0, parent
        while cursor.parent_index is not None:
            cursor = tree.nodes[cursor.parent_index]
            depth += 1
        depths.append(depth)
        px = vectors[parent.index]
        if rng.random() < family.p_dead:
            dead += 1
            node = tree.add_node(Program(f"dead{iteration}", iteration, parent.program.program_id,
                                         "", "", {"score": -math.inf}, False, "dead end"),
                                 -math.inf, parent.index)
            vectors[node.index] = px
        else:
            child_x = [v + family.step * rng.gauss(0.0, 1.0) for v in px]
            s = score(family, child_x, optimum)
            node = tree.add_node(Program(f"n{iteration}", iteration, parent.program.program_id,
                                         json.dumps(child_x), "", {"score": s}, True),
                                 s, parent.index)
            vectors[node.index] = child_x
            best = max(best, s)
        curve.append(best)
    return SearchTrace(family.name, seed, budget, curve, depths, dead, root_score,
                       best, len(tree.nodes))
