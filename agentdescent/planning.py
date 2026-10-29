"""Model-based planning over a learned simulator and value function.

`world_model` learns what happens next. `selection.MCTS` searches a candidate
tree. What was missing is the planner that turns a simulator + value estimate
into an action:

* a caller-supplied `Simulator` exposes `actions(state)` and
  `step(state, action) -> PlanTransition`;
* a caller-supplied value function scores non-terminal leaves (an LLM judge, a
  process reward model, a verifier, or a hand-written heuristic);
* `MCTSPlanner` runs UCT with a deterministic seed and a hard node budget.

The planner is intentionally independent of the engine. A state can be a prompt,
a code tree, a swarm topology or a sandbox world; the simulator adapts it.

    planner = MCTSPlanner(simulator, value_fn, budget=200)
    result = planner.plan(initial_state)
    result.action, result.value, result.visits

Each node stores the immediate reward of the edge that reached it. A leaf
return is ``edge + discount * value_fn(state)`` (terminal leaves use ``0`` as
the future value), and every backup re-propagates that same return, adding each
ancestor's own edge reward. Q-values therefore contain the path's rewards
instead of decaying as ``reward / visits``.

This is a vertical slice: no neural network, no distributed search. The
selection/expansion/backup contract is what a GPU batched planner keeps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "PlanTransition",
    "PlanResult",
    "MCTSNode",
    "MCTSPlanner",
]


@dataclass(frozen=True)
class PlanTransition:
    state: Any
    reward: float = 0.0
    terminal: bool = False
    cost: float = 0.0

    def __post_init__(self) -> None:
        for name in ("reward", "cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"PlanTransition.{name} must be finite")
        if self.cost < 0:
            raise ValueError("PlanTransition.cost must be >= 0")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "cost", float(self.cost))


@dataclass(frozen=True)
class PlanResult:
    action: Any
    value: float
    visits: int
    nodes: int
    root_visits: int
    detail: str = ""


@dataclass
class MCTSNode:
    state: Any
    parent: Optional["MCTSNode"] = None
    action: Any = None
    children: Dict[Any, "MCTSNode"] = field(default_factory=dict)
    actions: Optional[Tuple[Any, ...]] = None
    untried: List[Any] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    terminal: bool = False
    cost: float = 0.0
    edge: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSPlanner:
    """A deterministic UCT planner over a caller-supplied simulator."""

    def __init__(self, simulator: Any, value_fn: Callable[[Any], float], *,
                 budget: int = 100, c: float = 1.4,
                 discount: float = 1.0, cost_weight: float = 0.0,
                 seed: int = 0) -> None:
        if not callable(getattr(simulator, "actions", None)) or not callable(
                getattr(simulator, "step", None)):
            raise TypeError("simulator must expose actions(state) and step(state, action)")
        if not callable(value_fn):
            raise TypeError("value_fn must be callable")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if not math.isfinite(float(c)) or c <= 0:
            raise ValueError("c must be finite and > 0")
        if not 0.0 < float(discount) <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not math.isfinite(float(cost_weight)) or cost_weight < 0:
            raise ValueError("cost_weight must be finite and >= 0")
        self.simulator = simulator
        self.value_fn = value_fn
        self.budget = budget
        self.c = float(c)
        self.discount = float(discount)
        self.cost_weight = float(cost_weight)
        self.rng = random.Random(seed)
        self.nodes = 0

    def _value(self, state: Any) -> float:
        value = float(self.value_fn(state))
        if not math.isfinite(value):
            raise ValueError("value_fn returned a non-finite value")
        return value

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.actions is None:
            node.actions = tuple(self.simulator.actions(node.state))
            node.untried = list(node.actions)
        if not node.untried:
            return node
        index = self.rng.randrange(len(node.untried))
        action = node.untried.pop(index)
        transition = self.simulator.step(node.state, action)
        if not isinstance(transition, PlanTransition):
            raise TypeError("simulator.step must return PlanTransition")
        child = MCTSNode(state=transition.state, parent=node, action=action,
                         terminal=transition.terminal, cost=transition.cost,
                         edge=transition.reward - self.cost_weight * transition.cost)
        node.children[action] = child
        self.nodes += 1
        return child

    def _select(self, node: MCTSNode) -> MCTSNode:
        while True:
            if node.terminal:
                return node
            if node.actions is None:
                node.actions = tuple(self.simulator.actions(node.state))
                node.untried = list(node.actions)
            if node.untried:
                return self._expand(node)
            if not node.children:
                return node
            node = max(node.children.values(),
                       key=lambda child: self._uct(child, node))

    def _uct(self, child: MCTSNode, parent: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.value
        exploration = self.c * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        return exploitation + exploration

    def _backup(self, node: MCTSNode, future: float) -> None:
        while node is not None:
            value = node.edge + self.discount * future
            node.visits += 1
            node.value_sum += value
            future = value
            node = node.parent

    def plan(self, state: Any) -> PlanResult:
        root = MCTSNode(state=state)
        self.nodes = 1
        for _ in range(self.budget):
            leaf = self._select(root)
            if leaf is root:
                continue
            future = 0.0 if leaf.terminal else self._value(leaf.state)
            self._backup(leaf, future)
        if not root.children:
            raise ValueError("planner found no actions")
        best = max(root.children.values(),
                   key=lambda child: (child.visits, child.value, str(child.action)))
        detail = (f"{self.budget} simulations, {self.nodes} nodes, "
                  f"visits={best.visits}, value={best.value:.4f}")
        return PlanResult(action=best.action, value=best.value,
                          visits=best.visits, nodes=self.nodes,
                          root_visits=root.visits, detail=detail)
