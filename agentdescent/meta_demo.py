"""A tiny offline meta-learning domain for ``agentdescent meta-campaign --demo``.

The meta stack normally needs a real inner domain: tasks, an inner run, a
reward, an inner proposal and a slot spec. This module supplies the smallest
deterministic one so the campaign surface can be tried without a model or a
dataset. It is deliberately a *demo*, not a benchmark: every function is
deterministic and the inner problem is a synthetic reward curve.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .evolution import Task
from .meta import MetaOutcome, ParamSlot, Problem, SlotSpec
from .sampling import DifficultyWeighted
from .strategies import SingleSlot

__all__ = [
    "inner_propose",
    "inner_strategy",
    "problems",
    "propose",
    "reward",
    "run",
    "slot_spec",
    "tasks_for",
    "val_problems",
]


def _cap(rendered: str) -> int:
    match = re.search(r"cap:(\d+)", rendered)
    return int(match.group(1)) if match else 0


def _tasks() -> List[Task]:
    order = [1, 2, 3, 4, 8, 5, 6, 7]
    return [Task(id=f"demo{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a})
            for i, a in enumerate(order)]


def run(rendered: str, task: Task) -> str:
    """The inner agent: solve only when the rendered rule's cap is high enough."""
    return (str(task.meta["gold"])
            if task.meta.get("a", 0) <= _cap(rendered) else "?")


def reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def inner_propose(rendered: str, task: Task, output: str,
                  score: float) -> Optional[str]:
    if score >= 0.999:
        return "cap:0"
    needed = task.meta.get("a", 0)
    if needed > _cap(rendered):
        return f"cap:{needed}"
    return None


def slot_spec() -> SlotSpec:
    return ParamSlot(
        factory=lambda c: DifficultyWeighted(c=c),
        params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def propose(rendered: str, task: Task, output: str,
            score: float) -> Optional[str]:
    match = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(match.group(1)) if match else 0.2
    return f"c: {min(0.9, c + 0.1):.2f}"


def _climbing(value: Any, seed: int) -> MetaOutcome:
    c = getattr(value, "c", 0.2)
    score = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[score, min(1.0, score + 0.05)], final=score)


def _climbing_val(value: Any, seed: int) -> MetaOutcome:
    return _climbing(value, seed)


def problems() -> Dict[str, Problem]:
    return {"ta": _climbing, "tb": _climbing, "tc": _climbing}


def val_problems() -> Dict[str, Problem]:
    return {"va": _climbing_val}


def inner_strategy() -> SingleSlot:
    return SingleSlot(initial_value="cap:0")


def tasks_for(record: Any) -> List[Task]:
    return _tasks()
