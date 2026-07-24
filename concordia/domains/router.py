"""A deterministic reference Evolvable: the keyword-router skill.

Why a synthetic domain?  The framework's interesting behaviour (staleness,
conflict resolution, fusion, statistical acceptance, merge-vs-fork) is a
property of the *aggregation system*, not of any particular LLM.  A crisp,
deterministic task lets the whole parallel RSI loop run in-process and be unit
tested, with no external model calls -- while still producing genuine diffs that
measurably improve a metric.

The task: classify a piece of text by the labelled keyword it contains.  The
skill is a ``keyword -> label`` table.  The *optimal* skill maps every keyword
to its gold label; a fresh skill knows nothing and must evolve there.

* Two workers fixing *different* keywords -> complementary diffs -> fusion wins.
* A noisy worker proposing the *wrong* label for a keyword -> a contradiction
  the aggregator's conflict resolution must drop.

This is exactly the structure needed to demonstrate that *merging* concurrent
diffs beats *forking* them (design doc, RQ1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from ..evolvable import Contract, Diff, EvidenceCard


@dataclass(frozen=True)
class Task:
    text: str
    label: str
    keyword: str


class RouterSkill:
    """An :class:`~concordia.evolvable.Evolvable` keyword->label classifier."""

    def __init__(
        self,
        id: str,
        table: Optional[Dict[str, str]] = None,
        version: int = 1,
        blast_radius: float = 0.2,
    ) -> None:
        self.id = id
        self.table: Dict[str, str] = dict(table or {})
        self.version = version
        self.blast_radius = blast_radius
        self.contract = Contract(input_schema="text", output_schema="label", major=1)

    # -- classification -------------------------------------------------------

    def classify(self, text: str) -> str:
        # deterministic: first table keyword (sorted) that appears in the text.
        for kw in sorted(self.table):
            if kw in text:
                return self.table[kw]
        return "unknown"

    def accuracy(self, tasks: Sequence[Task]) -> float:
        if not tasks:
            return 0.0
        correct = sum(1 for t in tasks if self.classify(t.text) == t.label)
        return correct / len(tasks)

    # -- Evolvable protocol ---------------------------------------------------

    def diff(self, other: "RouterSkill") -> Diff:
        ops = {k: v for k, v in other.table.items() if self.table.get(k) != v}
        return Diff(diff_id=f"{self.id}:diff", target=self.id, ops=ops)

    def apply(self, diff: Diff) -> "RouterSkill":
        new_table = dict(self.table)
        new_table.update(diff.ops)
        return RouterSkill(self.id, new_table, self.version + 1, self.blast_radius)

    def cheap_eval(self, evidence: EvidenceCard) -> float:
        """Accuracy on the tasks the evidence card carries.

        The card's ``trajectory_refs`` hold the failing :class:`Task` objects
        that justified the diff; used by the aggregator's rebase re-verification
        (design doc, section 4.2)."""
        tasks = [t for t in evidence.trajectory_refs if isinstance(t, Task)]
        return self.accuracy(tasks)

    def full_eval(self, task_set: Sequence[Task]) -> Mapping[str, float]:
        return {"accuracy": self.accuracy(task_set)}


# -- serialization for the git-backed ledger ---------------------------------


def serialize_router(skill: RouterSkill) -> dict:
    return {"table": skill.table, "blast_radius": skill.blast_radius}


def deserialize_router(artifact_id: str, version: int, state: dict) -> RouterSkill:
    return RouterSkill(
        id=artifact_id,
        table=state.get("table", {}),
        version=version,
        blast_radius=state.get("blast_radius", 0.2),
    )


def router_eval(skill: RouterSkill, tasks: Sequence[Task]) -> float:
    """Ground-truth eval function handed to the three-layer verifier."""
    return skill.accuracy(tasks)


# -- synthetic task universe --------------------------------------------------

_LABELS = ["acidbase", "kinetics", "thermo", "spectro", "organic", "quantum"]


def make_task_universe(
    n_keywords: int = 24,
    tasks_per_keyword: int = 12,
    seed: int = 7,
) -> "TaskUniverse":
    """Build a keyword->label gold mapping and a pool of tasks over it."""
    rng = random.Random(seed)
    gold: Dict[str, str] = {}
    tasks: List[Task] = []
    for i in range(n_keywords):
        kw = f"kw{i:02d}"
        label = _LABELS[i % len(_LABELS)]
        gold[kw] = label
        for j in range(tasks_per_keyword):
            filler = rng.choice(["report", "log", "note", "trace", "run"])
            tasks.append(Task(text=f"{filler}-{kw}-{j}", label=label, keyword=kw))
    rng.shuffle(tasks)
    return TaskUniverse(gold=gold, tasks=tasks, seed=seed)


@dataclass
class TaskUniverse:
    gold: Dict[str, str]
    tasks: List[Task]
    seed: int = 7

    def split(self, held_out_frac: float = 0.4):
        rng = random.Random(self.seed + 1)
        pool = list(self.tasks)
        rng.shuffle(pool)
        cut = int(len(pool) * (1 - held_out_frac))
        return pool[:cut], pool[cut:]  # (train, held_out)

    def clusters(self, n_clusters: int = 6) -> List[List[Task]]:
        """Partition tasks into clusters by keyword hash (task long tail)."""
        buckets: List[List[Task]] = [[] for _ in range(n_clusters)]
        for t in self.tasks:
            buckets[hash(t.keyword) % n_clusters].append(t)
        return [b for b in buckets if b]
