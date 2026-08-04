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
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..evolvable import Contract, Diff, EvidenceCard, stable_hash


@dataclass(frozen=True)
class RouterTask:
    """One item of the reference domain: a text, its gold label, its keyword.

    Named ``Task`` until it collided with :class:`agentdescent.evolution.Task`
    -- ``Task(id, prompt, meta)``, the one every caller writes against. Disjoint
    fields, no relationship, same name, and a reader following
    ``AgentDescent -> Worker -> Task`` from the architecture page landed on this
    one with no signal. ``Task`` remains as an alias inside this module."""

    text: str
    label: str
    keyword: str


#: Back-compatible alias, kept only because it is a published name. It collides
#: with :class:`agentdescent.evolution.Task` -- the one every caller writes
#: against -- and the two share no fields and no relationship, so a reader who
#: followed `AgentDescent -> Worker -> Task` landed on the wrong class with no
#: signal. Nothing inside the package uses it any more; prefer `RouterTask`.
Task = RouterTask


class RouterSkill:
    """An :class:`~agentdescent.evolvable.Evolvable` keyword->label classifier."""

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

    def evidence_eval(self, evidence: EvidenceCard) -> float:
        """Accuracy on the tasks the evidence card carries.

        The card's ``trajectory_refs`` hold the failing :class:`Task` objects
        that justified the diff; used by the aggregator's rebase re-verification
        (design doc, section 4.2)."""
        tasks = [t for t in evidence.trajectory_refs if isinstance(t, Task)]
        return self.accuracy(tasks)


    #: Back-compatible alias for :meth:`evidence_eval`, which it was called until
    #: the name collided with the verifier's `cheap_eval(artifact)`.
    cheap_eval = evidence_eval

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
            buckets[stable_hash(t.keyword) % n_clusters].append(t)
        return [b for b in buckets if b]


# ---------------------------------------------------------------------------
# The same domain, expressed for the general engine
# ---------------------------------------------------------------------------
#
# `RouterSkill` + `Worker` are an `Evolvable` and a rollout written *for the
# reference runtime*. Everything below says the same thing in the vocabulary
# `evolve()` speaks -- a `Strategy` plus `run` / `reward` / `propose` -- so the
# domain stops needing a second engine to run it (see issue #93).
#
# One engine `Task` is one **cluster**, because that is what a reference rollout
# has always been: `Worker.run(skill, base_version, tasks)` classifies a whole
# cluster and proposes fixes for up to `max_ops` of the keywords that failed. A
# task-per-rollout port would have quietly changed the diff from "up to four
# complementary ops" to "one op", and the complementary-ops case is what the
# fusion tournament exists for.
#
# Three things do **not** survive the translation exactly, and are listed here
# rather than discovered later:
#
# 1. `before_after_delta` is measured on the whole cluster, not on the failing
#    subset, because `self_verify` re-runs `run(rendered, task)` and a task *is*
#    the cluster. Same sign, smaller magnitude -- it feeds `observe_delta`, whose
#    weight scales with magnitude.
# 2. `evidence_eval` likewise scores the cluster rather than the failing subset.
# 3. **Noise is per proposal, not per worker.** The reference makes a minority of
#    workers noisy (`noise if i % 3 == 0`); the general engine has one `propose`
#    for every worker and, by design, no worker identity to branch on -- that is
#    the rule `policies.py` states, and honouring it costs the per-worker
#    correlation. What the mechanism needs is contradictions, and a per-proposal
#    rate still produces them. It is seeded from `(seed, task, keyword)`, so it
#    is deterministic *and* order-independent, which the per-worker RNG was not
#    once workers ran concurrently.

ROUTER_TITLE = "# keyword -> label"


def render_table(table: Mapping[str, str]) -> str:
    """The artifact as text. Round-trips through :func:`parse_table`."""
    if not table:
        return f"{ROUTER_TITLE}\n(empty)"
    return "\n".join([ROUTER_TITLE] + [f"{k}: {table[k]}" for k in sorted(table)])


def parse_table(rendered: str) -> Dict[str, str]:
    """The inverse of :func:`render_table`, tolerant of the empty marker."""
    table: Dict[str, str] = {}
    for line in (rendered or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "(empty)":
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip():
            table[key.strip()] = value.strip()
    return table


@dataclass
class RouterStrategy:
    """The keyword table as an evolvable artifact, for :func:`evolve`.

    The op space is one key per keyword, so two workers fixing *different*
    keywords produce complementary diffs the aggregator fuses, and two workers
    proposing different labels for the *same* keyword contradict -- which is the
    whole point of the domain, preserved rather than reconstructed.

    ``max_ops`` is the reference ``Worker.max_ops``: a rollout proposes fixes for
    at most this many failing keywords, which keeps a diff inside the trust
    region.
    """

    max_ops: int = 4

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Mapping[str, str]) -> str:
        return render_table(state)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        ops: Dict[str, str] = {}
        for keyword, label in parse_table(proposal).items():
            if state.get(keyword) == label:
                continue                      # already says that; not a change
            ops[keyword] = label
            if len(ops) >= self.max_ops:
                break
        if not ops:
            return None
        # Same shape as the reference worker's, so a diff id still reads as
        # "who, which keywords, against which version".
        return Diff(diff_id=f"{author}:{'-'.join(sorted(ops))}:{base_version}",
                    target=target, ops=ops, author=author)


def _as_engine_task(prefix: str, index: int, items: Sequence[Task]):
    from ..evolution import Task as EngineTask

    keywords = sorted({t.keyword for t in items})
    return EngineTask(
        id=f"{prefix}{index}",
        prompt=f"{len(items)} texts over {len(keywords)} keyword(s)",
        meta={"items": list(items), "keywords": keywords})


def cluster_tasks(universe: "TaskUniverse", n_clusters: int = 6,
                  held_out_frac: float = 0.4) -> Tuple[List, List]:
    """``(train, held_out)`` engine tasks. One task is one **cluster**.

    A rollout in this domain is "classify a cluster and fix what failed", so the
    unit the engine schedules has to be the cluster.

    **The two sides are clustered differently, and that is not an oversight.**
    `evolve()` splits its task list positionally, so if both sides were built by
    :meth:`TaskUniverse.clusters` -- which buckets by *keyword* hash -- train and
    held-out would hold **disjoint keyword sets**, and a skill learned on one
    could never score on the other. Measured before this was fixed: 6 commits, a
    rising dev accuracy, and a held-out reward pinned at 0.000 for the whole run.

    So the train side keeps keyword-hash clusters, because that long-tail
    structure is what makes cluster leasing (and the whole L-task argument) mean
    anything, and the held-out side is chunked round-robin over the same
    keywords. Held-out chunks come out within one item of each other, so the
    engine's mean-over-tasks and the reference's flat mean-over-items agree to
    O(1/n).
    """
    train, held = universe.split(held_out_frac)
    buckets: List[List[Task]] = [[] for _ in range(max(1, n_clusters))]
    for item in train:
        buckets[stable_hash(item.keyword) % len(buckets)].append(item)
    train_tasks = [_as_engine_task("c", i, b)
                   for i, b in enumerate(b for b in buckets if b)]

    chunks: List[List[Task]] = [[] for _ in range(max(1, n_clusters))]
    for i, item in enumerate(held):
        chunks[i % len(chunks)].append(item)
    held_tasks = [_as_engine_task("h", i, c)
                  for i, c in enumerate(c for c in chunks if c)]
    return train_tasks, held_tasks


def cluster_split_frac(train_tasks: Sequence, held_tasks: Sequence) -> float:
    """The ``held_out_frac`` that makes `evolve()`'s positional split land here.

    The split is ``round(n * (1 - frac))``, so this is exact for integer counts
    rather than approximately right."""
    total = len(train_tasks) + len(held_tasks)
    return len(held_tasks) / total if total else 0.5


def router_run(rendered: str, task) -> str:
    """Classify every text in the cluster, in order."""
    skill = RouterSkill("probe", parse_table(rendered))
    return "\n".join(skill.classify(item.text) for item in task.meta["items"])


def router_reward(task, output: str) -> float:
    """Cluster accuracy -- exactly what ``RouterSkill.accuracy`` measures."""
    items = task.meta["items"]
    if not items:
        return 0.0
    predicted = (output or "").split("\n")
    correct = sum(1 for item, guess in zip(items, predicted) if guess == item.label)
    return correct / len(items)


def router_propose(gold: Mapping[str, str], noise: float = 0.0, seed: int = 0,
                   max_ops: int = 4):
    """The reference corrector, as a ``propose`` callable.

    Looks at what the rendered table gets wrong in this cluster, takes up to
    ``max_ops`` distinct failing keywords in cluster order -- the same choice
    ``Worker.run`` makes -- and proposes the gold label for each, or a
    plausible-but-wrong one with probability ``noise``.

    An LLM stands in for this in a real domain. Here it is deterministic, which
    is the entire reason the reference domain exists.

    **The noise is a rate drawn from a stream, not a fixed mask.** Seeding it per
    ``(seed, task, keyword)`` looks tidier -- deterministic *and* independent of
    the order concurrent workers run in -- and is wrong in a way that takes a
    while to see: the draw for a given keyword never changes, so a keyword the
    noise corrupts is corrupted *identically* on every later attempt, ``to_diff``
    sees the state already equals it, and the entry can never be corrected.
    Measured: the run learned all 24 keywords, committed a wrong label for three
    of them, then stopped proposing entirely and plateaued at 0.853 while the
    reference reached 1.000.

    A stream restores what the reference has: a noisy proposal is *transient*, so
    a later attempt gets it right and the tournament prefers the candidate that
    scores better. Deterministic while the loop is sequential -- which is what the
    reference runtime is and what the adapter asks for -- and a rate rather than a
    reproducible mask once it is not.
    """
    labels = sorted(set(gold.values()))
    rng = random.Random(stable_hash(("router-noise", seed)))
    rng_lock = threading.Lock()

    def _corrupt(true_label: str) -> Optional[str]:
        """Draw once; return a wrong label, or ``None`` to keep the gold one."""
        with rng_lock:
            if rng.random() >= noise:
                return None
            wrong = [l for l in labels if l != true_label]
            return rng.choice(wrong) if wrong else None

    def propose(rendered: str, task, output: str, reward: float) -> Optional[str]:
        skill = RouterSkill("probe", parse_table(rendered))
        chosen: Dict[str, object] = {}
        for item in task.meta["items"]:
            if skill.classify(item.text) == item.label:
                continue
            chosen.setdefault(item.keyword, item)
            if len(chosen) >= max_ops:
                break
        if not chosen:
            return None
        lines = []
        for keyword, item in chosen.items():
            corrupted = _corrupt(item.label)
            lines.append(f"{keyword}: {corrupted or gold[keyword]}")
        return "\n".join(lines)

    return propose
