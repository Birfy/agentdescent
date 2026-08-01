"""Parallel paradigms: DP / TP / PP (design doc, section 8).

Three ways to parallelize evolution, mirroring the three axes of model-parallel
training.  All three ride the same async runtime and aggregator; they differ only
in *how work is partitioned and recombined*.

* **DP (data parallel)** -- the default.  Every worker holds the same snapshot;
  tasks are sharded across workers; diffs flow back and are merged.  This is what
  :class:`~agentdescent.async_runtime.AsyncAgentDescent` runs.
* **TP (tensor parallel)** -- a single *hot* artifact is split along its internal
  structure into disjoint sections; each worker is authorized for one section, so
  edits are conflict-free *by construction* and the merge degrades to
  concatenation plus a lightweight consistency reviewer (the analogue of an
  all-reduce).  Use only for a few super-hot artifacts.
* **PP (pipeline parallel)** -- artifacts form a dependency chain
  (lit-review -> mol-engine -> hpc-submit); each stage has its own worker, and a
  downstream failure back-propagates blame to the earliest failing upstream stage
  (shared with the counterfactual-replay attribution of section 7).

This module provides the partition/recombine primitives; DP is exercised by the
async runtime, TP and PP by :mod:`tests.test_parallel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .evolvable import Diff, Evolvable, stable_hash


#: Names for the three paradigms. Descriptive -- nothing dispatches on it; the
#: strategy objects below are what `evolve(parallel=...)` takes.
class ParallelMode(Enum):
    DP = "data_parallel"
    TP = "tensor_parallel"
    PP = "pipeline_parallel"


# ---------------------------------------------------------------------------
# TP -- tensor parallel: section-split a hot artifact, conflict-free by design
# ---------------------------------------------------------------------------


def section_of(key: str, n_sections: int) -> int:
    """Hash an artifact key to a section id.

    A hash *bucket*, not a partition: with ``n_sections`` comparable to the number
    of keys, collisions leave some sections owning several keys and others owning
    none. Kept for callers that have no declared key space; prefer
    :func:`assign_key_sections`, which is what :class:`TensorParallel` uses when
    the key space is known.
    """
    return (stable_hash(key) & 0x7FFFFFFF) % n_sections


def assign_key_sections(keys: Sequence[str], n_sections: int) -> Dict[str, int]:
    """Partition a **known** artifact key space into balanced, disjoint sections.

    TP's guarantee is that each worker owns a disjoint section, so the merge is a
    conflict-free union. That only holds if the sections are a genuine partition of
    the keys the strategy will actually write. Hashing gave neither balance nor
    coverage -- on the four categories the tests use, two collided into one section
    and a third owned nothing at all, so the worker holding it could never commit.

    Round-robin over the sorted keys instead: deterministic, every section
    non-empty whenever ``n_sections <= len(keys)``, and sizes differ by at most one.
    """
    ordered = sorted(dict.fromkeys(keys))
    if not ordered:
        return {}
    return {k: i % n_sections for i, k in enumerate(ordered)}


def assign_sections(worker_ids: Sequence[str], n_sections: int) -> Dict[str, int]:
    """Authorize each worker for exactly one section (round-robin)."""
    return {wid: i % n_sections for i, wid in enumerate(worker_ids)}


class SectionViolation(Exception):
    """Raised when a worker's diff touches a key outside its section."""


@dataclass
class TensorParallelMerge:
    """Merge section-scoped diffs into one artifact (concatenation + review).

    Because sections are disjoint, there is no conflict to resolve -- the merge
    is a union.  The consistency reviewer only asserts the no-overlap invariant
    that makes that union sound (the cheap all-reduce analogue).

    ``keys`` is the artifact's declared key space, and passing it matters: with it
    the ownership map is :func:`assign_key_sections`, **the same partition
    `TensorParallel` and `evolve()` enforce**. Without it this falls back to
    :func:`section_of`, a hash bucket -- which is a different answer to the same
    question, so a diff legal on the `evolve()` path could be rejected here and
    vice versa. That divergence is why the argument exists."""

    n_sections: int
    keys: Optional[Sequence[str]] = None

    def owner_of(self, key: str) -> int:
        """Which section owns ``key`` -- via the declared partition when there is one."""
        if self.keys:
            return assign_key_sections(self.keys, self.n_sections).get(
                key, section_of(key, self.n_sections))
        return section_of(key, self.n_sections)

    def validate(self, diff: Diff, section: int) -> None:
        for key in diff.ops:
            if self.owner_of(key) != section:
                raise SectionViolation(
                    f"diff {diff.diff_id} touches {key!r} outside section {section}"
                )

    def merge(
        self, base: Evolvable, section_diffs: List[Tuple[int, Diff]]
    ) -> Tuple[Evolvable, bool]:
        """Return (merged_artifact, consistency_ok).

        Any :class:`~agentdescent.evolvable.Evolvable` works -- this used to be
        typed to the reference domain's ``RouterSkill``, which made a general
        parallelism primitive depend on the demo domain."""
        seen_keys: Dict[str, int] = {}
        merged = base
        for section, diff in section_diffs:
            self.validate(diff, section)
            for key in diff.ops:
                if key in seen_keys and seen_keys[key] != section:
                    return merged, False  # cross-section collision -> inconsistent
                seen_keys[key] = section
            merged = merged.apply(diff)
        return merged, True


# ---------------------------------------------------------------------------
# PP -- pipeline parallel: dependency chain with upstream blame propagation
# ---------------------------------------------------------------------------


@dataclass
class PipelineChain:
    """An ordered artifact dependency chain, upstream -> downstream."""

    stages: List[str]

    def upstream_of(self, stage: str) -> List[str]:
        i = self.stages.index(stage)
        return self.stages[:i]

    def blame(self, stage_success: Dict[str, bool]) -> Optional[str]:
        """Back-propagate blame to the *earliest* failing stage.

        A downstream failure is only charged to the downstream stage if every
        upstream stage succeeded; otherwise blame flows back through the pipeline
        to the first broken link (design doc, sections 7 and 8)."""
        for stage in self.stages:  # earliest first
            if not stage_success.get(stage, True):
                return stage
        return None

    def counterfactual_pairs(self, stage: str) -> List[Tuple[str, str]]:
        """The {old x new} version swaps to replay for minimal factor analysis.

        For the blamed ``stage`` we replay holding every other stage fixed and
        swapping just this stage's artifact version, isolating its contribution
        (the counterfactual-replay facility of section 7)."""
        return [("old:" + stage, "new:" + stage)]


# ---------------------------------------------------------------------------
# DP -- data parallel: task sharding (default; run by the async runtime)
# ---------------------------------------------------------------------------


def shard_round_robin(items: Sequence, n_shards: int) -> List[List]:
    """Split a task list into ``n_shards`` disjoint shards, round-robin."""
    shards: List[List] = [[] for _ in range(max(1, n_shards))]
    for i, item in enumerate(items):
        shards[i % len(shards)].append(item)
    return shards


# ---------------------------------------------------------------------------
# Pluggable parallelism: choose (or write) how work is partitioned across workers
# ---------------------------------------------------------------------------

from typing import Protocol, runtime_checkable  # noqa: E402


@dataclass
class WorkUnit:
    """What one worker is responsible for in one round of a parallel plan."""

    worker: int
    keys: List[str]                 # the artifact keys / tasks this worker owns
    stage: int = 0                  # PP: which pipeline stage/artifact
    section: Optional[int] = None   # TP: which section of the artifact


@runtime_checkable
class ParallelStrategy(Protocol):
    """How a round of work is partitioned across ``n_workers``.

    This is the customization point: implement ``plan`` to define your own
    parallelism method. The three classic paradigms are provided below."""

    name: str

    def plan(self, n_workers: int, round_index: int, keys: Sequence[str]) -> List[WorkUnit]: ...


@dataclass
class DataParallel:
    """DP -- every worker holds the same artifact; the *tasks* (keys) are sharded
    across workers and their diffs are merged. Coverage rotates each round."""

    name: str = "DP"

    def plan(self, n_workers, round_index, keys) -> List[WorkUnit]:
        shards = shard_round_robin(list(keys), n_workers)
        m = len(shards)
        return [WorkUnit(worker=i, keys=shards[(i + round_index) % m]) for i in range(n_workers)]


@dataclass
class TensorParallel:
    """TP -- one hot artifact is split into ``n_sections`` disjoint sections; each
    worker owns a section, so edits are conflict-free *by construction* and the
    merge is a union (concatenation + a consistency check).

    ``keys`` is the **artifact's** key space -- the keys the strategy writes. It is
    what the sections are carved out of, and :func:`~agentdescent.evolution.evolve`
    fills it in from the strategy when the strategy declares one
    (:class:`~agentdescent.evolution.KeyedRules` does; so does
    :class:`~agentdescent.evolution.SingleSlot`, with a single key, which is why it
    cannot be tensor-parallelised).

    The section and the *tasks* a worker rolls out are orthogonal, and conflating
    them was the bug this replaces: ``plan`` used to filter **task ids** through
    ``section_of``, then ``evolve`` enforced the section against the **artifact
    keys** the resulting diff wrote. Two unrelated key spaces, so a worker's legal
    tasks had nothing to do with its legal edits and 75-88% of proposals were
    dropped without a word. Tasks are now sharded data-parallel exactly as
    :class:`DataParallel` does; only the section is TP's business.

    ``route`` closes the loop. A worker owns a section of the *artifact*, but it is
    handed *tasks* -- and only the caller knows which artifact key a given task's
    failure will produce an edit for. Give ``route(task_id) -> artifact key`` and
    each worker is handed exactly the tasks whose edits land in its own section, so
    every proposal is legal and TP costs nothing in throughput. Without it, tasks
    are sharded data-parallel and any proposal that misses the worker's section is
    rejected and counted as ``section-violation`` in
    :meth:`~agentdescent.evolution.EvolutionResult.outcomes` -- visible, rather
    than the silent discard this replaces.
    """

    n_sections: int
    keys: Optional[Sequence[str]] = None
    route: Optional[Callable[[str], str]] = None
    name: str = "TP"

    def plan(self, n_workers, round_index, keys) -> List[WorkUnit]:
        # `keys` are TASK ids. The artifact section is a separate axis.
        sections = [i % self.n_sections for i in range(n_workers)]
        if self.route is not None:
            owner = self.section_map()
            by_section: Dict[int, List[str]] = {s: [] for s in range(self.n_sections)}
            for task_id in keys:
                sec = owner.get(self.route(task_id))
                if sec is not None:
                    by_section[sec].append(task_id)
            return [WorkUnit(worker=i, keys=list(by_section[sections[i]]),
                             section=sections[i])
                    for i in range(n_workers)]
        shards = shard_round_robin(list(keys), n_workers)
        m = len(shards)
        return [WorkUnit(worker=i, keys=shards[(i + round_index) % m],
                         section=sections[i])
                for i in range(n_workers)]

    def section_map(self) -> Dict[str, int]:
        """``artifact key -> section``. Empty when no key space was declared."""
        return assign_key_sections(self.keys or (), self.n_sections)


@dataclass
class PipelineParallel:
    """PP -- artifacts form a dependency chain; each worker drives one stage, and
    a downstream failure back-propagates blame to the earliest failing stage
    (via :class:`PipelineChain`)."""

    stages: Sequence[str]
    name: str = "PP"

    def plan(self, n_workers, round_index, keys) -> List[WorkUnit]:
        ns = len(self.stages)
        return [WorkUnit(worker=i, keys=list(keys), stage=i % ns) for i in range(n_workers)]

    def chain(self) -> "PipelineChain":
        return PipelineChain(list(self.stages))


STRATEGIES = {"DP": DataParallel, "TP": TensorParallel, "PP": PipelineParallel}
