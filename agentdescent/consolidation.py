"""Cross-run consolidation: what a run learned survives it.

An evolution run starts from an empty artifact, learns, and then the skill
dies with the process that made it. The trajectory ledger (#248) records *what
happened* during a run; :class:`ConsolidationStore` records *what it produced*
-- the artifact that survived, its measured reward, the configuration that
made it, and the trajectories that justify it -- in a durable archive a later
run can retrieve. That is the difference between self-improvement (one run)
and *continuing* self-improvement (skills that seed the next run).

The archive is deliberately conservative:

* **It stores the artifact, not the trajectories.** ``trajectory_store`` is a
  path, linked not copied -- evidence stays where the run left it, and the
  archive is small.
* **Seeding is gated.** ``seed_from_archive`` retrieves a candidate and only
  hands its state to the new run if a **mini-eval on a sample of the new
  tasks** shows it does not regress beyond ``min_reward`` against the empty
  seed. A skill learned on a different dataset is exactly the seed that helps
  nothing and hurts the cold start, so the gate counts the refusal instead of
  installing a regression.
* **Retrieval prefers the same configuration.** Archived runs whose
  ``config`` fingerprint matches the new run's are ranked first, then by
  measured reward. A caller names the dataset/scorer/params in ``config`` and
  the archive uses them to keep domains apart.

``archive`` is idempotent by ``run_id`` (atomic rewrite) and the store is
thread-safe, so two runs finishing at once both land.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task

__all__ = ["ConsolidationRecord", "ConsolidationStore", "SeedReport",
           "config_fingerprint", "seed_from_archive"]


def config_fingerprint(config: Optional[Dict[str, Any]]) -> str:
    """A stable id for the run configuration (dataset, scorer, model, params).

    ``None`` produces a sentinel that matches nothing, so a run without a
    declared configuration cannot silently adopt an archive entry from an
    unrelated domain."""
    if not config:
        return "no-config"
    try:
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    except (TypeError, ValueError):
        return "no-config"


@dataclass(frozen=True)
class ConsolidationRecord:
    """One archived run: the artifact it produced, and the configuration that
    produced it."""

    run_id: str
    artifact_id: str
    #: The strategy class name the state was written by; a new run adopts the
    #: state only through a strategy of the same name (its ``to_diff``/``render``
    #: must understand the keys).
    strategy: str
    #: The caller's run configuration, hashed for retrieval. The raw dict is
    #: kept so a human can read what the fingerprint means.
    config: Dict[str, Any]
    #: The artifact's flat state -- the seed a new run can adopt.
    state: Dict[str, str]
    rendered: str
    final_reward: float
    rounds: int
    #: Durable trajectory ledger path, linked not copied (the evidence lives
    #: where the run left it; the archive only points at it).
    trajectory_store: Optional[str] = None
    #: How to rebuild the inner problem(s) this product was produced or judged
    #: on. ``None`` for a plain artifact record; a rule record (written by
    #: ``agentdescent.meta.archive_rule``) carries the problem descriptor so a
    #: later session can rebuild the same inner problems via
    #: ``agentdescent.meta.problem_from_archive``. Optional and backwards
    #: compatible: an archive written before this field existed reads back as
    #: ``None`` and stays seedable.
    problem: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

    @property
    def config_hash(self) -> str:
        return config_fingerprint(self.config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "strategy": self.strategy,
            "config": self.config,
            "state": self.state,
            "rendered": self.rendered,
            "final_reward": self.final_reward,
            "rounds": self.rounds,
            "trajectory_store": self.trajectory_store,
            "problem": self.problem,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsolidationRecord":
        return cls(
            run_id=d["run_id"], artifact_id=d.get("artifact_id", "artifact"),
            strategy=d.get("strategy", ""),
            config=dict(d.get("config", {})),
            state=dict(d.get("state", {})),
            rendered=d.get("rendered", ""),
            final_reward=float(d.get("final_reward", 0.0)),
            rounds=int(d.get("rounds", 0)),
            trajectory_store=d.get("trajectory_store"),
            problem=dict(d["problem"]) if d.get("problem") else None,
            created_at=float(d.get("created_at", 0.0)),
        )


class ConsolidationStore:
    """A durable archive of what runs learned.

    One JSON record per run under ``<root>/runs/<sha256(run_id)>.json``.
    Filenames hash the complete ``run_id`` so distinct identities never share
    a path (the character-stripping mapper that preceded #338 collapsed
    ``"a/b"`` and ``"ab"`` onto the same file). Writes are atomic (tmp +
    rename) and idempotent by ``run_id``; ``find`` still accepts a pre-#338
    sanitized filename when the stored identity matches. A corrupt record is
    skipped on read, never raised -- this is derived data and a bad file must
    not take the next run down.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self._runs_dir = os.path.join(root, "runs")
        os.makedirs(self._runs_dir, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _digest_name(run_id: str) -> str:
        """Collision-resistant filename for the complete ``run_id``.

        Distinct identities never share a path: hashing the full string keeps
        filesystem safety without collapsing ``"a/b"`` and ``"ab"`` onto the
        same file the way character stripping did.
        """
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return f"{digest}.json"

    @staticmethod
    def _legacy_name(run_id: str) -> str:
        """Pre-#338 sanitized filename (unsafe chars deleted).

        Kept only so ``find`` can still locate archives written before the
        digest mapper, and so a colliding legacy file is never returned for a
        different identity.
        """
        safe = "".join(c for c in run_id if c.isalnum() or c in "._-") or "run"
        return f"{safe}.json"

    def _path(self, run_id: str) -> str:
        return os.path.join(self._runs_dir, self._digest_name(run_id))

    def _legacy_path(self, run_id: str) -> str:
        return os.path.join(self._runs_dir, self._legacy_name(run_id))

    def _load(self, path: str) -> Optional["ConsolidationRecord"]:
        try:
            with open(path, encoding="utf-8") as fh:
                return ConsolidationRecord.from_dict(json.load(fh))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def archive(
        self,
        result: EvolutionResult,
        *,
        run_id: str,
        artifact_id: str = "artifact",
        strategy: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        trajectory_store: Optional[str] = None,
        problem: Optional[Dict[str, Any]] = None,
    ) -> ConsolidationRecord:
        """Archive one run's product. Idempotent by ``run_id`` (atomic rewrite).

        ``strategy`` is the strategy class name that owns ``result.state``; a
        later run adopts the state only through the same strategy. ``config``
        names the dataset/scorer/model/params so retrieval keeps domains
        apart. ``trajectory_store`` links the run's durable evidence without
        copying it. ``problem`` is an optional descriptor of how to rebuild the
        inner problem(s) this product was produced or judged on (written by
        ``meta.archive_rule`` for an adopted slot rule); it is read back by
        ``meta.problem_from_archive`` and is ``None`` for an ordinary artifact
        record."""
        reward = float(result.final_reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError(
                f"cannot archive a run with non-finite or out-of-range "
                f"final_reward {result.final_reward!r}")
        record = ConsolidationRecord(
            run_id=run_id, artifact_id=artifact_id,
            strategy=strategy or "",
            config=dict(config or {}),
            state=dict(result.state or {}),
            rendered=result.rendered or "",
            final_reward=reward,
            rounds=len(result.history),
            trajectory_store=trajectory_store or result.trajectories_path,
            problem=dict(problem) if problem else None,
        )
        with self._lock:
            path = self._path(run_id)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(record.to_dict(), fh, sort_keys=True,
                              default=str, allow_nan=False)
                os.replace(tmp, path)
                # Drop a pre-#338 sanitized copy of *this* identity so
                # ``iter_archive`` / ``count`` stay unique after rewrite.
                legacy = self._legacy_path(run_id)
                if legacy != path and os.path.exists(legacy):
                    old = self._load(legacy)
                    if old is not None and old.run_id == run_id:
                        try:
                            os.remove(legacy)
                        except OSError:
                            pass
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
        return record

    def find(self, run_id: str) -> Optional[ConsolidationRecord]:
        """Lookup by identity.

        Prefer the digest path. Fall back to a pre-#338 sanitized filename
        only when the stored ``run_id`` matches -- never return a different
        run that shared the old character-stripped name.
        """
        seen = set()
        for path in (self._path(run_id), self._legacy_path(run_id)):
            if path in seen:
                continue
            seen.add(path)
            rec = self._load(path)
            if rec is not None and rec.run_id == run_id:
                return rec
        return None

    def iter_archive(self) -> Iterator[ConsolidationRecord]:
        try:
            names = sorted(os.listdir(self._runs_dir))
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._runs_dir, name),
                          encoding="utf-8") as fh:
                    yield ConsolidationRecord.from_dict(json.load(fh))
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def count(self) -> int:
        try:
            return sum(1 for n in os.listdir(self._runs_dir)
                       if n.endswith(".json"))
        except OSError:
            return 0


@dataclass(frozen=True)
class SeedReport:
    """What ``seed_from_archive`` decided, and why."""

    matched_run: Optional[str]
    archived_reward: float
    seeded_score: float
    empty_score: float
    accepted: bool
    reason: str
    sample_n: int


def seed_from_archive(
    store: ConsolidationStore,
    tasks: Sequence[Task],
    run: Any,
    reward: Any,
    *,
    strategy: Any,
    artifact_id: str = "artifact",
    config: Optional[Dict[str, Any]] = None,
    sample: int = 8,
    min_reward: float = 0.0,
    seed: int = 0,
) -> Tuple[Optional[Dict[str, str]], SeedReport]:
    """Retrieve a seed for a new run, gated by a mini-eval on its own tasks.

    Returns ``(initial_state, report)``: pass ``initial_state`` to
    ``evolve(initial_state=...)`` when it is not ``None``. The gate evaluates
    the archived artifact and the empty seed on a seeded sample of the new
    tasks; the archived state is handed over only if it does not regress
    beyond ``min_reward``. A harmful cross-domain seed is refused and counted,
    never installed.

    ``config`` is the new run's configuration (dataset/scorer/model/params);
    archived runs with the same fingerprint rank first, then by measured
    reward. A candidate is only adopted through a strategy of the same name.
    """
    from .evalcache import MemoryCache
    from .evolution import EvolvingArtifact, _Runtime

    if not tasks or sample < 1:
        return None, SeedReport(None, 0.0, 0.0, 0.0, False,
                                "no tasks / no sample", 0)
    want_hash = config_fingerprint(config)
    strategy_name = type(strategy).__name__
    records = [
        r for r in store.iter_archive()
        if r.artifact_id == artifact_id
        and (not strategy_name or r.strategy == strategy_name)
    ]
    if not records:
        return None, SeedReport(None, 0.0, 0.0, 0.0, False,
                                "no archived run for this artifact/strategy", 0)
    records.sort(key=lambda r: (r.config_hash == want_hash, r.final_reward),
                 reverse=True)
    best = records[0]

    rng = random.Random(seed)
    sample_tasks = [tasks[i] for i in rng.sample(range(len(tasks)),
                                                 min(sample, len(tasks)))]
    runtime = _Runtime(run=run, reward=reward, cache=MemoryCache())
    seeded = EvolvingArtifact(artifact_id, dict(best.state),
                              runtime=runtime, strategy=strategy)
    empty = EvolvingArtifact(artifact_id, dict(strategy.initial()),
                             runtime=runtime, strategy=strategy)
    seeded_score = seeded.score(sample_tasks)
    empty_score = empty.score(sample_tasks)
    accepted = bool(seeded_score + min_reward >= empty_score)
    if accepted:
        return dict(best.state), SeedReport(
            best.run_id, best.final_reward, seeded_score, empty_score,
            True, "seed accepted", len(sample_tasks))
    return None, SeedReport(
        best.run_id, best.final_reward, seeded_score, empty_score,
        False, f"seed rejected: {seeded_score:.3f} below empty {empty_score:.3f} "
               f"by more than min_reward {min_reward:.3f}",
        len(sample_tasks))
