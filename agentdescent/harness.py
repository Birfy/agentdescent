"""Harness composition: combine algorithm ports, learn which wins where.

Every algorithm port in this repository is one ``evolve()`` call with a
different ``Policies`` / ``aggregator_factory`` -- RRSI, solver-challenger
self-play, population islands, metasearch -- and nothing ever chose between
them, or remembered which one won on which domain. This module is the
composition layer: a harness is a named ``(tasks, repo) -> EvolutionResult``
callable, the archive is the leaderboard (one record per ``(config, harness)``
with the measured reward), and the selector routes a new domain to the harness
the evidence says wins there -- and, when the evidence is empty, falls back to
round-robin rather than guessing.

The archive is reused unchanged: a harness outcome is a
:class:`~agentdescent.consolidation.ConsolidationRecord` with
``artifact_id="harness:<name>"``, the domain's ``config``, and the harness's
measured ``final_reward``, written atomically and idempotently by
``run_id``. ``config_fingerprint`` keeps domains apart exactly as
``seed_from_archive`` does, so a harness that wins on arithmetic is never
mistaken for one that wins on code.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore, config_fingerprint
from .evolution import EvolutionResult, Task

__all__ = [
    "Harness",
    "harness",
    "record_harness",
    "harness_score",
    "select_harness",
    "HarnessRun",
    "HarnessReport",
    "compose_run",
    "harness_leaderboard",
]

#: The archive ``artifact_id`` prefix a harness outcome is written under. The
#: harness name follows the colon, so retrieval filters on ``"harness:<name>"``
#: the same way ``seed_from_archive`` filters on an artifact id.
HARNESS_PREFIX = "harness:"


@dataclass(frozen=True)
class Harness:
    """One algorithm port, named and runnable.

    ``run(tasks, repo_path) -> EvolutionResult``: the caller binds everything
    the port needs (``reward``, ``policies``, ``aggregator_factory``, the
    challenger for self-play ...) into the closure, so a harness is a black box
    that turns a task set into a measured result. ``repo_path`` is supplied per
    invocation so every run gets a fresh ledger.
    """

    name: str
    run: Callable[[Sequence[Task], str], EvolutionResult]
    description: str = ""


def harness(name: str, run: Callable[[Sequence[Task], str], EvolutionResult],
            description: str = "") -> Harness:
    """The least a harness can be: a name and a ``(tasks, repo) -> result`` callable."""
    if not name or any(c.isspace() for c in name):
        raise ValueError(f"harness name must be a non-empty, whitespace-free str")
    return Harness(name=name, run=run, description=description)


def record_harness(
    store: ConsolidationStore,
    *,
    harness_name: str,
    config: Optional[Dict[str, Any]],
    result: EvolutionResult,
    run_id: Optional[str] = None,
) -> str:
    """Record one harness outcome for a domain. Idempotent by ``run_id``.

    Writes ``result`` to the archive under ``artifact_id="harness:<name>"`` with
    the domain's ``config``, so ``harness_score`` / ``select_harness`` can rank
    harnesses per domain fingerprint by measured reward. Returns the run id.
    """
    rid = run_id or f"harness-{harness_name}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    store.archive(
        result, run_id=rid, artifact_id=f"{HARNESS_PREFIX}{harness_name}",
        strategy="harness", config=config)
    return rid


def harness_score(store: ConsolidationStore, harness_name: str,
                  config: Optional[Dict[str, Any]]) -> Optional[float]:
    """The best measured reward this harness has recorded for the domain.

    ``None`` when there is no evidence -- never zero, so a harness with no runs
    on this domain cannot beat one that has actually been measured here.
    """
    want = config_fingerprint(config)
    best: Optional[float] = None
    for rec in store.iter_archive():
        if rec.artifact_id != f"{HARNESS_PREFIX}{harness_name}":
            continue
        if rec.config_hash != want:
            continue
        if best is None or rec.final_reward > best:
            best = rec.final_reward
    return best


def harness_leaderboard(
    store: ConsolidationStore,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """``harness_name -> {rewards, best}`` over the archive's recorded outcomes.

    ``config=None`` aggregates across every domain; pass a config to restrict
    to its fingerprint. ``rewards`` is every measured ``final_reward`` (newest
    last), ``best`` the highest. A harness with no records is absent.
    """
    want = config_fingerprint(config)
    out: Dict[str, Dict[str, Any]] = {}
    for rec in store.iter_archive():
        if not rec.artifact_id.startswith(HARNESS_PREFIX):
            continue
        if config is not None and rec.config_hash != want:
            continue
        name = rec.artifact_id[len(HARNESS_PREFIX):]
        entry = out.setdefault(name, {"rewards": [], "best": float("-inf")})
        entry["rewards"].append(float(rec.final_reward))
        if float(rec.final_reward) > entry["best"]:
            entry["best"] = float(rec.final_reward)
    for entry in out.values():
        if entry["best"] == float("-inf"):
            entry["best"] = None
    return out


def select_harness(
    store: ConsolidationStore,
    harnesses: Sequence[Harness],
    config: Optional[Dict[str, Any]],
    *,
    default: Optional[str] = None,
) -> Tuple[Harness, str]:
    """Route a domain to the harness the archive's evidence says wins there.

    Among harnesses with any recorded reward for this ``config`` fingerprint,
    the highest measured reward wins; a tie goes to the harness that appears
    first in ``harnesses``. When no harness has evidence, ``default`` (or the
    first harness) is returned -- round-robin rather than a guess. Returns
    ``(chosen, reason)``; the reason says whether the choice is evidence-based
    or a fallback, so a caller can log it.
    """
    want = config_fingerprint(config)
    best: Tuple[Optional[float], int, Optional[Harness]] = (None, 0, None)
    for index, h in enumerate(harnesses):
        score = harness_score(store, h.name, config)
        if score is None:
            continue
        if best[0] is None or score > best[0]:
            best = (score, index, h)
    if best[2] is not None:
        return best[2], (f"evidence-based: {best[2].name} best on "
                         f"config {want} at {best[0]:.3f}")
    if default is not None:
        chosen = next((h for h in harnesses if h.name == default), None)
        if chosen is not None:
            return chosen, f"no evidence; caller default {default}"
    return harnesses[0], "no evidence; round-robin"


@dataclass(frozen=True)
class HarnessRun:
    """One harness's measured outcome in a :func:`compose_run`."""

    harness: str
    reward: float
    time_to_quality: Optional[float]
    run_id: Optional[str] = None


@dataclass(frozen=True)
class HarnessReport:
    """What a :func:`compose_run` did, and why.

    ``selected`` is the harness the selector chose (or the first when running
    all); ``winner`` the harness with the highest measured reward this round;
    ``runs`` one row per harness invoked. ``reason`` says how the selection was
    made (evidence-based / round-robin / caller-specified).
    """

    config_hash: str
    selected: str
    winner: str
    runs: Tuple[HarnessRun, ...]
    reason: str


def compose_run(
    store: ConsolidationStore,
    tasks: Sequence[Task],
    harnesses: Sequence[Harness],
    config: Optional[Dict[str, Any]],
    *,
    mode: str = "all",
    default: Optional[str] = None,
    target: float = 0.999,
    repo_dir: Optional[str] = None,
    **run_kwargs: Any,
) -> Tuple[HarnessReport, EvolutionResult]:
    """Run the harness(es) for a domain, record the outcomes, pick a winner.

    ``mode="best"`` runs only the harness :func:`select_harness` chooses for
    ``config``; ``mode="all"`` runs every harness (and thus builds evidence for
    all of them -- the leaderboard's training mode). Each harness runs on a
    fresh ledger repo under ``repo_dir``, and every outcome is recorded via
    :func:`record_harness`, so the next ``select_harness`` on this domain sees
    it. Returns ``(report, winner_result)`` where ``winner_result`` is the run
    with the highest ``final_reward``.
    """
    if not tasks:
        raise ValueError("compose_run() got no tasks")
    if not harnesses:
        raise ValueError("compose_run() got no harnesses")
    if mode not in ("all", "best"):
        raise ValueError(f"mode must be 'all' or 'best', got {mode!r}")
    if repo_dir is not None:
        os.makedirs(repo_dir, exist_ok=True)

    if mode == "best":
        chosen, reason = select_harness(store, harnesses, config, default=default)
        to_run = [chosen]
    else:
        to_run = list(harnesses)
        reason = "running all harnesses (leaderboard training)"

    runs: List[HarnessRun] = []
    winner_result: Optional[EvolutionResult] = None
    winner_name = ""
    winner_reward = float("-inf")
    for h in to_run:
        repo = tempfile.mkdtemp(prefix=f"agentdescent-harness-{h.name}-",
                                dir=repo_dir)
        result = h.run(list(tasks), repo)
        rid = record_harness(store, harness_name=h.name, config=config,
                             result=result)
        ttq = result.time_to_quality(target) if hasattr(result, "time_to_quality") else None
        runs.append(HarnessRun(harness=h.name, reward=float(result.final_reward),
                               time_to_quality=ttq, run_id=rid))
        if float(result.final_reward) > winner_reward:
            winner_reward = float(result.final_reward)
            winner_result = result
            winner_name = h.name

    report = HarnessReport(
        config_hash=config_fingerprint(config), selected=to_run[0].name,
        winner=winner_name, runs=tuple(runs), reason=reason)
    return report, winner_result  # type: ignore[return-value]
