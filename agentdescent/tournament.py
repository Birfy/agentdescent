"""Run the same spec under several named algorithms and pick a winner.

A registry of algorithms is useful only if a user can answer "which one works
for *my* skill?" without reading each paper. `run_tournament` runs the same
`EvolveSpec` once per requested algorithm and seed, under the same evolve
budget, and reports the measured result. It is an ordinary composition --
`EvolveSpec` -> `compose()` -> `evolve()` per arm -- not a second engine.

The default arm is the spec's own recipe; name it ``"default"``. A named arm
sets `spec.algorithm` for that run only (a deep copy), so the file on disk is
untouched. The winner is the highest final reward among arms that did not error;
ties go to fewer agent calls.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import algorithms as _algorithms
from .evolvespec import EvolveSpec, estimate, compose

__all__ = ["AlgorithmArm", "AlgorithmScore", "AlgorithmTournamentReport",
           "estimate_tournament", "run_tournament"]

DEFAULT_ARM = "default"


@dataclass(frozen=True)
class AlgorithmArm:
    """One algorithm/seed arm of a tournament."""

    algorithm: str
    seed: int
    final_reward: float
    stop_reason: str
    outcomes: Dict[str, int]
    calls: int
    error: Optional[str] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmScore:
    """One algorithm's aggregate over its seeds."""

    algorithm: str
    mean_reward: float
    min_reward: float
    max_reward: float
    mean_calls: float
    seeds: int


@dataclass(frozen=True)
class AlgorithmTournamentReport:
    """Every arm, the per-algorithm aggregate, the winner and any tie."""

    kind: str
    arms: Tuple[AlgorithmArm, ...] = ()
    algorithms: Tuple[AlgorithmScore, ...] = ()
    winner: Optional[str] = None
    tied: Tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "winner": self.winner,
            "tied": list(self.tied),
            "reason": self.reason,
            "algorithms": [
                {"algorithm": a.algorithm, "mean_reward": a.mean_reward,
                 "min_reward": a.min_reward, "max_reward": a.max_reward,
                 "mean_calls": a.mean_calls, "seeds": a.seeds}
                for a in self.algorithms
            ],
            "arms": [
                {"algorithm": e.algorithm, "seed": e.seed,
                 "final_reward": e.final_reward, "stop_reason": e.stop_reason,
                 "outcomes": dict(e.outcomes), "calls": e.calls,
                 "error": e.error, "notes": list(e.notes)}
                for e in self.arms
            ],
        }


def _arm_spec(spec: EvolveSpec, name: str, seed: int, *, rounds: Optional[int],
              max_seconds: Optional[float]) -> EvolveSpec:
    arm = copy.deepcopy(spec)
    # `default` means the spec's own recipe -- including its own `algorithm`.
    # A named arm replaces the method (and therefore its params) for that run.
    arm.algorithm = spec.algorithm if name == DEFAULT_ARM else name
    evolve = dict(getattr(arm, "evolve", None) or {})
    evolve["seed"] = int(seed)
    if rounds is not None:
        evolve["rounds"] = int(rounds)
    if max_seconds is not None:
        evolve["max_seconds"] = float(max_seconds)
    arm.evolve = evolve
    return arm


def run_tournament(spec: EvolveSpec, algorithm_names: Sequence[str], *,
                   seeds: Sequence[int] = (0,),
                   rounds: Optional[int] = None,
                   max_seconds: Optional[float] = None,
                   repo_root: Optional[str] = None,
                   keep_repos: bool = False) -> TournamentReport:
    """Run every ``(algorithm, seed)`` arm and return the measured ranking.

    ``algorithm_names`` may include ``"default"`` for the spec's own recipe.
    Unknown names, names for an unsupported kind, and an empty list raise before
    any run starts. ``repo_root`` keeps the temporary ledgers in one place;
    ``keep_repos`` leaves them behind for diagnosis.
    """
    names = list(dict.fromkeys(str(name) for name in algorithm_names))
    if not names:
        raise ValueError("run_tournament() needs at least one algorithm (or 'default')")
    for name in names:
        if name == DEFAULT_ARM:
            continue
        alg = _algorithms.get(name, where="tournament")
        if spec.kind not in alg.kinds:
            raise ValueError(
                f"algorithm {name!r} applies to kind {list(alg.kinds)}, "
                f"not {spec.kind!r}")
    if not seeds:
        raise ValueError("run_tournament() needs at least one seed")
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")
    if repo_root is not None:
        os.makedirs(repo_root, exist_ok=True)

    entries: List[AlgorithmArm] = []
    for name in names:
        for seed in seeds:
            arm = _arm_spec(spec, name, seed, rounds=rounds,
                            max_seconds=max_seconds)
            repo = tempfile.mkdtemp(prefix=f"agentdescent-tournament-{name}-",
                                    dir=repo_root)
            try:
                comp = compose(arm, repo_path=repo)
                result = comp.run()
                entry = AlgorithmArm(
                    algorithm=name, seed=int(seed),
                    final_reward=float(result.final_reward),
                    stop_reason=str(result.stop_reason),
                    outcomes=dict(result.outcomes()),
                    calls=int(getattr(result.usage, "calls", 0)),
                    error=result.error, notes=tuple(comp.notes))
            except Exception as e:  # noqa: BLE001 - one arm must not hide the others
                entry = AlgorithmArm(
                    algorithm=name, seed=int(seed), final_reward=0.0,
                    stop_reason="error", outcomes={}, calls=0,
                    error=f"{type(e).__name__}: {e}")
            finally:
                if not keep_repos:
                    shutil.rmtree(repo, ignore_errors=True)
            entries.append(entry)

    grouped: Dict[str, List[AlgorithmArm]] = {}
    for entry in entries:
        if entry.error is None:
            grouped.setdefault(entry.algorithm, []).append(entry)
    scores = tuple(sorted(
        (AlgorithmScore(
            algorithm=name,
            mean_reward=sum(e.final_reward for e in arms) / len(arms),
            min_reward=min(e.final_reward for e in arms),
            max_reward=max(e.final_reward for e in arms),
            mean_calls=sum(e.calls for e in arms) / len(arms),
            seeds=len(arms)) for name, arms in grouped.items()),
        key=lambda a: (-a.mean_reward, a.mean_calls, a.algorithm)))
    if not scores:
        reason = "every arm failed; the first error is: " + (
            entries[0].error if entries else "no entries")
        return AlgorithmTournamentReport(
            kind=spec.kind, arms=tuple(entries), reason=reason)
    best = scores[0]
    tied = tuple(a.algorithm for a in scores
                 if abs(a.mean_reward - best.mean_reward) <= 1e-9
                 and abs(a.mean_calls - best.mean_calls) <= 1e-9)
    if len(tied) > 1:
        winner = None
        reason = (f"no measured difference between {', '.join(tied)} "
                  f"(mean reward {best.mean_reward:.3f}, "
                  f"{best.mean_calls:.0f} calls)")
    else:
        winner = best.algorithm
        reason = (f"mean reward {best.mean_reward:.3f} over "
                  f"{best.seeds} seed(s), {best.mean_calls:.0f} calls; "
                  f"range {best.min_reward:.3f}-{best.max_reward:.3f}")
    return AlgorithmTournamentReport(
        kind=spec.kind, arms=tuple(entries), algorithms=scores,
        winner=winner, tied=tied, reason=reason)


def estimate_tournament(spec: EvolveSpec, algorithm_names: Sequence[str], *,
                        seeds: Sequence[int] = (0,),
                        rounds: Optional[int] = None,
                        max_seconds: Optional[float] = None) -> Dict[str, Any]:
    """The per-arm plan estimate without running anything.

    Each arm is composed (loads data/tree, no model call) and priced with the
    same `estimate()` `plan` uses. The total is the sum over arms.
    """
    names = list(dict.fromkeys(str(name) for name in algorithm_names))
    if not names:
        raise ValueError("estimate_tournament() needs at least one algorithm")
    if not seeds:
        raise ValueError("estimate_tournament() needs at least one seed")
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")
    rows = []
    for name in names:
        if name != DEFAULT_ARM:
            alg = _algorithms.get(name, where="tournament")
            if spec.kind not in alg.kinds:
                raise ValueError(
                    f"algorithm {name!r} applies to kind {list(alg.kinds)}, "
                    f"not {spec.kind!r}")
        arm = _arm_spec(spec, name, int(seeds[0]), rounds=rounds,
                        max_seconds=max_seconds)
        comp = compose(arm)
        est = estimate(comp)
        rows.append({"algorithm": name, "tasks": len(comp.tasks),
                     "calls_per_arm": est["agent_calls_upper_bound"],
                     "seeds": len(seeds),
                     "total_calls": est["agent_calls_upper_bound"] * len(seeds)})
    return {"kind": spec.kind, "arms": rows,
            "total_calls": sum(r["total_calls"] for r in rows)}
