"""A bounded campaign: iterate evolution, carrying the artifact forward.

One `evolve()` run is a session. A campaign runs several sessions on the same
spec, each starting from the artifact the previous session produced, and stops
when a session fails to improve. That is the smallest honest "keep improving"
loop for a spec: no new engine, no hidden state -- the next session's target is
the previous session's output.

The campaign works on a **copy** of the target in a temporary workspace, so the
user's file/directory is never touched. The workspace is returned in the report
(and survives the call) so the best artifact can be inspected or applied
deliberately. A session that does not improve is not applied: the workspace
keeps the best artifact, not the last one.

``policy_slot`` specs are refused: their target is a slot name, not a file, and
"carry the artifact forward" has no file to point the next session at.
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import algorithms as _algorithms
from .evolvespec import EvolveSpec, SpecError, compose, estimate

__all__ = ["CampaignReport", "CampaignSession", "estimate_campaign",
           "run_campaign"]


@dataclass(frozen=True)
class CampaignSession:
    """One evolution session inside a campaign."""

    index: int
    reward: float
    best_reward: float
    stop_reason: str
    outcomes: Dict[str, int]
    calls: int
    #: ``None`` on session 0, which establishes the campaign's baseline; later
    #: sessions say whether they beat the best so far.
    improved: Optional[bool]
    error: Optional[str] = None


@dataclass(frozen=True)
class CampaignReport:
    """Every session, the best artifact and where it lives."""

    kind: str
    sessions: Tuple[CampaignSession, ...] = ()
    workspace: str = ""
    final_target: str = ""
    best_reward: float = 0.0
    baseline_reward: float = 0.0
    final_reward: float = 0.0
    stopped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "workspace": self.workspace,
            "final_target": self.final_target,
            "best_reward": self.best_reward,
            "baseline_reward": self.baseline_reward,
            "final_reward": self.final_reward,
            "stopped_reason": self.stopped_reason,
            "sessions": [
                {"index": s.index, "reward": s.reward,
                 "best_reward": s.best_reward, "stop_reason": s.stop_reason,
                 "outcomes": dict(s.outcomes), "calls": s.calls,
                 "improved": s.improved, "error": s.error}
                for s in self.sessions
            ],
        }


def _check_target(spec: EvolveSpec) -> None:
    if spec.kind == "policy_slot":
        raise ValueError("campaign does not apply to kind 'policy_slot'")
    if spec.kind == "text":
        target = os.path.abspath(os.path.expanduser(str(spec.target)))
        if not os.path.isfile(target):
            raise ValueError(
                "campaign needs a text target that is a file; this looks like "
                "an inline instruction -- put the prompt in a file first")


def _copy_target(source: str, dest: str) -> str:
    source = os.path.abspath(os.path.expanduser(source))
    if os.path.isdir(source):
        shutil.copytree(source, dest)
    else:
        shutil.copy2(source, dest)
    return dest


def _apply(result: Any, target: str, kind: str) -> None:
    if kind == "text":
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(result.rendered)
    else:
        result.write_to(target, backup=False)


def _arm(spec: EvolveSpec, target: str, *, index: int, algorithm: Optional[str],
         rounds: Optional[int]) -> EvolveSpec:
    arm = copy.deepcopy(spec)
    arm.target = target
    if algorithm is not None and algorithm != "default":
        arm.algorithm = algorithm
    evolve = dict(getattr(arm, "evolve", None) or {})
    # Keep the split's seed fixed across sessions: `shuffle` derives the
    # held-out split from it, so changing it would make session rewards
    # incomparable. Sessions differ because the artifact they start from does.
    evolve["seed"] = int(evolve.get("seed", 0))
    if rounds is not None:
        evolve["rounds"] = int(rounds)
    arm.evolve = evolve
    return arm


def estimate_campaign(spec: EvolveSpec, *, sessions: int = 3,
                      rounds: Optional[int] = None,
                      algorithm: Optional[str] = None) -> Dict[str, Any]:
    """Price a campaign without running it (one composed arm per session)."""
    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1")
    _check_target(spec)
    if algorithm not in (None, "default"):
        alg = _algorithms.get(algorithm, where="campaign")
        if spec.kind not in alg.kinds:
            raise ValueError(
                f"algorithm {algorithm!r} applies to kind {list(alg.kinds)}, "
                f"not {spec.kind!r}")
    arm = _arm(spec, str(spec.target), index=0, algorithm=algorithm,
               rounds=rounds)
    comp = compose(arm)
    one = estimate(comp)["agent_calls_upper_bound"]
    return {"kind": spec.kind, "sessions": sessions,
            "calls_per_session": one, "total_calls": one * sessions}


def run_campaign(spec: EvolveSpec, *, sessions: int = 3,
                 rounds: Optional[int] = None,
                 algorithm: Optional[str] = None,
                 patience: int = 1,
                 workdir: Optional[str] = None,
                 keep_ledgers: bool = False) -> CampaignReport:
    """Run up to ``sessions`` sessions, keeping the best artifact.

    A session is ``compose()`` + ``run()`` on a copy of the current target. An
    improving session's artifact is written back to that copy; a non-improving
    one is discarded and the campaign stops after ``patience`` such sessions.
    ``workdir`` is created when omitted and always returned.
    """
    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1")
    _check_target(spec)
    if algorithm not in (None, "default"):
        alg = _algorithms.get(algorithm, where="campaign")
        if spec.kind not in alg.kinds:
            raise ValueError(
                f"algorithm {algorithm!r} applies to kind {list(alg.kinds)}, "
                f"not {spec.kind!r}")

    if workdir is not None and os.path.isdir(os.path.expanduser(workdir)) and \
            os.listdir(os.path.expanduser(workdir)):
        raise ValueError(
            f"campaign workdir {workdir!r} is not empty; choose a fresh path "
            "(campaign does not resume a previous workspace yet)")
    workspace = os.path.abspath(os.path.expanduser(workdir)) if workdir else \
        tempfile.mkdtemp(prefix="agentdescent-campaign-")
    os.makedirs(workspace, exist_ok=True)
    target = _copy_target(spec.target,
                          os.path.join(workspace, os.path.basename(
                              spec.target.rstrip(os.sep)) or "target"))
    best = -math.inf
    baseline = None
    history: List[CampaignSession] = []
    stale = 0
    stopped = "sessions"
    for index in range(sessions):
        arm = _arm(spec, target, index=index, algorithm=algorithm,
                   rounds=rounds)
        repo = tempfile.mkdtemp(prefix="agentdescent-campaign-run-",
                                dir=workspace)
        try:
            comp = compose(arm, repo_path=repo)
            result = comp.run()
            reward = float(result.final_reward)
            if result.error:
                improved = None
                stale += 1
            elif index == 0:
                # Session 0 establishes the baseline; do not claim it beat the
                # user's artifact, which the campaign never measured.
                improved = None
                baseline = reward
                _apply(result, target, spec.kind)
                best = reward
            else:
                improved = reward > best + 1e-9
                if improved:
                    _apply(result, target, spec.kind)
                    best = reward
                    stale = 0
                else:
                    stale += 1
            history.append(CampaignSession(
                index=index, reward=reward, best_reward=best,
                stop_reason=str(result.stop_reason),
                outcomes=dict(result.outcomes()),
                calls=int(getattr(result.usage, "calls", 0)),
                improved=improved, error=result.error))
            if result.error:
                stopped = "error"
                break
        except Exception as e:  # noqa: BLE001 - a session is one attempt
            history.append(CampaignSession(
                index=index, reward=0.0, best_reward=best,
                stop_reason="error", outcomes={}, calls=0, improved=False,
                error=f"{type(e).__name__}: {e}"))
            stopped = "error"
            break
        finally:
            if not keep_ledgers:
                shutil.rmtree(repo, ignore_errors=True)
        if stale >= patience:
            stopped = "patience"
            break

    last = history[-1].reward if history else 0.0
    return CampaignReport(
        kind=spec.kind, sessions=tuple(history), workspace=workspace,
        final_target=target, best_reward=0.0 if best == -math.inf else best,
        baseline_reward=0.0 if baseline is None else baseline,
        final_reward=last, stopped_reason=stopped)
