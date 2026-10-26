"""A bounded multi-session controller for an RSI loop.

`run_guarded` runs one session. A product needs the loop around it: session
budget, call budget, kill switch, stagnation stop and a decision record per
session. This controller is deliberately generic -- ``run_session(index, seed)``
is supplied by the caller, so it can wrap a guarded run, a campaign step, a
population generation or a remote job without duplicating the engine.

    report = run_controller(run_session, ControllerConfig(max_sessions=5,
                          max_calls=500, kill_switch=switch, decision_log=log))
    report.best_reward, report.stopped_reason

It does not decide *what* a session is or how to apply its artifact. It decides
when another session is no longer justified and records why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

__all__ = ["ControllerConfig", "ControllerReport", "SessionOutcome",
           "SessionRecord", "run_controller"]


@dataclass(frozen=True)
class SessionOutcome:
    """What one session produced."""

    reward: float
    calls: int = 0
    error: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("session reward must be finite")
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) \
                or self.calls < 0:
            raise ValueError("session calls must be a non-negative int")


@dataclass(frozen=True)
class SessionRecord:
    """One session's outcome inside the controller report."""

    index: int
    seed: int
    reward: float
    calls: int
    improved: Optional[bool]
    error: Optional[str] = None


@dataclass(frozen=True)
class ControllerReport:
    """The sessions run, the best reward and why the loop stopped."""

    sessions: Tuple[SessionRecord, ...] = ()
    best_reward: Optional[float] = None
    total_calls: int = 0
    stopped_reason: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessions": [
                {"index": s.index, "seed": s.seed, "reward": s.reward,
                 "calls": s.calls, "improved": s.improved, "error": s.error}
                for s in self.sessions],
            "best_reward": self.best_reward,
            "total_calls": self.total_calls,
            "stopped_reason": self.stopped_reason,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ControllerConfig:
    """Bounds and hooks for the multi-session loop."""

    max_sessions: int = 3
    max_calls: Optional[int] = None
    patience: int = 1
    base_seed: int = 0
    kill_switch: Optional[Any] = None
    decision_log: Optional[Any] = None

    def __post_init__(self) -> None:
        for name, value in (("max_sessions", self.max_sessions),
                            ("patience", self.patience)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1")
        if self.max_calls is not None and (
                isinstance(self.max_calls, bool)
                or not isinstance(self.max_calls, int) or self.max_calls < 1):
            raise ValueError("max_calls must be an int >= 1 or None")


def run_controller(run_session: Callable[[int, int], SessionOutcome], config: ControllerConfig
                   ) -> ControllerReport:
    """Run sessions until budget, a kill switch, an error or stagnation.

    An errored session is recorded and stops the loop; it is not counted as an
    improvement. The first session establishes the baseline. A decision record
    (``kind="session"``) is written per session when ``decision_log`` is given.
    """
    switch = config.kill_switch
    if switch is not None and switch.engaged():
        return ControllerReport(stopped_reason="kill-switch",
                                reason=switch.reason() or "operator stop")
    history = []
    best = float("-inf")
    total_calls = 0
    stagnant = 0
    stopped = "sessions"
    reason = "reached max_sessions"
    for index in range(config.max_sessions):
        if switch is not None and switch.engaged():
            stopped, reason = "kill-switch", switch.reason() or "operator stop"
            break
        if config.max_calls is not None and total_calls >= config.max_calls:
            stopped, reason = "budget", (
                f"{total_calls} calls reached max_calls {config.max_calls}")
            break
        seed = config.base_seed + index
        try:
            outcome = run_session(index, seed)
            if not isinstance(outcome, SessionOutcome):
                raise TypeError(
                    f"run_session returned {type(outcome).__name__}, expected "
                    "SessionOutcome")
        except Exception as e:  # noqa: BLE001 - one session is one attempt
            outcome = SessionOutcome(0.0, 0, f"{type(e).__name__}: {e}")
        total_calls += int(outcome.calls)
        if outcome.error is not None:
            improved = False
        elif index == 0:
            improved = None
            best = outcome.reward
        else:
            improved = best is not None and outcome.reward > best + 1e-9
            if improved:
                best = outcome.reward
                stagnant = 0
            else:
                stagnant += 1
        history.append(SessionRecord(
            index=index, seed=seed, reward=float(outcome.reward),
            calls=int(outcome.calls), improved=improved, error=outcome.error))
        log = config.decision_log
        if log is not None:
            log.record(
                kind="session", artifact_id="controller", accepted=outcome.error is None,
                category=("error" if outcome.error else "session"),
                detail=str(outcome.error or ""),
                evidence={"reward": float(outcome.reward),
                          "improved": improved, **dict(outcome.evidence)},
                cost={"calls": int(outcome.calls)})
        if outcome.error is not None:
            stopped, reason = "error", outcome.error or "session failed"
            break
        if config.max_calls is not None and total_calls >= config.max_calls:
            stopped, reason = "budget", (
                f"{total_calls} calls reached max_calls {config.max_calls}")
            break
        if stagnant >= config.patience:
            stopped, reason = "patience", (
                f"{stagnant} session(s) without a new best")
            break
    return ControllerReport(
        sessions=tuple(history),
        best_reward=(None if best == float("-inf") else best),
        total_calls=total_calls, stopped_reason=stopped, reason=reason)
