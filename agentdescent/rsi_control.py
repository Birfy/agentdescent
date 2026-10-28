"""The control plane around a self-improvement session.

The framework already has the pieces: :func:`run_controller` bounds a
multi-session loop, :func:`run_guarded` runs one guarded session, and
``portfolio`` / ``evidence`` / ``admit`` read the archive. What it did not have
is the policy seam that turns one session's outcome into the next action:
**accept it, reject it, defer it, roll it back, keep going, or stop**.

This module is deliberately small and duck-typed. A *gate* can be a
``ConformalGate``, an ``EProcess``-backed function, a ``RiskPolicy``, a plain
callable, or a test double; this class only normalises the answer into a
`ControlAction`. An unrecognised non-``None`` gate answer **fails closed**:
a safety seam must not turn a typo into an acceptance.

    controller = RSIController(ControlConfig(target=0.9, max_sessions=6),
                               gate=my_gate, budget_policy=my_budget)
    report = controller.run(run_session)
    report.best_reward, report.stopped_reason, report.decisions

A session result is one measured attempt: reward, cost, risk, and optional
evidence. The first accepted session establishes the baseline. A session that
does not beat the best by ``min_gain`` is a rejection, not an improvement.
``stop`` is separate from the per-session verdict: an accepted candidate that
also reaches the target has ``action="accept", stop=True`` so callers can still
commit it. ``DEFER`` does not count as stagnation; a human queue is not the same
as a failed search.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ControlAction",
    "ControlConfig",
    "ControlDecision",
    "ControlReport",
    "SessionResult",
    "RSIController",
]


class ControlAction:
    """The small vocabulary a controller emits."""

    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    ROLLBACK = "rollback"
    CONTINUE = "continue"
    STOP = "stop"

    ALL = (ACCEPT, REJECT, DEFER, ROLLBACK, CONTINUE, STOP)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class ControlConfig:
    """Bounds and thresholds for one control session.

    ``target`` is the reward at which the loop stops as successful.
    ``max_cost`` is a budget on the session result's own cost unit (calls,
    tokens or dollars; the controller never converts). ``patience`` is the
    number of consecutive rejections that stop the loop; deferrals do not count.
    ``risk_limit`` is the highest acceptable session risk in ``[0, 1]``; the
    default ``1.0`` is off, so a caller that does not supply risk is not
    surprised by every session being rejected.
    """

    target: Optional[float] = None
    max_sessions: int = 6
    max_cost: Optional[float] = None
    min_gain: float = 0.0
    patience: int = 2
    risk_limit: float = 1.0

    def __post_init__(self) -> None:
        if self.target is not None:
            _finite(self.target, "target")
        if isinstance(self.max_sessions, bool) or not isinstance(self.max_sessions, int) \
                or self.max_sessions < 1:
            raise ValueError("max_sessions must be an int >= 1")
        if self.max_cost is not None:
            cost = _finite(self.max_cost, "max_cost")
            if cost <= 0:
                raise ValueError("max_cost must be > 0 or None")
        if _finite(self.min_gain, "min_gain") < 0:
            raise ValueError("min_gain must be >= 0")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) \
                or self.patience < 1:
            raise ValueError("patience must be an int >= 1")
        risk = _finite(self.risk_limit, "risk_limit")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk_limit must be in [0, 1]")


@dataclass(frozen=True)
class SessionResult:
    """One measured session attempt."""

    reward: float
    cost: float = 0.0
    risk: float = 0.0
    error: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.reward, "reward")
        cost = _finite(self.cost, "cost")
        if cost < 0:
            raise ValueError("cost must be >= 0")
        risk = _finite(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")


@dataclass(frozen=True)
class ControlDecision:
    """What the controller decided after one session.

    ``action`` is the session verdict; ``stop`` says whether the loop should
    stop after this decision. A target reached on an accepted session therefore
    keeps ``action="accept"`` and sets ``stop=True``; the decision log can still
    commit the candidate.
    """

    index: int
    action: str
    reason: str
    reward: Optional[float]
    gain: Optional[float]
    cost: float
    budget_scale: float = 1.0
    stop: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ControlAction.ALL:
            raise ValueError(f"unknown control action {self.action!r}")

    @property
    def accepted(self) -> bool:
        return self.action == ControlAction.ACCEPT


@dataclass(frozen=True)
class ControlReport:
    """The decisions, the best reward and why the loop stopped."""

    decisions: Tuple[ControlDecision, ...] = ()
    best_reward: Optional[float] = None
    total_cost: float = 0.0
    accepted: int = 0
    stopped_reason: str = ""
    reason: str = ""

    @property
    def stopped(self) -> bool:
        return bool(self.stopped_reason)


Gate = Callable[[SessionResult, Optional[float], float], Any]
BudgetPolicy = Callable[[Mapping[str, Any]], float]

_STOP_ACTIONS = (ControlAction.ROLLBACK, ControlAction.STOP,
                 ControlAction.DEFER, ControlAction.REJECT)


def _decision_like(value: Any) -> Optional[Tuple[str, str]]:
    """Unwrap an ``AcceptDecision``-like object into ``(action, reason)``."""
    if value is None:
        return None
    if isinstance(value, str):
        action = value.strip().lower()
        if action in ControlAction.ALL:
            return action, ""
        raise ValueError(f"unrecognised gate action {value!r}")
    if isinstance(value, bool):
        return (ControlAction.ACCEPT if value else ControlAction.REJECT), ""
    for attr in ("accept", "accepted"):
        accepted = getattr(value, attr, None)
        if isinstance(accepted, bool):
            return ((ControlAction.ACCEPT if accepted else ControlAction.REJECT),
                    str(getattr(value, "detail", "") or getattr(value, "reason", "")))
    return None


def _normalise_gate(value: Any) -> Tuple[Optional[str], str]:
    """Turn a gate result into ``(action, reason)``; fail closed on unknowns.

    Supported: ``None``, a control-action string, a bool, an object with
    ``action``/``decision``, or an object with ``accept``/``accepted``. A nested
    ``AcceptDecision`` (``GuardDecision.decision``) is unwrapped. Any other
    non-``None`` value raises; silently treating it as "no opinion" would make a
    safety gate fail open.
    """
    if value is None:
        return None, ""
    direct = _decision_like(value)
    if direct is not None:
        return direct
    for attr in ("action", "decision"):
        nested = getattr(value, attr, None)
        if nested is None:
            continue
        result = _decision_like(nested)
        if result is not None:
            return result
    raise ValueError(
        f"unrecognised gate result {type(value).__name__}; expected a control "
        "action, bool, AcceptDecision-like object or None")


class RSIController:
    """A bounded, gate-aware policy seam around a sequence of sessions."""

    def __init__(self, config: ControlConfig, *,
                 gate: Optional[Gate] = None,
                 risk_gate: Optional[Gate] = None,
                 budget_policy: Optional[BudgetPolicy] = None,
                 kill_switch: Optional[Any] = None,
                 decision_log: Optional[Any] = None) -> None:
        self.config = config
        self.gate = gate
        self.risk_gate = risk_gate
        self.budget_policy = budget_policy
        self.kill_switch = kill_switch
        self.decision_log = decision_log
        self._decisions: List[ControlDecision] = []
        self._accepted_rewards: List[float] = []
        self._best: Optional[float] = None
        self._total_cost = 0.0
        self._accepted = 0
        self._stagnant = 0
        self._stopped: Optional[Tuple[str, str]] = None

    # -- state ---------------------------------------------------------------

    @property
    def decisions(self) -> Tuple[ControlDecision, ...]:
        return tuple(self._decisions)

    @property
    def best_reward(self) -> Optional[float]:
        return self._best

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def stopped(self) -> bool:
        return self._stopped is not None

    def reset(self) -> "RSIController":
        self._decisions.clear()
        self._accepted_rewards.clear()
        self._best = None
        self._total_cost = 0.0
        self._accepted = 0
        self._stagnant = 0
        self._stopped = None
        return self

    # -- one observation -----------------------------------------------------

    def _state(self, result: SessionResult, gain: Optional[float]) -> Dict[str, Any]:
        return {
            "index": len(self._decisions),
            "best": self._best,
            "gain": gain,
            "cost": result.cost,
            "total_cost": self._total_cost,
            "accepted": self._accepted,
            "risk": result.risk,
        }

    def _budget_scale(self, state: Mapping[str, Any]) -> float:
        if self.budget_policy is None:
            return 1.0
        scale = float(self.budget_policy(state))
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("budget_policy must return a finite scale >= 0")
        return scale

    def _rollback_best(self) -> None:
        if self._accepted_rewards:
            self._accepted_rewards.pop()
        self._accepted = len(self._accepted_rewards)
        self._best = self._accepted_rewards[-1] if self._accepted_rewards else None
        self._stagnant = 0

    def observe(self, result: SessionResult) -> ControlDecision:
        """Consume one session result and return the next decision."""
        if not isinstance(result, SessionResult):
            raise TypeError(
                f"observe() expects SessionResult, got {type(result).__name__}")
        index = len(self._decisions)
        if self._stopped is not None:
            decision = ControlDecision(
                index=index, action=ControlAction.STOP,
                reason="already stopped", reward=None, gain=None,
                cost=0.0, stop=True)
            self._decisions.append(decision)
            return decision

        gain: Optional[float] = (None if self._best is None
                                 else float(result.reward) - self._best)
        state = self._state(result, gain)
        budget_scale = self._budget_scale(state)

        action: Optional[str] = None
        reason = ""
        stop_category = ""
        stop_reason = ""

        if self.kill_switch is not None and self.kill_switch.engaged():
            action, reason = ControlAction.STOP, (
                self.kill_switch.reason() or "operator stop")
            stop_category, stop_reason = "kill_switch", reason
        if action is None and result.error is not None:
            action, reason = ControlAction.STOP, f"session error: {result.error}"
            stop_category, stop_reason = "error", reason
        if action is None and result.risk > self.config.risk_limit:
            action = ControlAction.REJECT
            reason = f"risk {result.risk:.3f} > {self.config.risk_limit:.3f}"
        if action is None and self.risk_gate is not None:
            gate_action, gate_reason = _normalise_gate(
                self.risk_gate(result, gain, result.risk))
            if gate_action in _STOP_ACTIONS:
                action, reason = gate_action, gate_reason or "risk gate"
        if action is None and self.gate is not None:
            gate_action, gate_reason = _normalise_gate(
                self.gate(result, gain, result.risk))
            if gate_action in _STOP_ACTIONS:
                action, reason = gate_action, gate_reason or "acceptance gate"

        if action is None:
            if self._best is None:
                action, reason = ControlAction.ACCEPT, "baseline accepted"
            elif gain is not None and gain > self.config.min_gain:
                action, reason = ControlAction.ACCEPT, (
                    f"gain {gain:+.6g} > min_gain {self.config.min_gain:.6g}")
            else:
                action, reason = ControlAction.REJECT, (
                    f"gain {gain:+.6g} <= min_gain {self.config.min_gain:.6g}")

        if action == ControlAction.ACCEPT:
            self._accepted_rewards.append(float(result.reward))
            self._best = float(result.reward)
            self._accepted += 1
            self._stagnant = 0
        elif action == ControlAction.REJECT:
            self._stagnant += 1
        elif action == ControlAction.ROLLBACK:
            self._rollback_best()

        self._total_cost += float(result.cost)

        # A per-session verdict can also end the loop. Keep the verdict.
        if action != ControlAction.STOP:
            if action == ControlAction.ACCEPT and self.config.target is not None \
                    and result.reward >= self.config.target:
                stop_category, stop_reason = "target", (
                    f"target {self.config.target:g} reached at {result.reward:g}")
            elif self.config.max_cost is not None and self._total_cost >= self.config.max_cost:
                stop_category, stop_reason = "budget", (
                    f"cost {self._total_cost:g} reached max_cost {self.config.max_cost:g}")
            elif self._stagnant >= self.config.patience:
                stop_category, stop_reason = "patience", (
                    f"{self._stagnant} rejection(s) without a new best")
            elif index + 1 >= self.config.max_sessions:
                stop_category, stop_reason = "max_sessions", "reached max_sessions"

        stop = action == ControlAction.STOP or bool(stop_category)
        if stop and action != ControlAction.STOP:
            reason = f"{reason}; stopping: {stop_reason}"
        elif stop and not stop_reason:
            stop_reason = reason
            stop_category = stop_category or "stop"
        decision = ControlDecision(
            index=index, action=action, reason=reason,
            reward=float(result.reward), gain=gain, cost=float(result.cost),
            budget_scale=budget_scale, stop=stop,
            evidence=dict(result.evidence))
        self._decisions.append(decision)
        if stop:
            self._stopped = (stop_category or "stop", stop_reason or reason)
        if self.decision_log is not None:
            self._log(decision)
        return decision

    def _log(self, decision: ControlDecision) -> None:
        try:
            self.decision_log.record(
                kind="control", artifact_id="rsi-controller",
                accepted=decision.accepted, category=decision.action,
                detail=decision.reason,
                evidence={"reward": decision.reward, "gain": decision.gain,
                          "stop": decision.stop,
                          "budget_scale": decision.budget_scale,
                          **dict(decision.evidence)},
                cost={"session_cost": decision.cost})
        except Exception:  # noqa: BLE001 - a broken log must not kill the loop
            pass

    # -- the loop ------------------------------------------------------------

    def run(self, run_session: Callable[..., SessionResult], *,
            base_seed: int = 0) -> ControlReport:
        """Run ``run_session(index, seed[, budget_scale])`` until it stops.

        If the callable accepts a third positional argument, the previous
        decision's ``budget_scale`` is passed; otherwise the two-argument form is
        used, so existing callers keep working.
        """
        if self._stopped is not None:
            return self.report()
        try:
            params = list(inspect.signature(run_session).parameters.values())
            wants_budget = len(params) >= 3
        except (TypeError, ValueError):
            wants_budget = False
        budget_scale = 1.0
        for index in range(len(self._decisions), self.config.max_sessions):
            if wants_budget:
                result = run_session(index, base_seed + index, budget_scale)
            else:
                result = run_session(index, base_seed + index)
            if not isinstance(result, SessionResult):
                result = SessionResult(0.0, error=(
                    f"run_session returned {type(result).__name__}, expected SessionResult"))
            decision = self.observe(result)
            budget_scale = decision.budget_scale
            if decision.stop:
                break
        return self.report()

    def report(self) -> ControlReport:
        stopped_reason, reason = self._stopped or ("", "")
        return ControlReport(decisions=self.decisions, best_reward=self._best,
                             total_cost=self._total_cost, accepted=self._accepted,
                             stopped_reason=stopped_reason, reason=reason)
