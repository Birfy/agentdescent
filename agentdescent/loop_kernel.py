"""A resumable RSI loop kernel over durable run state.

The platform already has the pieces: ``task_foundry`` produces a stream,
``verifier_ensemble`` scores a candidate, ``RSIController`` turns one
observation into accept/reject/defer/rollback/stop, ``durable_runs`` keeps a
leased event log, and ``rsi_telemetry`` / ``DecisionLog`` are the read models.
What was missing is one loop that keeps those seams in the same order across a
crash.

This module is that kernel, deliberately generic:

* the caller supplies ``step(context) -> LoopObservation`` (generate + evaluate
  one candidate);
* the controller decides what the observation means;
* each generation is a durable event carrying the observation **and the
  control decision**, so a new process can resume without repeating completed
  generations or re-running non-deterministic gates;
* optional ``apply`` / ``rollback`` callbacks turn a decision into an external
  effect, and optional telemetry / decision log record the story.

    store = SqliteRunStore("runs.sqlite3")
    kernel = LoopKernel(store, LoopConfig("tenant-a", "ws-1",
                                          max_generations=4, target=0.9))
    run_id = kernel.submit()
    report = kernel.run(run_id, "worker-1", step, apply=apply_artifact)
    report.best_reward, report.stopped_reason, report.generations

Durable execution is **at-least-once**: ``step`` and ``apply`` must be
idempotent for ``(run_id, generation)``. The kernel records a generation only
after both return; a crash between the external effect and the event replays
that generation. That is the contract of ``durable_runs``, not a hidden retry.

This is a vertical slice, not a distributed scheduler: one SQLite lease, one
worker at a time. The event schema is what a Postgres/Temporal adapter keeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .durable_runs import DurableRunService, RunSnapshot, SqliteRunStore, StepOutcome
from .rsi_control import (
    ControlAction,
    ControlConfig,
    ControlDecision,
    RSIController,
    SessionResult,
)

__all__ = [
    "LoopConfig",
    "LoopObservation",
    "LoopContext",
    "LoopGeneration",
    "LoopReport",
    "LoopKernel",
]


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class LoopConfig:
    """Identity, bounds and controller thresholds for one durable loop."""

    tenant: str
    workspace: str
    spec_hash: str = "loop"
    max_generations: int = 3
    target: Optional[float] = None
    max_cost: Optional[float] = None
    min_gain: float = 0.0
    patience: int = 2
    risk_limit: float = 1.0
    base_seed: int = 0
    lease_seconds: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, str) or not self.tenant.strip():
            raise ValueError("tenant is required")
        if not isinstance(self.workspace, str) or not self.workspace.strip():
            raise ValueError("workspace is required")
        if not isinstance(self.spec_hash, str) or not self.spec_hash.strip():
            raise ValueError("spec_hash must be a non-empty string")
        if isinstance(self.max_generations, bool) or not isinstance(
                self.max_generations, int) or self.max_generations < 1:
            raise ValueError("max_generations must be an int >= 1")
        if self.target is not None:
            _finite(self.target, "target")
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
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an int")
        lease = _finite(self.lease_seconds, "lease_seconds")
        if lease <= 0:
            raise ValueError("lease_seconds must be > 0")
        if isinstance(self.max_retries, bool) or not isinstance(
                self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be an int >= 0")

    def control_config(self) -> ControlConfig:
        return ControlConfig(
            target=self.target, max_sessions=self.max_generations,
            max_cost=self.max_cost, min_gain=self.min_gain,
            patience=self.patience, risk_limit=self.risk_limit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant": self.tenant,
            "workspace": self.workspace,
            "spec_hash": self.spec_hash,
            "max_generations": self.max_generations,
            "target": self.target,
            "max_cost": self.max_cost,
            "min_gain": self.min_gain,
            "patience": self.patience,
            "risk_limit": self.risk_limit,
            "base_seed": self.base_seed,
        }


@dataclass(frozen=True)
class LoopObservation:
    """One measured generation: reward, cost, risk and the artifact it produced."""

    reward: float
    cost: float = 0.0
    risk: float = 0.0
    error: Optional[str] = None
    artifact_id: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.reward, "reward")
        cost = _finite(self.cost, "cost")
        if cost < 0:
            raise ValueError("cost must be >= 0")
        risk = _finite(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if not isinstance(self.artifact_id, str):
            raise TypeError("artifact_id must be a string")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "evidence", dict(self.evidence or {}))

    def to_session_result(self) -> SessionResult:
        evidence = dict(self.evidence)
        evidence.setdefault("artifact_id", self.artifact_id)
        return SessionResult(reward=self.reward, cost=self.cost,
                             risk=self.risk, error=self.error,
                             evidence=evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "cost": self.cost,
            "risk": self.risk,
            "error": self.error,
            "artifact_id": self.artifact_id,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopObservation":
        if not isinstance(payload, Mapping) or "reward" not in payload:
            raise ValueError("a recorded loop observation needs a reward")
        return cls(
            reward=payload["reward"],
            cost=payload.get("cost", 0.0),
            risk=payload.get("risk", 0.0),
            error=payload.get("error"),
            artifact_id=str(payload.get("artifact_id", "")),
            evidence=payload.get("evidence") or {})


@dataclass(frozen=True)
class LoopGeneration:
    """One generation's durable record, reconstructed from the event log."""

    generation: int
    seed: int
    action: str
    reason: str
    reward: float
    cost: float
    risk: float
    gain: Optional[float]
    artifact_id: str
    stop: bool
    applied: bool = False
    error: Optional[str] = None
    budget_scale: float = 1.0
    span_id: str = ""

    @property
    def accepted(self) -> bool:
        return self.action == ControlAction.ACCEPT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "seed": self.seed,
            "action": self.action,
            "reason": self.reason,
            "reward": self.reward,
            "cost": self.cost,
            "risk": self.risk,
            "gain": self.gain,
            "artifact_id": self.artifact_id,
            "accepted": self.accepted,
            "stop": self.stop,
            "applied": self.applied,
            "error": self.error,
            "budget_scale": self.budget_scale,
            "span_id": self.span_id,
        }


@dataclass(frozen=True)
class LoopContext:
    """What the caller's ``step`` sees for one generation."""

    run_id: str
    generation: int
    seed: int
    budget_scale: float
    best_reward: Optional[float]
    history: Tuple[LoopGeneration, ...]
    snapshot: RunSnapshot


@dataclass(frozen=True)
class LoopReport:
    """The durable loop's generations and why it stopped."""

    run_id: str
    status: str
    generations: Tuple[LoopGeneration, ...] = ()
    best_reward: Optional[float] = None
    total_cost: float = 0.0
    accepted: int = 0
    stopped_reason: str = ""
    reason: str = ""
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "generations": [row.to_dict() for row in self.generations],
            "best_reward": self.best_reward,
            "total_cost": self.total_cost,
            "accepted": self.accepted,
            "stopped_reason": self.stopped_reason,
            "reason": self.reason,
            "detail": self.detail,
        }


Step = Callable[[LoopContext], LoopObservation]
Apply = Callable[[LoopContext, LoopObservation], None]
Rollback = Callable[[LoopContext, LoopObservation], None]
ControllerFactory = Callable[[], RSIController]


class LoopKernel:
    """Run a durable loop of ``step -> control -> apply`` generations."""

    def __init__(self, store: SqliteRunStore, config: LoopConfig, *,
                 decision_log: Optional[Any] = None,
                 telemetry: Optional[Any] = None,
                 gate: Optional[Callable[..., Any]] = None,
                 risk_gate: Optional[Callable[..., Any]] = None,
                 controller_factory: Optional[ControllerFactory] = None) -> None:
        if not isinstance(config, LoopConfig):
            raise TypeError("config must be a LoopConfig")
        self.store = store
        self.config = config
        self.service = DurableRunService(store)
        self.decision_log = decision_log
        self.telemetry = telemetry
        self._control = config.control_config()
        self._gate = gate
        self._risk_gate = risk_gate
        self._controller_factory = controller_factory or self._default_controller

    def _default_controller(self) -> RSIController:
        return RSIController(self._control, gate=self._gate,
                             risk_gate=self._risk_gate)

    # -- lifecycle -----------------------------------------------------------

    def submit(self, *, run_id: Optional[str] = None,
               payload: Optional[Mapping[str, Any]] = None) -> str:
        data = {"loop": self.config.to_dict()}
        data.update(dict(payload or {}))
        return self.service.submit(
            self.config.tenant, self.config.workspace,
            spec_hash=self.config.spec_hash, run_id=run_id, payload=data)

    def snapshot(self, run_id: str) -> RunSnapshot:
        return self.service.snapshot(run_id)

    def cancel(self, run_id: str, *, reason: str = "operator cancel") -> LoopReport:
        """Cancel a pending/running loop and return the durable report."""
        return self._report(self.service.cancel(run_id, reason=reason))

    def report(self, run_id: str) -> LoopReport:
        return self._report(self.service.snapshot(run_id))

    def run(self, run_id: str, worker_id: str, step: Step, *,
            apply: Optional[Apply] = None,
            rollback: Optional[Rollback] = None) -> LoopReport:
        """Resume/run until the controller stops or the budget ends.

        A crashed generation is retried by ``durable_runs``; every completed
        generation is replayed from the event log (decision included, gates are
        not re-called) before the next ``step``. If retryable failures exhaust
        the step budget the run can return with ``status == "running"``; call
        ``run`` again until ``report.terminal``.
        """
        if not callable(step):
            raise TypeError("step must be callable")
        snapshot = self.service.run_step(
            run_id, worker_id,
            lambda snap: self._step(snap, run_id, step, apply, rollback),
            lease_seconds=self.config.lease_seconds,
            max_steps=(self.config.max_generations + 1) * (
                self.config.max_retries + 1),
            max_retries=self.config.max_retries)
        return self._report(snapshot)

    # -- one durable step ----------------------------------------------------

    def _step(self, snapshot: RunSnapshot, run_id: str, step: Step,
              apply: Optional[Apply],
              rollback: Optional[Rollback]) -> StepOutcome:
        controller, history = self._replay(snapshot)
        if controller.stopped:
            report = controller.report()
            return StepOutcome(done=True, payload={
                "stopped_reason": report.stopped_reason,
                "reason": report.reason})
        generation = len(history)
        if generation >= self.config.max_generations:
            return StepOutcome(done=True, payload={
                "stopped_reason": "max_generations",
                "reason": f"reached {self.config.max_generations} generation(s)"})

        budget_scale = history[-1].budget_scale if history else 1.0
        context = LoopContext(
            run_id=run_id, generation=generation,
            seed=self.config.base_seed + generation,
            budget_scale=budget_scale,
            best_reward=controller.best_reward,
            history=history, snapshot=snapshot)

        span = None
        if self.telemetry is not None:
            span = self.telemetry.start_span(
                "generation", kind="loop",
                attributes={"run_id": run_id, "generation": generation,
                            "seed": context.seed})
        observation: LoopObservation
        try:
            observation = step(context)
            if not isinstance(observation, LoopObservation):
                raise TypeError(
                    f"step returned {type(observation).__name__}, expected "
                    "LoopObservation")
        except Exception as exc:  # noqa: BLE001 - an observation error is a result
            observation = LoopObservation(
                0.0, error=f"{type(exc).__name__}: {exc}")

        decision = controller.observe(observation.to_session_result())
        applied = False
        try:
            if decision.accepted and apply is not None:
                apply(context, observation)
                applied = True
            elif decision.action == ControlAction.ROLLBACK and rollback is not None:
                rollback(context, observation)
        finally:
            if span is not None:
                self.telemetry.event(
                    span, decision.action,
                    {"generation": generation, "reward": observation.reward,
                     "applied": applied, "stop": decision.stop})
                self.telemetry.end_span(
                    span, "error" if observation.error else "ok")

        self._record_decision(run_id, generation, observation, decision,
                              applied, controller.total_cost)
        payload = {
            "observation": observation.to_dict(),
            "decision": {
                "action": decision.action,
                "reason": decision.reason,
                "gain": decision.gain,
                "stop": decision.stop,
                "budget_scale": decision.budget_scale,
                "stopped_reason": controller.report().stopped_reason,
                "stop_reason": controller.report().reason,
            },
            "applied": applied,
            "seed": context.seed,
            "span_id": span.span_id if span is not None else "",
        }
        # Always record the generation as a ``step`` event. When this decision
        # stops the loop, the next (bounded) iteration sees ``controller.stopped``
        # and completes the run, so the final generation is not lost inside the
        # ``completed`` payload.
        return StepOutcome(
            done=False, payload=payload, cost=observation.cost,
            event_key=f"generation:{generation}")

    def _record_decision(self, run_id: str, generation: int,
                         observation: LoopObservation, decision: Any,
                         applied: bool, total_cost: float) -> None:
        log = self.decision_log
        if log is None:
            return
        try:
            log.record(
                kind="loop",
                artifact_id=observation.artifact_id or f"run:{run_id}",
                accepted=decision.accepted,
                category=decision.action,
                detail=decision.reason,
                evidence={
                    "run_id": run_id, "generation": generation,
                    "reward": observation.reward, "gain": decision.gain,
                    "stop": decision.stop, "applied": applied,
                    **dict(observation.evidence),
                },
                cost={"generation_cost": observation.cost,
                      "total_cost": total_cost},
                parent_id=run_id,
                record_id=f"{run_id}:generation:{generation}")
        except ValueError as exc:
            # A crash after the log write but before the event append replays the
            # generation; the deterministic record id makes the replay a no-op.
            if "duplicate record_id" not in str(exc):
                raise

    # -- replay / reporting --------------------------------------------------

    def _replay(self, snapshot: RunSnapshot) -> Tuple[RSIController,
                                                       Tuple[LoopGeneration, ...]]:
        controller = self._controller_factory()
        rows = []
        for event in snapshot.events:
            if event.kind != "step":
                continue
            payload = event.payload
            body = payload.get("result")
            if not isinstance(body, Mapping):
                body = payload
            observation = LoopObservation.from_dict(
                body.get("observation") or {})
            recorded = body.get("decision")
            if not isinstance(recorded, Mapping):
                raise ValueError(
                    "a recorded loop generation needs its decision; re-running "
                    "the gates on replay would not be durable")
            decision = ControlDecision(
                index=len(rows),
                action=str(recorded.get("action", "")),
                reason=str(recorded.get("reason", "")),
                reward=observation.reward,
                gain=recorded.get("gain"),
                cost=observation.cost,
                budget_scale=float(recorded.get("budget_scale", 1.0)),
                stop=bool(recorded.get("stop", False)))
            controller.restore(
                decision,
                stopped_reason=str(recorded.get("stopped_reason", "")),
                stop_reason=str(recorded.get("stop_reason", "")))
            rows.append(LoopGeneration(
                generation=len(rows),
                seed=int(body.get("seed", len(rows))),
                action=decision.action,
                reason=decision.reason,
                reward=observation.reward,
                cost=observation.cost,
                risk=observation.risk,
                gain=decision.gain,
                artifact_id=observation.artifact_id,
                stop=decision.stop,
                applied=bool(body.get("applied", False)),
                error=observation.error,
                budget_scale=decision.budget_scale,
                span_id=str(body.get("span_id", ""))))
        return controller, tuple(rows)

    def _report(self, snapshot: RunSnapshot) -> LoopReport:
        controller, rows = self._replay(snapshot)
        summary = controller.report()
        stopped_reason = summary.stopped_reason
        reason = summary.reason
        if not stopped_reason and snapshot.terminal:
            stopped_reason = snapshot.status
            reason = snapshot.error or snapshot.status
        detail = (f"{len(rows)} generation(s), best={summary.best_reward}, "
                  f"cost={summary.total_cost:g}, status={snapshot.status}")
        return LoopReport(
            run_id=snapshot.run_id, status=snapshot.status,
            generations=rows, best_reward=summary.best_reward,
            total_cost=summary.total_cost, accepted=summary.accepted,
            stopped_reason=stopped_reason, reason=reason, detail=detail)
