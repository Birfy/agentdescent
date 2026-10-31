"""One RSI system cycle over the five integration layers.

The five feature modules are deliberately independent: a loop kernel, a compute
allocator, a promotion decision, a capability foundry and a continual
experience store. This module is the narrow system seam that makes them one
cycle instead of five disconnected tools:

``compute -> loop -> promote -> admit capability -> learn experience``.

The caller still supplies the domain callbacks (one step, promotion evidence,
an optional capability candidate). The platform owns the hand-offs, the
ordering, the automatic experience records and the completeness check.

    platform = RSIPlatform(
        store, PlatformConfig("tenant", "ws", max_generations=3, target=0.9),
        compute=controller, foundry=foundry, experience=engine,
        decision_log=log, telemetry=telemetry)
    run_id = platform.submit()
    report = platform.run_cycle(
        run_id, "worker-1",
        step=my_step,                  # (LoopContext, ComputeDecision) -> LoopObservation
        evidence=my_promotion_evidence,  # (LoopReport, best) -> PromotionRequest
        capability=my_capability,        # (LoopReport, PromotionDecision) -> CapabilityRequest
    )
    report.complete, report.to_dict()

This is an orchestrator, not a sixth feature: it calls the tested modules and
adds no new decision logic. A layer that is not configured is simply absent
from the report; a configured layer that did not produce a decision makes
``complete`` false instead of being silently skipped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .capability_foundry import (
    CapabilityFoundry,
    CapabilityOutcome,
    CapabilityRequest,
)
from .compute_policy import ComputeContext, ComputeController, ComputeDecision
from .durable_runs import SqliteRunStore
from .experience import ExperienceEngine, ExperienceRecord, ExperienceUpdate
from .loop_kernel import (
    LoopConfig,
    LoopContext,
    LoopGeneration,
    LoopKernel,
    LoopObservation,
    LoopReport,
)
from .promotion import (
    PromotionDecision,
    PromotionRequest,
    PromotionRules,
    promote,
)

__all__ = [
    "PlatformConfig",
    "PlatformReport",
    "RSIPlatform",
]

StepFn = Callable[[LoopContext, ComputeDecision], LoopObservation]
EvidenceFn = Callable[[LoopReport, LoopGeneration], PromotionRequest]
CapabilityFn = Callable[[LoopReport, Optional[PromotionDecision]],
                        Optional[CapabilityRequest]]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class PlatformConfig:
    """The loop identity and bounds; the layers keep their own policies."""

    tenant: str
    workspace: str
    max_generations: int = 3
    target: Optional[float] = None
    max_cost: Optional[float] = None
    min_gain: float = 0.0
    patience: int = 2
    risk_limit: float = 1.0
    base_seed: int = 0
    spec_hash: str = "platform"
    lease_seconds: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, str) or not self.tenant.strip():
            raise ValueError("tenant is required")
        if not isinstance(self.workspace, str) or not self.workspace.strip():
            raise ValueError("workspace is required")
        if isinstance(self.max_generations, bool) or not isinstance(
                self.max_generations, int) or self.max_generations < 1:
            raise ValueError("max_generations must be an int >= 1")
        if self.target is not None:
            _finite(self.target, "target")
        if self.max_cost is not None and _finite(self.max_cost, "max_cost") <= 0:
            raise ValueError("max_cost must be > 0 or None")
        if _finite(self.min_gain, "min_gain") < 0:
            raise ValueError("min_gain must be >= 0")
        if isinstance(self.patience, bool) or not isinstance(
                self.patience, int) or self.patience < 1:
            raise ValueError("patience must be an int >= 1")
        risk = _finite(self.risk_limit, "risk_limit")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk_limit must be in [0, 1]")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an int")
        if not isinstance(self.spec_hash, str) or not self.spec_hash.strip():
            raise ValueError("spec_hash must be a non-empty string")
        if _finite(self.lease_seconds, "lease_seconds") <= 0:
            raise ValueError("lease_seconds must be > 0")
        if isinstance(self.max_retries, bool) or not isinstance(
                self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be an int >= 0")

    def loop_config(self) -> LoopConfig:
        return LoopConfig(
            tenant=self.tenant, workspace=self.workspace,
            spec_hash=self.spec_hash, max_generations=self.max_generations,
            target=self.target, max_cost=self.max_cost, min_gain=self.min_gain,
            patience=self.patience, risk_limit=self.risk_limit,
            base_seed=self.base_seed, lease_seconds=self.lease_seconds,
            max_retries=self.max_retries)


@dataclass(frozen=True)
class PlatformReport:
    """The five layers' decisions for one cycle, and whether all configured ones ran."""

    run_id: str
    loop: LoopReport
    promotion: Optional[PromotionDecision] = None
    foundry: Optional[CapabilityOutcome] = None
    experience: Optional[ExperienceUpdate] = None
    skipped: Tuple[str, ...] = ()
    self_validated: bool = False
    complete: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "complete": self.complete,
            "loop": self.loop.to_dict(),
            "promotion": (None if self.promotion is None
                          else self.promotion.to_dict()),
            "foundry": (None if self.foundry is None
                        else self.foundry.to_dict()),
            "experience": (None if self.experience is None
                           else asdict(self.experience)),
            "skipped": list(self.skipped),
            "self_validated": self.self_validated,
            "detail": self.detail,
        }


class RSIPlatform:
    """Compose the loop, compute, promotion, foundry and experience layers."""

    def __init__(self, store: SqliteRunStore, config: PlatformConfig, *,
                 compute: ComputeController,
                 promotion_rules: Optional[PromotionRules] = None,
                 foundry: Optional[CapabilityFoundry] = None,
                 experience: Optional[ExperienceEngine] = None,
                 decision_log: Optional[Any] = None,
                 telemetry: Optional[Any] = None,
                 gate: Optional[Callable[..., Any]] = None,
                 risk_gate: Optional[Callable[..., Any]] = None,
                 _processed: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if not isinstance(config, PlatformConfig):
            raise TypeError("config must be a PlatformConfig")
        if not isinstance(compute, ComputeController):
            raise TypeError("compute must be a ComputeController")
        if promotion_rules is not None and not isinstance(
                promotion_rules, PromotionRules):
            raise TypeError("promotion_rules must be a PromotionRules or None")
        if foundry is not None and not isinstance(foundry, CapabilityFoundry):
            raise TypeError("foundry must be a CapabilityFoundry or None")
        if experience is not None and not isinstance(experience, ExperienceEngine):
            raise TypeError("experience must be an ExperienceEngine or None")
        self.store = store
        self.config = config
        self.compute = compute
        self.promotion_rules = promotion_rules or PromotionRules()
        self.foundry = foundry
        self.experience = experience
        self.decision_log = decision_log
        self.telemetry = telemetry
        self.gate = gate
        self.risk_gate = risk_gate
        self._processed = _processed if _processed is not None else {}
        self.kernel = LoopKernel(
            store, config.loop_config(), decision_log=decision_log,
            telemetry=telemetry, gate=gate, risk_gate=risk_gate)

    def submit(self, *, run_id: Optional[str] = None,
               payload: Optional[Mapping[str, Any]] = None) -> str:
        data = {"platform": self.config.loop_config().to_dict()}
        data.update(dict(payload or {}))
        return self.kernel.submit(run_id=run_id, payload=data)

    def clone(self, *, base_seed: Optional[int] = None) -> "RSIPlatform":
        """A new platform with a new loop seed, sharing every other component."""
        config = self.config if base_seed is None else replace(
            self.config, base_seed=base_seed)
        return RSIPlatform(
            self.store, config, compute=self.compute,
            promotion_rules=self.promotion_rules, foundry=self.foundry,
            experience=self.experience, decision_log=self.decision_log,
            telemetry=self.telemetry, gate=self.gate,
            risk_gate=self.risk_gate, _processed=self._processed)

    def report(self, run_id: str) -> LoopReport:
        return self.kernel.report(run_id)

    def run_cycle(
            self, run_id: str, worker_id: str, *,
            step: StepFn,
            evidence: Optional[EvidenceFn] = None,
            capability: Optional[CapabilityFn] = None,
            validation: Sequence[ExperienceRecord] = (),
            anchors: Sequence[ExperienceRecord] = (),
            compute_context: Optional[ComputeContext] = None,
            apply: Optional[Callable[[LoopContext, LoopObservation], None]] = None,
            rollback: Optional[Callable[[LoopContext, LoopObservation], None]] = None,
    ) -> PlatformReport:
        """Run one full cycle; every configured layer must produce a decision."""
        if not callable(step):
            raise TypeError("step must be callable")
        if evidence is not None and not callable(evidence):
            raise TypeError("evidence must be callable or None")
        if capability is not None and not callable(capability):
            raise TypeError("capability must be callable or None")
        if capability is not None and self.foundry is None:
            raise RuntimeError("a capability callback needs a CapabilityFoundry")

        def platform_step(context: LoopContext) -> LoopObservation:
            base = compute_context or ComputeContext()
            context_decision = replace(
                base, task_id=f"{run_id}:{context.generation}")
            decision = self.compute.choose(context_decision)
            observation = step(context, decision)
            if not isinstance(observation, LoopObservation):
                raise TypeError(
                    f"step returned {type(observation).__name__}, expected "
                    "LoopObservation")
            self.compute.observe(
                decision, reward=observation.reward, cost=observation.cost)
            data = dict(observation.evidence)
            data.setdefault("compute_action", decision.action)
            data.setdefault("compute_option", decision.option)
            data.setdefault("compute_model", decision.model)
            data.setdefault("features",
                            [context.generation, observation.reward,
                             observation.cost])
            return replace(observation, evidence=data)

        if self.kernel.snapshot(run_id).terminal and run_id not in self._processed:
            raise RuntimeError(
                f"run {run_id!r} is already terminal and its post-loop "
                "decisions are not cached; use a new run_id")
        loop = self.kernel.run(
            run_id, worker_id, platform_step, apply=apply, rollback=rollback)
        if not loop.terminal:
            return PlatformReport(
                run_id=run_id, loop=loop, complete=False,
                detail=(f"{len(loop.generations)} generation(s), status="
                        f"{loop.status}; call run_cycle again to finish"))
        cached = self._processed.get(run_id)
        if cached is not None:
            return PlatformReport(
                run_id=run_id, loop=loop,
                promotion=cached["promotion"], foundry=cached["foundry"],
                experience=cached["experience"],
                skipped=tuple(cached["skipped"]),
                self_validated=bool(cached["self_validated"]),
                complete=bool(cached["complete"]),
                detail=str(cached["detail"]))
        best = max(
            (row for row in loop.generations if row.accepted),
            key=lambda row: row.reward, default=None)
        promotion = None
        if evidence is not None and best is not None:
            request = evidence(loop, best)
            if not isinstance(request, PromotionRequest):
                raise TypeError(
                    "evidence callback must return a PromotionRequest")
            promotion = promote(request, self.promotion_rules)

        foundry = None
        skipped = []
        if capability is not None:
            if evidence is not None and (
                    promotion is None or promotion.action != "promote"):
                skipped.append("foundry:promotion_not_accepted")
            else:
                request = capability(loop, promotion)
                if request is not None:
                    if not isinstance(request, CapabilityRequest):
                        raise TypeError(
                            "capability callback must return a CapabilityRequest "
                            "or None")
                    foundry = self.foundry.admit(request)

        experience = None
        self_validated = False
        experience_requested = self.experience is not None
        if self.experience is not None:
            records = [self._experience_record(run_id, row)
                       for row in loop.generations]
            known = {row.run_id for row in self.experience.store.rows}
            for record in records:
                if record.run_id not in known:
                    self.experience.ingest(record)
                    known.add(record.run_id)
            val = list(validation)
            if not val:
                val = records
                self_validated = True
            if anchors:
                anc = list(anchors)
            else:
                current = {record.run_id for record in records}
                history = [row for row in self.experience.store.rows
                           if row.run_id not in current]
                # A real forgetting guard uses a pre-cycle slice. On the very
                # first cycle there is no history, so the cycle's own records
                # are the only bootstrap anchors.
                anc = history[-1:] or val[:1]
            if val:
                experience = self.experience.update(
                    validation=val, anchors=anc)

        requested = ["loop"]
        if evidence is not None:
            requested.append("promotion")
        if capability is not None:
            requested.append("foundry")
        if experience_requested:
            requested.append("experience")
        foundry_resolved = capability is None or foundry is not None or bool(skipped)
        complete = (
            loop.terminal
            and (promotion is not None if evidence is not None else True)
            and foundry_resolved
            and (experience is not None if experience_requested else True))
        detail = (f"{len(loop.generations)} generation(s), "
                  f"promotion={None if promotion is None else promotion.action}, "
                  f"foundry={None if foundry is None else foundry.status}, "
                  f"experience={None if experience is None else experience.reason}"
                  + (f", skipped={','.join(skipped)}" if skipped else "")
                  + (", experience=self-validation" if self_validated else ""))
        result = PlatformReport(
            run_id=run_id, loop=loop, promotion=promotion, foundry=foundry,
            experience=experience, skipped=tuple(skipped),
            self_validated=self_validated, complete=complete, detail=detail)
        # Only a terminal run's post-loop decisions are safe to cache.
        self._processed[run_id] = {
            "promotion": promotion, "foundry": foundry,
            "experience": experience, "skipped": tuple(skipped),
            "self_validated": self_validated, "complete": complete,
            "detail": detail}
        return result

    @staticmethod
    def _experience_record(run_id: str,
                           generation: LoopGeneration) -> ExperienceRecord:
        raw_features = generation.evidence.get(
            "features", (generation.generation, generation.reward,
                         generation.cost))
        try:
            features = tuple(float(value) for value in raw_features)
        except (TypeError, ValueError):
            features = (float(generation.generation), generation.reward,
                        generation.cost)
        action = str(generation.evidence.get(
            "compute_action", generation.evidence.get("action", "control")))
        raw_rewards = generation.evidence.get("rewards")
        if isinstance(raw_rewards, Mapping) and raw_rewards:
            rewards = {str(key): float(value)
                       for key, value in raw_rewards.items()}
        else:
            rewards = {action: generation.reward}
        return ExperienceRecord(
            run_id=f"{run_id}:generation:{generation.generation}",
            features=features,
            rewards=rewards,
            cost=generation.cost,
            accepted=generation.accepted,
            artifact_id=generation.artifact_id,
            created_at=0.0,
            metadata={"run_id": run_id,
                      "generation": generation.generation})

