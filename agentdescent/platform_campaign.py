"""Multi-cycle campaigns on top of the five-layer RSI platform.

:class:`agentdescent.platform.RSIPlatform` runs one bounded cycle across
compute, loop, promotion, foundry and experience. This module adds the missing
long-running seam: :class:`RSIController` decides whether another cycle is
worth running, and each cycle is a fresh clone of the platform.

A campaign shares the platform's run store, compute controller, capability
foundry and experience store, so learning and admitted capabilities accumulate.
Only the loop seed changes per cycle, which keeps each cycle reproducible while
still exploring a different point. An incomplete cycle or an exception is
reported to the controller as a failed session; it is never silently skipped.

    campaign = PlatformCampaign(
        platform, CampaignConfig(max_cycles=5, target=0.9, patience=2))
    report = campaign.run(step=my_step, evidence=my_evidence)
    report.best_cycle, report.control.stopped_reason, report.complete
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .compute_policy import ComputeContext
from .experience import ExperienceRecord
from .loop_kernel import LoopContext, LoopObservation
from .platform import (
    CapabilityFn,
    EvidenceFn,
    PlatformReport,
    RSIPlatform,
    StepFn,
)
from .rsi_control import (
    ControlConfig,
    ControlReport,
    RSIController,
    SessionResult,
)

__all__ = [
    "CampaignConfig",
    "PlatformCampaign",
    "PlatformCampaignReport",
]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class CampaignConfig:
    """Bounds and stop thresholds for a multi-cycle platform campaign."""

    max_cycles: int = 3
    target: Optional[float] = None
    max_cost: Optional[float] = None
    patience: int = 1
    min_gain: float = 0.0
    base_seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_cycles, bool) or not isinstance(
                self.max_cycles, int) or self.max_cycles < 1:
            raise ValueError("max_cycles must be an int >= 1")
        if self.target is not None:
            _finite(self.target, "target")
        if self.max_cost is not None:
            cost = _finite(self.max_cost, "max_cost")
            if cost <= 0:
                raise ValueError("max_cost must be > 0 or None")
        if _finite(self.min_gain, "min_gain") < 0:
            raise ValueError("min_gain must be >= 0")
        if isinstance(self.patience, bool) or not isinstance(
                self.patience, int) or self.patience < 1:
            raise ValueError("patience must be an int >= 1")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an int")

    def control_config(self) -> ControlConfig:
        return ControlConfig(
            target=self.target, max_sessions=self.max_cycles,
            max_cost=self.max_cost, min_gain=self.min_gain,
            patience=self.patience)


@dataclass(frozen=True)
class PlatformCampaignReport:
    """The cycle reports and the controller's stop decision."""

    cycles: Tuple[PlatformReport, ...]
    control: ControlReport
    best_cycle: Optional[int] = None
    complete: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycles": [cycle.to_dict() for cycle in self.cycles],
            "control": asdict(self.control),
            "best_cycle": self.best_cycle,
            "complete": self.complete,
            "detail": self.detail,
        }


class PlatformCampaign:
    """Run several platform cycles under the RSIController stop policy."""

    def __init__(self, platform: RSIPlatform,
                 config: Optional[CampaignConfig] = None) -> None:
        if not isinstance(platform, RSIPlatform):
            raise TypeError("platform must be an RSIPlatform")
        if config is not None and not isinstance(config, CampaignConfig):
            raise TypeError("config must be a CampaignConfig or None")
        self.platform = platform
        self.config = CampaignConfig() if config is None else config

    def run(self, *, worker_id: str = "campaign-worker",
            step: StepFn,
            evidence: Optional[EvidenceFn] = None,
            capability: Optional[CapabilityFn] = None,
            validation: Sequence[ExperienceRecord] = (),
            anchors: Sequence[ExperienceRecord] = (),
            compute_context: Optional[ComputeContext] = None,
            apply: Optional[Callable[[LoopContext, LoopObservation], None]] = None,
            rollback: Optional[Callable[[LoopContext, LoopObservation], None]] = None,
    ) -> PlatformCampaignReport:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if not callable(step):
            raise TypeError("step must be callable")
        if evidence is not None and not callable(evidence):
            raise TypeError("evidence must be callable or None")
        if capability is not None and not callable(capability):
            raise TypeError("capability must be callable or None")
        if apply is not None and not callable(apply):
            raise TypeError("apply must be callable or None")
        if rollback is not None and not callable(rollback):
            raise TypeError("rollback must be callable or None")

        reports: List[PlatformReport] = []

        def run_session(index: int, seed: int) -> SessionResult:
            run_id = ""
            try:
                session_platform = self.platform.clone(base_seed=seed)
                run_id = session_platform.submit(
                    payload={"campaign_cycle": index})
                report = session_platform.run_cycle(
                    run_id, worker_id, step=step, evidence=evidence,
                    capability=capability, validation=validation,
                    anchors=anchors, compute_context=compute_context,
                    apply=apply, rollback=rollback)
            except Exception as exc:  # noqa: BLE001 - one cycle is one attempt
                return SessionResult(
                    0.0, error=f"{type(exc).__name__}: {exc}",
                    evidence={"run_id": run_id, "cycle": index})
            reports.append(report)
            if not report.complete:
                return SessionResult(
                    0.0, cost=float(report.loop.total_cost),
                    error=f"incomplete cycle: {report.detail}",
                    evidence={"run_id": run_id, "cycle": index})
            reward = (0.0 if report.loop.best_reward is None
                      else float(report.loop.best_reward))
            return SessionResult(
                reward=reward, cost=float(report.loop.total_cost),
                evidence={"run_id": run_id, "cycle": index})

        controller = RSIController(self.config.control_config())
        control = controller.run(run_session, base_seed=self.config.base_seed)

        best_cycle = None
        best_reward = None
        for index, report in enumerate(reports):
            reward = report.loop.best_reward
            if reward is None:
                continue
            if best_reward is None or reward > best_reward:
                best_reward = reward
                best_cycle = index

        complete = (
            bool(reports)
            and all(report.complete for report in reports)
            and control.stopped_reason != "error")
        detail = (
            f"{len(reports)} cycle(s), best_cycle={best_cycle}, "
            f"stopped={control.stopped_reason or 'none'}, "
            f"reason={control.reason or 'none'}")
        return PlatformCampaignReport(
            cycles=tuple(reports), control=control, best_cycle=best_cycle,
            complete=complete, detail=detail)
