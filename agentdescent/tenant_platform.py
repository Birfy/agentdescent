"""Tenancy around the platform seam: reserve, authorize and audit one cycle.

``RSIPlatform.run_cycle`` knows how to hand work between the loop, compute,
promotion, foundry and experience layers, but it has no notion of who may run
it or what the cycle is allowed to spend. ``WorkspaceRegistry`` already owns
roles, quota reservations and an append-only audit trail. This module composes
the two without reimplementing either:

* construct a ``TenantPlatform`` with one principal and one workspace, both of
  which must belong to the platform's tenant;
* ``run_cycle`` reserves the caller's stated tokens/cost (which also checks the
  ``run:write`` permission and every quota), runs the platform, and commits the
  actual usage when the platform succeeds;
* a failed platform call (including a loop generation that records a step
  error) releases the reservation, and a commit that would exceed the
  reservation also releases it and propagates the denial.

``TenantPlatform`` wraps ``RSIPlatform.run_cycle``; only cycles routed through
it are reserved, authorized and audited. The reservation is the prepaid charge.
If ``usage_of`` is omitted the reported usage is exactly the reserved amount;
when it is supplied the platform report is turned into actual ``(tokens, cost)``
and committed, capped at the reservation. A cycle that overruns the reservation
is charged the reserved amount and then refused, so a tenant cannot consume
compute indefinitely with a tiny reservation. A recorded generation error is
charged the same way. Only a platform failure before any generation ran is
released without a charge, because no work was observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .platform import PlatformReport, RSIPlatform
from .workspaces import (
    PermissionDenied,
    Principal,
    QuotaReservation,
    QuotaUsage,
    WorkspaceRegistry,
)

__all__ = ["TenantCycleReport", "TenantPlatform"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


@dataclass(frozen=True)
class TenantCycleReport:
    """One authorized platform cycle, its reservation and the committed usage."""

    report: PlatformReport
    reservation: QuotaReservation
    usage: QuotaUsage
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        report = self.report
        reservation = self.reservation
        return {
            "run_id": report.run_id,
            "complete": report.complete,
            "loop": report.loop.to_dict(),
            "promotion": (None if report.promotion is None
                          else report.promotion.action),
            "foundry": (None if report.foundry is None
                        else report.foundry.status),
            "experience": (None if report.experience is None
                           else report.experience.reason),
            "reservation_id": reservation.reservation_id,
            "reservation": {
                "reservation_id": reservation.reservation_id,
                "tenant_id": reservation.tenant_id,
                "workspace_id": reservation.workspace_id,
                "tokens": reservation.tokens,
                "cost": reservation.cost,
                "active": reservation.active,
                "committed": reservation.committed,
            },
            "usage": self.usage.to_dict(),
            "detail": self.detail,
        }


class TenantPlatform:
    """Bind one principal and workspace to an existing ``RSIPlatform``."""

    def __init__(
            self,
            platform: RSIPlatform,
            registry: WorkspaceRegistry,
            principal: Principal,
            *,
            workspace_id: Optional[str] = None,
            usage_of: Optional[Callable[[PlatformReport],
                                        Tuple[float, float]]] = None,
    ) -> None:
        if not isinstance(platform, RSIPlatform):
            raise TypeError("platform must be an RSIPlatform")
        if not isinstance(registry, WorkspaceRegistry):
            raise TypeError("registry must be a WorkspaceRegistry")
        if not isinstance(principal, Principal):
            raise TypeError("principal must be a Principal")
        if usage_of is not None and not callable(usage_of):
            raise TypeError("usage_of must be callable or None")
        if workspace_id is None:
            workspace_id = platform.config.workspace
        if not isinstance(workspace_id, str):
            raise TypeError("workspace_id must be a string")
        if not workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if principal.tenant_id != platform.config.tenant:
            raise ValueError(
                "principal tenant does not match the platform tenant")
        try:
            registry.permissions(principal.principal_id, workspace_id)
        except KeyError:
            raise ValueError(
                f"unknown workspace {principal.tenant_id}/{workspace_id}"
            ) from None
        self.platform = platform
        self.registry = registry
        self.principal = principal
        self.workspace_id = workspace_id
        self.usage_of = usage_of

    @staticmethod
    def _loop_error(report: PlatformReport) -> Optional[str]:
        for generation in report.loop.generations:
            if generation.error:
                return generation.error
        return None

    def _used(self, report: PlatformReport, tokens: float,
              cost: float) -> Tuple[float, float]:
        if self.usage_of is None:
            return (_finite(tokens, "tokens"), _finite(cost, "cost"))
        try:
            used = self.usage_of(report)
            tokens_used, cost_used = used
        except (TypeError, ValueError):
            raise TypeError("usage_of must return a (tokens, cost) pair") from None
        return (_finite(tokens_used, "used tokens"),
                _finite(cost_used, "used cost"))

    def _release(self, reservation: QuotaReservation) -> None:
        self.registry.release(
            self.principal.principal_id, reservation.reservation_id)

    def _charge_reserved(self, reservation: QuotaReservation) -> QuotaUsage:
        return self.registry.commit(
            self.principal.principal_id, reservation.reservation_id,
            tokens_used=reservation.tokens, cost_used=reservation.cost)

    def _charge(self, reservation: QuotaReservation, tokens_used: float,
                cost_used: float) -> QuotaUsage:
        return self.registry.commit(
            self.principal.principal_id, reservation.reservation_id,
            tokens_used=min(tokens_used, reservation.tokens),
            cost_used=min(cost_used, reservation.cost))

    def run_cycle(
            self, run_id: str, worker_id: str, *,
            tokens: float, cost: float, **kwargs: Any) -> TenantCycleReport:
        """Reserve, run one platform cycle, then commit its usage.

        Once the platform has run, a failure or an overrun is charged the
        reserved amount instead of releasing it: release-without-charge is only
        for a platform call that never produced a generation.
        """
        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(worker_id, str):
            raise TypeError("worker_id must be a string")
        if not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        reservation = self.registry.reserve(
            self.principal.principal_id, self.workspace_id,
            tokens=tokens, cost=cost)
        try:
            report = self.platform.run_cycle(run_id, worker_id, **kwargs)
        except Exception:
            did_work = False
            try:
                did_work = bool(self.platform.report(run_id).generations)
            except Exception:
                did_work = False
            if did_work:
                try:
                    self._charge_reserved(reservation)
                except Exception:
                    pass
            else:
                self._release(reservation)
            raise

        try:
            error = self._loop_error(report)
            tokens_used, cost_used = self._used(report, tokens, cost)
        except Exception:
            # the platform ran; charge the prepaid amount before propagating
            try:
                self._charge_reserved(reservation)
            except Exception:
                pass
            raise

        overrun = (tokens_used > reservation.tokens + 1e-12
                   or cost_used > reservation.cost + 1e-12)
        usage = self._charge(reservation, tokens_used, cost_used)
        if error is not None:
            raise RuntimeError(
                f"{error}; charged tokens={min(tokens_used, reservation.tokens)} "
                f"cost={min(cost_used, reservation.cost)}")
        if overrun:
            raise PermissionDenied(
                f"usage ({tokens_used} tokens, {cost_used} cost) exceeds the "
                f"reservation ({reservation.tokens} tokens, "
                f"{reservation.cost} cost); charged the reserved amount")
        detail = (f"reservation={reservation.reservation_id}; "
                  f"{report.detail}; "
                  f"used tokens={tokens_used} cost={cost_used}")
        return TenantCycleReport(
            report=report, reservation=reservation, usage=usage, detail=detail)
