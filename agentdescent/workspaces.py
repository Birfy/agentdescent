"""Tenant/workspace isolation, RBAC, quotas and audit for an RSI product.

A single-user CLI does not need tenancy. A team does: every run, artifact,
approval, trace and dollar must belong to a tenant and a workspace, and every
decision must answer "who was allowed to do this?".

This is the smallest useful control-plane slice:

* **principals** carry a tenant and per-workspace roles;
* **roles** map to a fixed permission set (`viewer`, `operator`, `approver`,
  `admin`, `owner`);
* **workspaces** own quotas (tokens, cost, concurrent runs);
* **reservations** make quota checks atomic at the app level and idempotent on
  commit/release;
* **audit** is append-only and exportable as JSONL or dicts.

It is deliberately in-memory for this slice. The interface is the contract a
SQLite/Postgres adapter keeps: tenant/workspace columns, a permission check
before every write, and an audit record for allow/deny and quota changes.

    reg = WorkspaceRegistry()
    reg.add_workspace(Workspace("t1", "ws1", "Research", Quota(max_cost=50)))
    reg.add_principal(Principal("u1", "t1", {"ws1": ("operator",)}))
    reservation = reg.reserve("u1", "ws1", tokens=1000, cost=2.5)
    reg.commit(reservation.reservation_id, tokens_used=900, cost_used=2.2)
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from types import MappingProxyType
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "Permission",
    "Role",
    "PermissionDenied",
    "Principal",
    "Quota",
    "Workspace",
    "QuotaUsage",
    "QuotaReservation",
    "AuditEvent",
    "WorkspaceRegistry",
]


class Role:
    VIEWER = "viewer"
    OPERATOR = "operator"
    APPROVER = "approver"
    ADMIN = "admin"
    OWNER = "owner"

    ALL = (VIEWER, OPERATOR, APPROVER, ADMIN, OWNER)


class Permission:
    RUN_READ = "run:read"
    RUN_WRITE = "run:write"
    RUN_CANCEL = "run:cancel"
    APPROVAL_READ = "approval:read"
    APPROVAL_DECIDE = "approval:decide"
    POLICY_WRITE = "policy:write"
    AUDIT_READ = "audit:read"
    WORKSPACE_ADMIN = "workspace:admin"
    QUOTA_READ = "quota:read"
    QUOTA_WRITE = "quota:write"

    ALL = (RUN_READ, RUN_WRITE, RUN_CANCEL, APPROVAL_READ, APPROVAL_DECIDE,
           POLICY_WRITE, AUDIT_READ, WORKSPACE_ADMIN, QUOTA_READ, QUOTA_WRITE)


ROLE_PERMISSIONS: Dict[str, frozenset] = {
    Role.VIEWER: frozenset({Permission.RUN_READ}),
    Role.OPERATOR: frozenset({
        Permission.RUN_READ, Permission.RUN_WRITE, Permission.RUN_CANCEL,
        Permission.QUOTA_READ,
    }),
    Role.APPROVER: frozenset({
        Permission.RUN_READ, Permission.APPROVAL_READ,
        Permission.APPROVAL_DECIDE, Permission.AUDIT_READ,
    }),
    Role.ADMIN: frozenset(Permission.ALL),
    Role.OWNER: frozenset(Permission.ALL),
}


class PermissionDenied(Exception):
    """Raised when a principal is not allowed to perform an action."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


@dataclass(frozen=True)
class Principal:
    """A user/service identity and its per-workspace roles.

    ``roles`` keys are workspace ids; ``"*"`` grants the roles in every
    workspace of the principal's tenant. Cross-tenant access is always denied.
    """

    principal_id: str
    tenant_id: str
    roles: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.principal_id or not self.tenant_id:
            raise ValueError("principal_id and tenant_id are required")
        for workspace, roles in self.roles.items():
            for role in roles:
                if role not in Role.ALL:
                    raise ValueError(f"unknown role {role!r}")
        object.__setattr__(self, "roles", MappingProxyType({
            str(workspace): tuple(roles) for workspace, roles in self.roles.items()}))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def roles_for(self, workspace_id: str) -> Tuple[str, ...]:
        return tuple(self.roles.get(workspace_id, ())) + tuple(self.roles.get("*", ()))


@dataclass(frozen=True)
class Quota:
    """Per-workspace limits; zero means unlimited."""

    max_tokens: int = 0
    max_cost: float = 0.0
    max_concurrent: int = 0

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_concurrent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Quota.{name} must be an int >= 0")
        if not math.isfinite(float(self.max_cost)) or float(self.max_cost) < 0:
            raise ValueError("Quota.max_cost must be finite and >= 0")


@dataclass(frozen=True)
class Workspace:
    tenant_id: str
    workspace_id: str
    name: str
    quota: Quota = field(default_factory=Quota)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.workspace_id or not self.name:
            raise ValueError("tenant_id, workspace_id and name are required")


@dataclass
class QuotaUsage:
    tokens: float = 0.0
    cost: float = 0.0
    active: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {"tokens": self.tokens, "cost": self.cost, "active": self.active}


@dataclass
class QuotaReservation:
    reservation_id: str
    tenant_id: str
    workspace_id: str
    tokens: float
    cost: float
    created_at: float
    active: bool = True
    committed: bool = False


@dataclass(frozen=True)
class AuditEvent:
    tenant_id: str
    workspace_id: str
    actor: str
    action: str
    resource: str
    allowed: bool
    reason: str = ""
    at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "allowed": self.allowed,
            "reason": self.reason,
            "at": self.at,
            "metadata": dict(self.metadata),
        }


class WorkspaceRegistry:
    """In-memory tenants, roles, quotas and audit."""

    def __init__(self, clock: Optional[Any] = None) -> None:
        self.clock = clock or time.time
        self._workspaces: Dict[Tuple[str, str], Workspace] = {}
        self._principals: Dict[str, Principal] = {}
        self._usage: Dict[Tuple[str, str], QuotaUsage] = {}
        self._reservations: Dict[str, QuotaReservation] = {}
        self._audit: List[AuditEvent] = []
        self._lock = threading.RLock()

    # -- registration --------------------------------------------------------

    def add_workspace(self, workspace: Workspace) -> Workspace:
        key = (workspace.tenant_id, workspace.workspace_id)
        if key in self._workspaces:
            raise ValueError(f"workspace {key} already exists")
        self._workspaces[key] = workspace
        self._usage[key] = QuotaUsage()
        self._record(workspace.tenant_id, workspace.workspace_id, "system",
                     "workspace.create", workspace.workspace_id, True)
        return workspace

    def add_principal(self, principal: Principal) -> Principal:
        if principal.principal_id in self._principals:
            raise ValueError(f"principal {principal.principal_id!r} already exists")
        self._principals[principal.principal_id] = principal
        self._record(principal.tenant_id, "*", principal.principal_id,
                     "principal.create", principal.principal_id, True)
        return principal

    def grant(self, actor: str, principal_id: str, workspace_id: str,
              role: str) -> Principal:
        if role not in Role.ALL:
            raise ValueError(f"unknown role {role!r}")
        principal = self._principal(principal_id)
        self.authorize(actor, principal.tenant_id, workspace_id,
                       Permission.WORKSPACE_ADMIN)
        roles = dict(principal.roles)
        current = list(roles.get(workspace_id, ()))
        if role not in current:
            current.append(role)
        roles[workspace_id] = tuple(current)
        updated = Principal(principal.principal_id, principal.tenant_id,
                            roles, principal.metadata)
        self._principals[principal_id] = updated
        self._record(principal.tenant_id, workspace_id, actor, "role.grant",
                     principal_id, True, role)
        return updated

    def revoke(self, actor: str, principal_id: str, workspace_id: str,
               role: str) -> Principal:
        principal = self._principal(principal_id)
        self.authorize(actor, principal.tenant_id, workspace_id,
                       Permission.WORKSPACE_ADMIN)
        roles = dict(principal.roles)
        current = [item for item in roles.get(workspace_id, ()) if item != role]
        roles[workspace_id] = tuple(current)
        updated = Principal(principal.principal_id, principal.tenant_id,
                            roles, principal.metadata)
        self._principals[principal_id] = updated
        self._record(principal.tenant_id, workspace_id, actor, "role.revoke",
                     principal_id, True, role)
        return updated

    # -- authorization -------------------------------------------------------

    def _principal(self, principal_id: str) -> Principal:
        try:
            return self._principals[principal_id]
        except KeyError:
            raise KeyError(f"unknown principal {principal_id!r}") from None

    def _workspace(self, tenant_id: str, workspace_id: str) -> Workspace:
        try:
            return self._workspaces[(tenant_id, workspace_id)]
        except KeyError:
            raise KeyError(f"unknown workspace {tenant_id}/{workspace_id}") from None

    def permissions(self, principal_id: str, workspace_id: str) -> frozenset:
        principal = self._principal(principal_id)
        workspace = self._workspace(principal.tenant_id, workspace_id)
        allowed = set()
        for role in principal.roles_for(workspace.workspace_id):
            allowed.update(ROLE_PERMISSIONS.get(role, ()))
        return frozenset(allowed)

    def authorize(self, principal_id: str, tenant_id: str, workspace_id: str,
                  permission: str) -> bool:
        with self._lock:
            principal = self._principals.get(principal_id)
            workspace = self._workspaces.get((tenant_id, workspace_id))
            if principal is None or workspace is None:
                self._record(tenant_id, workspace_id, principal_id, "authorize",
                             permission, False, "unknown principal or workspace")
                raise PermissionDenied("unknown principal or workspace")
            if principal.tenant_id != tenant_id:
                self._record(tenant_id, workspace_id, principal_id, "authorize",
                             permission, False, "cross-tenant access")
                raise PermissionDenied("cross-tenant access")
            allowed = permission in self.permissions(principal_id, workspace_id)
            if not allowed:
                self._record(tenant_id, workspace_id, principal_id, "authorize",
                             permission, False, f"{permission} not granted")
                raise PermissionDenied(f"{permission} not granted")
            self._record(tenant_id, workspace_id, principal_id, "authorize",
                         permission, True)
            return True

    # -- quotas --------------------------------------------------------------

    def _quota_key(self, reservation: QuotaReservation) -> Tuple[str, str]:
        return reservation.tenant_id, reservation.workspace_id

    def _active_totals(self, key: Tuple[str, str]) -> Tuple[float, float]:
        tokens = cost = 0.0
        for reservation in self._reservations.values():
            if reservation.active and self._quota_key(reservation) == key:
                tokens += reservation.tokens
                cost += reservation.cost
        return tokens, cost

    def reserve(self, principal_id: str, workspace_id: str, *,
                tokens: float, cost: float) -> QuotaReservation:
        with self._lock:
            principal = self._principal(principal_id)
            workspace = self._workspace(principal.tenant_id, workspace_id)
            self.authorize(principal_id, principal.tenant_id, workspace_id,
                           Permission.RUN_WRITE)
            tokens = _finite(tokens, "tokens")
            cost = _finite(cost, "cost")
            usage = self._usage[(principal.tenant_id, workspace_id)]
            reserved_tokens, reserved_cost = self._active_totals(
                (principal.tenant_id, workspace_id))
            quota = workspace.quota
            if quota.max_tokens and usage.tokens + reserved_tokens + tokens > quota.max_tokens:
                self._record(principal.tenant_id, workspace_id, principal_id,
                             "quota.reserve", "tokens", False, "max_tokens")
                raise PermissionDenied("token quota exceeded")
            if quota.max_cost and usage.cost + reserved_cost + cost > quota.max_cost + 1e-12:
                self._record(principal.tenant_id, workspace_id, principal_id,
                             "quota.reserve", "cost", False, "max_cost")
                raise PermissionDenied("cost quota exceeded")
            if quota.max_concurrent and usage.active + 1 > quota.max_concurrent:
                self._record(principal.tenant_id, workspace_id, principal_id,
                             "quota.reserve", "concurrency", False, "max_concurrent")
                raise PermissionDenied("concurrency quota exceeded")
            reservation = QuotaReservation(
                reservation_id=uuid.uuid4().hex,
                tenant_id=principal.tenant_id, workspace_id=workspace_id,
                tokens=tokens, cost=cost, created_at=float(self.clock()))
            self._reservations[reservation.reservation_id] = reservation
            usage.active += 1
            self._record(principal.tenant_id, workspace_id, principal_id,
                         "quota.reserve", reservation.reservation_id, True)
            return reservation

    def commit(self, actor: str, reservation_id: str, *, tokens_used: float,
               cost_used: float) -> QuotaUsage:
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise KeyError(f"unknown reservation {reservation_id!r}")
            self.authorize(actor, reservation.tenant_id, reservation.workspace_id,
                           Permission.RUN_WRITE)
            if reservation.committed:
                return self._usage[(reservation.tenant_id, reservation.workspace_id)]
            if not reservation.active:
                raise ValueError("reservation was released")
            tokens_used = _finite(tokens_used, "tokens_used")
            cost_used = _finite(cost_used, "cost_used")
            if tokens_used > reservation.tokens + 1e-12 or cost_used > reservation.cost + 1e-12:
                self._record(reservation.tenant_id, reservation.workspace_id,
                             actor, "quota.commit", reservation_id, False,
                             "commit exceeds reservation")
                raise PermissionDenied("commit exceeds reservation")
            usage = self._usage[(reservation.tenant_id, reservation.workspace_id)]
            usage.tokens += tokens_used
            usage.cost += cost_used
            usage.active = max(0, usage.active - 1)
            reservation.active = False
            reservation.committed = True
            self._record(reservation.tenant_id, reservation.workspace_id, actor,
                         "quota.commit", reservation_id, True,
                         f"tokens={tokens_used} cost={cost_used}")
            return usage

    def release(self, actor: str, reservation_id: str) -> QuotaUsage:
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise KeyError(f"unknown reservation {reservation_id!r}")
            self.authorize(actor, reservation.tenant_id, reservation.workspace_id,
                           Permission.RUN_WRITE)
            if reservation.active:
                usage = self._usage[(reservation.tenant_id, reservation.workspace_id)]
                usage.active = max(0, usage.active - 1)
                reservation.active = False
                self._record(reservation.tenant_id, reservation.workspace_id, actor,
                             "quota.release", reservation_id, True)
            return self._usage[(reservation.tenant_id, reservation.workspace_id)]

    def set_quota(self, actor: str, tenant_id: str, workspace_id: str,
                  quota: Quota) -> Workspace:
        with self._lock:
            self.authorize(actor, tenant_id, workspace_id, Permission.QUOTA_WRITE)
            workspace = self._workspace(tenant_id, workspace_id)
            updated = Workspace(workspace.tenant_id, workspace.workspace_id,
                                workspace.name, quota, workspace.metadata)
            self._workspaces[(tenant_id, workspace_id)] = updated
            self._record(tenant_id, workspace_id, actor, "quota.set",
                         workspace_id, True)
            return updated

    def usage(self, actor: str, tenant_id: str, workspace_id: str) -> QuotaUsage:
        with self._lock:
            self.authorize(actor, tenant_id, workspace_id, Permission.QUOTA_READ)
            return self._usage[(tenant_id, workspace_id)]

    # -- audit ---------------------------------------------------------------

    def _record(self, tenant_id: str, workspace_id: str, actor: str,
                action: str, resource: str, allowed: bool,
                reason: str = "") -> AuditEvent:
        event = AuditEvent(tenant_id, workspace_id, actor, action, resource,
                           allowed, reason, float(self.clock()))
        self._audit.append(event)
        return event

    def _audit_scope(self, actor: str, tenant_id: str,
                     workspace_id: Optional[str]) -> Tuple[str, ...]:
        if workspace_id is not None:
            self.authorize(actor, tenant_id, workspace_id, Permission.AUDIT_READ)
            return (workspace_id,)
        principal = self._principals.get(actor)
        if principal is None or principal.tenant_id != tenant_id:
            self._record(tenant_id, "*", actor, "authorize",
                         Permission.AUDIT_READ, False,
                         "unknown principal or cross-tenant access")
            raise PermissionDenied("audit:read not granted in this tenant")
        allowed = tuple(sorted(
            ws for (tenant, ws) in self._workspaces
            if tenant == tenant_id
            and Permission.AUDIT_READ in self.permissions(actor, ws)))
        if not allowed:
            self._record(tenant_id, "*", actor, "authorize",
                         Permission.AUDIT_READ, False,
                         "audit:read not granted in any workspace")
            raise PermissionDenied("audit:read not granted in this tenant")
        return allowed

    def audit_events(self, actor: str, *, tenant_id: str,
                     workspace_id: Optional[str] = None) -> Tuple[AuditEvent, ...]:
        with self._lock:
            scopes = self._audit_scope(actor, tenant_id, workspace_id)
            return tuple(event for event in self._audit
                         if event.tenant_id == tenant_id
                         and event.workspace_id in scopes)

    def export_audit(self, actor: str, *, tenant_id: str,
                     workspace_id: Optional[str] = None,
                     fmt: str = "jsonl") -> Union[str, Tuple[Dict[str, Any], ...]]:
        records = tuple(event.to_dict() for event in
                        self.audit_events(actor, tenant_id=tenant_id,
                                          workspace_id=workspace_id))
        if fmt == "dicts":
            return records
        if fmt == "jsonl":
            return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True)
                             for record in records)
        raise ValueError("fmt must be jsonl or dicts")
