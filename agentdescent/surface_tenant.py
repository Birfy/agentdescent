"""The ``agentdescent tenant`` surface: one quota-checked platform cycle.

`TenantPlatform` already puts a `WorkspaceRegistry` (roles, quota reservations,
audit) around `RSIPlatform.run_cycle`. This module makes that composition
runnable from a JSON spec, the CLI and the MCP server:

    platform spec + tenant/workspace/principal/roles/quota
        -> reserve quota -> one platform cycle -> commit usage

The nested `platform` object is exactly the spec `agentdescent platform` takes;
the tenancy fields say who may run it and what the cycle may spend. The surface
does not reimplement RBAC or quota logic; it builds the registry/principal the
tested `TenantPlatform` expects and returns its `TenantCycleReport`.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, load_json,
    parse_csv, require_callable, require_mapping, require_text, resolve_ref,
)
from . import surface_platform
from .tenant_platform import TenantPlatform
from .workspaces import (
    Permission, Principal, Quota, Role, Workspace, WorkspaceRegistry,
)

__all__ = ["tenant_init", "tenant_plan", "tenant_run"]

_TENANT_KEYS = ("schema_version", "platform", "tenant", "workspace",
                "workspace_name", "principal", "roles", "quota", "tokens",
                "cost", "usage_of", "worker_id")


def _label(value: Any) -> str:
    return (os.path.abspath(os.path.expanduser(value))
            if isinstance(value, str) else "<inline>")


def _config(cls: Any, payload: Any, name: str) -> Any:
    if payload is None:
        return cls()
    data = require_mapping(payload, name)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SurfaceError(
            f"unknown {name} key(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(sorted(allowed)))
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid {name}: {error}") from None


def load_spec(value: Any) -> Dict[str, Any]:
    payload = load_json(value, name="tenant spec")
    spec = require_mapping(payload, "tenant spec")
    unknown = sorted(set(spec) - set(_TENANT_KEYS))
    if unknown:
        raise SurfaceError(
            "unknown tenant key(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(_TENANT_KEYS))
    for key in ("platform", "tenant", "workspace", "principal"):
        if key not in spec:
            raise SurfaceError(f"tenant spec needs {key!r}")
    return spec


def _tenancy(spec: Mapping[str, Any], *,
             allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    tenant = require_text(spec.get("tenant"), "tenant")
    workspace = require_text(spec.get("workspace"), "workspace")
    principal_id = require_text(spec.get("principal"), "principal")
    platform_spec = require_mapping(spec.get("platform"), "platform")
    platform = _config(surface_platform.PlatformConfig, platform_spec.get(
        "platform"), "platform.platform")
    if platform.tenant != tenant or platform.workspace != workspace:
        raise SurfaceError(
            "tenant/workspace must match platform.platform "
            f"({platform.tenant}/{platform.workspace})")
    roles = [str(role) for role in parse_csv(
        spec.get("roles") or ("operator",), name="roles")]
    for role in roles:
        if role not in Role.ALL:
            raise SurfaceError(
                f"unknown role {role!r}; allowed: {', '.join(Role.ALL)}")
    quota = _config(Quota, spec.get("quota"), "quota")
    if "tokens" not in spec or "cost" not in spec:
        raise SurfaceError(
            "tenant spec needs explicit tokens and cost (the reservation)")
    if isinstance(spec.get("tokens"), bool) or isinstance(spec.get("cost"), bool):
        raise SurfaceError("tokens and cost must be numbers, not bool")
    try:
        tokens = float(spec["tokens"])
        cost = float(spec["cost"])
    except (TypeError, ValueError):
        raise SurfaceError("tokens and cost must be numbers") from None
    if not math.isfinite(tokens) or not math.isfinite(cost) or tokens < 0 or cost < 0:
        raise SurfaceError("tokens and cost must be finite and >= 0")
    usage_of = spec.get("usage_of")
    if usage_of is not None:
        usage_of = require_callable(
            resolve_ref(usage_of, name="usage_of", allow=allow), "usage_of")
    return {
        "tenant": tenant,
        "workspace": workspace,
        "principal": principal_id,
        "roles": tuple(roles),
        "quota": quota,
        "workspace_name": str(spec.get("workspace_name") or workspace),
        "tokens": tokens,
        "cost": cost,
        "worker_id": str(spec.get("worker_id") or "tenant-worker"),
        "usage_of": usage_of,
        "platform": platform_spec,
    }


def _registry(tenancy: Mapping[str, Any]) -> Tuple[WorkspaceRegistry, Principal]:
    registry = WorkspaceRegistry()
    registry.add_workspace(Workspace(
        tenancy["tenant"], tenancy["workspace"], tenancy["workspace_name"],
        tenancy["quota"]))
    principal = Principal(
        tenancy["principal"], tenancy["tenant"],
        {tenancy["workspace"]: tenancy["roles"]})
    registry.add_principal(principal)
    return registry, principal


def tenant_plan(path: Any, *, allow: Optional[Sequence[str]] = None
                ) -> Dict[str, Any]:
    spec = load_spec(path)
    tenancy = _tenancy(spec, allow=allow)
    checked = surface_platform._validate(tenancy["platform"], allow=allow)
    registry, principal = _registry(tenancy)
    permissions = registry.permissions(
        principal.principal_id, tenancy["workspace"])
    if Permission.RUN_WRITE not in permissions:
        raise SurfaceError(
            f"principal {principal.principal_id!r} has no run:write in "
            f"{tenancy['workspace']!r}")
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "tenant": tenancy["tenant"],
        "workspace": tenancy["workspace"],
        "principal": tenancy["principal"],
        "roles": list(tenancy["roles"]),
        "quota": asdict(tenancy["quota"]),
        "reserve": {"tokens": tenancy["tokens"], "cost": tenancy["cost"]},
        "runs_db": checked["runs_db"],
        "platform": checked["platform"],
        "next": ["agentdescent tenant run --spec <spec.json>"],
    }


def tenant_run(path: Any, *, out: Optional[str] = None,
               allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    spec = load_spec(path)
    tenancy = _tenancy(spec, allow=allow)
    platform, _, callbacks = surface_platform._build(
        tenancy["platform"], allow=allow)
    registry, principal = _registry(tenancy)
    tenant = TenantPlatform(
        platform, registry, principal, workspace_id=tenancy["workspace"],
        usage_of=tenancy["usage_of"])
    run_id = platform.submit()
    report = tenant.run_cycle(
        run_id, tenancy["worker_id"], tokens=tenancy["tokens"],
        cost=tenancy["cost"], step=callbacks["step"],
        evidence=callbacks["evidence"], capability=callbacks["capability"],
        validation=callbacks["validation"], anchors=callbacks["anchors"])
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "tenant": tenancy["tenant"],
        "workspace": tenancy["workspace"],
        "principal": tenancy["principal"],
        "report": report.to_dict(),
        "next": [
            "inspect report.usage for committed tokens/cost",
            "agentdescent ops publish --runs-db <runs_db> ...",
        ],
    }
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def tenant_init(out: Any, *, demo: bool = False) -> Dict[str, Any]:
    target = os.path.abspath(os.path.expanduser(require_text(out, "out")))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(target))[0]
    if demo:
        platform_spec = {
            "runs_db": os.path.join(directory, f"{stem}-runs.sqlite3"),
            "platform": {
                "tenant": "demo", "workspace": "default",
                "max_generations": 3, "target": 0.75,
            },
            "campaign": {"max_cycles": 1, "target": 0.75, "patience": 1},
            "compute": {
                "models": [{"name": "local", "cost": 0.0}],
                "options": [{"name": "direct", "action": "direct", "calls": 1}],
                "cost_weight": 0.0, "exploration": 0.0,
            },
            "step": "agentdescent.surface_platform:demo_step",
        }
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "platform": platform_spec,
            "tenant": "demo",
            "workspace": "default",
            "principal": "operator",
            "roles": ["operator"],
            "quota": {"max_tokens": 1000, "max_cost": 10.0,
                      "max_concurrent": 1},
            "tokens": 50,
            "cost": 1.0,
        }
    else:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "platform": {
                "runs_db": os.path.join(directory, f"{stem}-runs.sqlite3"),
                "platform": {"tenant": "t1", "workspace": "default",
                             "max_generations": 3},
                "step": "your_package.steps:step",
            },
            "tenant": "t1",
            "workspace": "default",
            "principal": "operator",
            "roles": ["operator"],
            "quota": {"max_tokens": 10000, "max_cost": 100.0,
                      "max_concurrent": 2},
            "tokens": 100,
            "cost": 1.0,
        }
    atomic_write_json(target, spec)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": target,
        "demo": demo,
        "next": [
            f"agentdescent tenant plan --spec {target}",
            f"agentdescent tenant run  --spec {target} --json",
        ],
    }
