"""The ``agentdescent governed-harness`` surface: one governed self-modification.

`governed_harness.govern_harness_change` already composes the safety gate, proof
runner, human approval store, canary/rollback plan and promotion decision. This
module makes that path usable from a JSON spec, the CLI and the MCP server.

A spec carries the modification, the paired scores, and the gates:

    {
      "artifact_id": "harness-1",
      "modification": {
        "modification_id": "m1",
        "source": "def f(x):\\n    return x + 1\\n",
        "target": "harness/f.py",
        "risk": 0.05
      },
      "before": "before.json",
      "after": "after.json",
      "gate": {"max_risk": 0.5, "canary_risk": 0.2},
      "prover": "mypkg.wiring:prover",
      "approval_store": "approvals.jsonl",
      "rules": {"min_effect": 0.01},
      "contaminated": false,
      "risk": 0.05,
      "rollback_current": {"harness/f.py": "new"},
      "rollback_previous": {"harness/f.py": "old"}
    }

`agentdescent governed-harness init --demo --out governed.json` writes a
runnable offline example. `plan` validates every ref and loads the scores without
running the gates; `run` returns the full `GovernedHarnessReport`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Sequence

from .approval import ApprovalRequest, ApprovalStore
from .governed_harness import GovernedHarnessReport, govern_harness_change
from .promotion import PromotionRules
from .proofs import ProofRunner
from .release_surface import load_rules, load_scores
from .safe_selfmod import ModificationSpec, SafeSelfModifier
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, load_json,
    parse_csv, require_instance, require_mapping, require_text,
    resolve_builder,
)

__all__ = [
    "governed_init",
    "governed_plan",
    "governed_run",
    "approvals_list",
    "approvals_decide",
]

_APPROVAL_STATUSES = ("pending", "approved", "rejected", "all")


def _approval_store(value: Any) -> ApprovalStore:
    store = _approval(value)
    if store is None:
        raise SurfaceError("an approval store path is required")
    return store


def _approval_row(request: ApprovalRequest) -> Dict[str, Any]:
    return {
        "key": request.key,
        "artifact_id": request.artifact_id,
        "reason": request.reason,
        "status": request.status,
        "created_at": request.created_at,
        "decided_at": request.decided_at,
        "note": request.note,
        "summary": request.summary,
    }


def _approval_counts(store: ApprovalStore) -> Dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0, "total": 0}
    for request in store.all():
        counts[request.status] += 1
        counts["total"] += 1
    return counts


def approvals_list(store: Any, *, status: Optional[str] = None
                   ) -> Dict[str, Any]:
    """List approval requests, newest decisions included.

    ``status`` is ``pending`` (the default), ``approved``, ``rejected`` or
    ``all``. Read-only: a missing store is an empty queue, not a file to create.
    """
    approval_store = _approval_store(store)
    wanted = "pending" if status is None else require_text(status, "status")
    if wanted not in _APPROVAL_STATUSES:
        raise SurfaceError(
            f"status must be one of {', '.join(_APPROVAL_STATUSES)}")
    requests = approval_store.all()
    rows = [request for request in requests
            if wanted == "all" or request.status == wanted]
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": approval_store.path,
        "status": wanted,
        "approvals": [_approval_row(request) for request in rows],
        "counts": _approval_counts(approval_store),
        "next": (
            ["agentdescent governed-harness approve --store <store> --key <key>",
             "agentdescent governed-harness reject  --store <store> --key <key>"]
            if wanted == "pending" and rows else []),
    }


def approvals_decide(store: Any, key: Any, *, approve: bool,
                     note: str = "") -> Dict[str, Any]:
    """Approve or reject one pending request; an unknown key is an input error.

    Deliberately CLI-only: a host model must not be able to approve the change
    it proposed for itself.
    """
    approval_store = _approval_store(store)
    key = require_text(key, "key")
    if not isinstance(approve, bool):
        raise SurfaceError("approve must be a bool")
    if not isinstance(note, str):
        raise SurfaceError("note must be a string")
    existing = approval_store.status(key)
    if existing is None:
        raise SurfaceError(f"no approval request for key {key!r}")
    if existing != "pending":
        raise SurfaceError(
            f"approval request {key!r} is already {existing}; only pending "
            "requests can be decided")
    approval_store.decide(key, approve=approve, note=note)
    request = approval_store._requests()[key]
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "store": approval_store.path,
        "approval": _approval_row(request),
        "counts": _approval_counts(approval_store),
    }



def _label(value: Any) -> str:
    return (os.path.abspath(os.path.expanduser(value))
            if isinstance(value, str) else "<inline>")


def _modification(value: Any) -> ModificationSpec:
    data = require_mapping(value, "modification")
    allowed = {field.name for field in fields(ModificationSpec)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SurfaceError(
            "unknown modification key(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(sorted(allowed)))
    try:
        return ModificationSpec(**data)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid modification: {error}") from None


def _gate(value: Any, *, allow: Optional[Sequence[str]]) -> SafeSelfModifier:
    if value is None:
        raise SurfaceError("governed-harness spec needs a gate")
    if isinstance(value, str):
        return require_instance(
            resolve_builder(value, name="gate", allow=allow),
            SafeSelfModifier, "gate")
    data = require_mapping(value, "gate")
    capabilities = parse_csv(
        data.get("granted_capabilities") or (), name="gate.granted_capabilities")
    try:
        return SafeSelfModifier(
            capabilities,
            max_risk=float(data.get("max_risk", 0.4)),
            canary_risk=float(data.get("canary_risk", 0.1)),
            tolerance=float(data.get("tolerance", 0.0)),
            allowed_imports=data.get("allowed_imports"),
            forbidden_imports=data.get("forbidden_imports"),
            forbidden_calls=data.get("forbidden_calls"))
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid gate: {error}") from None


def _prover(value: Any, *, allow: Optional[Sequence[str]]) -> Optional[ProofRunner]:
    if value is None:
        return None
    return require_instance(
        resolve_builder(value, name="prover", allow=allow),
        ProofRunner, "prover")


def _approval(value: Any) -> Optional[ApprovalStore]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = require_mapping(value, "approval_store").get("path")
    path = require_text(value, "approval_store")
    return ApprovalStore(os.path.abspath(os.path.expanduser(path)))


def _rollback(value: Any, name: str) -> Optional[Mapping[str, str]]:
    if value is None:
        return None
    data = load_json(value, name=name)
    rows = require_mapping(data, name)
    for path, content in rows.items():
        if not isinstance(path, str) or not path:
            raise SurfaceError(f"{name} keys must be non-empty strings")
        if not isinstance(content, str):
            raise SurfaceError(f"{name} values must be strings")
    return rows


def _rules(value: Any) -> Optional[PromotionRules]:
    if value is None:
        return None
    return load_rules(value)


def load_spec(value: Any) -> Dict[str, Any]:
    payload = load_json(value, name="governed-harness spec")
    spec = require_mapping(payload, "governed-harness spec")
    for key in ("artifact_id", "modification", "before", "after", "gate"):
        if key not in spec:
            raise SurfaceError(f"governed-harness spec needs {key!r}")
    return spec


def _build(spec: Mapping[str, Any], *, allow: Optional[Sequence[str]] = None
           ) -> Dict[str, Any]:
    artifact_id = require_text(spec.get("artifact_id"), "artifact_id")
    modification = _modification(spec.get("modification"))
    before = load_scores(spec.get("before"), name="before")
    after = load_scores(spec.get("after"), name="after")
    gate = _gate(spec.get("gate"), allow=allow)
    prover = _prover(spec.get("prover"), allow=allow)
    approval = _approval(spec.get("approval_store"))
    rules = _rules(spec.get("rules"))
    contaminated = spec.get("contaminated", False)
    if not isinstance(contaminated, bool):
        raise SurfaceError("contaminated must be a bool")
    risk = spec.get("risk", modification.risk)
    if isinstance(risk, bool) or not isinstance(risk, (int, float)):
        raise SurfaceError("risk must be a number")
    kind = require_text(spec.get("kind") or "harness", "kind")
    return {
        "artifact_id": artifact_id,
        "modification": modification,
        "before": before,
        "after": after,
        "gate": gate,
        "prover": prover,
        "approval": approval,
        "rules": rules,
        "contaminated": contaminated,
        "risk": float(risk),
        "kind": kind,
        "approval_key": spec.get("approval_key"),
        "approval_reason": spec.get("approval_reason") or "high-risk harness change",
        "rollback_current": _rollback(spec.get("rollback_current"), "rollback_current"),
        "rollback_previous": _rollback(spec.get("rollback_previous"), "rollback_previous"),
    }


def governed_plan(path: Any, *, allow: Optional[Sequence[str]] = None
                  ) -> Dict[str, Any]:
    spec = load_spec(path)
    built = _build(spec, allow=allow)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "artifact_id": built["artifact_id"],
        "tasks": len(built["before"]),
        "gate": type(built["gate"]).__name__,
        "prover": None if built["prover"] is None else "configured",
        "approval_store": (None if built["approval"] is None
                           else built["approval"].path),
        "rules": (None if built["rules"] is None
                  else asdict(built["rules"])),
        "next": ["agentdescent governed-harness run --spec <spec.json>"],
    }


def governed_run(path: Any, *, out: Optional[str] = None,
                 allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    spec = load_spec(path)
    built = _build(spec, allow=allow)
    report: GovernedHarnessReport = govern_harness_change(
        built["artifact_id"], built["modification"], built["before"],
        built["after"], gate=built["gate"], rules=built["rules"],
        prover=built["prover"], approval=built["approval"],
        approval_key=built["approval_key"],
        approval_reason=built["approval_reason"],
        rollback_current=built["rollback_current"],
        rollback_previous=built["rollback_previous"],
        kind=built["kind"], contaminated=built["contaminated"],
        risk=built["risk"])
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "promoted": report.promoted,
        "report": report.to_dict(),
        "next": (
            ["apply the change through the harness materialiser, then canary it"]
            if report.promoted else
            ["inspect report.verdict, report.safety and report.decision"]),
    }
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def governed_init(out: Any, *, demo: bool = False) -> Dict[str, Any]:
    target = os.path.abspath(os.path.expanduser(require_text(out, "out")))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(target))[0]
    before_path = os.path.join(directory, f"{stem}-before.json")
    after_path = os.path.join(directory, f"{stem}-after.json")
    if demo:
        before = {f"t{i}": 0.5 for i in range(20)}
        after = {f"t{i}": 0.7 for i in range(20)}
        atomic_write_json(before_path, before)
        atomic_write_json(after_path, after)
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "artifact_id": "demo-harness",
            "modification": {
                "modification_id": "demo-m1",
                "source": "def demo_harness(value):\n    return value + 1\n",
                "target": "harness/demo.py",
                "risk": 0.0,
            },
            "before": before_path,
            "after": after_path,
            "gate": {"max_risk": 0.5, "canary_risk": 0.2},
            "contaminated": False,
            "risk": 0.0,
            "rollback_current": {"harness/demo.py": "return value + 2"},
            "rollback_previous": {"harness/demo.py": "return value"},
        }
    else:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "artifact_id": "harness-1",
            "modification": {
                "modification_id": "m1",
                "source": "your new harness source",
                "target": "harness/f.py",
                "requested_capabilities": [],
                "risk": 0.05,
            },
            "before": before_path,
            "after": after_path,
            "gate": {"max_risk": 0.5, "canary_risk": 0.2},
            "prover": "your_package.wiring:prover",
            "approval_store": os.path.join(directory, "approvals.jsonl"),
            "rules": {"min_effect": 0.01},
            "contaminated": False,
            "risk": 0.05,
        }
    atomic_write_json(target, spec)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": target,
        "before": before_path,
        "after": after_path,
        "demo": demo,
        "next": [
            (f"write the before/after score files {before_path} and {after_path}"
             if not demo else f"before/after scores: {before_path}, {after_path}"),
            f"agentdescent governed-harness plan --spec {target}",
            f"agentdescent governed-harness run  --spec {target} --json",
        ],
    }
