"""Tenancy must isolate data, permissions and spend before it is too late."""

import json

import pytest

from agentdescent.workspaces import (
    Permission,
    PermissionDenied,
    Principal,
    Quota,
    Role,
    Workspace,
    WorkspaceRegistry,
)


def _registry():
    reg = WorkspaceRegistry(clock=lambda: 100.0)
    reg.add_workspace(Workspace("t1", "ws1", "Research",
                                Quota(max_tokens=1000, max_cost=10,
                                      max_concurrent=1)))
    reg.add_principal(Principal("owner", "t1", {"*": (Role.OWNER,)}))
    reg.add_principal(Principal("operator", "t1", {"ws1": (Role.OPERATOR,)}))
    reg.add_principal(Principal("approver", "t1", {"ws1": (Role.APPROVER,)}))
    reg.add_principal(Principal("viewer", "t1", {"ws1": (Role.VIEWER,)}))
    reg.add_principal(Principal("outsider", "t2", {"*": (Role.OWNER,)}))
    return reg


def test_role_permissions_and_cross_tenant_denial():
    reg = _registry()
    assert reg.authorize("viewer", "t1", "ws1", Permission.RUN_READ)
    with pytest.raises(PermissionDenied):
        reg.authorize("viewer", "t1", "ws1", Permission.RUN_WRITE)
    assert reg.authorize("operator", "t1", "ws1", Permission.RUN_WRITE)
    assert reg.authorize("approver", "t1", "ws1", Permission.APPROVAL_DECIDE)
    with pytest.raises(PermissionDenied):
        reg.authorize("outsider", "t1", "ws1", Permission.RUN_READ)
    with pytest.raises(PermissionDenied):
        reg.authorize("missing", "t1", "ws1", Permission.RUN_READ)
    assert any(not event.allowed for event in reg.audit_events("owner", tenant_id="t1"))


def test_grant_and_revoke_change_permissions_and_audit():
    reg = _registry()
    reg.grant("owner", "viewer", "ws1", Role.OPERATOR)
    assert reg.authorize("viewer", "t1", "ws1", Permission.RUN_WRITE)
    reg.revoke("owner", "viewer", "ws1", Role.OPERATOR)
    with pytest.raises(PermissionDenied):
        reg.authorize("viewer", "t1", "ws1", Permission.RUN_WRITE)
    actions = [event.action for event in reg.audit_events("owner", tenant_id="t1")]
    assert "role.grant" in actions and "role.revoke" in actions


def test_quota_reservation_commit_release_and_concurrency():
    reg = _registry()
    first = reg.reserve("operator", "ws1", tokens=400, cost=4)
    assert reg.usage("operator", "t1", "ws1").active == 1
    with pytest.raises(PermissionDenied):
        reg.reserve("operator", "ws1", tokens=100, cost=1)
    usage = reg.commit("operator", first.reservation_id, tokens_used=350, cost_used=3.5)
    assert usage.tokens == 350 and usage.cost == 3.5 and usage.active == 0
    # Idempotent commit does not charge twice.
    again = reg.commit("operator", first.reservation_id, tokens_used=999, cost_used=999)
    assert again.tokens == 350

    second = reg.reserve("operator", "ws1", tokens=400, cost=4)
    released = reg.release("operator", second.reservation_id)
    assert released.active == 0 and released.tokens == 350
    assert reg.release("operator", second.reservation_id).active == 0


def test_quota_limits_tokens_cost_and_concurrency():
    reg = _registry()
    with pytest.raises(PermissionDenied):
        reg.reserve("operator", "ws1", tokens=1001, cost=0)
    with pytest.raises(PermissionDenied):
        reg.reserve("operator", "ws1", tokens=0, cost=11)
    first = reg.reserve("operator", "ws1", tokens=100, cost=1)
    with pytest.raises(PermissionDenied):
        reg.reserve("operator", "ws1", tokens=100, cost=1)
    reg.release("operator", first.reservation_id)


def test_audit_export_is_jsonl_or_dicts():
    reg = _registry()
    reg.authorize("viewer", "t1", "ws1", Permission.RUN_READ)
    text = reg.export_audit("owner", tenant_id="t1")
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows and rows[-1]["action"] == "authorize"
    assert reg.export_audit("owner", tenant_id="t1", fmt="dicts") == tuple(rows)
    with pytest.raises(ValueError):
        reg.export_audit("owner", tenant_id="t1", fmt="xml")


def test_duplicates_and_invalid_records_are_refused():
    reg = _registry()
    with pytest.raises(ValueError):
        reg.add_workspace(Workspace("t1", "ws1", "duplicate"))
    with pytest.raises(ValueError):
        reg.add_principal(Principal("owner", "t1", {}))
    with pytest.raises(ValueError):
        Principal("u", "t", {"ws": ("magic",)})
    with pytest.raises(ValueError):
        Quota(max_tokens=-1)
    with pytest.raises(ValueError):
        Quota(max_cost=float("nan"))
    with pytest.raises(ValueError):
        Workspace("", "ws", "name")
    with pytest.raises(KeyError):
        reg.reserve("missing", "ws1", tokens=1, cost=1)


def test_roles_are_deep_frozen_after_the_principal_is_added():
    reg = WorkspaceRegistry(clock=lambda: 1.0)
    reg.add_workspace(Workspace("t1", "ws1", "W"))
    reg.add_principal(Principal("owner", "t1", {"*": ("owner",)}))
    roles = {"ws1": ("viewer",)}
    reg.add_principal(Principal("u", "t1", roles))
    roles["ws1"] = ("owner",)
    assert Permission.RUN_WRITE not in reg.permissions("u", "ws1")


def test_audit_and_usage_are_tenant_and_permission_scoped():
    reg = _registry()
    reg.authorize("viewer", "t1", "ws1", Permission.RUN_READ)
    with pytest.raises(PermissionDenied):
        reg.export_audit("viewer", tenant_id="t1")
    with pytest.raises(PermissionDenied):
        reg.usage("viewer", "t1", "ws1")
    with pytest.raises(PermissionDenied):
        reg.export_audit("owner", tenant_id="t2")
    rows = reg.export_audit("owner", tenant_id="t1", fmt="dicts")
    assert rows and all(row["tenant_id"] == "t1" for row in rows)


def test_tenant_audit_export_only_covers_workspaces_with_audit_read():
    reg = WorkspaceRegistry(clock=lambda: 1.0)
    reg.add_workspace(Workspace("t1", "ws1", "W1"))
    reg.add_workspace(Workspace("t1", "ws2", "W2"))
    reg.add_principal(Principal("a1", "t1", {"ws1": (Role.OWNER,)}))
    reg.add_principal(Principal("a2", "t1", {"ws2": (Role.OWNER,)}))
    reg.authorize("a1", "t1", "ws1", Permission.RUN_READ)
    reg.reserve("a2", "ws2", tokens=1, cost=0.1)
    events = reg.audit_events("a1", tenant_id="t1")
    assert {event.workspace_id for event in events} == {"ws1"}
    with pytest.raises(PermissionDenied):
        reg.audit_events("missing", tenant_id="t1")


def test_commit_cannot_exceed_reservation_and_actor_is_checked():
    reg = _registry()
    reservation = reg.reserve("operator", "ws1", tokens=10, cost=1)
    with pytest.raises(PermissionDenied):
        reg.commit("operator", reservation.reservation_id, tokens_used=11, cost_used=1)
    with pytest.raises(PermissionDenied):
        reg.commit("viewer", reservation.reservation_id, tokens_used=10, cost_used=1)
    reg.commit("operator", reservation.reservation_id, tokens_used=10, cost_used=1)


def test_set_quota_requires_quota_write_and_is_audited():
    reg = _registry()
    with pytest.raises(PermissionDenied):
        reg.set_quota("operator", "t1", "ws1", Quota(max_cost=2))
    updated = reg.set_quota("owner", "t1", "ws1", Quota(max_cost=2))
    assert updated.quota.max_cost == 2
    assert any(e.action == "quota.set" for e in reg.audit_events("owner", tenant_id="t1"))
