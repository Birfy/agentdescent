"""The file-backed registry preserves the in-memory registry's snapshot."""

import json

import pytest

from agentdescent.workspace_store import PersistentWorkspaceRegistry
from agentdescent.workspaces import (
    Principal, Quota, Role, Workspace, WorkspaceRegistry,
)


def _registry():
    registry = WorkspaceRegistry()
    registry.add_workspace(Workspace("t1", "ws", "Research",
                                     Quota(max_tokens=10, max_cost=2.0)))
    registry.add_principal(Principal("u1", "t1", {"ws": (Role.OWNER,)}))
    reservation = registry.reserve("u1", "ws", tokens=3, cost=1.0)
    registry.commit("u1", reservation.reservation_id,
                    tokens_used=2, cost_used=0.5)
    return registry


def test_registry_snapshot_round_trips():
    registry = _registry()
    payload = registry.to_dict()
    json.dumps(payload)
    restored = WorkspaceRegistry.from_dict(payload)
    assert restored.workspace_exists("t1", "ws")
    assert restored.principal_exists("u1")
    assert restored.to_dict()["audit"] == payload["audit"]
    assert restored.usage("u1", "t1", "ws").to_dict() == {
        "tokens": 2.0, "cost": 0.5, "active": 0}


def test_snapshot_rejects_unknown_schema():
    with pytest.raises(ValueError, match="schema_version"):
        WorkspaceRegistry.from_dict({"schema_version": 99})


def test_persistent_registry_survives_reload(tmp_path):
    path = tmp_path / "registry.json"
    first = PersistentWorkspaceRegistry(str(path))
    assert isinstance(first, WorkspaceRegistry)
    first.add_workspace(Workspace("t1", "ws", "Research",
                                  Quota(max_tokens=10)))
    first.add_principal(Principal("u1", "t1", {"ws": (Role.OPERATOR,)}))
    reservation = first.reserve("u1", "ws", tokens=4, cost=0.0)
    first.commit("u1", reservation.reservation_id,
                 tokens_used=3, cost_used=0.0)

    second = PersistentWorkspaceRegistry(str(path))
    assert second.workspace_exists("t1", "ws")
    assert second.principal_exists("u1")
    assert second.usage("u1", "t1", "ws").tokens == 3
    assert second.snapshot()["audit"]


def test_malformed_file_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not readable"):
        PersistentWorkspaceRegistry(str(path))
