"""A file-backed :class:`WorkspaceRegistry` for the tenant surface.

The CLI/MCP `tenant` surface creates a fresh registry per invocation, so quota
and audit do not accumulate across runs. This adapter subclasses the tested
in-memory registry and persists a JSON snapshot atomically after every
mutation. The registry remains the owner of the RBAC/quota rules; this module
only owns durability.

    registry = PersistentWorkspaceRegistry("~/.agentdescent/workspaces.json")
    registry.reserve("operator", "default", tokens=100, cost=1.0)
    # the reservation and its audit event are on disk

Use `WorkspaceRegistry.to_dict` / `.from_dict` for the snapshot format. The file
is not a multi-process database: concurrent writers need their own lock or a
real database adapter, just as the in-memory registry is a per-process slice.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

from .workspaces import (
    Principal, Quota, Workspace, WorkspaceRegistry,
)

__all__ = ["PersistentWorkspaceRegistry"]


class PersistentWorkspaceRegistry(WorkspaceRegistry):
    """A ``WorkspaceRegistry`` whose mutations are persisted to one JSON file."""

    def __init__(self, path: str, *, clock: Optional[Any] = None) -> None:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        super().__init__(clock=clock)
        self.path = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                restored = WorkspaceRegistry.from_dict(payload, clock=clock)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"workspace registry {path!r} is not readable: {error}"
                ) from None
            self._workspaces = restored._workspaces
            self._principals = restored._principals
            self._usage = restored._usage
            self._reservations = restored._reservations
            self._audit = restored._audit

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def snapshot(self) -> dict:
        return self.to_dict()

    def save(self) -> None:
        """Persist the current snapshot atomically."""
        with self._lock:
            payload = self.to_dict()
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".agentdescent-workspaces-", dir=directory, text=True)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True,
                              ensure_ascii=False, default=str)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise

    # -- mutators: persist only after the registry accepted the change --------

    def add_workspace(self, workspace: Workspace) -> Workspace:
        result = super().add_workspace(workspace)
        self.save()
        return result

    def add_principal(self, principal: Principal) -> Principal:
        result = super().add_principal(principal)
        self.save()
        return result

    def grant(self, actor: str, principal_id: str, workspace_id: str,
              role: str) -> Principal:
        result = super().grant(actor, principal_id, workspace_id, role)
        self.save()
        return result

    def revoke(self, actor: str, principal_id: str, workspace_id: str,
               role: str) -> Principal:
        result = super().revoke(actor, principal_id, workspace_id, role)
        self.save()
        return result

    def set_quota(self, actor: str, tenant_id: str, workspace_id: str,
                  quota: Quota) -> Workspace:
        result = super().set_quota(actor, tenant_id, workspace_id, quota)
        self.save()
        return result

    def reserve(self, principal_id: str, workspace_id: str, *,
                tokens: float, cost: float) -> Any:
        result = super().reserve(
            principal_id, workspace_id, tokens=tokens, cost=cost)
        self.save()
        return result

    def commit(self, actor: str, reservation_id: str, *, tokens_used: float,
               cost_used: float) -> Any:
        result = super().commit(
            actor, reservation_id, tokens_used=tokens_used,
            cost_used=cost_used)
        self.save()
        return result

    def release(self, actor: str, reservation_id: str) -> Any:
        result = super().release(actor, reservation_id)
        self.save()
        return result
