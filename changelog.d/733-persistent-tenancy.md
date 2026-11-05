Added
`PersistentWorkspaceRegistry` and the tenant spec's optional `registry` path:
the tested in-memory `WorkspaceRegistry` can now be saved as one atomically
replaced JSON snapshot, so quota usage, reservations and audit accumulate across
`agentdescent tenant` runs instead of being discarded with the process.
`WorkspaceRegistry` gains `to_dict`/`from_dict` and small existence/getters
(`workspace_exists`, `principal_exists`, `get_workspace`, `get_principal`); the
tenant surface uses the persistent registry when `registry` is set and keeps the
in-memory behavior when it is not. The file is authoritative: when it already
has a workspace or principal, `plan`/`run` compare the persisted quota and roles
with the spec and refuse a mismatch instead of silently using either value. The
file is a single-writer JSON slice, not a multi-process database.
