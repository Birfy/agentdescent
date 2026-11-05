# Tenant — one platform cycle under roles, quota and audit

`agentdescent tenant` puts the existing `WorkspaceRegistry` around one platform
cycle: the principal must have `run:write` in the workspace, the cycle reserves
the stated tokens/cost, and the usage is committed through the audited quota
path. It composes `WorkspaceRegistry` + `Principal`/`Workspace`/`Quota` +
`TenantPlatform` + the platform spec.

## Quickstart

```bash
agentdescent tenant init --out tenant.json --demo
agentdescent tenant plan --spec tenant.json
agentdescent tenant run  --spec tenant.json --json --out report.json
```

The demo runs offline on the platform demo step and commits its reserved
tokens/cost.

## Spec

```json
{
  "platform": {
    "runs_db": "tenant-runs.sqlite3",
    "platform": {"tenant": "t1", "workspace": "default",
                 "max_generations": 3},
    "step": "mypkg.steps:step"
  },
  "tenant": "t1",
  "workspace": "default",
  "principal": "operator",
  "roles": ["operator"],
  "quota": {"max_tokens": 10000, "max_cost": 100.0, "max_concurrent": 2},
  "tokens": 100,
  "cost": 1.0
}
```

- The nested `platform` object is exactly the `agentdescent platform` spec.
- `tenant`/`workspace` must match `platform.platform`; the principal belongs to
  that tenant and gets `roles` on that workspace.
- `roles` is a comma-separated list or JSON array; `operator` is the minimum for
  a run, `viewer` is refused before the platform starts.
- `tokens`/`cost` are the reservation (the prepaid charge); `usage_of` is an
  optional `module:attribute` callable returning actual `(tokens, cost)`.
- `quota` is a `Quota` object: `max_tokens`, `max_cost`, `max_concurrent`
  (zero means unlimited).
- `registry` is an optional JSON path. When set, quota usage, reservations and
  audit persist across runs through `PersistentWorkspaceRegistry`; when absent,
  the registry is in-memory and per-invocation. The file is an atomic JSON
  snapshot, not a multi-process database; serialize concurrent writers or use a
  real database adapter.

`plan` validates the nested platform refs and checks `run:write` without
creating the SQLite DB. `run` exits 1 only for an incomplete cycle; a denied
role or exhausted quota is an input error (exit 2). On MCP, refs use
`AGENTDESCENT_SURFACE_ALLOW` and `runs_db`/`out` writes use
`AGENTDESCENT_SURFACE_WRITE_DIRS`.

With `registry` set, the file is authoritative. Whenever the file already has
the workspace, `plan` and `run` compare its quota; whenever it already has the
principal, they compare that principal's roles in the workspace. A mismatch is
refused (exit 2) instead of silently using the file's value, including when only
one half already exists: adding a new workspace for an existing principal needs
a matching role entry in the file, or a new registry file. `plan` reads the
persisted usage when the workspace exists, so it predicts what `run` will see.

Roles are asserted by the spec, not authenticated: the `viewer` refusal catches a
misconfigured spec, not a caller (including an MCP model) that writes
`"roles": ["owner"]`. With `registry` set, quota and audit accumulate across
runs; without it, the registry is in-memory and per-invocation. `tokens` and
`cost` are required explicitly, so a typo cannot turn a metered cycle into a
free one. For real multi-process tenancy, replace the JSON snapshot with the
persistent adapter described in `docs/governance.md`.
