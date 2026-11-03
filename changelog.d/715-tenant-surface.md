Added
`agentdescent tenant` and the MCP `tenant` tool: one platform cycle under a
`WorkspaceRegistry` principal/workspace/roles/quota. The spec nests the normal
platform spec plus tenancy fields and the tokens/cost to reserve; `plan`
validates refs and checks `run:write` without creating the SQLite DB, and `run`
reserves quota, executes the cycle and commits usage through `TenantPlatform`.
A denied role or exhausted quota fails closed. This gives the previously
isolated `workspaces` / `tenant_platform` pair a product consumer. JSON results
carry `schema_version: 1`; MCP refs and writes use the operator allowlists.
