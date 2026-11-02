Added
`TenantPlatform`: every `RSIPlatform.run_cycle` is now a reserved, authorized
and audited tenant action. A cycle reserves quota at the stated tokens/cost
(which enforces `run:write` and the workspace limits), commits actual usage on
success, and releases the reservation if the platform or commit fails closed.
`TenantCycleReport` exposes the platform report, reservation, usage and a
JSON-safe summary. Viewer denial and quota exhaustion happen before the platform
runs, and a commit above the reservation releases rather than overcharges
(#682).
