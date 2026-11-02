Added
`TenantPlatform` wraps `RSIPlatform.run_cycle` so each cycle routed through it
is a reserved, authorized and audited tenant action. A cycle reserves quota at
the stated tokens/cost (which enforces `run:write` and the workspace limits)
and commits actual usage on success. An overrun or a recorded generation error
is charged the reserved amount before the refusal, so repeated overruns still
consume quota; only a platform call that produced no generation is released
without charge. `TenantCycleReport` exposes the platform report, reservation,
usage and a JSON-safe summary (#682).
