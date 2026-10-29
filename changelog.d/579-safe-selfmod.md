Added
`SafeSelfModifier`, a capability gate plus canary/rollback promotion path for
self-modifying code: allow-listed imports, forbidden name/attribute references
(not only calls), inferred files/network/process capabilities, risk tiers, and a
canary measurement before promotion. Static screening is defense-in-depth; a
sandbox is still required for execution.
