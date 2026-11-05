Added
`agentdescent tenant report --registry FILE`: a read-only view of a persistent
workspace registry with per-workspace quota and accumulated usage, principals
and their roles, and the most recent audit events when `--principal` names an
actor with `audit:read`. A missing registry is an input error and is never
created; a viewer or operator gets `audit: null` with a note. The command is
CLI-only so the local audit trail is not exposed to a host model. The report also makes stale `active` reservations visible, which a hard kill can otherwise leave behind.
