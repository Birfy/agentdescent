Added
`agentdescent governed-harness approvals|approve|reject`: list pending,
approved or rejected human-approval requests, and record a decision with an
optional note. A missing store is an empty queue; an unknown key or an
already-decided request is an input error, so a rejection cannot be silently
flipped into an approval. The decide verbs stay CLI-only, so a host model cannot
approve the change it proposed for itself, and the next `run` of the same
governed-harness spec reads the decision.
