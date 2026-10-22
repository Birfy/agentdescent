Added
Joint multi-slot evolution (`agentdescent/joint_meta.py`): search the whole
decision plane as one artifact. A `PoliciesBundle` holds several `SLOTS`
together; each outer rollout installs **all** the slots on `Policies` and runs
the inner search, so an effect that only appears when two rules change
together is visible to the search (an effect two independently evolved rules
never see). A proposal names one or more slots with `=== <slot> ===` blocks, so
the aggregator's existing union-on-distinct-keys / resolve-on-same-key rules
apply unchanged. `joint_meta_validate` attributes a joint change to each slot by
its **marginal** held-out gain and, with `isolate=True`, rolls back a slot that
regresses while adopting the rest -- so one over-eager rule cannot smuggle a
regression into the archive. An adopted bundle archives one ordinary
`meta-rule` record per slot, so `_archived_rules`, `exam_archive` and
`continuous_meta_run` keep seeding and examining it unchanged.

Based on #388 (P7 multi-slot joint evolution + isolating untrusted rules).
