Added
`ApprovalGate` + `ApprovalStore`: high-risk self-modifications (blast radius or
diff size above a threshold) are not committed; a request is appended to a JSONL
store, and a human approval lets the same **content** through while a rejection
stays sticky. The key fingerprints target, ops and base version instead of the
producer's `diff_id`, `needs-approval` / `approval-rejected` are real
`MergeOutcome` members, requests carry a reviewable summary, and malformed store
lines are skipped. Low-risk candidates delegate to the inner acceptance policy.
