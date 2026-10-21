Added
Chain audit (`agentdescent/chain_audit.py`): one adoption, traced through every
evidence ring. `chain_audit` walks an archived rule's rings — governance
(decision plane vs machinery), the recorded decision, the meta-gain oracle
bias (from a `MetaAuditStore`, #426), the self-exam stamp (passed/flagged/
demoted), the inner verifier's residual bias (from an `AuditStore`, #273), and
harness routing evidence — and reports each ring as `verified` (evidence
present and within `bias_threshold`), `flagged` (present and outside it), or
`unverified` (the evidence simply is not there, reported rather than silently
assumed). `ChainAuditReport.verified` is true only when no ring is flagged or
unverified, and the reason names the rings that keep it false. An unknown run
is refused with `KeyError`. The promise "no number in the loop is unverified"
becomes inspectable end to end.
