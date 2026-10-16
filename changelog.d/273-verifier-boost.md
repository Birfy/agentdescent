Added
Verifier bootstrap flywheel (`agentdescent/audit/boost.py`, #273):
`VerifierBoost` wraps any verifier and learns a monotone calibration curve
(`isotonic_fit`, numpy-only PAV) from the audit's **IMPROVEMENT** pool, then
activates it **only if it provably reduces residual error on a held-out slice
of that pool** (the bootstrap gate). When active it corrects the cheap and
learned layers the aggregator uses to rank candidates and the audit uses for
its trust signal; `eval_counts` / `full_eval` are delegated **raw**, so the
commit gate keeps the oracle's authority over what commits. It trains on the
IMPROVEMENT pool only (never the CALIBRATION pool that measures it), filters
to the current verifier version, and declares itself as a new verifier version
(`version_tag`) so `attach(boost=...)` / `Audit.recompute()` refresh the audit
fingerprint and `VerifierWatch` withdraws the old calibration. Default
behaviour unchanged -- no boost means no fitting and no correction.
