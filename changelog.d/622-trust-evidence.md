Added
`TrustEvidence` + `build_trust_evidence`: one paired evaluation becomes the
eval/bootstrap, canary, conformal, sequential, Goodhart, metamorphic, verifier
and calibration reports in one bundle, ready for `to_promotion_request()`.

Changed
`promotion` now accepts `goodhart`, `metamorphic`, `verifier` and `calibration`
evidence and has matching `require_*` rules: gaming/failed relations/failed
panel/overconfidence reject, suspicious/incomplete evidence defers, and a green
bundle still promotes. The evidence hash covers the new reports. The builder
leaves `risk`/`contaminated` as `None` when not supplied (so the fail-closed
gates defer), and computes eval means/CI/per-task over paired keys only.
