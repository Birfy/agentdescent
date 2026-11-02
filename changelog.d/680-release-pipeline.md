Added
`release_evidence` and `release_candidate` compose trust evidence, promotion
and the `PromotionLedger` into one fail-closed release seam. An accepted
promotion is signed and written to lineage; a reject or defer returns a report
without touching the ledger. An accepted promotion requires an artifact hash (`artifact_hash=` or
`content_hash=`); missing signing material or a missing/mismatched artifact
hash is refused rather than silently signed (#680).
