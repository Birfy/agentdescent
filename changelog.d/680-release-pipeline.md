Added
`release_evidence` and `release_candidate` compose trust evidence, promotion
and the `PromotionLedger` into one fail-closed release seam. An accepted
promotion is signed and written to lineage; a reject or defer returns a report
without touching the ledger. Missing signing material is refused (#680).
