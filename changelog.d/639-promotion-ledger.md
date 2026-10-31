Added
`PromotionLedger`: only an accepted `promote()` decision becomes a release. It
builds an `ArtifactManifest` carrying the evidence/decision hashes and reasons,
HMAC-signs it, verifies the signature, and writes a lineage node with the parent
artifacts, run id, manifest hash and key id. Refused promotions are not signed.
