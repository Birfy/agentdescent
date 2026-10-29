Added
`PolicyBundle` + `AdmissionController` + signed `ArtifactManifest`: deny-overrides
policy evaluation, approval obligations, canonical manifest hashes, HMAC-SHA256
detached signatures (demo signer; swap for cosign/Sigstore), and fail-closed
admission when the signature is missing or invalid.
