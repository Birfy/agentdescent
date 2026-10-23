Added
`VerifierCurriculum` and `VerifierAgreement`: a curriculum that selects the
tasks where a verifier disagrees most with the oracle (abs / overestimate /
underestimate), feeds the selected oracle labels back into the verifier, and
reports the gap before/after (one oracle label per task per round); plus an
acceptance policy that audits a sample of commits and refuses a verifier-only
gain or an oracle regression (`oracle-disagrees`). This targets the
Goodhart failure mode of recursive self-improvement without adding a second
verifier.
