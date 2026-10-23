Added
`CapabilityGuard`, an acceptance policy that protects capabilities outside the
task being evolved: it scores the base artifact and the candidate on named task
suites and rejects a self-modification that misses a suite's absolute floor or
falls below the base by more than its tolerance, then delegates to the inner
acceptance policy. This is the anti-catastrophic-forgetting gate for recursive
self-improvement rather than a second verifier.
