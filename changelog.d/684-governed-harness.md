Added
`agentdescent.governed_harness` composes the capability gate, proof runner,
human approval store, canary/rollback plan and promotion rules into one governed
path for harness and code self-modifications. A denied gate, failed proof,
pending or rejected approval never reaches promotion, and approval keys are
content-bound so an old approval cannot be replayed for a different change. The
returned report is JSON-safe for manifests and decision logs (#684).
