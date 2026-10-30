# Capability foundry — from a synthesized tool to an admitted capability

`tool_synthesis` proposes a tool, `deliberation` reviews it, `safe_selfmod`
gates its authority, `proofs` certifies its invariants and `artifact_graph`
stores it. `CapabilityFoundry` owns the order and the decision to admit.

```python
from agentdescent import (
    CapabilityFoundry, CapabilityRequest, CapabilityStatus,
)
from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.tool_synthesis import ToolSpec, ToolValidation

foundry = CapabilityFoundry(
    SafeSelfModifier({"files:read"}),
    prover=ProofRunner([ProofInvariant("no-network", no_network)]),
    graph=ArtifactGraph(),
)
outcome = foundry.admit(CapabilityRequest(
    gap_id="csv-failures",
    tool=ToolSpec("csv-tool", "parse a CSV line", source,
                  capabilities=("files:read",)),
    validation=ToolValidation(True, 0.9),
    review=debate_report,             # agentdescent.deliberation.DeliberationReport
    modification=ModificationSpec(
        "csv-tool", source, "tool",
        requested_capabilities=("files:read",)),
    before=0.5, after=0.7, canary=0.65,
))
outcome.status, outcome.artifact_id, outcome.reasons
```

## The admission order

1. **Validation.** The tool must be accepted and clear `min_valid_score`.
2. **Review.** A supplied review is always checked: the debate winner must be
   `review_winner`, and agreement below `min_review_agreement` defers the
   candidate. `require_review=False` only stops a *missing* review from
   deferring; a supplied review with the wrong winner still rejects.
3. **Safety.** `SafeSelfModifier.review()` checks syntax, forbidden imports,
   private-attribute access and the requested capabilities; `promote()` then
   rolls back when the full metric (`after < before - tolerance`) **or** the
   canary regresses, and holds when a required canary is missing.
4. **Proof.** `ProofRunner.prove()` runs the invariant suite and
   `verify()` checks the digest and error-severity obligations.
5. **Gain.** `after - before` must clear `min_holdout_gain`.
6. **Registration.** The admitted tool is added to `ArtifactGraph`.

## Binding rules

The gate must review the same artifact that gets registered:

- `modification.modification_id == tool.name`;
- `modification.source == tool.source`;
- every `tool.capabilities` entry is in
  `modification.requested_capabilities`.

A mismatch can never be promoted or registered. It is reported as `rejected`
unless the safety gate also measured a regression, in which case the status is
`rolled_back` (rollback outranks the binding failure); the reasons still name
the mismatch.

## Status lattice

| Evidence state | Status |
| --- | --- |
| a required review/prover is missing | `deferred` |
| review agreement is below the threshold | `deferred` |
| safety says it must hold (a required canary is missing) | `deferred` |
| a review winner, validation, safety, proof or gain fails | `rejected` |
| the tool name is already registered | `rejected` |
| safety measured a full-metric or canary regression | `rolled_back` |
| every gate is green and the tool is registered | `promoted` |

`require_registration` defaults to `True`: a green tool with no artifact graph is
`deferred` with `missing=("graph",)`. Set it `False` only when the caller
registers elsewhere.

The registered artifact metadata carries the validation score and holdout gain,
plus the proof digest and artifact hash when a proof was produced
(`require_proof=True`), so a later audit can see what was checked. With
`require_proof=False` those two metadata fields are `None`.
