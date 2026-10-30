# Trusted promotion — one fail-closed answer to "may this ship?"

Evaluation, statistics, proofs, canaries and policy admission exist as separate
modules. `promote()` collects their reports into one decision and refuses to
guess.

```python
from agentdescent import (
    PromotionRequest, PromotionRules, ProofEvidence, promote,
)

decision = promote(PromotionRequest(
    artifact_id="skill-1",
    artifact_hash=proof.certificate.artifact_hash,
    eval_result=eval_result,          # agentdescent.evalops.EvalResult
    contaminated=False,
    conformal=conformal_report,       # agentdescent.conformal.ConformalReport
    sequential=sequential_report,     # agentdescent.sequential.SequentialReport
    canary=canary_result,             # agentdescent.safety_controller.CanaryResult
    proof=ProofEvidence(certificate, verification),
    admission=admission,              # agentdescent.policy_release.AdmissionResult
    risk=0.2,
    approved=True,
), PromotionRules(require_conformal=True, require_proof=True,
                  require_admission=True))

decision.action, decision.reasons, decision.decision_hash
```

Assume the reports come from `evalops`, `conformal`, `sequential`,
`safety_controller.canary`, `proofs` and `policy_release`; this module consumes
them and never re-runs them.

## The decision lattice

| Evidence state | Action |
| --- | --- |
| a required report is missing | `defer` |
| a gate fails, a proof is invalid, risk is too high, approval was denied | `reject` |
| a canary says roll back | `rollback` |
| every required gate is green | `promote` |

`rollback` outranks the other failures: a measured regression is an incident,
not a pending approval. An admission whose signature is invalid is a hard
`reject`.

Two approval controls are separate:

- `admission.needs_approval` (the policy effect `require_approval`) defers the
  decision regardless of the local `require_approval` flag;
- `require_approval` governs `request.approved`: `None` defers, `True` passes,
  and an explicit `False` is a hard `reject` **even when the flag is off**.
  A caller that supplies "no" is not asking for the default.

## Rules, not a feature list

`PromotionRules` says what must be present and the bars it must clear:

- `require_eval`, `require_contamination`, `require_risk`,
  `require_conformal`, `require_sequential`, `require_canary`,
  `require_proof`, `require_admission`, `require_approval`;
- `min_effect`, `min_tasks`, `max_risk`, `allow_contaminated`.

`require_risk` defaults to `True`, so a request without `risk` defers by
default. Supplied evidence is judged even when its `require_*` flag is off: a
supplied canary rollback or invalid proof still rejects. A flag only decides
whether the absence of that evidence defers the decision.

## Evidence is bound to the artifact

A green certificate for `skill-1` must not promote `skill-EVIL`:

- `proof.certificate.artifact_id` must equal `request.artifact_id` (always);
- if `artifact_hash` is supplied, the proof certificate and the admission
  manifest hash must equal it.

A **mismatched** binding is a `reject`, not a warning. Hash binding only applies
when `artifact_hash` is supplied, so always pass it when a proof or admission is
present. The decision is then safe to carry into a manifest or a lineage
record.

## Two hashes

- `evidence_hash` identifies the **inputs** (the reports, risk, approval,
  metadata). The same evidence always hashes the same.
- `decision_hash` covers `evidence_hash` + `asdict(PromotionRules)` + `action`.
  Two different rule sets that reach different verdicts cannot share a decision
  hash.

`PromotionDecision.to_dict()` is JSON-serialisable and is what a decision log
should store.

## What it does not do

It does not re-run the gates. `evalops`, `conformal`, `sequential`, `canary`,
`proofs` and `policy_release` produce the reports; this module owns the
ordering, the missing-evidence policy and the audit record. That keeps a
non-deterministic judge or a billable verifier from being called twice.
