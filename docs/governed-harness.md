# Governed harness change — safety, proof, approval, canary, promotion

`agentdescent governed-harness` is the product surface for a high-risk
self-modification of a harness or code artifact. It composes the existing
seams in order:

```text
SafeSelfModifier gate -> proof invariants -> human approval
    -> trusted evidence (canary/eval) -> promotion decision -> rollback plan
```

It does not apply the change. It answers *may this change be promoted, and what
exactly did it pass?* The caller still writes the artifact through its
materialiser and runs the canary.

## Quickstart

```bash
agentdescent governed-harness init --out governed.json --demo
agentdescent governed-harness plan --spec governed.json
agentdescent governed-harness run  --spec governed.json --json --out receipt.json
```

The demo writes `governed-before.json` and `governed-after.json` and a spec that
promotes offline. `run` exits 0 only when the change is promoted; a defer,
reject or rollback exits 1.

## Spec

```json
{
  "artifact_id": "harness-1",
  "modification": {
    "modification_id": "m1",
    "source": "def f(x):\n    return x + 1\n",
    "target": "harness/f.py",
    "requested_capabilities": [],
    "risk": 0.05
  },
  "before": "before.json",
  "after": "after.json",
  "gate": {"max_risk": 0.5, "canary_risk": 0.2},
  "prover": "mypkg.wiring:prover",
  "approval_store": "approvals.jsonl",
  "approval_key": "optional-prefix",
  "rules": {"min_effect": 0.01},
  "contaminated": false,
  "risk": 0.05,
  "rollback_current": {"harness/f.py": "new"},
  "rollback_previous": {"harness/f.py": "old"}
}
```

- `gate` is an inline `SafeSelfModifier` config (`granted_capabilities`,
  `max_risk`, `canary_risk`, `tolerance`, import/call policies) or a
  `module:attribute` builder/instance.
- `prover` is an optional `ProofRunner` builder/instance.
- `approval_store` is an `ApprovalStore` JSONL path. A pending request defers;
  a rejected request rejects; an approved one proceeds.
- `rules` is a `PromotionRules` object or JSON path. Supplying a prover or
  approval store ORs `require_proof` / `require_approval` in, so a custom rule
  cannot silently ignore a gate you passed.
- `before`/`after` are task -> score objects or JSON paths.
- `rollback_current`/`rollback_previous` are mappings or JSON paths; the report
  carries the non-destructive `rollback_plan`.

## Fail-closed semantics

- A denied capability/syntax check rejects before proof or approval.
- A failed proof rejects and never reaches promotion.
- A pending or rejected human approval never reaches promotion.
- A canary rollback wins over a failed eval and is reported as rollback.
- Optional conformal/sequential *non-decisions* are ignored unless the rules
  require them; a conformal/sequential **reject is never dropped**.

## Approval queue

A pending approval used to stop the workflow with no product entry point to
inspect or decide it. The queue is now a first-class verb:

```bash
agentdescent governed-harness approvals --store approvals.jsonl \
  [--status pending|approved|rejected|all] [--json]
agentdescent governed-harness approve --store approvals.jsonl --key KEY \
  [--note "..."] [--json]
agentdescent governed-harness reject  --store approvals.jsonl --key KEY \
  [--note "..."] [--json]
```

`approvals` lists the requests and returns `counts` per status; a missing store
is an empty queue, not an error. `approve`/`reject` require an existing
**pending** key and write one append-only decision event; the next `run` of the
same spec reads it. An unknown key or an already-decided request is an input
error (exit 2), so a typo or a re-run from shell history cannot silently flip a
rejection into an approval.

The decide verbs are **CLI-only**, with no MCP counterpart: a host model must
not be able to approve the change it proposed for itself. Listing is also
CLI-only so the approval queue does not become a tool the model can poll.

## MCP

The `governed_harness` tool takes the same spec object and `dry_run=true` for
the plan step. Refs are gated by `AGENTDESCENT_SURFACE_ALLOW` (empty refuses
every ref); report writes are confined to `AGENTDESCENT_SURFACE_WRITE_DIRS`.
The tool returns `{"ok": true, "promoted": false, ...}` for a normal
defer/reject; ask the user before applying anything.
