# Approval gate — high-risk self-modifications wait for a human

Automatic gates are good at local edits and bad at judging a change that
rewrites the harness, the verifier or a large part of the artifact.
`ApprovalGate` turns that risk into a file-based queue: a high-risk candidate is
not committed, a request is written to a JSONL store, and a human approves or
rejects it out of band. A later merge of the same **content** reads the decision.

```python
from agentdescent import ApprovalGate, ApprovalStore, Policies

store = ApprovalStore("~/.agentdescent/approvals.jsonl")
policies = Policies(acceptance=ApprovalGate(store, risk_threshold=0.5))
```

- **Risk** is the artifact's `blast_radius >= risk_threshold` (L1/L0 governance)
  or `len(diff.ops) >= ops_threshold` (default 6). The blast radius is the
  artifact's static property; it says "this artifact needs humans", not "this
  individual change is large". `ops_threshold` is the per-change size trigger.
- The approval key is a fingerprint of **target + ops + base version**, not the
  producer's `diff_id`. Approving one change therefore cannot let a different
  change with the same id through.
- A **pending** request rejects the merge with `category="needs-approval"`; the
  same content is not requested twice.
- An **approved** diff passes to the inner policy; a **rejected** one is refused
  with `category="approval-rejected"` and stays rejected until the content
  changes.
- `inner=None` is the shipped `DefaultAcceptance`, configured through the
  forwarded hooks, so the gate can only make acceptance stricter.
- The request includes a short `summary` of the rendered ops so a human has
  something to review beyond an opaque key.

```python
for request in store.pending():
    print(request.key, request.artifact_id, request.summary)
store.decide(request.key, approve=True, note="reviewed the harness change")
```

The store is append-only JSONL, so an operator or a host tool can read and decide
without a server, and the history of who approved what survives the run.
Malformed lines are skipped rather than breaking the queue, writers in one
process share a per-path lock, and `decide()` refuses a key with no matching
request.

An approved diff is re-offered by the host loop or a resumed run: the gate does
not resurrect a settled candidate by itself.
