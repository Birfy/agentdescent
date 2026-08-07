# Proposal policies — evidence into proposals

*Module:* [`agentdescent.policies`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/policies.py)
· *Contract:* `ProposalPolicy.propose(ctx: ProposalContext) -> Sequence[str]`

The seam between a rollout's outcome and the proposal the strategy turns into
a diff. The engine consumes **one proposal per rollout**; a policy returning
several is refused rather than truncated — batched rollouts are what makes
k > 1 usable, and silently keeping the first would report work that never ran.

## Implemented

No shipped implementations yet — the field takes any object with the
protocol's `propose` method. The MethodPolicy ports do not use this seam:
their proposal logic is plain actor code (a `propose` callable), which is the
right home when there is no *decision* to swap, only text.

Write one when the *rule* for proposing needs to vary independently of the
prompts — e.g. propose only below a reward threshold, or route to different
reflectors by task cluster.
