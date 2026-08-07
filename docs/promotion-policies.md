# Promotion policies — when dev reaches stable

*Module:* [`agentdescent.defaults`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/defaults.py)
· *Contract:* `PromotionPolicy.promote(ctx) -> Sequence[Promotion]`

The dual-branch rule: `dev` absorbs candidates, `stable` is what production
reads, and promotion is the EMA-style confirmation between them.

## Implemented

| Policy | Rule | Reach for it when |
|---|---|---|
| `DefaultPromotion(promote_after_k)` | promote after K regression-free rounds on dev; the counter resets on any regression | the default; read its docstring before replacing it — the reset semantics are the part everyone gets wrong |

A clean run also promotes on `finalize()`: `target_reward` can fire on the
very commit that reaches it, and the artifact the run was *for* must reach the
branch production reads.

## What the default knows that a replacement must be told

**Promotion counts rounds *survived*, not commits.** Counting commits inverts
the incentive: an artifact that converges stops committing and so can never be
promoted, while one that thrashes promotes every K commits.
