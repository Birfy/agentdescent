# Trusted evidence — one bundle from the independent trust modules

Promotion can judge evidence, but a caller still has to assemble it. This module
turns one paired run (incumbent vs candidate task scores) plus optional side
data into every report the promotion seam understands.

```python
from agentdescent import build_trust_evidence, promote, PromotionRules

evidence = build_trust_evidence(
    before_scores, after_scores,
    verifier=ensemble, candidate=candidate,
    calibration=(confidences, outcomes),
    proxy_gains=proxy, oracle_gains=oracle,
    metamorphic=suite, metamorphic_run=solver,
    metamorphic_tasks=tasks,
    contaminated=False, risk=0.2,
)
decision = promote(
    evidence.to_promotion_request("skill-1"),
    PromotionRules(require_goodhart=True, require_metamorphic=True,
                   require_verifier=True, require_calibration=True))
```

## What it computes

| Report | Source | Signal |
|---|---|---|
| eval + bootstrap | `stats_eval` | paired gain, CI, significance |
| canary | `safety_controller.canary` | promote / hold / rollback |
| conformal | `conformal.ConformalGate` | finite-sample lower bound |
| sequential | `sequential.EProcess` | anytime-valid e-process |
| Goodhart | `goodhart.detect_goodhart` | proxy vs oracle: aligned / suspicious / gaming |
| metamorphic | `metamorphic.MetamorphicSuite` | transformed-input relations |
| verifier | `verifier_ensemble.VerifierEnsemble` | cost-aware panel vote |
| calibration | `self_model.SelfModel` | ECE/Brier and overconfidence |

All reports are carried in `TrustEvidence`; `to_promotion_request()` places the
extra four in the promotion evidence/metadata, so the evidence hash covers them.

## Promotion gates

`PromotionRules` now has `require_goodhart`, `require_metamorphic`,
`require_verifier` and `require_calibration` (default off). When supplied:

- Goodhart `gaming` → reject; `suspicious`/`insufficient` → defer; `aligned` passes.
- Metamorphic `fail` → reject; `insufficient` → defer; `pass` passes.
- Verifier errors or a failed panel → reject; an incomplete panel → defer.
- Calibration overconfident or anti-correlated → reject; insufficient or
  miscalibrated → defer; calibrated/underconfident passes.

The eval means, CI, `n` and `per_task` are computed over the keys present in
**both** before and after; extra unpaired tasks are not folded into the means.
Omitting `risk` or `contaminated` leaves those gates missing, so the default
rules defer rather than treating the absence as zero risk.

This is the point of the bundle: every independent check has a defined
fail-closed consequence in the same decision, instead of five dashboards.
