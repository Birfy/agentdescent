# Guard pipeline — one ordered acceptance gate

The repo has several independent acceptance checks (leakage, capability
regression, verifier/oracle agreement, human approval, the default statistical
gate). Stacking them by hand makes the order implicit and loses which ring fired.
`GuardPipeline` is the one place the order lives.

```python
from agentdescent import GuardPipeline, Policies
from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision

def no_gold_in_candidate(ctx):
    ...  # supply the guard; shipped guards live in their own modules
    return AcceptDecision(True)

policies = Policies(acceptance=GuardPipeline([
    ("leakage", no_gold_in_candidate),
    ("stats", DefaultAcceptance()),
]))
```

- Guards run in the listed order; the first refusal is the pipeline's decision.
  `category` stays a real merge category accepted by the aggregator (falling
  back to `below-threshold`), while the refusing guard's name and detail are
  carried in `detail`.
- `collect_all=True` keeps evaluating after a refusal for a full trail; the
  candidate is still refused, and `accept()` reports the **first** refusal
  rather than whichever guard ran last.
- An empty pipeline is the identity gate and accepts.
- `bind` / `configure` are forwarded to every guard, so the engine's verifier
  and thresholds reach nested policies. `checkpoint_state`, `restore_state`
  and `on_screen_reject` are forwarded to the guards that implement them.
- `evaluate(ctx)` returns a `GuardPipelineResult` with every named decision and
  `failed_at`; `accept(ctx)` is the ordinary acceptance-policy shape.

The pipeline is deliberately generic: it imports no specific guard, so a
third-party guard is just another `(name, policy)` pair and the order is visible
in one place rather than in caller code.
