# Adaptive compute — which strategy deserves this task?

`model_router` chooses a model. An RSI loop also chooses a **strategy**: answer
directly, rerank with a process reward model, search with MCTS, or pay for a
verifier ensemble. `ComputeController` is the cost-aware bandit over those
options, with `ModelRouter` choosing the concrete model.

```python
from agentdescent import (
    ComputeAction, ComputeContext, ComputeController, ComputeOption,
)
from agentdescent.model_router import ModelRouter, ModelSpec

router = ModelRouter([
    ModelSpec("cheap", cost=1, quality_prior=(5, 5)),
    ModelSpec("strong", cost=10, quality_prior=(5, 5)),
], cost_weight=0.0, exploration=0.0)

controller = ComputeController([
    ComputeOption("direct", ComputeAction.DIRECT, calls=1,
                  max_difficulty=0.5),
    ComputeOption("prm", ComputeAction.PRM_RERANK, calls=2,
                  quality_prior=(4, 2), min_difficulty=0.3),
    ComputeOption("mcts", ComputeAction.MCTS, calls=8,
                  quality_prior=(6, 2), min_difficulty=0.6),
], router)

decision = controller.choose(ComputeContext(difficulty=0.8, tokens=4000))
# decision.action / decision.model / decision.expected_cost / decision.calls

controller.observe(decision, reward=0.9)   # updates strategy + model posteriors
report = controller.report()
```

## The allocation contract

- **Difficulty bands.** An option runs only inside `[min_difficulty,
  max_difficulty]`. The model router also receives the difficulty and makes
  harder tasks pay more for quality.
- **Per-call cost.** `option.calls` multiplies the model price. `max_cost` is
  passed to the router as `max_cost / calls`, so a multi-call strategy can still
  choose a cheaper model instead of disappearing from the candidate set.
- **Calls are reserved at `choose` time.** Each `ComputeDecision` carries a
  deterministic `reservation_id`; `observe` settles it. Several decisions
  without an `observe` cannot exceed `max_calls`, and a double-observe is
  refused.
- **One decision, one audit row.** `ComputeDecision.to_dict()` has the option,
  action, model, expected and sampled quality, cost, calls and exploration flag.
- **Router cost stays per call.** `observe` forwards `spent / calls` to
  `ModelRouter`, so its `mean_cost` is a model price, not a strategy price.

## Actions

| Action | What the caller does |
| --- | --- |
| `direct` | one completion |
| `prm_rerank` | sample candidates, score with `process_reward`, pick the best |
| `mcts` | run `planning.MCTSPlanner` (often with a PRM as the value function) |
| `verifier_ensemble` | run `verifier_ensemble.VerifierEnsemble` |

The controller does not execute any of them. It owns allocation, the posteriors
and the audit record.

## What it is and is not

It is a Thompson-sampling bandit over strategies with priors you can seed from
offline replay. It is not a contextual model over task text: the context enters
through difficulty, tokens, tags, cost and the model filter. A caller that
needs task embeddings can wrap `choose()` in its own policy.

A successful strategy also updates the model posterior, so model quality is
confounded with strategy quality. Use a router dedicated to one strategy when
those signals must stay separate.
