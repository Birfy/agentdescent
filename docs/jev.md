# JEV — a distilled Judge–Evaluator–Verifier student

An oracle is expensive, and a cheap verifier drifts. The third option is a
**student** verifier trained on the oracle labels a curriculum selects. `JEVModel`
is the smallest model that can do that in a dependency-free package: a linear,
L2-regularised logistic scorer whose features come from the task and the
artifact being judged.

It is deliberately not a neural network. There is no embedding stack, no
training loop hidden behind a framework, and no claim to replace the oracle. It
is the piece that turns a stream of oracle labels into a fast scorer that can be
checkpointed as JSON and used where a `(artifact, task) -> float` verifier
callable is expected. Features are standardised online (running mean/variance,
saved in the checkpoint), so raw lengths and difficulty counts work without
manual scaling.

```python
from agentdescent import JEVModel

features = lambda artifact, task: [len(task.prompt), task.meta.get("difficulty", 0)]
jev = JEVModel(features, learning_rate=0.1)
jev.fit([(artifact, task, oracle(artifact, task)) for task in labelled], epochs=10)
score = jev(artifact, task)          # sigmoid(w·x + b), clipped to [0, 1]
jev.observe(artifact, task, 1.0)     # one online correction from a new label
checkpoint = jev.to_dict()
JEVModel.from_dict(checkpoint, features)
```

## Where it fits

- **Distillation**: feed it the oracle labels from a disagreement-selected task
  loop (the tasks where the current verifier is least trustworthy), so the
  student spends its capacity on the blind spots rather than the easy majority.
- **Fast verifier**: use `jev` as the `(artifact, task)` verifier callable in a
  label loop once it has learned enough; keep the oracle for the audited sample.
  The `(output, prompt)` audit tap and the object-shaped `VerifierProtocol` need
  a small adapter -- this model's call shape is `(artifact, task)`.
- **Portable checkpoint**: `to_dict` / `from_dict` carry the weights, bias and
  update count; the feature function stays in code because it is the task
  contract, not learned state.

`JEVModel` learns online: `observe` is one logistic-gradient step, `fit` repeats
over labelled triples, and `predict` is the `(artifact, task) -> float` call.
Feature length and finiteness are checked before any update, as are non-finite
and out-of-range targets, so a broken feature function or oracle fails at the
label, not deep inside a run. The first `predict` on an untrained model fixes
the feature length, the same way `observe` does.
