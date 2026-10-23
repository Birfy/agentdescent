"""A distilled Judge–Evaluator–Verifier (JEV) student for oracle labels.

An oracle is expensive and a cheap verifier drifts. The third option is a
*student* verifier trained on the oracle labels the curriculum selects: a small,
dependency-free linear model whose features come from the task and the artifact
the caller is judging. It is not a neural network and does not pretend to be
one; it is the smallest model that can turn a stream of oracle labels into a
fast scorer, checkpoint it as JSON, and sit in the same seats as the verifier in
:mod:`agentdescent.verifier_curriculum` and
:mod:`agentdescent.policies` acceptance.

    features = lambda artifact, task: [len(task.prompt), task.meta.get("difficulty", 0)]
    jev = JEVModel(features)
    jev.fit([(artifact, task, oracle(artifact, task)) for task in labelled])
    for task in next_batch:
        jev.observe(artifact, task, oracle(artifact, task))

The model is online: ``observe`` is one logistic-gradient step, so a label loop
can keep correcting it while it explores. Features are standardised online
(running mean/variance, checkpointed with the weights), so raw lengths and
difficulty counts are usable without manual scaling. ``predict`` is
``sigmoid(w·x + b)`` in standardised space, clipped to ``[0, 1]``. ``to_dict`` /
``from_dict`` make a student checkpoint portable; the features callable stays in
code. The call shape is ``(artifact, task) -> float`` -- the seats that take
that pair (for example a curriculum's verifier); the `(output, prompt)` audit
tap and the object-shaped `VerifierProtocol` need an adapter.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["JEVModel"]

Features = Callable[[Any, Any], Sequence[float]]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class JEVModel:
    """A linear student verifier trained on oracle labels.

    ``features(artifact, task) -> sequence[float]`` is the only caller-supplied
    piece; its length is fixed by the first observation. ``predict`` returns a
    score in ``[0, 1]`` -- use it where a verifier score is expected. ``observe``
    applies one L2-regularised logistic step; ``fit`` repeats over labelled
    samples. ``seed`` only controls the optional shuffle order in ``fit``.
    """

    def __init__(self, features: Features, *, learning_rate: float = 0.1,
                 l2: float = 1e-4, seed: int = 0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.features = features
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.rng = random.Random(seed)
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.updates: int = 0
        #: Running standardisation state, in raw feature space.
        self.mean: List[float] = []
        self.m2: List[float] = []
        self.count: int = 0
        self.seed = int(seed)

    # -- projection ----------------------------------------------------------

    def _raw(self, artifact: Any, task: Any) -> List[float]:
        values = [float(v) for v in self.features(artifact, task)]
        if not values:
            raise ValueError("JEVModel features returned an empty vector")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("JEVModel features must be finite")
        if not self.weights:
            self.weights = [0.0] * len(values)
            self.mean = [0.0] * len(values)
            self.m2 = [0.0] * len(values)
        if len(values) != len(self.weights):
            raise ValueError(
                f"JEVModel got {len(values)} features, expected {len(self.weights)}")
        return values

    def _standardise(self, values: List[float]) -> List[float]:
        out = []
        for index, value in enumerate(values):
            if self.count > 1:
                sd = math.sqrt(max(0.0, self.m2[index] / self.count))
            else:
                sd = 0.0
            out.append((value - self.mean[index]) / (sd or 1.0))
        return out

    def _vector(self, artifact: Any, task: Any) -> List[float]:
        return self._standardise(self._raw(artifact, task))

    def _observe_stats(self, values: List[float]) -> None:
        self.count += 1
        for index, value in enumerate(values):
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count
            self.m2[index] += delta * (value - self.mean[index])

    def predict(self, artifact: Any, task: Any) -> float:
        """The student's score for one ``(artifact, task)`` pair, in ``[0, 1]``.

        The first call fixes the feature length (as ``observe`` does); a later
        different length is an error rather than a silent reshape.
        """
        values = self._vector(artifact, task)
        logit = self.bias + sum(w * v for w, v in zip(self.weights, values))
        return min(1.0, max(0.0, _sigmoid(logit)))

    __call__ = predict

    # -- learning ------------------------------------------------------------

    def observe(self, artifact: Any, task: Any, target: float) -> float:
        """One logistic-gradient step toward ``target``; returns the new loss."""
        target = float(target)
        if not math.isfinite(target):
            raise ValueError("JEVModel target must be finite")
        if not 0.0 <= target <= 1.0:
            raise ValueError("JEVModel target must be in [0, 1]")
        values = self._raw(artifact, task)
        self._observe_stats(values)
        values = self._standardise(values)
        logit = self.bias + sum(w * v for w, v in zip(self.weights, values))
        prediction = min(1.0, max(0.0, _sigmoid(logit)))
        error = prediction - target
        for index, value in enumerate(values):
            gradient = error * value + self.l2 * self.weights[index]
            self.weights[index] -= self.learning_rate * gradient
        self.bias -= self.learning_rate * error
        self.updates += 1
        # Binary cross-entropy for the caller's log; clamped to keep log finite.
        p = min(1.0 - 1e-12, max(1e-12, prediction))
        return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))

    update = observe

    def fit(self, samples: Iterable[Tuple[Any, Any, float]], *,
            epochs: int = 1) -> "JEVModel":
        """Train on ``(artifact, task, target)`` triples for ``epochs`` passes."""
        rows = list(samples)
        for _ in range(max(1, int(epochs))):
            order = list(rows)
            self.rng.shuffle(order)
            for artifact, task, target in order:
                self.observe(artifact, task, target)
        return self

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "weights": list(self.weights), "bias": self.bias,
                "updates": self.updates, "learning_rate": self.learning_rate,
                "l2": self.l2, "seed": self.seed, "mean": list(self.mean),
                "m2": list(self.m2), "count": self.count,
                "feature_count": len(self.weights)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], features: Features) -> "JEVModel":
        if int(payload.get("version", 1)) != 1:
            raise ValueError(f"unsupported JEVModel checkpoint version {payload.get('version')!r}")
        model = cls(features,
                    learning_rate=float(payload.get("learning_rate", 0.1)),
                    l2=float(payload.get("l2", 1e-4)),
                    seed=int(payload.get("seed", 0)))
        model.weights = [float(w) for w in payload.get("weights", [])]
        model.bias = float(payload.get("bias", 0.0))
        model.mean = [float(v) for v in payload.get("mean", [])]
        model.m2 = [float(v) for v in payload.get("m2", [])]
        model.count = int(payload.get("count", 0))
        model.updates = int(payload.get("updates", 0))
        numbers = [*model.weights, model.bias, *model.mean, *model.m2]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("JEVModel checkpoint contains a non-finite value")
        expected = payload.get("feature_count")
        if expected is not None and int(expected) != len(model.weights):
            raise ValueError("JEVModel checkpoint feature_count does not match weights")
        model.rng = random.Random(model.seed)
        return model

    def __repr__(self) -> str:
        return (f"JEVModel(features={len(self.weights)}, updates={self.updates}, "
                f"bias={self.bias:+.3f})")
