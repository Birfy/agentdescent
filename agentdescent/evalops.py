"""Continuous evaluation as a service: datasets, specs, gates and drift.

A single `evaluate()` call is an experiment. EvalOps is the product around it:
versioned datasets, reusable eval specs, a regression gate that a CI job can
block on, scheduled reruns, bootstrap confidence intervals and drift detection.

The vertical slice is deliberately storage-light: an in-memory registry plus
stable content hashes. The contract is the part a Postgres/S3 adapter keeps:
dataset version + content hash + spec version + result row.

    ops = EvalOps()
    ops.add_dataset(Dataset("gsm", 1, tasks))
    ops.add_spec(EvalSpec("gsm-gate", "gsm", 1, min_effect=0.01))
    result = ops.run("gsm-gate", evaluate, baseline_evaluate)
    result.passed, result.ci, result.detail

`evaluate(task, seed)` may return a float or an `EvalOutput(score, cost)`. The
runner averages seeds per task, then bootstraps the paired per-task delta with
:func:`agentdescent.stats_eval.bootstrap_ci`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stats_eval import BootstrapResult, bootstrap_ci

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "EvalSpec",
    "EvalOutput",
    "EvalResult",
    "GateDecision",
    "RegressionGate",
    "DriftReport",
    "DriftMonitor",
    "EvalOps",
]

Task = Mapping[str, Any]
EvaluateFn = Callable[[Task, int], Any]
BaselineFn = Callable[[Task, int], Any]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in dataset")
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("dataset keys must be strings")
            out[key] = _json_safe(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"non-JSON dataset value: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class Dataset:
    """A versioned task set with a content hash."""

    dataset_id: str
    version: int
    tasks: Tuple[Task, ...]
    contaminated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.dataset_id or isinstance(self.version, bool) \
                or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("dataset_id and version >= 1 are required")
        tasks = tuple(_freeze(dict(task)) for task in self.tasks)
        if not tasks:
            raise ValueError("Dataset needs at least one task")
        ids = [str(task.get("task_id", index)) for index, task in enumerate(tasks)]
        if len(set(ids)) != len(ids):
            raise ValueError("Dataset task_id values must be unique")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "_content_hash",
                           hashlib.sha256(_canonical(list(tasks)).encode()).hexdigest())

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return tuple(str(task.get("task_id", index))
                     for index, task in enumerate(self.tasks))

    def to_dict(self) -> Dict[str, Any]:
        return {"dataset_id": self.dataset_id, "version": self.version,
                "content_hash": self.content_hash,
                "contaminated": self.contaminated,
                "tasks": [_json_safe(task) for task in self.tasks],
                "metadata": _json_safe(self.metadata)}


class DatasetRegistry:
    """Immutable dataset versions with stable content hashes."""

    def __init__(self) -> None:
        self._datasets: Dict[Tuple[str, int], Dataset] = {}

    def add(self, dataset: Dataset) -> Dataset:
        key = (dataset.dataset_id, dataset.version)
        if key in self._datasets:
            raise ValueError(f"dataset {key} already exists")
        self._datasets[key] = dataset
        return dataset

    def get(self, dataset_id: str, version: Optional[int] = None) -> Dataset:
        if version is not None:
            try:
                return self._datasets[(dataset_id, version)]
            except KeyError:
                raise KeyError(f"unknown dataset {dataset_id}/{version}") from None
        versions = sorted(v for (name, v) in self._datasets if name == dataset_id)
        if not versions:
            raise KeyError(f"unknown dataset {dataset_id!r}")
        return self._datasets[(dataset_id, versions[-1])]

    def versions(self, dataset_id: str) -> Tuple[int, ...]:
        return tuple(sorted(v for (name, v) in self._datasets if name == dataset_id))

    def all(self) -> Tuple[Dataset, ...]:
        return tuple(self._datasets[key] for key in sorted(self._datasets))


@dataclass(frozen=True)
class EvalSpec:
    """How a dataset is evaluated and gated."""

    eval_id: str
    dataset_id: str
    version: Optional[int] = None
    seeds: Tuple[int, ...] = (0,)
    min_effect: float = 0.0
    alpha: float = 0.05
    bootstrap: int = 2000
    schedule_seconds: Optional[float] = None
    allow_contaminated: bool = False

    def __post_init__(self) -> None:
        if not self.eval_id or not self.dataset_id:
            raise ValueError("eval_id and dataset_id are required")
        if not self.seeds:
            raise ValueError("EvalSpec needs at least one seed")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.bootstrap < 1:
            raise ValueError("bootstrap must be >= 1")
        if self.schedule_seconds is not None and (
                not math.isfinite(float(self.schedule_seconds))
                or float(self.schedule_seconds) <= 0):
            raise ValueError("schedule_seconds must be finite and > 0 or None")
        if not isinstance(self.allow_contaminated, bool):
            raise ValueError("allow_contaminated must be a bool")


@dataclass(frozen=True)
class EvalOutput:
    score: float
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _finite(self.score, "score"))
        cost = _finite(self.cost, "cost")
        if cost < 0:
            raise ValueError("EvalOutput.cost must be >= 0")
        object.__setattr__(self, "cost", cost)


@dataclass(frozen=True)
class EvalResult:
    eval_id: str
    dataset_id: str
    version: int
    content_hash: str
    n: int
    score: float
    baseline: float
    delta: float
    ci: Tuple[float, float]
    passed: bool
    cost: float
    per_task: Mapping[str, float]
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id, "dataset_id": self.dataset_id,
            "version": self.version, "content_hash": self.content_hash,
            "n": self.n, "score": self.score, "baseline": self.baseline,
            "delta": self.delta, "ci": list(self.ci), "passed": self.passed,
            "cost": self.cost, "per_task": dict(self.per_task),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    result: Optional[EvalResult] = None


class RegressionGate:
    """Turn an EvalResult into a CI gate with a plain-language reason."""

    def __init__(self, *, min_effect: Optional[float] = None,
                 require_significance: bool = True) -> None:
        self.min_effect = 0.0 if min_effect is None else float(min_effect)
        self.require_significance = require_significance

    def check(self, result: EvalResult) -> GateDecision:
        if result.n < 2 and self.require_significance:
            return GateDecision(False, "fewer than two paired tasks", result)
        effect = self.min_effect
        if result.delta < effect:
            return GateDecision(False, f"delta {result.delta:.6g} < {effect:.6g}", result)
        if self.require_significance and result.ci[0] <= effect:
            return GateDecision(False, f"CI lower {result.ci[0]:.6g} <= {effect:.6g}", result)
        return GateDecision(True, "effect cleared the gate", result)


@dataclass(frozen=True)
class DriftReport:
    shared: int
    mean_shift: float
    max_shift: float
    fraction_beyond: float
    drift: bool
    detail: str = ""


class DriftMonitor:
    """Compare a reference task-score map with a current run."""

    def __init__(self, *, threshold: float = 0.1, fraction: float = 0.2) -> None:
        if threshold < 0 or not math.isfinite(float(threshold)):
            raise ValueError("threshold must be finite and >= 0")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        self.threshold = float(threshold)
        self.fraction = float(fraction)

    def observe(self, reference: Mapping[str, float],
                current: Mapping[str, float]) -> DriftReport:
        keys = sorted(set(reference) & set(current))
        if not keys:
            return DriftReport(0, 0.0, 0.0, 0.0, False, "no shared tasks")
        shifts = [float(current[key]) - float(reference[key]) for key in keys]
        mean_shift = sum(shifts) / len(shifts)
        max_shift = max(abs(shift) for shift in shifts)
        beyond = sum(1 for shift in shifts if abs(shift) > self.threshold) / len(shifts)
        drift = abs(mean_shift) > self.threshold or beyond > self.fraction
        return DriftReport(len(keys), mean_shift, max_shift, beyond, drift,
                           f"{len(keys)} shared, mean={mean_shift:+.4f}, "
                           f"max={max_shift:.4f}, beyond={beyond:.2%}")


class EvalOps:
    """The registry + runner + gate + schedule read model."""

    def __init__(self, clock: Optional[Any] = None) -> None:
        self.clock = clock or time.time
        self.datasets = DatasetRegistry()
        self._specs: Dict[str, EvalSpec] = {}
        self._results: Dict[str, EvalResult] = {}
        self.last_run: Dict[str, float] = {}

    @property
    def specs(self) -> Mapping[str, EvalSpec]:
        return MappingProxyType(self._specs)

    @property
    def results(self) -> Mapping[str, EvalResult]:
        return MappingProxyType(self._results)

    def add_dataset(self, dataset: Dataset) -> Dataset:
        return self.datasets.add(dataset)

    def add_spec(self, spec: EvalSpec) -> EvalSpec:
        if spec.eval_id in self._specs:
            raise ValueError(f"eval {spec.eval_id!r} already exists")
        self._specs[spec.eval_id] = spec
        return spec

    def _output(self, value: Any, name: str) -> EvalOutput:
        if isinstance(value, EvalOutput):
            return value
        return EvalOutput(_finite(value, name))

    def run(self, eval_id: str, evaluate: EvaluateFn,
            baseline_evaluate: Optional[BaselineFn] = None, *,
            now: Optional[float] = None) -> EvalResult:
        if eval_id not in self._specs:
            raise KeyError(f"unknown eval {eval_id!r}")
        spec = self._specs[eval_id]
        dataset = self.datasets.get(spec.dataset_id, spec.version)
        if dataset.contaminated and not spec.allow_contaminated:
            raise ValueError(f"dataset {dataset.dataset_id}/{dataset.version} is contaminated")
        per_task: Dict[str, float] = {}
        baseline: Dict[str, float] = {}
        cost = 0.0
        for index, task in enumerate(dataset.tasks):
            task_id = str(task.get("task_id", index))
            scores = []
            for seed in spec.seeds:
                output = self._output(evaluate(task, seed), "evaluate")
                scores.append(output.score)
                cost += output.cost
            per_task[task_id] = sum(scores) / len(scores)
            if baseline_evaluate is not None:
                base_scores = []
                for seed in spec.seeds:
                    output = self._output(baseline_evaluate(task, seed), "baseline")
                    base_scores.append(output.score)
                    cost += output.cost
                baseline[task_id] = sum(base_scores) / len(base_scores)
            else:
                baseline[task_id] = 0.0
        after = per_task
        before = baseline
        boot = bootstrap_ci(before, after, n=spec.bootstrap, alpha=spec.alpha)
        delta = boot.mean
        passed = delta >= spec.min_effect and (
            boot.n < 2 or boot.lo > spec.min_effect)
        result = EvalResult(
            eval_id=eval_id, dataset_id=dataset.dataset_id,
            version=dataset.version, content_hash=dataset.content_hash,
            n=boot.n, score=sum(after.values()) / len(after),
            baseline=sum(before.values()) / len(before), delta=delta,
            ci=(boot.lo, boot.hi), passed=passed, cost=cost,
            per_task=dict(per_task),
            detail=f"delta={delta:+.4f}, CI=[{boot.lo:+.4f},{boot.hi:+.4f}], "
                   f"cost={cost:g}")
        self._results[eval_id] = result
        self.last_run[eval_id] = float(self.clock() if now is None else now)
        return result

    def gate(self, eval_id: str, gate: Optional[RegressionGate] = None) -> GateDecision:
        if eval_id not in self._results:
            raise KeyError(f"no result for eval {eval_id!r}")
        if gate is None:
            spec = self._specs[eval_id]
            gate = RegressionGate(min_effect=spec.min_effect)
        return gate.check(self._results[eval_id])

    def due(self, *, now: Optional[float] = None) -> Tuple[str, ...]:
        at = float(self.clock() if now is None else now)
        out = []
        for eval_id, spec in sorted(self._specs.items()):
            if spec.schedule_seconds is None:
                continue
            last = self.last_run.get(eval_id)
            if last is None or at >= last + spec.schedule_seconds:
                out.append(eval_id)
        return tuple(out)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "datasets": [d.to_dict() for d in self.datasets.all()],
            "specs": {key: {"dataset_id": spec.dataset_id,
                            "version": spec.version,
                            "schedule_seconds": spec.schedule_seconds}
                      for key, spec in sorted(self._specs.items())},
            "results": {key: result.to_dict()
                        for key, result in sorted(self._results.items())},
            "due": list(self.due()),
        }
