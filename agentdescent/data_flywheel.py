"""Turn production traces into a curated dataset candidate.

The evolution loop starts from a task set. A product starts from **traffic**:
successes, failures, human corrections, tool errors, and the long tail the
benchmark never contained. The data flywheel is the bridge:

1. filter (reward band, error state, deterministic sampling);
2. redact (secrets/PII-shaped keys recursively);
3. deduplicate (exact normalized task+output, optional similarity hook);
4. cap cluster dominance so one repeated failure does not become the dataset;
5. emit a content-hashed dataset candidate for `TaskFoundry` / `EvalOps`.

This module is storage-light and deterministic. It does not call a model; the
caller supplies trace rows from `runstore`, `telemetry`, or a product log.

    wheel = DataFlywheel(FlywheelConfig(min_reward=0.0, max_reward=0.4))
    report = wheel.mine(trace_rows)
    candidate = wheel.to_dataset(report, dataset_id="prod-failures", version=1)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .evolvable import stable_hash

__all__ = [
    "TraceSample",
    "FlywheelConfig",
    "FlywheelReport",
    "FlywheelDataset",
    "DataFlywheel",
    "redact",
]

_WS = re.compile(r"\s+")
_SECRET_KEY = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|cookie|email|phone|ssn)",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return _WS.sub(" ", value.strip().lower())
    try:
        return _WS.sub(" ", json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        default=str).strip().lower())
    except Exception:
        return _WS.sub(" ", str(value).strip().lower())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in trace")
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    raise TypeError(f"non-JSON trace value: {type(value).__name__}")


def redact(value: Any, *, keys: Sequence[str] = ()) -> Any:
    """Recursively mask secret/PII-shaped keys.

    Matching is by key name (case-insensitive substring), not by value, so a
    normal task string containing "email" is untouched unless its key looks
    sensitive. ``keys`` adds extra key fragments.
    """
    extra = tuple(str(key).lower() for key in keys)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_KEY.search(name) or any(part in name.lower() for part in extra):
                out[name] = "[REDACTED]"
            else:
                out[name] = redact(item, keys=keys)
        return out
    if isinstance(value, list):
        return [redact(item, keys=keys) for item in value]
    if isinstance(value, tuple):
        return [redact(item, keys=keys) for item in value]
    return value


@dataclass(frozen=True)
class TraceSample:
    trace_id: str
    task: Any
    output: Any = None
    reward: float = 0.0
    cost: float = 0.0
    tags: Tuple[str, ...] = ()
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("TraceSample.trace_id is required")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and >= 0")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))


@dataclass(frozen=True)
class FlywheelConfig:
    min_reward: float = 0.0
    max_reward: float = 1.0
    sample_rate: float = 1.0
    max_samples: int = 1000
    near_duplicate_threshold: float = 0.95
    max_per_cluster: int = 5
    seed: int = 0
    redact_keys: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("min_reward", "max_reward"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_reward < self.min_reward:
            raise ValueError("max_reward must be >= min_reward")
        rate = float(self.sample_rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        if self.max_samples < 1 or self.max_per_cluster < 1:
            raise ValueError("max_samples and max_per_cluster must be >= 1")
        threshold = float(self.near_duplicate_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")


@dataclass(frozen=True)
class FlywheelReport:
    accepted: Tuple[TraceSample, ...]
    rejected: Tuple[Tuple[str, str], ...]
    clusters: int
    detail: str = ""

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    def rejection_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for reason, _ in self.rejected:
            counts[reason] = counts.get(reason, 0) + 1
        return counts


@dataclass(frozen=True)
class FlywheelDataset:
    dataset_id: str
    version: int
    rows: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.dataset_id or isinstance(self.version, bool) or not isinstance(self.version, int) \
                or self.version < 1:
            raise ValueError("dataset_id and version >= 1 are required")
        rows = tuple(_json_safe(dict(row)) for row in self.rows)
        if not rows:
            raise ValueError("FlywheelDataset needs at least one row")
        object.__setattr__(self, "rows", rows)

    @property
    def content_hash(self) -> str:
        body = json.dumps(list(self.rows), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"dataset_id": self.dataset_id, "version": self.version,
                "content_hash": self.content_hash,
                "rows": [dict(row) for row in self.rows]}


class DataFlywheel:
    """Filter, redact, deduplicate and emit trace-derived dataset candidates."""

    def __init__(self, config: Optional[FlywheelConfig] = None, *,
                 similarity: Optional[Callable[[TraceSample, TraceSample], float]] = None) -> None:
        self.config = config or FlywheelConfig()
        if similarity is not None and not callable(similarity):
            raise TypeError("similarity must be callable or None")
        self.similarity = similarity

    def _sample(self, sample: TraceSample) -> bool:
        rate = self.config.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return stable_hash((self.config.seed, sample.trace_id)) % 10_000 < rate * 10_000

    def _row(self, sample: TraceSample) -> Dict[str, Any]:
        return {
            "trace_id": sample.trace_id,
            "task": redact(_json_safe(sample.task), keys=self.config.redact_keys),
            "output": redact(_json_safe(sample.output), keys=self.config.redact_keys),
            "reward": sample.reward,
            "cost": sample.cost,
            "tags": list(sample.tags),
            "metadata": redact(_json_safe(sample.metadata), keys=self.config.redact_keys),
        }

    def mine(self, samples: Iterable[TraceSample]) -> FlywheelReport:
        cfg = self.config
        accepted: List[TraceSample] = []
        rejected: List[Tuple[str, str]] = []
        fingerprints: Dict[str, str] = {}
        cluster_counts: Dict[str, int] = {}
        cluster_of: Dict[str, str] = {}
        seen_ids = set()
        for sample in samples:
            if not isinstance(sample, TraceSample):
                raise TypeError(f"mine() expects TraceSample, got {type(sample).__name__}")
            if sample.trace_id in seen_ids:
                rejected.append(("duplicate-trace-id", sample.trace_id))
                continue
            seen_ids.add(sample.trace_id)
            if sample.error is not None:
                rejected.append(("error", sample.trace_id))
                continue
            if not (cfg.min_reward <= sample.reward <= cfg.max_reward):
                rejected.append(("reward-band", sample.trace_id))
                continue
            if not self._sample(sample):
                rejected.append(("sampled-out", sample.trace_id))
                continue
            fingerprint = _text(sample.task) + "\x00" + _text(sample.output)
            previous = fingerprints.get(fingerprint)
            if previous is not None:
                # Exact duplicates keep one representative; near-duplicates use
                # the per-cluster cap below.
                rejected.append(("duplicate", sample.trace_id))
                continue
            if self.similarity is not None:
                duplicate_of = ""
                for existing in accepted:
                    score = float(self.similarity(sample, existing))
                    if not math.isfinite(score):
                        raise ValueError("similarity returned a non-finite score")
                    if score >= cfg.near_duplicate_threshold:
                        duplicate_of = existing.trace_id
                        break
                if duplicate_of:
                    root = cluster_of.get(duplicate_of, duplicate_of)
                    count = cluster_counts.get(root, 1)
                    if count >= cfg.max_per_cluster:
                        rejected.append(("cluster-cap", sample.trace_id))
                        continue
                    cluster_counts[root] = count + 1
                    cluster_of[sample.trace_id] = root
                    fingerprints[fingerprint] = root
                    accepted.append(sample)
                    if len(accepted) >= cfg.max_samples:
                        break
                    continue
            fingerprints[fingerprint] = sample.trace_id
            cluster_of[sample.trace_id] = sample.trace_id
            cluster_counts.setdefault(sample.trace_id, 1)
            accepted.append(sample)
            if len(accepted) >= cfg.max_samples:
                break
        detail = (f"{len(accepted)} accepted / {len(rejected)} rejected, "
                  f"{len(cluster_counts)} cluster(s)")
        return FlywheelReport(tuple(accepted), tuple(rejected),
                              len(cluster_counts), detail)

    def to_dataset(self, report: FlywheelReport, *, dataset_id: str,
                   version: int) -> FlywheelDataset:
        return FlywheelDataset(dataset_id, version,
                               tuple(self._row(sample) for sample in report.accepted))
