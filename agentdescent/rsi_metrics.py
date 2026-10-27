"""RSI metrics: is the loop producing accepted improvements at a sane price?

A run store answers what happened; a product needs the aggregate: acceptance
rate, cost per accepted change, which categories appear most often, and a stable
text export for Prometheus. This module derives those from decision records (an
iterable of mappings, or objects with the same fields) and calls no model.

    metrics = metrics_from_log(records)
    print(metrics.acceptance_rate, metrics.cost_per_accepted)
    print(to_prometheus(metrics))

The record schema is ``accepted`` (bool), ``category`` (str) and ``cost`` (a
mapping/object with ``calls``, ``prompt_tokens``, ``completion_tokens`` and
``wallclock``). Missing numeric fields count as zero; a present non-numeric,
negative or non-finite value is refused with a ``ValueError`` naming the record.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = ["RSIMetrics", "alerts", "metrics_from_log", "metrics_from_records",
           "to_prometheus"]


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _field_error(index: int, field: str, message: str) -> ValueError:
    return ValueError(f"record {index}: {field} {message}")


def _non_negative_int(cost: Any, key: str, index: int) -> int:
    value = _get(cost, key, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _field_error(
            index, f"cost.{key}", "must be a non-negative int")
    return value


def _non_negative_float(cost: Any, key: str, index: int) -> float:
    value = _get(cost, key, 0.0)
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _field_error(index, f"cost.{key}", "must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise _field_error(
            index, f"cost.{key}", "must be finite and non-negative")
    return number


@dataclass(frozen=True)
class RSIMetrics:
    """Aggregate counts and cost for a stream of decisions."""

    decisions: int = 0
    accepted: int = 0
    rejected: int = 0
    acceptance_rate: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wallclock: float = 0.0
    cost_per_accepted: Optional[float] = None
    categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": self.decisions, "accepted": self.accepted,
            "rejected": self.rejected, "acceptance_rate": self.acceptance_rate,
            "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wallclock": self.wallclock,
            "cost_per_accepted": self.cost_per_accepted,
            "categories": dict(self.categories),
        }


def metrics_from_records(records: Iterable[Any]) -> RSIMetrics:
    """Aggregate decision records.

    Missing fields count as zero. A malformed present field is refused rather
    than silently coerced, because these numbers drive alerts and dashboards.
    """
    rows = list(records)
    accepted = 0
    categories: Counter = Counter()
    calls = prompt = completion = 0
    wallclock = 0.0
    for index, row in enumerate(rows):
        flag = _get(row, "accepted", None)
        if flag is None:
            flag = False
        if not isinstance(flag, bool):
            raise _field_error(index, "accepted", "must be a bool")
        accepted += int(flag)
        category = str(_get(row, "category", "") or "unknown")
        categories[category] += 1
        cost = _get(row, "cost", None)
        if cost is None:
            cost = {}
        elif not (isinstance(cost, Mapping)
                  or any(hasattr(cost, key) for key in (
                      "calls", "prompt_tokens", "completion_tokens",
                      "wallclock"))):
            raise _field_error(
                index, "cost", "must be a mapping or an object with cost fields")
        calls += _non_negative_int(cost, "calls", index)
        prompt += _non_negative_int(cost, "prompt_tokens", index)
        completion += _non_negative_int(cost, "completion_tokens", index)
        wallclock += _non_negative_float(cost, "wallclock", index)
    total = len(rows)
    return RSIMetrics(
        decisions=total, accepted=accepted, rejected=total - accepted,
        acceptance_rate=(accepted / total if total else 0.0),
        calls=calls, prompt_tokens=prompt, completion_tokens=completion,
        wallclock=wallclock,
        cost_per_accepted=(calls / accepted if accepted else None),
        categories=dict(sorted(categories.items())))


def metrics_from_log(log: Any) -> RSIMetrics:
    """Aggregate any iterable log (a JSONL-backed log, list, or generator)."""
    return metrics_from_records(log)


def _label(value: str) -> str:
    return (value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n"))


def to_prometheus(metrics: RSIMetrics) -> str:
    """A deterministic Prometheus text exposition (no timestamps).

    Each family has one HELP/TYPE pair. The per-category counter has its own
    family name so a Prometheus/OpenMetrics parser never sees two types for the
    same sample name.
    """
    lines = [
        "# HELP agentdescent_rsi_decisions_total Total decision records.",
        "# TYPE agentdescent_rsi_decisions_total counter",
        f"agentdescent_rsi_decisions_total {metrics.decisions}",
        "# HELP agentdescent_rsi_decisions_by_category_total Decision records by category.",
        "# TYPE agentdescent_rsi_decisions_by_category_total counter",
    ]
    for category, count in metrics.categories.items():
        lines.append(
            "agentdescent_rsi_decisions_by_category_total"
            f'{{category="{_label(category)}"}} {count}')
    lines += [
        "# HELP agentdescent_rsi_accepted_total Accepted decisions.",
        "# TYPE agentdescent_rsi_accepted_total counter",
        f"agentdescent_rsi_accepted_total {metrics.accepted}",
        "# HELP agentdescent_rsi_rejected_total Rejected decisions.",
        "# TYPE agentdescent_rsi_rejected_total counter",
        f"agentdescent_rsi_rejected_total {metrics.rejected}",
        "# HELP agentdescent_rsi_acceptance_rate Accepted / total.",
        "# TYPE agentdescent_rsi_acceptance_rate gauge",
        f"agentdescent_rsi_acceptance_rate {metrics.acceptance_rate:.6f}",
        "# HELP agentdescent_rsi_calls_total Agent calls spent.",
        "# TYPE agentdescent_rsi_calls_total counter",
        f"agentdescent_rsi_calls_total {metrics.calls}",
        "# HELP agentdescent_rsi_prompt_tokens_total Prompt tokens spent.",
        "# TYPE agentdescent_rsi_prompt_tokens_total counter",
        f"agentdescent_rsi_prompt_tokens_total {metrics.prompt_tokens}",
        "# HELP agentdescent_rsi_completion_tokens_total Completion tokens spent.",
        "# TYPE agentdescent_rsi_completion_tokens_total counter",
        f"agentdescent_rsi_completion_tokens_total {metrics.completion_tokens}",
        "# HELP agentdescent_rsi_wallclock_seconds_total Wall clock spent.",
        "# TYPE agentdescent_rsi_wallclock_seconds_total counter",
        f"agentdescent_rsi_wallclock_seconds_total {metrics.wallclock:.6f}",
        "# HELP agentdescent_rsi_cost_per_accepted Calls per accepted decision.",
        "# TYPE agentdescent_rsi_cost_per_accepted gauge",
        "agentdescent_rsi_cost_per_accepted "
        + ("NaN" if metrics.cost_per_accepted is None
           else f"{metrics.cost_per_accepted:.6f}"),
    ]
    return "\n".join(lines) + "\n"


def alerts(metrics: RSIMetrics, *, min_acceptance: float = 0.0,
           max_cost_per_accepted: Optional[float] = None) -> List[str]:
    """Human-readable alert conditions; empty means no alert.

    ``max_cost_per_accepted`` also fires when calls were spent but nothing was
    accepted; ``None`` would otherwise hide the worst case.
    """
    if (isinstance(min_acceptance, bool)
            or not isinstance(min_acceptance, (int, float))
            or not math.isfinite(float(min_acceptance))
            or not 0.0 <= float(min_acceptance) <= 1.0):
        raise ValueError("min_acceptance must be a finite number in [0, 1]")
    if max_cost_per_accepted is not None:
        if (isinstance(max_cost_per_accepted, bool)
                or not isinstance(max_cost_per_accepted, (int, float))
                or not math.isfinite(float(max_cost_per_accepted))
                or float(max_cost_per_accepted) < 0):
            raise ValueError(
                "max_cost_per_accepted must be a finite non-negative number")
    out = []
    if metrics.decisions and metrics.acceptance_rate < min_acceptance:
        out.append(
            f"acceptance rate {metrics.acceptance_rate:.3f} is below "
            f"{min_acceptance:.3f}")
    if (max_cost_per_accepted is not None
            and metrics.decisions and metrics.accepted == 0
            and metrics.calls > 0):
        out.append(
            f"no accepted decisions after {metrics.calls} calls")
    if (max_cost_per_accepted is not None
            and metrics.cost_per_accepted is not None
            and metrics.cost_per_accepted > max_cost_per_accepted):
        out.append(
            f"cost per accepted {metrics.cost_per_accepted:.1f} exceeds "
            f"{max_cost_per_accepted:.1f}")
    return out
