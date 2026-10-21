"""Portfolio governance: watch every layer's measured return, shift the budget to what pays.

The framework spends across layers -- inner ``evolve()`` runs, meta sessions,
audit sampling, harness reruns, population islands -- and nothing decides how
the next window's budget is split. This module is that rung: a *layer* is one
spending line (a spend limit, what it actually spent, and the measured return
that spend bought), the portfolio view ranks layers by return-per-cost, and the
reallocation rule is deterministic and conservative:

* a layer whose return-per-cost is at or above ``floor`` and whose return rests
  on at least ``min_samples`` measurements **grows** (``budget * (1 + grow)``,
  up to ``cap``);
* a layer below the floor (it is not paying for its cost) or flagged
  ``diminishing`` **shrinks** (``budget * (1 - shrink)``);
* a layer with too few measurements **holds** -- nothing is concluded from a
  sample of one, and the budget must not be moved on a guess;
* no layer is ever zeroed (a held line keeps a floor budget), so a temporarily
  quiet layer is not killed by a busy one.

The rule is deliberately a fixed, honest policy first -- it is the product the
framework can deploy today; evolving the reallocation policy itself via
``meta_run`` is the next rung, and the fixed rule is its baseline.
``layers_from_archive`` turns the consolidation archive's rule records into
layer rows (per decision slot: return = mean measured reward, spend = total
rounds, samples = rule count), and any other layer -- audit bias, harness
leaderboard, population islands -- is a caller-constructed :class:`LayerBudget`
added to the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore

__all__ = ["LayerBudget", "PortfolioReport", "portfolio_report",
           "layers_from_archive"]


@dataclass(frozen=True)
class LayerBudget:
    """One spending line of the framework: a limit, what it spent, what it bought."""

    name: str
    #: The spend limit for this layer in the window (tokens, calls, seconds --
    #: any one unit; the rule only ever uses ratios and finite numbers).
    budget: float
    #: What the layer actually spent.
    spent: float
    #: The measured return that spend bought (mean reward, mean gain, negative
    #: audit bias ...). Any finite number; higher is better.
    return_: float
    #: How many measurements the return rests on. Below ``min_samples`` the
    #: portfolio holds the layer rather than moving its budget.
    samples: int
    #: Spread of the measurements, for the diminishing-returns look.
    return_sd: float = 0.0
    #: A caller may flag a layer as diminishing (its return has fallen off its
    #: own peak); the portfolio then shrinks it regardless of this window's mean.
    diminishing: bool = False
    #: Free-form label for logs.
    note: str = ""


@dataclass(frozen=True)
class PortfolioReport:
    """What a :func:`portfolio_report` pass saw, and decided."""

    layers: Tuple[LayerBudget, ...]
    #: Sum of the layers' current budgets.
    total_budget: float
    #: ``layer name -> new budget`` for the next window.
    reallocated: Dict[str, float]
    #: Why each line moved the way it did.
    reasons: Dict[str, str]


def portfolio_report(
    layers: Sequence[LayerBudget],
    *,
    floor: float = 0.05,
    min_budget: Optional[float] = None,
    min_samples: int = 3,
    grow: float = 0.5,
    shrink: float = 0.5,
    cap: Optional[float] = None,
) -> PortfolioReport:
    """Score the layers by return-per-cost and reallocate the next window.

    A layer's return-per-cost is ``return_ / max(spent, 1e-9)``; that is the
    signal everything below uses. ``floor`` is the minimum return-per-cost a
    layer must clear to earn more budget; ``min_samples`` the minimum evidence
    before any movement at all; ``grow`` / ``shrink`` the movement sizes; ``cap``
    an optional absolute ceiling on a grown budget. A layer below the floor (or
    flagged ``diminishing``) is shrunk, one clearing it with enough samples is
    grown, one with too few samples is held at its current budget, and one that
    has not spent anything yet is held too (nothing to price).

    ``min_budget`` is the absolute floor a shrunk layer is never pushed below;
    ``None`` (default) keeps at least ``5%`` of the layer's *current* budget, so
    "no layer is ever zeroed" is a real floor and not an effectively-zero
    absolute amount. ``floor`` is only ever a return-per-cost ratio. The
    reallocation is per layer, so the total can grow or shrink with the
    movements; pass the same ``budget``s and a fixed total to conserve it.
    Deterministic: same inputs, same outcome.
    """
    total = sum(_finite(l.budget, f"{l.name}.budget") for l in layers)
    reallocated: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    for layer in layers:
        budget = _finite(layer.budget, f"{layer.name}.budget")
        spent = max(_finite(layer.spent, f"{layer.name}.spent"), 0.0)
        ret = _finite(layer.return_, f"{layer.name}.return_")
        samples = int(layer.samples)
        rpc = ret / spent if spent > 0 else 0.0
        if samples < min_samples:
            reallocated[layer.name] = budget
            reasons[layer.name] = (
                f"held: {samples}/{min_samples} measurements, not enough to "
                "move budget on")
        elif spent <= 0:
            reallocated[layer.name] = budget
            reasons[layer.name] = (
                "held: nothing was spent, so there is no return-per-cost to "
                "move budget on")
        elif layer.diminishing or rpc < floor:
            floor_budget = budget * 0.05 if min_budget is None else min_budget
            new_budget = max(budget * (1.0 - shrink), floor_budget)
            reallocated[layer.name] = new_budget
            reasons[layer.name] = (
                f"shrunk: return-per-cost {rpc:.3f} below floor {floor:.3f}"
                + (" (flagged diminishing)" if layer.diminishing else ""))
        else:
            new_budget = budget * (1.0 + grow)
            if cap is not None:
                new_budget = min(new_budget, cap)
            reallocated[layer.name] = new_budget
            reasons[layer.name] = (
                f"grown: return-per-cost {rpc:.3f} at or above floor "
                f"{floor:.3f} over {samples} measurements")
    return PortfolioReport(layers=tuple(layers), total_budget=total,
                           reallocated=reallocated, reasons=reasons)


def layers_from_archive(
    store: ConsolidationStore,
    *,
    artifact_id: str = "meta-rule",
) -> List[LayerBudget]:
    """One layer per decision slot, aggregated from the archive's rule records.

    For each distinct ``problem.slot`` among the ``artifact_id`` records: the
    layer's return is the mean measured ``final_reward``, its spend the mean
    ``rounds`` per rule (matching aggregates, so return-per-cost is per-rule and
    does not decay as a slot accumulates rules), its budget the total rounds the
    slot spent this window, its samples the rule count, and its spread the
    standard deviation of the rewards. A slot with no rules contributes nothing.
    This is the archive's honest view of "which decision's evolution pays";
    other layers (audit bias, harness leaderboard, population islands) are
    caller rows added to the same table.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for rec in store.iter_archive():
        if rec.artifact_id != artifact_id:
            continue
        slot = (rec.problem or {}).get("slot")
        if not slot:
            continue
        row = rows.setdefault(slot, {"rewards": [], "n": 0, "spent": 0.0})
        row["n"] += 1
        row["rewards"].append(float(rec.final_reward))
        row["spent"] += float(rec.rounds)
    out = []
    for slot, row in rows.items():
        rewards = row["rewards"]
        mean = sum(rewards) / len(rewards)
        sd = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        n = int(row["n"])
        out.append(LayerBudget(
            name=f"meta:{slot}", budget=row["spent"], spent=row["spent"] / n,
            return_=mean, samples=n, return_sd=sd,
            note=f"{len(rewards)} rule(s) aggregated"))
    return out


def _finite(value: Any, name: str) -> float:
    v = float(value)
    if v != v or abs(v) == float("inf"):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v
