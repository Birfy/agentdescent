"""Which edit actually caused the improvement?

A self-evolution run sees one scalar: the candidate scored better than the
incumbent. It does **not** see which of the candidate's edits paid for that
gain. That is the credit-assignment problem, and it is the difference between a
reflector that learns "add a tool example" and one that learns "the third edit
in this particular diff was the useful one".

Two attribution rules are shipped, because they answer different questions:

* **Leave-one-out** -- remove one edit, re-evaluate, and read the drop. Cheap,
  intuitive, and the right diagnostic when edits are independent. With
  interactions it over-counts: two edits that only work together each look
  responsible for the whole gain.
* **Shapley value** -- the average marginal contribution over orderings. Exact
  for small edit sets and sampled deterministically for larger ones. The
  credits sum to ``full - baseline`` by construction, which makes budgets and
  reports add up.

The evaluator is any callable ``Sequence[Edit] -> float``. It is the caller's
job to decide how to run the artifact (a model call, a sandbox, a cached
replay); this module only controls *which subsets* are measured. A per-subset
cache means an edit set evaluated for one marginal is not paid for twice.

    report = shapley_credit(edits, evaluate, permutations=128, seed=0)
    print(report.ranking[0].edit_id, report.ranking[0].share)
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Edit",
    "EditCredit",
    "CreditReport",
    "leave_one_out_credit",
    "shapley_credit",
]

Evaluator = Callable[[Sequence["Edit"]], float]


@dataclass(frozen=True)
class Edit:
    """One independently attributable change inside a candidate.

    ``payload`` is opaque; it may be a diff op, a source span, a prompt block,
    or a skill file. ``edit_id`` must be unique within one attribution run, and
    ``parent_id`` is optional lineage metadata.
    """

    edit_id: str
    payload: Any = None
    parent_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.edit_id):
            raise ValueError("Edit.edit_id must be non-empty")


@dataclass(frozen=True)
class EditCredit:
    """The attributed contribution of one edit."""

    edit_id: str
    credit: float
    share: float


@dataclass(frozen=True)
class CreditReport:
    """Attribution over a candidate's edits.

    ``credits`` are sorted by descending credit. ``share`` is the credit divided
    by the total measured gain (zero when there was no gain), so a report with
    one dominant edit reads ``0.92`` without a second arithmetic step.
    ``interactions`` is ``sum(leave-one-out) - gain`` for leave-one-out reports
    and ``0`` for Shapley (whose credits are efficient by construction).
    """

    method: str
    baseline: float
    full: float
    credits: Tuple[EditCredit, ...]
    interactions: float = 0.0
    samples: int = 0
    detail: str = ""

    @property
    def gain(self) -> float:
        return self.full - self.baseline

    @property
    def total_credit(self) -> float:
        return sum(row.credit for row in self.credits)

    @property
    def ranking(self) -> Tuple[EditCredit, ...]:
        return self.credits

    def by_id(self) -> Dict[str, EditCredit]:
        return {row.edit_id: row for row in self.credits}

    def top(self, n: int = 1) -> Tuple[EditCredit, ...]:
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.credits[:n]


def _validate_edits(edits: Sequence[Edit]) -> Tuple[Edit, ...]:
    items = tuple(edits)
    ids = [edit.edit_id for edit in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"edit ids must be unique, got {ids}")
    return items


def _clean(value: Any, where: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{where} returned a non-finite score: {score!r}")
    return score


def _report(method: str, edits: Sequence[Edit], baseline: float, full: float,
            raw: Mapping[str, float], *, interactions: float = 0.0,
            samples: int = 0, detail: str = "") -> CreditReport:
    gain = full - baseline
    rows = []
    for edit in edits:
        credit = float(raw.get(edit.edit_id, 0.0))
        share = credit / gain if gain != 0 else 0.0
        rows.append(EditCredit(edit_id=edit.edit_id, credit=credit, share=share))
    rows.sort(key=lambda row: (-row.credit, row.edit_id))
    return CreditReport(method=method, baseline=baseline, full=full,
                        credits=tuple(rows), interactions=interactions,
                        samples=samples, detail=detail)


def leave_one_out_credit(edits: Sequence[Edit], evaluate: Evaluator, *,
                         baseline: Optional[float] = None,
                         full: Optional[float] = None) -> CreditReport:
    """Attribute by the drop when each edit is removed.

    Every edit is measured exactly twice: once in the full candidate and once
    without it, so an ``n``-edit candidate costs at most ``n + 2`` evaluations
    (the full score and an empty baseline are shared). The report's
    ``interactions`` is the amount by which the per-edit credits over-count the
    measured gain -- positive means edits reinforce each other or overlap.
    """
    items = _validate_edits(edits)
    if not items:
        raise ValueError("leave_one_out_credit needs at least one edit")
    base = _clean(evaluate(()) if baseline is None else baseline,
                  "baseline")
    whole = _clean(evaluate(items) if full is None else full, "full")
    raw: Dict[str, float] = {}
    for index, edit in enumerate(items):
        without = items[:index] + items[index + 1:]
        raw[edit.edit_id] = whole - _clean(evaluate(without), f"without {edit.edit_id}")
    interactions = sum(raw.values()) - (whole - base)
    return _report("leave-one-out", items, base, whole, raw,
                   interactions=interactions,
                   detail=("independent edits" if abs(interactions) < 1e-9
                           else f"interaction/overlap = {interactions:.6g}"))


def _permutations(n: int, samples: int, rng: random.Random,
                  exact_limit: int) -> List[Tuple[int, ...]]:
    if n <= exact_limit:
        return list(itertools.permutations(range(n)))
    seen = set()
    out: List[Tuple[int, ...]] = []
    # Sample without replacement while the space is small enough; for huge edit
    # sets the birthday bound is irrelevant next to the evaluation cost.
    attempts = 0
    while len(out) < samples and attempts < samples * 20:
        attempts += 1
        order = list(range(n))
        rng.shuffle(order)
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def shapley_credit(edits: Sequence[Edit], evaluate: Evaluator, *,
                   permutations: int = 64, seed: int = 0,
                   exact_limit: int = 7,
                   baseline: Optional[float] = None,
                   full: Optional[float] = None) -> CreditReport:
    """Average marginal contribution over permutations (sampled if needed).

    Exact when ``len(edits) <= exact_limit``; otherwise ``permutations`` random
    orderings are sampled with ``random.Random(seed)``. Credits sum to
    ``full - baseline`` even when the evaluator is non-additive, which is the
    property leave-one-out lacks.
    """
    items = _validate_edits(edits)
    if not items:
        raise ValueError("shapley_credit needs at least one edit")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if exact_limit < 0:
        raise ValueError("exact_limit must be >= 0")
    base = _clean(evaluate(()) if baseline is None else baseline, "baseline")
    whole = _clean(evaluate(items) if full is None else full, "full")
    cache: Dict[Tuple[int, ...], float] = {(): base, tuple(range(len(items))): whole}

    def score(subset: Sequence[int]) -> float:
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = _clean(evaluate(tuple(items[i] for i in key)),
                                f"subset {key}")
        return cache[key]

    orders = _permutations(len(items), permutations, random.Random(seed),
                           exact_limit)
    raw = {edit.edit_id: 0.0 for edit in items}
    for order in orders:
        prefix: List[int] = []
        for index in order:
            before = score(prefix)
            prefix.append(index)
            after = score(prefix)
            raw[items[index].edit_id] += after - before
    for edit_id in raw:
        raw[edit_id] /= len(orders)
    return _report("shapley", items, base, whole, raw, samples=len(orders),
                   detail=f"{len(orders)} permutation(s)"
                          + (" (exact)" if len(items) <= exact_limit else " (sampled)"))
