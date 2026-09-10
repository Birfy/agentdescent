"""The card that has to be filled in before a new verifier replaces the old one.

A verifier is the instrument every other number in a run is measured with, so
changing it invalidates the run's history in a way nothing in the run can see.
The plan asks for a scorecard at that moment. This is it -- with the top row
changed, because the row the plan puts first is the one that lies.

The plan's first row is ``delta_hat``, "the only real target: it should fall".
Measured on the Phase 0 audit by ``scripts/audit_diagnose.py``, one
obviously-correct hard rule against an LLM judge -- reject an answer that echoes
the question -- did this:

=====================  ==========  ==========
metric                 before      after
=====================  ==========  ==========
``delta``              +0.175      **+0.045**
``sigma``              0.381       **0.410**
disagreement           0.175       0.130
false-negative rate    0.0%        **22.4%**
=====================  ==========  ==========

Twelve corrections, eleven fresh mistakes, a 74% fall in the bias and a *worse*
verifier. A mean goes to zero when errors cancel, and errors cancelling is not
an improvement. So :func:`scorecard` leads with ``sigma``, treats a rise in it
as a **blocker**, and reports ``delta_hat`` below it with that sentence attached.
The reasoning is in :mod:`agentdescent.audit.diagnose`, where it was measured.

And measure one change at a time. Bundled with a second rule that genuinely
works, the same rule passes every row of this card: the bundle's ``sigma``
improves, so nothing blocks, and the harmful half is invisible in every number
the bundle reports. This card grades what it is handed.

``delta_hat`` is not useless -- it is what the calibrator subtracts, and it is
what a drift monitor watches over generations. It is just not what a *change to
the verifier* should be judged by, because it is the one metric a change can
improve by breaking things.

The other rows are the plan's, unchanged:

``gain_factor``
    Should rise. Approaching 1 means the verifier carries no usable signal about
    the truth any more, at which point the answer is a different verifier rather
    than a bigger audit.
``agreement with the previous version``
    From :func:`rescan`. Large disagreement means the run's recorded history was
    scored by an instrument that no longer exists, and it has to be said out
    loud rather than discovered later.
``cost per decision``
    A verifier that costs what the oracle costs is not a cheap verifier, and the
    whole design rests on it being one.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .calibrator import Rectification
from .diagnose import residual_stats
from .estimate import hajek_mean
from .records import AuditRecord

__all__ = ["FLIP_ALARM", "Cost", "Goal", "Metric", "RescanReport", "Scorecard",
           "rescan", "scorecard"]

#: Flip rate above which a rescan is called out rather than reported. The plan's
#: number. It is a policy choice and not a measurement -- there is no rate at
#: which silently re-scoring a run's history becomes fine.
FLIP_ALARM = 0.10


class Goal(Enum):
    """Which way a metric is supposed to move."""

    LOWER = "lower is better"
    HIGHER = "higher is better"
    WATCH = "no target; read it"


@dataclass(frozen=True)
class Metric:
    """One row. ``previous`` is ``None`` when there is nothing to compare to."""

    name: str
    value: float
    goal: Goal
    previous: Optional[float] = None
    note: str = ""
    #: This row alone can refuse the change.
    blocking: bool = False
    #: This row *did*. Set by :func:`scorecard` after the rows are read, because
    #: "can block" and "blocked" render the same otherwise, and a baseline card
    #: with no comparison would show every gate as though it had fired.
    triggered: bool = False

    @property
    def change(self) -> float:
        if self.previous is None:
            return float("nan")
        return self.value - self.previous

    @property
    def verdict(self) -> str:
        if self.previous is None or self.goal is Goal.WATCH:
            return "--"
        delta = self.change
        if delta != delta or delta == 0.0:
            return "unchanged"
        better = delta < 0.0 if self.goal is Goal.LOWER else delta > 0.0
        return "better" if better else "worse"

    @property
    def regressed(self) -> bool:
        return self.verdict == "worse"


@dataclass(frozen=True)
class Cost:
    """Seconds per decision, for the verifier and for the thing it stands in for.

    Both measured the same way by the caller. The ratio is the point: a verifier
    at a tenth of the oracle's cost can be run on every rollout, one at half of
    it cannot, and nothing else in the package notices the difference.
    """

    verifier_seconds: float
    oracle_seconds: float = float("nan")

    @property
    def ratio(self) -> float:
        if not self.oracle_seconds or self.oracle_seconds != self.oracle_seconds:
            return float("nan")
        return self.verifier_seconds / self.oracle_seconds


# ---------------------------------------------------------------------------
# The rescan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RescanReport:
    """What a new verifier would have said about outputs the run already scored.

    **Not a re-decide, and the difference matters.** The plan describes replaying
    the Ledger: "the verifier is cheap and the artifacts are all stored, so this
    sweep is nearly free". The Ledger stores artifact *states*. The outputs those
    artifacts produced -- the things a verifier scores -- were never kept, so
    re-deciding a past merge means re-running the agent over both sides' held-out
    sets, which is a whole run's worth of rollouts and the expensive half.

    What *is* nearly free is re-scoring the outputs the **audit store** kept,
    which is what this does. Those are a probability sample of the run rather
    than all of it, so the numbers here are estimates with an ``n``, weighted by
    ``inclusion_prob`` because the sample was deliberately not representative.

    That is a weaker claim than the plan's and an honest one. It still answers
    the question the row exists for: how much of what the run recorded rests on
    scores this verifier would now give differently.
    """

    n: int
    n_artifacts: int
    #: Fraction of stored outputs the two verifiers score identically.
    agreement: float
    #: ``E[f_new] - E[f_old]`` over the sample, inclusion-weighted.
    mean_shift: float
    #: Sd of the per-unit change. Zero with a non-zero ``mean_shift`` is a
    #: uniform re-scaling; large with a zero mean shift is a verifier that
    #: changed its mind in both directions and looks unchanged in every average.
    sigma_shift: float
    #: Artifact signature -> ``{"n", "old", "new", "shift"}``, inclusion-weighted.
    by_artifact: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: ``(base, cand, was, now)`` for every compared pair whose ordering reverses.
    flipped: List[Tuple[str, str, float, float]] = field(default_factory=list)
    n_pairs: int = 0
    #: Accuracy against the oracle labels, where the sample has them.
    sigma_before: float = float("nan")
    sigma_after: float = float("nan")

    @property
    def flip_rate(self) -> float:
        return len(self.flipped) / self.n_pairs if self.n_pairs else float("nan")

    @property
    def alarming(self) -> bool:
        rate = self.flip_rate
        return rate == rate and rate > FLIP_ALARM

    def to_markdown(self) -> str:
        rows = [
            f"# Rescan -- {self.n} stored outputs, {self.n_artifacts} artifacts",
            "",
            f"- agreement with the previous verifier: **{self.agreement:.1%}**",
            f"- mean shift: {self.mean_shift:+.4f}  (sd of the per-unit change "
            f"{self.sigma_shift:.4f})",
        ]
        if self.sigma_before == self.sigma_before:
            rows.append(f"- residual sd against ground truth: "
                        f"{self.sigma_before:.4f} -> {self.sigma_after:.4f}")
        if self.n_pairs:
            mark = " **(above the alarm)**" if self.alarming else ""
            rows.append(f"- artifact pairs whose ordering reverses: "
                        f"**{len(self.flipped)}/{self.n_pairs}** "
                        f"({self.flip_rate:.1%}){mark}")
            for base, cand, was, now in self.flipped[:10]:
                rows.append(f"    - `{base[:12]}` vs `{cand[:12]}`: "
                            f"{was:+.3f} -> {now:+.3f}")
        if self.alarming:
            rows += ["", "The recorded history was scored by an instrument that "
                     "no longer exists. Mark the affected version chain "
                     "explicitly; do not let the old numbers stand unlabelled."]
        return "\n".join(rows)


def rescan(records: Sequence[AuditRecord],
           new_verifier: Callable[[AuditRecord, Any], float],
           context: Optional[Mapping[str, Any]] = None,
           *, pairs: Optional[Sequence[Tuple[str, str]]] = None) -> RescanReport:
    """Re-score the outputs the audit kept, and see what would have moved.

    ``new_verifier(record, context_for_task)`` returns the score the candidate
    verifier gives that stored output -- the same shape as
    :func:`~agentdescent.audit.diagnose.evaluate_fix`'s ``fix``, so a proposed
    rule can be run through both without adapting it.

    ``pairs`` are ``(base_signature, candidate_signature)`` as the run actually
    compared them, when the caller kept them. Without it every artifact is
    compared against every other, which over-counts pairs that no merge ever
    considered -- so a flip rate from the fallback is a property of the audited
    sample, not of the run's merge history, and is labelled that way in the
    report.
    """
    context = context or {}
    rows = list(records)
    if not rows:
        nan = float("nan")
        return RescanReport(0, 0, nan, nan, nan)

    old, new, probs, sigs = [], [], [], []
    for rec in rows:
        old.append(rec.verifier_score)
        new.append(float(new_verifier(rec, context.get(rec.task_id))))
        probs.append(rec.inclusion_prob)
        sigs.append(rec.artifact_signature)

    changes = [b - a for a, b in zip(old, new)]
    by_artifact: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        grouped[sig].append(i)
    for sig, idx in grouped.items():
        p = [probs[i] for i in idx]
        was, now = (hajek_mean([old[i] for i in idx], p),
                    hajek_mean([new[i] for i in idx], p))
        by_artifact[sig] = {"n": float(len(idx)), "old": was, "new": now,
                            "shift": now - was}

    if pairs is None:
        names = sorted(by_artifact)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    flipped, n_pairs = [], 0
    for base, cand in pairs:
        if base not in by_artifact or cand not in by_artifact:
            continue
        n_pairs += 1
        was = by_artifact[cand]["old"] - by_artifact[base]["old"]
        now = by_artifact[cand]["new"] - by_artifact[base]["new"]
        if was and now and (was > 0) != (now > 0):
            flipped.append((base, cand, was, now))

    labelled = [(i, r) for i, r in enumerate(rows) if r.oracle_score is not None]
    if len(labelled) > 1:
        before = statistics.stdev([old[i] - r.oracle_score for i, r in labelled])
        after = statistics.stdev([new[i] - r.oracle_score for i, r in labelled])
    else:
        before = after = float("nan")

    return RescanReport(
        n=len(rows), n_artifacts=len(by_artifact),
        agreement=sum(1 for c in changes if c == 0.0) / len(changes),
        mean_shift=hajek_mean(new, probs) - hajek_mean(old, probs),
        sigma_shift=statistics.stdev(changes) if len(changes) > 1 else 0.0,
        by_artifact=by_artifact, flipped=flipped, n_pairs=n_pairs,
        sigma_before=before, sigma_after=after)


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scorecard:
    """The rows, and whether they add up to a change worth making.

    :attr:`blockers` is the whole verdict. Not a score -- a weighted total would
    let a large fall in the metric that lies buy a small rise in the one that
    does not, which is exactly the trade this card exists to refuse.
    """

    version: str
    previous_version: Optional[str]
    metrics: List[Metric]
    blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    rescan: Optional[RescanReport] = None
    computed_at: float = 0.0

    @property
    def ship(self) -> bool:
        """True only when nothing blocks. Absent a comparison, nothing blocks."""
        return not self.blockers

    def get(self, name: str) -> Optional[Metric]:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_markdown(self) -> str:
        head = f"# Verifier scorecard -- `{self.version}`"
        if self.previous_version:
            head += f" against `{self.previous_version}`"
        rows = [head, "",
                "| metric | value | previous | change | verdict |",
                "|---|---|---|---|---|"]
        for m in self.metrics:
            prev = "--" if m.previous is None else f"{m.previous:.4f}"
            change = "--" if m.previous is None else f"{m.change:+.4f}"
            flag = (" **(blocking)**" if m.triggered
                    else " (can block)" if m.blocking else "")
            rows.append(f"| {m.name} | {m.value:.4f} | {prev} | {change} | "
                        f"{m.verdict}{flag} |")
        for m in self.metrics:
            if m.note:
                rows += ["", f"**{m.name}** -- {m.note}"]
        if self.blockers:
            rows += ["", "## Do not ship", ""]
            rows += [f"- {b}" for b in self.blockers]
        else:
            rows += ["", "## Nothing blocks", ""]
            rows.append("Every row that has a target moved the right way, or "
                        "there was nothing to compare against.")
        for note in self.notes:
            rows += ["", note]
        if self.rescan is not None:
            rows += ["", "---", "", self.rescan.to_markdown()]
        return "\n".join(rows)


def _false_negative_rate(records: Iterable[AuditRecord]) -> float:
    """Of the answers ground truth called right, how many did the verifier mark
    down. The direction a bias-driven "improvement" buys its improvement with."""
    right = [r for r in records
             if r.oracle_score is not None and r.oracle_score > 0.0]
    if not right:
        return float("nan")
    return sum(1 for r in right if r.verifier_score < r.oracle_score) / len(right)


def scorecard(current: Rectification, records: Sequence[AuditRecord], *,
              previous: Optional[Rectification] = None,
              previous_records: Sequence[AuditRecord] = (),
              rescan_report: Optional[RescanReport] = None,
              cost: Optional[Cost] = None,
              previous_cost: Optional[Cost] = None,
              max_false_negative: float = 0.05,
              max_cost_ratio: float = 0.25) -> Scorecard:
    """Fill the card for ``current``, against ``previous`` where there is one.

    ``records`` must be **fresh CALIBRATION labels** for the new verifier -- the
    improvement pool was chosen to make its units agree with the verifier, so
    scoring a change on it reads as more honest than the truth.
    :meth:`~agentdescent.audit.store.AuditStore.for_calibration` asserts that
    split; this function trusts what it is handed.

    ``max_false_negative`` and ``max_cost_ratio`` are **policy dials, not
    measurements**. Nothing here can tell you what rate of correct answers your
    loop can afford to have rejected; the defaults are a starting point that
    will refuse the Phase 0 echo rule, which rejected 22.4% of them.
    """
    now = residual_stats(records)
    was = residual_stats(previous_records) if previous_records else None
    metrics: List[Metric] = []
    blockers: List[str] = []
    notes: List[str] = []

    metrics.append(Metric(
        "sigma", now["sigma"], Goal.LOWER,
        previous=was["sigma"] if was else None,
        note="The spread of `f - Y`, and the row to read first. It is what the "
             "acceptance gate's variance is built from, and unlike the bias it "
             "cannot be improved by making errors in both directions.",
        blocking=True))

    fn_now = _false_negative_rate(records)
    metrics.append(Metric(
        "false-negative rate", fn_now, Goal.LOWER,
        previous=_false_negative_rate(previous_records) if previous_records
        else None,
        note=f"Correct answers the verifier marks down. Bounded at "
             f"{max_false_negative:.0%} by policy, not by measurement.",
        blocking=True))

    metrics.append(Metric(
        "delta_hat", current.delta_hat, Goal.WATCH,
        previous=previous.delta_hat if previous else None,
        note="Reported, not targeted. A mean error goes to zero when errors "
             "cancel: on the audit that motivated this card, a correct-looking "
             "rule cut it 74% while making the verifier worse. It is what the "
             "calibrator subtracts and what a drift monitor watches -- not what "
             "a change to the verifier should be judged by."))

    metrics.append(Metric(
        "disagreement", now["disagree"], Goal.LOWER,
        previous=was["disagree"] if was else None,
        note="How often the two scorers differ at all, in either direction."))

    metrics.append(Metric(
        "gain_factor", current.gain_factor, Goal.HIGHER,
        previous=previous.gain_factor if previous else None,
        note="Approaching 1 means the verifier carries no usable signal about "
             "the truth, at which point the answer is a different verifier "
             "rather than a bigger audit."))

    metrics.append(Metric("labels", float(now["n"]), Goal.WATCH,
                          previous=float(was["n"]) if was else None))

    cost_blocked = False
    if cost is not None:
        metrics.append(Metric(
            "seconds per decision", cost.verifier_seconds, Goal.LOWER,
            blocking=True,
            previous=previous_cost.verifier_seconds if previous_cost else None,
            note="A verifier that costs what the oracle costs is not a cheap "
                 "verifier, and the whole design rests on it being one."))
        ratio = cost.ratio
        if ratio == ratio and ratio > max_cost_ratio:
            blockers.append(
                f"a decision now costs {ratio:.0%} of an oracle call "
                f"(cap {max_cost_ratio:.0%}); at that price audit every unit "
                f"and skip the proxy")
            cost_blocked = True

    if current.is_stale:
        notes.append(
            f"> The rectification is stale ({current.stale_reason}). Every row "
            f"drawn from it is the last one computed, not a measurement of this "
            f"verifier.")

    fn_metric = metrics[1]
    triggered = set()
    if fn_now == fn_now and fn_now > max_false_negative:
        blockers.append(
            f"{fn_now:.1%} of correct answers are marked down, above the "
            f"{max_false_negative:.0%} policy bound")
        triggered.add(fn_metric.name)
    sigma_metric = metrics[0]
    if sigma_metric.regressed:
        blockers.append(
            f"the residual grew {sigma_metric.previous:.4f} -> "
            f"{sigma_metric.value:.4f}; whatever else improved was bought with "
            f"new errors in the opposite direction")
        triggered.add(sigma_metric.name)
    if fn_metric.regressed and not sigma_metric.regressed:
        notes.append(
            "> The false-negative rate rose while the residual fell. That is a "
            "real trade and can be the right one -- a verifier that is stricter "
            "and more accurate -- but it is a change in what the loop is "
            "allowed to commit, not a free improvement.")

    if rescan_report is not None and rescan_report.alarming:
        blockers.append(
            f"{rescan_report.flip_rate:.0%} of compared artifact pairs reverse "
            f"order under the new verifier (alarm at {FLIP_ALARM:.0%}); the "
            f"recorded history was scored by an instrument that no longer "
            f"exists and the affected version chain has to be marked")

    if previous is None and not previous_records:
        notes.append(
            "> Nothing to compare against, so nothing blocks. This is a "
            "baseline, not a passing grade.")

    if cost_blocked:
        triggered.add("seconds per decision")
    metrics = [replace(m, triggered=m.name in triggered) for m in metrics]

    return Scorecard(
        version=current.verifier_version,
        previous_version=previous.verifier_version if previous else None,
        metrics=metrics, blockers=blockers, notes=notes,
        rescan=rescan_report, computed_at=time.time())
