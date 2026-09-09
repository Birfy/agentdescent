"""What the verifier gets wrong, sorted by what it would take to fix.

The calibration half of this package corrects the verifier's *mean* error. This
half is for the other question: can the verifier be made less wrong in the first
place, and is it worth trying?

Two findings from running it on a real audit shape everything below.

**Do not optimise the verifier against ``delta_hat``.** A mean can be driven to
zero by adding errors in the opposite direction, and that is not an improvement.
Measured on 177 HotpotQA pairs, two hard rules that looked like a clean win --
reject an answer that echoes the question, reject one far shorter than the
reference -- moved the numbers like this:

    Delta       +0.175  ->  +0.051   (down 71%)
    sigma_eps    0.381  ->   0.417   (UP)
    disagreement 0.175  ->   0.175   (unchanged)

Eleven corrections and eleven fresh mistakes. The bias fell because the errors
now cancel, not because the verifier learned anything. ``sigma`` is the target
here; ``delta`` is what the calibrator already handles, and optimising the thing
that is already handled breaks the thing that is not.

**Some of the residual is not the verifier's fault and must not be "fixed".**
From the same audit, gold ``'Robert Erskine Childers DSC'`` against an answer of
``'Robert Erskine Childers'``, or ``'from 1986 to 2013'`` against
``'1986 to 2013'``: the judge said these were right and exact match said they
were wrong, and the judge has the better case. Driving those out means training
the judge *into* exact match, which is what having a judge was supposed to avoid.
They are :attr:`Kind.AMBIGUOUS`, they set a floor on ``sigma_eps``, and the
honest report is where that floor is rather than a plan to cross it.

So the two things this module does: sort the disagreements by **what it would
take** to fix them, and -- through :func:`evaluate_fix` -- measure a proposed fix
on the **whole labelled set** rather than on the disagreements it targets. The
second is what turns the table above from a success into a warning.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .records import AuditRecord

__all__ = [
    "Direction",
    "Disagreement",
    "DisagreementReport",
    "FixReport",
    "Kind",
    "classify_disagreements",
    "evaluate_fix",
    "reference_classifier",
    "residual_stats",
]


class Direction(str, Enum):
    """Which way the verifier was wrong.

    Recorded separately from :class:`Kind` because a fix that trades one for the
    other looks like progress in every summary that does not carry it. The
    example in the module docstring is exactly that: eleven ``OVER`` errors
    turned into eleven ``UNDER`` errors, and every mean-based number improved.
    """

    #: ``f > Y`` -- credit for an answer that was wrong. The direction that lets
    #: a loop accept changes which improved nothing.
    OVER = "over"
    #: ``f < Y`` -- a correct answer marked wrong. Costs progress rather than
    #: soundness, and is the usual price of a rule written to cure ``OVER``.
    UNDER = "under"


class Kind(str, Enum):
    """What it would take to fix this disagreement. Ordered by increasing cost.

    The order is the point. The first three need no model and no training, and on
    most verifiers they are most of the residual -- so a diagnosis that jumps
    straight to "the judge needs to be smarter" is usually skipping the cheap
    majority.
    """

    #: The two agree about the answer and disagree about how it is written --
    #: units, significant figures, an equivalent spelling, whitespace. Fixed by
    #: normalising both sides, which costs nothing and cannot introduce a
    #: judgement.
    FORMATTING = "formatting"
    #: The verifier never checked a condition it should have. Fixed by a hard
    #: rule. Cheap, and the category most likely to look like a free win --
    #: :func:`evaluate_fix` before believing it.
    SPEC_GAP = "spec_gap"
    #: A second oracle would disagree too. Not the verifier's error; a
    #: disagreement about what "correct" means. **Not fixable by improving the
    #: verifier**, and the attempt makes it worse at the thing it was for. Sets
    #: the floor on ``sigma_eps``.
    AMBIGUOUS = "ambiguous"
    #: Genuinely needs to understand the content. Only here is a better judge --
    #: a better prompt, a better model, more thinking -- the answer.
    JUDGMENT = "judgment"
    #: Not sorted yet. The honest default: a classifier that cannot see the
    #: reference answer cannot tell formatting from judgement, and guessing would
    #: put a number on the report that nobody measured.
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Disagreement:
    record: AuditRecord
    kind: Kind
    direction: Direction
    note: str = ""

    @property
    def residual(self) -> float:
        return self.record.residual


def residual_stats(records: Iterable[AuditRecord]) -> Dict[str, float]:
    """``n``, ``delta``, ``sigma``, ``disagree`` over resolved records.

    ``sigma`` is the sample sd of ``f - Y``, and it is the number to watch: it is
    what the acceptance gate's missing variance term is built from, and unlike
    ``delta`` it cannot be improved by making errors in both directions.
    """
    resid = [r.residual for r in records if r.oracle_score is not None]
    n = len(resid)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "delta": nan, "sigma": nan, "disagree": nan}
    return {
        "n": n,
        "delta": statistics.fmean(resid),
        "sigma": statistics.stdev(resid) if n > 1 else 0.0,
        "disagree": sum(1 for x in resid if x != 0.0) / n,
    }


# ---------------------------------------------------------------------------
# Classifying
# ---------------------------------------------------------------------------

#: ``(record, reference) -> (Kind, note)``. ``reference`` is whatever the caller
#: keeps per task -- the gold answer, the question, a rubric. The store does not
#: hold it: it is domain-shaped, sometimes large, and the caller has the tasks.
Classifier = Callable[[AuditRecord, Any], Tuple[Kind, str]]


def reference_classifier(
        normalise: Callable[[str], str],
        reference_of: Callable[[Any], str],
        *,
        ambiguous_when: Optional[Callable[[str, str, Any], bool]] = None,
        spec_gap_when: Optional[Callable[[str, str, Any], bool]] = None
) -> Classifier:
    """A classifier for the common case: a reference answer and a normaliser.

    Sorts what it can and leaves the rest :attr:`Kind.UNCLASSIFIED` rather than
    guessing:

    * equal after ``normalise`` -> :attr:`Kind.FORMATTING`
    * ``spec_gap_when(output, reference, context)`` -> :attr:`Kind.SPEC_GAP`
    * ``ambiguous_when(output, reference, context)`` -> :attr:`Kind.AMBIGUOUS`
    * anything else -> :attr:`Kind.UNCLASSIFIED`

    The two predicates are the caller's because they are the two judgements a
    library cannot make. Whether ``'Robert Erskine Childers'`` for
    ``'Robert Erskine Childers DSC'`` is the oracle being pedantic or the answer
    being incomplete is a fact about the domain, and a default that guessed would
    be putting a number nobody measured onto the report that decides what to fix.
    """
    def classify(record: AuditRecord, context: Any) -> Tuple[Kind, str]:
        reference = reference_of(context) if context is not None else ""
        output = record.output or ""
        if reference and normalise(output) == normalise(reference):
            return Kind.FORMATTING, "equal after normalisation"
        if spec_gap_when is not None and spec_gap_when(output, reference, context):
            return Kind.SPEC_GAP, "matched a spec-gap rule"
        if ambiguous_when is not None and ambiguous_when(output, reference, context):
            return Kind.AMBIGUOUS, "the reference is arguable here"
        return Kind.UNCLASSIFIED, ""

    return classify


@dataclass(frozen=True)
class DisagreementReport:
    """The residual, sorted by what fixing it would cost.

    :attr:`floor_sigma` is the number to read first. It is what ``sigma_eps``
    would be if every disagreement except the ambiguous ones were fixed
    perfectly -- the best this verifier can do against **this oracle**. A plan
    that targets a number below it is not a plan to improve the verifier, it is a
    plan to redefine correctness.
    """

    n_pairs: int
    n_disagree: int
    delta: float
    sigma: float
    by_kind: Dict[Kind, int]
    by_direction: Dict[Direction, int]
    by_kind_direction: Dict[Tuple[Kind, Direction], int]
    #: ``sigma_eps`` if every non-ambiguous disagreement were fixed.
    floor_sigma: float
    #: ``sigma_eps`` if this one kind were fixed and nothing else changed. Read
    #: with :attr:`by_kind`: a large bucket that is mostly ambiguous is a floor,
    #: not an opportunity.
    sigma_without: Dict[Kind, float]
    items: List[Disagreement] = field(default_factory=list)

    def of_kind(self, kind: Kind) -> List[Disagreement]:
        return [d for d in self.items if d.kind is kind]

    def to_markdown(self) -> str:
        lines = [
            "| kind | over | under | total | sigma if fixed |",
            "|---|---|---|---|---|",
        ]
        for kind in Kind:
            over = self.by_kind_direction.get((kind, Direction.OVER), 0)
            under = self.by_kind_direction.get((kind, Direction.UNDER), 0)
            if not (over or under):
                continue
            sig = self.sigma_without.get(kind, float("nan"))
            marker = " (floor)" if kind is Kind.AMBIGUOUS else ""
            lines.append(f"| {kind.value}{marker} | {over} | {under} | "
                         f"{over + under} | {sig:.4f} |")
        lines += [
            "",
            f"{self.n_disagree} disagreements in {self.n_pairs} pairs "
            f"({self.n_disagree / max(1, self.n_pairs):.1%}); "
            f"delta {self.delta:+.4f}, sigma {self.sigma:.4f}.",
            "",
            f"**Floor: sigma cannot go below {self.floor_sigma:.4f}** against this "
            "oracle -- that is what remains once every non-ambiguous disagreement "
            "is fixed. Aiming lower means changing the definition of correct, "
            "not improving the verifier.",
        ]
        return "\n".join(lines)


def classify_disagreements(
        records: Iterable[AuditRecord],
        classifier: Optional[Classifier] = None,
        context: Optional[Mapping[str, Any]] = None) -> DisagreementReport:
    """Sort a store's resolved disagreements by what fixing them would take.

    ``context`` maps ``task_id`` to whatever ``classifier`` needs. Without a
    classifier every disagreement lands in :attr:`Kind.UNCLASSIFIED`, which is
    still worth running: the direction split and the counts are the two numbers
    that catch a fix trading ``OVER`` errors for ``UNDER`` ones.
    """
    resolved = [r for r in records if r.oracle_score is not None]
    stats = residual_stats(resolved)

    items: List[Disagreement] = []
    for rec in resolved:
        if rec.residual == 0.0:
            continue
        direction = Direction.OVER if rec.residual > 0 else Direction.UNDER
        kind, note = Kind.UNCLASSIFIED, ""
        if classifier is not None:
            kind, note = classifier(rec, (context or {}).get(rec.task_id))
        items.append(Disagreement(rec, kind, direction, note))

    by_kind = Counter(d.kind for d in items)
    by_direction = Counter(d.direction for d in items)
    by_kd = Counter((d.kind, d.direction) for d in items)

    def sigma_if_fixed(fixed: Sequence[Disagreement]) -> float:
        drop = {id(d.record) for d in fixed}
        resid = [0.0 if id(r) in drop else r.residual for r in resolved]
        return statistics.stdev(resid) if len(resid) > 1 else 0.0

    sigma_without = {kind: sigma_if_fixed([d for d in items if d.kind is kind])
                     for kind in by_kind}
    floor = sigma_if_fixed([d for d in items if d.kind is not Kind.AMBIGUOUS])

    return DisagreementReport(
        n_pairs=stats["n"], n_disagree=len(items),
        delta=stats["delta"], sigma=stats["sigma"],
        by_kind=dict(by_kind), by_direction=dict(by_direction),
        by_kind_direction=dict(by_kd),
        floor_sigma=floor, sigma_without=sigma_without, items=items)


# ---------------------------------------------------------------------------
# Evaluating a proposed fix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixReport:
    """What a proposed change to the verifier actually costs.

    Read :attr:`sigma_after` against :attr:`sigma_before`, not
    :attr:`delta_after` against :attr:`delta_before`. The delta pair can improve
    dramatically while the verifier gets worse -- see the module docstring.
    """

    n_pairs: int
    delta_before: float
    delta_after: float
    sigma_before: float
    sigma_after: float
    disagree_before: float
    disagree_after: float
    #: Pairs the verifier got wrong and the fix gets right.
    fixed: int
    #: Pairs the verifier got **right** and the fix gets wrong. The number a fix
    #: evaluated only on the disagreements it targets can never report.
    broken: int
    #: ``broken`` over the pairs that were correct before -- what fraction of the
    #: verifier's good judgements this fix destroys.
    false_negative_rate: float
    unchanged: int

    @property
    def helps(self) -> bool:
        """Did the residual actually shrink?

        The only question worth asking of a fix, and deliberately not "did delta
        shrink": a fix that trades over-crediting for under-crediting improves
        every mean-based number while leaving the verifier no more accurate.
        """
        return self.sigma_after < self.sigma_before

    def to_markdown(self) -> str:
        verdict = "**helps**" if self.helps else "**does not help**"
        return "\n".join([
            f"{verdict}: sigma {self.sigma_before:.4f} -> {self.sigma_after:.4f}",
            "",
            "| | before | after |",
            "|---|---|---|",
            f"| sigma (the target) | {self.sigma_before:.4f} | {self.sigma_after:.4f} |",
            f"| delta | {self.delta_before:+.4f} | {self.delta_after:+.4f} |",
            f"| disagreement | {self.disagree_before:.3f} | {self.disagree_after:.3f} |",
            "",
            f"fixed {self.fixed}, broke {self.broken}, unchanged {self.unchanged}; "
            f"false-negative rate {self.false_negative_rate:.1%}.",
        ])


def evaluate_fix(records: Iterable[AuditRecord],
                 fix: Callable[[AuditRecord, Any], float],
                 context: Optional[Mapping[str, Any]] = None) -> FixReport:
    """Score a proposed verifier change against **every** labelled pair.

    ``fix(record, context_for_task) -> new verifier score``.

    Every pair, and that is the whole point. Evaluated on the disagreements it
    targets, the two-rule fix in the module docstring removes eleven errors and
    looks like a 35% win; evaluated on all 177 pairs it also breaks eleven
    correct judgements, and the residual it was meant to shrink goes **up**.

    The rule that produced that result was not a bad rule -- "reject an answer
    that echoes the question" is obviously right. It was a rule nobody had
    measured against the answers it was not aimed at.
    """
    resolved = [r for r in records if r.oracle_score is not None]
    if not resolved:
        nan = float("nan")
        return FixReport(0, nan, nan, nan, nan, nan, nan, 0, 0, nan, 0)

    before = [r.residual for r in resolved]
    after: List[float] = []
    fixed = broken = unchanged = 0
    correct_before = 0
    for rec in resolved:
        new_f = float(fix(rec, (context or {}).get(rec.task_id)))
        was_right = rec.verifier_score == rec.oracle_score
        is_right = new_f == rec.oracle_score
        correct_before += was_right
        if was_right and not is_right:
            broken += 1
        elif not was_right and is_right:
            fixed += 1
        else:
            unchanged += 1
        after.append(new_f - rec.oracle_score)

    n = len(resolved)
    return FixReport(
        n_pairs=n,
        delta_before=statistics.fmean(before), delta_after=statistics.fmean(after),
        sigma_before=statistics.stdev(before) if n > 1 else 0.0,
        sigma_after=statistics.stdev(after) if n > 1 else 0.0,
        disagree_before=sum(1 for x in before if x) / n,
        disagree_after=sum(1 for x in after if x) / n,
        fixed=fixed, broken=broken, unchanged=unchanged,
        false_negative_rate=(broken / correct_before) if correct_before else 0.0)
