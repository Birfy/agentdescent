"""What the verifier gets wrong, sorted by what it would take to fix.

The calibration half of this package corrects the verifier's *mean* error. This
half is for the other question: can the verifier be made less wrong in the first
place, and is it worth trying?

Two findings from running it on a real audit shape everything below.

**Do not optimise the verifier against ``delta_hat``.** A mean can be driven to
zero by adding errors in the opposite direction, and that is not an improvement.
Measured on 177 HotpotQA pairs by ``scripts/audit_diagnose.py``, which anyone
can re-run offline, two hard rules that both look like a clean win:

    rule                                  fixed broke   sigma           Delta
    A  answer echoes the question           12    11    0.381 -> 0.410  -74%
    B  answer far shorter than the gold     10     0    0.381 -> 0.324  -32%
    A + B, as anyone would ship them        19    11    0.381 -> 0.362  -97%

**Rule A cuts the bias by three quarters and makes the verifier worse.** Twelve
corrections, eleven fresh mistakes: the mean falls because the errors now cancel,
and the spread -- which is what the acceptance gate's variance is built from --
goes up. Its false-negative rate goes from nothing to 22.4%.

**And bundled with a rule that works, it passes.** B alone is a clean win, the
bundle's ``sigma`` improves, so the bundle "helps" -- while still containing A
and still rejecting 22.4% of correct answers. A bundle launders whatever is in
it, so :func:`evaluate_fix` is meant to be run on one rule at a time.

``sigma`` is the target throughout; ``delta`` is what the calibrator already
handles, and optimising the thing that is already handled breaks the thing that
is not.

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
    other looks like progress in every summary that does not carry it. Rule A in
    the module docstring is exactly that: twelve ``OVER`` errors corrected,
    eleven ``UNDER`` errors created, and every mean-based number improved.
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

    ``sigma`` is the sd of ``f - Y``, and it is the number to watch: it is what
    the acceptance gate's missing variance term is built from, and unlike
    ``delta`` it cannot be improved by making errors in both directions.

    **Weighted by ``1 / inclusion_prob``.** These were raw sample statistics over
    records drawn with deliberately unequal probabilities: Neyman allocation
    oversamples the stratum whose residual varies most, *because* it varies most,
    so an unweighted sd read far above the population value that
    ``Rectification.resid_sd`` reports to the gate -- two numbers called
    ``sigma`` in one report, disagreeing, with a rise in this one a hard blocker.

    At uniform inclusion the weights cancel and every value is bit-identical to
    the unweighted statistic, which is what every measurement committed here was
    drawn at.
    """
    rows = [(r.residual, 1.0 / max(1e-12, float(getattr(r, "inclusion_prob", 1.0) or 1.0)))
            for r in records if r.oracle_score is not None]
    n = len(rows)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "delta": nan, "sigma": nan, "disagree": nan}
    total = sum(w for _, w in rows)
    delta = sum(x * w for x, w in rows) / total
    if n > 1 and total > 1.0:
        sigma = math.sqrt(sum(w * (x - delta) ** 2 for x, w in rows) / (total - 1.0))
    else:
        sigma = 0.0
    return {
        "n": n,
        "delta": delta,
        "sigma": sigma,
        "disagree": sum(w for x, w in rows if x != 0.0) / total,
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
    #: verifier's good judgements this fix destroys. **Not** the false-negative
    #: rate, which it was called until the two were measured side by side on
    #: real data and came out 7.5% and 22.4%: the denominators are the
    #: judgements that were right and the *answers* that were right, and those
    #: are different sets whenever the verifier is wrong in only one direction.
    breakage_rate: float
    #: Correct answers the verifier scores below the truth, over all correct
    #: answers -- the classical false-negative rate, and the same definition
    #: :func:`agentdescent.audit.scorecard.scorecard` reports. Before and after,
    #: because a fix that raises it is spending something the summary rows do
    #: not show.
    false_negative_before: float
    false_negative_after: float
    unchanged: int
    #: Units whose **score** the fix moved at all, verdict flip or not.
    #:
    #: Not the same as ``fixed + broken``, which counts verdict flips, and the
    #: difference is the whole of `changed`: a fix taking 1.0 to 0.4 against a
    #: 0.0 reference moved the residual by 0.6 and flipped nothing, so it landed
    #: in ``unchanged``. The noise floor it is compared against is a raw count of
    #: score differences, so the two sides of `above_the_noise` were being
    #: measured differently. Identical to ``fixed + broken`` on binary scores,
    #: which is every measurement committed here.
    moved: int = 0
    #: Units the **unchanged** verifier flips when it is simply re-run. Zero for
    #: a deterministic one. A stochastic verifier -- an LLM judge is one -- moves
    #: ``sigma`` on its own, so a fix that moves fewer units than this has not
    #: been shown to do anything. See :func:`evaluate_fix`.
    noise_floor: int = 0

    @property
    def changed(self) -> int:
        """Units this fix scored differently, verdict flip or not.

        Reads ``moved``, which counts score differences -- the same thing the
        noise floor counts. It was ``fixed + broken``, which counts verdict
        flips, so a continuous-scored fix could move every unit and be declared
        "not shown to help" against a floor built on the other definition.
        """
        return self.moved

    @property
    def above_the_noise(self) -> bool:
        """Did it move more units than re-running the same verifier does?

        A **necessary** condition, not a sufficient one: one control run gives a
        count, not a distribution, so clearing it by one unit clears nothing in
        particular. What it rules out is the case that has no other symptom --
        a candidate whose whole effect is the verifier disagreeing with itself.
        """
        return self.changed > self.noise_floor

    @property
    def helps(self) -> bool:
        """Did the residual actually shrink, by more than noise?

        The only question worth asking of a fix, and deliberately not "did delta
        shrink": a fix that trades over-crediting for under-crediting improves
        every mean-based number while leaving the verifier no more accurate.

        **And not by less than the verifier's own noise.** An LLM judge re-run
        on the same outputs with the same prompt does not give the same scores:
        measured, 6 of 177 units on HotpotQA. Those six move ``sigma`` by
        themselves, so without a ``noise_floor`` this property will happily
        report that a prompt improves on *itself*. It caught a real reading:
        a one-clause fix moved 8 units on a workload the clause cannot apply
        to, against a floor of 6 -- which is not an effect, it is a re-run.
        """
        return self.sigma_after < self.sigma_before and self.above_the_noise

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
            f"fixed {self.fixed}, broke {self.broken}, unchanged "
            f"{self.unchanged}; {self.breakage_rate:.1%} of the verifier's good "
            f"judgements destroyed; false negatives "
            f"{self.false_negative_before:.1%} -> {self.false_negative_after:.1%}."
            + ("" if not self.noise_floor else
               f" Noise floor {self.noise_floor}: re-running the unchanged "
               f"verifier moves that many units by itself, and this fix moved "
               f"{self.changed}."),
        ])


def evaluate_fix(records: Iterable[AuditRecord],
                 fix: Callable[[AuditRecord, Any], float],
                 context: Optional[Mapping[str, Any]] = None,
                 *, noise_floor: int = 0) -> FixReport:
    """Score a proposed verifier change against **every** labelled pair.

    ``fix(record, context_for_task) -> new verifier score``.

    Every pair, and that is the whole point. Evaluated on the disagreements it
    targets, rule A in the module docstring removes twelve errors and breaks
    nothing -- a clean win by every number a person reaches for. Evaluated on
    all 177 pairs it also breaks eleven correct judgements, and the residual it
    was meant to shrink goes **up**.

    Rule A was not a bad rule -- "reject an answer that echoes the question" is
    obviously right. It was a rule nobody had measured against the answers it
    was not aimed at.

    ``noise_floor`` is how many units the **unchanged** verifier flips when it is
    simply re-run. Leave it at 0 for a deterministic verifier -- a rule, a
    normalisation, exact match. Set it for a stochastic one: an LLM judge
    re-scoring the same stored outputs with the same prompt flipped 6 of 177
    units on HotpotQA, and six flips move ``sigma`` without anything having been
    fixed. Get it by running the unchanged verifier through this same function.

    One rule at a time, too. Bundled with a rule that works, A passes: the
    bundle's residual improves, the verdict reads "helps", and the harmful half
    is invisible in every number the bundle reports.
    """
    resolved = [r for r in records if r.oracle_score is not None]
    if not resolved:
        nan = float("nan")
        return FixReport(0, nan, nan, nan, nan, nan, nan, 0, 0, nan, nan, nan, 0,
                         noise_floor=noise_floor)

    before = [r.residual for r in resolved]
    after: List[float] = []
    fixed = broken = unchanged = 0
    correct_before = 0
    moved = 0
    right_answers = fn_before = fn_after = 0
    for rec in resolved:
        new_f = float(fix(rec, (context or {}).get(rec.task_id)))
        was_right = rec.verifier_score == rec.oracle_score
        is_right = new_f == rec.oracle_score
        correct_before += was_right
        if rec.oracle_score > 0.0:
            right_answers += 1
            fn_before += rec.verifier_score < rec.oracle_score
            fn_after += new_f < rec.oracle_score
        if new_f != rec.verifier_score:
            moved += 1
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
        fixed=fixed, broken=broken, unchanged=unchanged, moved=moved,
        breakage_rate=(broken / correct_before) if correct_before else 0.0,
        false_negative_before=(fn_before / right_answers) if right_answers
        else float("nan"),
        false_negative_after=(fn_after / right_answers) if right_answers
        else float("nan"),
        noise_floor=noise_floor)
