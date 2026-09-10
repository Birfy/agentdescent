"""Searching for a verifier fix, instead of guessing one and measuring it once.

:mod:`~agentdescent.audit.diagnose` sorts the verifier's errors by what fixing
them would cost and scores **a** proposed fix. It does not propose one, so the
improvement pool's labels were being spent on a diagnosis nobody acted on.

This is the first rung of the plan's ladder, automated: enumerate the hard rules
a person reaches for, score every combination of them on the whole labelled set,
and rank by the residual. No model, no training, and no judgement about the
domain beyond the reference the caller already has.

Run on the Phase 0 audit, over eight candidate rules taken two at a time:

    sigma   fixed broke   FN    rules
    0.3812      -     -    -    (the verifier as it is)
    0.2421     20     0   0%    far-shorter(0.6) + far-longer(1.6)
    0.2521     19     0   0%    far-shorter(0.6) + shares-no-token-with-reference
    0.2955     14     0   0%    far-longer(1.6) + shares-no-token-with-reference

**The best pair cuts the residual 36% and breaks nothing.** It is better than
either rule a person picked by hand, and `echoes-the-question` -- the rule that
cut the bias 74% while making the verifier worse -- does not appear anywhere in
the ranking, because the ranking is on the residual and that is what the residual
is for.

It also lands at 0.2421 against a
:attr:`~agentdescent.audit.diagnose.DisagreementReport.floor_sigma` of 0.2203:
the search gets to within two points of what the classifier said was reachable,
and the rest is the AMBIGUOUS bucket, which is the part that must not be fixed.
A search that goes *below* the floor has not found something clever -- it has
found that the classifier is wrong, and :class:`SearchReport` says so.

Three rules about the search itself, each of which was a way to be wrong:

* **Ranked by the residual, never by the bias.** Same reason as
  :attr:`~agentdescent.audit.diagnose.FixReport.helps`: a mean goes to zero when
  errors cancel.
* **Every member of a winning combination is also scored alone**, and a
  combination that helps while containing a member that does not is **flagged**.
  A bundle launders whatever is in it, and the search is a bundle generator.
* **A rule may only reject.** It never raises a score. A "hard rule" that could
  raise one would be free to buy a lower residual with a higher false-negative
  rate in the same move, and the report would show only the first.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import (Any, Callable, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from .diagnose import FixReport, evaluate_fix
from .records import AuditRecord

__all__ = ["Candidate", "MAX_COMBINATIONS", "Rule", "SearchReport",
           "length_rules", "search"]

#: Combinations the search will build before it complains. Not a performance
#: limit -- each one is a pass over the labelled set, which is a few hundred
#: rows -- but a statistical one: every combination scored is another chance for
#: one to look good by luck, and a search over a thousand of them on two hundred
#: labels is fitting the noise.
MAX_COMBINATIONS = 200

#: ``(record, context_for_task) -> True to reject this output``.
Predicate = Callable[[AuditRecord, Any], bool]


@dataclass(frozen=True)
class Rule:
    """A named reason to mark an output wrong that the verifier marked right."""

    name: str
    predicate: Predicate

    def __call__(self, record: AuditRecord, context: Any) -> bool:
        return bool(self.predicate(record, context))


@dataclass(frozen=True)
class Candidate:
    """One combination of rules and what it does to the whole labelled set."""

    rules: Tuple[str, ...]
    report: FixReport
    #: Members that do not help on their own. A combination carrying one is
    #: passing because of its other half.
    passengers: Tuple[str, ...] = ()

    @property
    def sigma(self) -> float:
        return self.report.sigma_after

    @property
    def helps(self) -> bool:
        return self.report.helps

    @property
    def launders(self) -> bool:
        """Does this combination pass while carrying a rule that does not?"""
        return self.helps and bool(self.passengers)

    def __str__(self) -> str:
        return " + ".join(self.rules)


@dataclass
class SearchReport:
    """Every combination tried, best first, and the two things to distrust."""

    sigma_before: float
    candidates: List[Candidate] = field(default_factory=list)
    n_rules: int = 0
    n_combinations: int = 0
    #: From :attr:`~agentdescent.audit.diagnose.DisagreementReport.floor_sigma`,
    #: when the caller had one. What remains once every non-ambiguous
    #: disagreement is fixed.
    floor: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def clean(self) -> List[Candidate]:
        """Combinations that help and carry no passenger."""
        return [c for c in self.candidates if c.helps and not c.launders]

    @property
    def below_the_floor(self) -> List[Candidate]:
        """Combinations that beat what the classifier said was reachable.

        Not a triumph. Either some disagreement classified AMBIGUOUS is in fact
        fixable, or -- far more likely on a handful of labels -- the combination
        is fitting the sample. Look at the units it changed before believing it.
        """
        if self.floor != self.floor:
            return []
        return [c for c in self.candidates if c.sigma < self.floor]

    def to_markdown(self, limit: int = 10) -> str:
        rows = [f"# Rule search -- {self.n_rules} rules, "
                f"{self.n_combinations} combinations", "",
                f"The verifier as it is: sigma **{self.sigma_before:.4f}**"
                + ("" if self.floor != self.floor
                   else f"; floor {self.floor:.4f}"),
                "",
                "| sigma | fixed | broke | false negatives | rules |",
                "|---|---|---|---|---|"]
        for cand in self.candidates[:limit]:
            mark = " **(launders)**" if cand.launders else ""
            rows.append(
                f"| {cand.sigma:.4f} | {cand.report.fixed} | "
                f"{cand.report.broken} | "
                f"{cand.report.false_negative_after:.1%} | {cand}{mark} |")
        if self.below_the_floor:
            rows += ["", "!!! warning \"Below the floor\"",
                     "    These beat what the classifier said was reachable. "
                     "Either a disagreement",
                     "    classified AMBIGUOUS is fixable after all, or the "
                     "combination is fitting",
                     "    the sample. Read the units it changed before "
                     "believing it.", ""]
            rows += [f"    - {c} ({c.sigma:.4f})" for c in self.below_the_floor]
        launderers = [c for c in self.candidates[:limit] if c.launders]
        if launderers:
            rows += ["", "## Passing on someone else's work", ""]
            rows += [f"- **{c}** helps, and `{'`, `'.join(c.passengers)}` does "
                     f"not help alone" for c in launderers]
            rows.append("")
            rows.append("A bundle launders whatever is in it. Ship the members "
                        "that earn their place.")
        for warning in self.warnings:
            rows += ["", f"> {warning}"]
        return "\n".join(rows)


def _fix_for(rules: Sequence[Rule]):
    def fix(record: AuditRecord, context: Any) -> float:
        # Reject only. A rule that could raise a score would be free to buy a
        # lower residual with a higher false-negative rate in one move, and the
        # report would show only the first.
        if record.verifier_score <= 0.0:
            return record.verifier_score
        return 0.0 if any(r(record, context) for r in rules) \
            else record.verifier_score
    return fix


def search(records: Sequence[AuditRecord], rules: Sequence[Rule],
           context: Optional[Mapping[str, Any]] = None, *,
           max_size: int = 2, floor: float = float("nan"),
           max_combinations: int = MAX_COMBINATIONS) -> SearchReport:
    """Score every combination of ``rules`` up to ``max_size``, best first.

    ``records`` must be the **improvement** pool. Labels used to choose a rule
    cannot then measure the bias it leaves behind -- that is constraint 2, and
    a search is the most thorough way there is to violate it.
    """
    rows = [r for r in records if r.oracle_score is not None]
    warnings: List[str] = []
    if not rows or not rules:
        return SearchReport(float("nan"), n_rules=len(rules),
                            warnings=["nothing to search"])

    baseline = evaluate_fix(rows, lambda rec, ctx: rec.verifier_score, context)
    alone = {r.name: evaluate_fix(rows, _fix_for([r]), context) for r in rules}

    combos: List[Tuple[str, ...]] = []
    for size in range(1, max(1, max_size) + 1):
        combos.extend(itertools.combinations([r.name for r in rules], size))
    if len(combos) > max_combinations:
        warnings.append(
            f"{len(combos)} combinations over {len(rows)} labels is fitting the "
            f"noise; truncated to {max_combinations}. Every combination scored "
            f"is another chance for one to look good by luck")
        combos = combos[:max_combinations]

    by_name = {r.name: r for r in rules}
    candidates: List[Candidate] = []
    for combo in combos:
        report = (alone[combo[0]] if len(combo) == 1
                  else evaluate_fix(rows, _fix_for([by_name[n] for n in combo]),
                                    context))
        passengers = tuple(n for n in combo if not alone[n].helps)
        candidates.append(Candidate(combo, report, passengers))

    candidates.sort(key=lambda c: (c.sigma, len(c.rules)))
    return SearchReport(sigma_before=baseline.sigma_before,
                        candidates=candidates, n_rules=len(rules),
                        n_combinations=len(combos), floor=floor,
                        warnings=warnings)


# ---------------------------------------------------------------------------
# The rules a person reaches for
# ---------------------------------------------------------------------------

def length_rules(normalise: Callable[[str], str],
                 reference_of: Callable[[Any], str], *,
                 question_of: Optional[Callable[[Any], str]] = None,
                 short: Sequence[float] = (0.6, 0.4),
                 long: Sequence[float] = (1.6, 3.0)) -> List[Rule]:
    """Candidate rules that need only a reference and a normaliser.

    Every one of them is a *shape* argument -- the answer is far shorter than
    the reference, far longer, shares no word with it, is empty. None needs a
    model and none is a judgement about the domain, which is what puts them on
    the first rung of the ladder.

    ``normalise`` must be the **oracle's own**. A rule that measures length in
    one normalisation while the truth compares in another is answering a
    different question with the same word.

    ``question_of`` adds ``echoes-the-question``. It is included because it is
    the first rule everyone writes, and it is the rule measured in
    :mod:`~agentdescent.audit.diagnose` that cuts the bias by three quarters and
    makes the verifier worse -- so a search that ranks it last is the search
    doing its job, and one that ranks it first is a reason to look at the data.
    """
    rules: List[Rule] = []

    if question_of is not None:
        def echoes(record, context):
            if context is None:
                return False
            out = normalise(record.output)
            return bool(out) and out in normalise(question_of(context))
        rules.append(Rule("echoes-the-question", echoes))

    def _pair(record, context):
        if context is None:
            return "", ""
        return normalise(record.output), normalise(reference_of(context))

    for ratio in short:
        def shorter(record, context, ratio=ratio):
            out, ref = _pair(record, context)
            return bool(out) and bool(ref) and out != ref \
                and len(out) < ratio * len(ref)
        rules.append(Rule(f"far-shorter({ratio})", shorter))

    for ratio in long:
        def longer(record, context, ratio=ratio):
            out, ref = _pair(record, context)
            return bool(out) and bool(ref) and out != ref \
                and len(out) > ratio * len(ref)
        rules.append(Rule(f"far-longer({ratio})", longer))

    def empty(record, context):
        return not normalise(record.output)
    rules.append(Rule("empty", empty))

    def disjoint(record, context):
        out, ref = _pair(record, context)
        words, reference = set(out.split()), set(ref.split())
        return bool(words) and bool(reference) and not (words & reference)
    rules.append(Rule("shares-no-token-with-reference", disjoint))

    return rules
