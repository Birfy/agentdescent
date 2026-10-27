"""Leakage guard: a self-modification may not memorise gold answers.

A recursive system that evolves against examples can learn the answers instead
of the rule: an appended skill line that contains a task id or a gold string
scores perfectly on the archive and generalises to nothing. This is the
contamination failure the RRSI screen addresses inside one algorithm; this
module makes it a general acceptance policy, so every adoption path can share
the same check.

    Policies(screen=LeakageGuard(gold_answers + task_ids))

It also implements the acceptance-policy shape as a backstop, but the screen is
the right seam: a leaking card is settled before it buys a held-out sweep and
before its inflated score reaches the merge path.

``protected`` strings shorter than ``min_length`` are ignored (a one-character
gold would match everything). Matching is case-insensitive and whitespace is
collapsed; the candidate's whole rendered artifact is checked, because a leak
can be in any touched key. When an acceptance-form decision has a base
artifact, only newly introduced hits are refused, so a protected string that
was already in the seed does not freeze every future candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .advantage import _ForwardsInstall
from .policies import AcceptDecision, MergeContext

__all__ = ["LeakageGuard", "LeakageReport", "find_leakage"]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def find_leakage(text: str, protected: Sequence[str], *,
                 min_length: int = 3) -> Tuple[str, ...]:
    """The protected strings that appear in ``text`` (normalised match)."""
    haystack = _normalise(text)
    hits = []
    seen = set()
    for raw in protected:
        needle = _normalise(raw)
        if len(needle) < min_length or needle in seen:
            continue
        seen.add(needle)
        if needle and needle in haystack:
            hits.append(str(raw))
    return tuple(hits)


def _diff_text(diff: Any) -> str:
    ops = getattr(diff, "ops", None) or {}
    return "\n".join([str(key) for key in ops]
                     + [str(value) for value in ops.values()])


@dataclass(frozen=True)
class LeakageReport:
    """What the guard found and which protected strings matched."""

    leaked: Tuple[str, ...] = ()
    detail: str = ""

    @property
    def passed(self) -> bool:
        return not self.leaked


class LeakageGuard(_ForwardsInstall):
    """Reject a candidate whose rendered artifact contains protected text.

    ``inner`` is the acceptance policy that would have run without the guard;
    it is consulted only after no protected string is found. ``inner=None`` is
    the shipped ``DefaultAcceptance`` (configured through the forwarded hooks),
    so adding the guard can only make acceptance stricter. ``LeakageReport``
    carries the matching strings for a log; the detail names them without
    printing the artifact.
    """

    def __init__(self, protected: Sequence[str], *, inner: Optional[Any] = None,
                 min_length: int = 3) -> None:
        from .defaults import DefaultAcceptance

        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        self.protected = tuple(str(value) for value in protected)
        self.inner = inner if inner is not None else DefaultAcceptance()
        self.min_length = int(min_length)

    def check(self, artifact: Any, *, base: Optional[Any] = None) -> LeakageReport:
        """Protected strings in ``artifact`` that are not already in ``base``."""
        text = artifact.render() if hasattr(artifact, "render") else str(artifact)
        leaked = find_leakage(text, self.protected, min_length=self.min_length)
        if base is not None and leaked:
            base_text = (base.render() if hasattr(base, "render") else str(base))
            existing = {_normalise(value) for value in find_leakage(
                base_text, self.protected, min_length=self.min_length)}
            leaked = tuple(value for value in leaked
                           if _normalise(value) not in existing)
        if not leaked:
            return LeakageReport()
        shown = ", ".join(repr(value) for value in leaked)
        return LeakageReport(
            leaked=leaked,
            detail=f"candidate contains protected string(s) {shown}")

    def screen(self, card: Any) -> Optional[str]:
        """Reject a card whose diff introduces a protected string."""
        diff = getattr(card, "diff", None)
        if diff is None:
            return "card carries no diff"
        leaked = find_leakage(_diff_text(diff), self.protected,
                              min_length=self.min_length)
        if not leaked:
            return None
        shown = ", ".join(repr(value) for value in leaked)
        return f"candidate diff contains protected string(s) {shown}"

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        report = self.check(ctx.candidate, base=ctx.artifact)
        if not report.passed:
            # ``critic-rejected`` is an existing MergeOutcome, so the
            # aggregator does not have to learn a new category for this gate.
            return AcceptDecision(False, "critic-rejected", report.detail)
        if hasattr(self.inner, "accept"):
            return self.inner.accept(ctx)
        return self.inner(ctx)
