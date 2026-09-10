"""Can this verifier put candidates in the right order? Nothing else measures it.

The acceptance gate does exactly one thing: decide whether a candidate is better
than a baseline. Everything else in this package measures how *far* the verifier
is from the truth -- :attr:`~agentdescent.audit.calibrator.Rectification.delta_hat`
its mean error, ``resid_sd`` the spread of it,
:attr:`~agentdescent.audit.ppi.PPIResult.gain_factor` how much an estimator can
borrow from it. A verifier can be badly wrong on all three and still order every
comparison correctly -- add 0.2 to every score and nothing the gate decides
changes -- and it can be close on all three and still pick the wrong winner.

Measured on the Phase 0 audit, five artifacts from one run:

    artifact          n   mean f   mean Y
    0bf0b7ead111...  32    0.594    0.469
    26ecdd4b7262...  32    0.625    0.375
    434f37209 8a0...  32    0.188    0.062
    4aab01c9a0d2...  32    0.000    0.000
    bab6bec25105...  49    0.714    0.408

**Eight of ten artifact pairs are ordered the same way; two are reversed.** The
artifact the verifier ranks *first* (0.714) is third by ground truth, and the
reversal against `0bf0b7ea` is on an apparent 12-point improvement -- exactly the
size of gap the gate commits on. ``delta_hat`` and ``resid_sd`` both say "this
judge is generous". Neither says "it picks the wrong winner in one comparison
out of five", and that is the only thing the gate does.

Three things to be careful about, because each makes the number look better or
worse than it is:

**Unit-level Kendall tau is nearly uninformative here, and it is the number
people ask for.** On the same data: 4753 concordant pairs, **zero** discordant,
``tau_b = 0.681``. That is not evidence of good ordering -- it is a restatement
of the bias being one-directional. Two units can only be discordant if the
verifier prefers one and the truth prefers the other, and when every error runs
the same way (``f > Y``, never ``f < Y``) no such pair can exist. A verifier that
answers 1.0 to everything also scores zero discordant pairs.
:attr:`RankReport.one_directional` flags it.

**A reversal on a gap smaller than the gate's own noise is not a failure.** The
gate refuses both candidates there. What matters is agreement among the pairs
the gate would actually act on, which is :meth:`RankReport.above`.

**Artifacts from one run are a lineage, not independent draws**, and five of them
make ten pairs. This reports counts and refuses to put an interval around them:
a binomial interval on 2-of-10 spans 0.03 to 0.56 and would be wrong about the
dependence on top of that. Treat a flip rate as a flag to go and look, not as an
estimate.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .estimate import hajek_mean
from .records import AuditRecord

__all__ = ["Flip", "RankReport", "kendall_tau_b", "rank_agreement"]


def kendall_tau_b(x: Sequence[float], y: Sequence[float]
                  ) -> Tuple[float, int, int]:
    """``(tau_b, concordant, discordant)``, tie-corrected.

    The ``-b`` form because the data here is usually binary, where the
    uncorrected coefficient is dominated by ties and reads far lower than the
    agreement it is describing. It is still a weak statistic on this data -- see
    the module docstring -- which is why the concordant and discordant counts
    come back beside it rather than being summarised away.
    """
    if len(x) != len(y):
        raise ValueError(f"{len(x)} and {len(y)} are not the same length")
    n = len(x)
    conc = disc = tied_x = tied_y = tied_both = 0
    for i, j in itertools.combinations(range(n), 2):
        a, b = x[i] - x[j], y[i] - y[j]
        if a == 0.0 and b == 0.0:
            tied_both += 1
        elif a == 0.0:
            tied_x += 1
        elif b == 0.0:
            tied_y += 1
        elif (a > 0.0) == (b > 0.0):
            conc += 1
        else:
            disc += 1
    pairs = n * (n - 1) / 2
    denom = (pairs - tied_x - tied_both) * (pairs - tied_y - tied_both)
    if denom <= 0:
        return float("nan"), conc, disc
    return (conc - disc) / math.sqrt(denom), conc, disc


@dataclass(frozen=True)
class Flip:
    """One artifact pair the verifier orders backwards.

    ``verifier_gap`` is what the gate would have seen. A flip on a gap the gate
    would have refused anyway costs nothing; a flip on a large one is a commit
    that made the artifact worse.
    """

    base: str
    candidate: str
    verifier_gap: float
    oracle_gap: float
    n_base: int
    n_candidate: int

    def __str__(self) -> str:
        return (f"`{self.base[:12]}` -> `{self.candidate[:12]}`: verifier "
                f"{self.verifier_gap:+.3f}, truth {self.oracle_gap:+.3f}")


@dataclass(frozen=True)
class RankReport:
    """Ordering agreement, at the unit level and at the level the gate acts on."""

    n_units: int
    tau_b: float
    concordant: int
    discordant: int
    #: Every residual has the same sign, so no unit pair *can* be discordant and
    #: :attr:`tau_b` says nothing about ordering. The usual case for a generous
    #: judge, and the reason this flag exists.
    one_directional: bool
    n_artifacts: int
    #: Signature -> ``{"n", "verifier", "oracle"}``, inclusion-weighted.
    by_artifact: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_pairs: int = 0
    agree: int = 0
    ties: int = 0
    flips: List[Flip] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        """Fraction of decided artifact pairs the verifier orders correctly."""
        decided = self.agree + len(self.flips)
        return self.agree / decided if decided else float("nan")

    def above(self, gap: float) -> Tuple[int, int]:
        """``(agree, flip)`` among pairs whose *verifier* gap is at least ``gap``.

        The gate only acts on gaps it can distinguish from noise, so this is the
        agreement that costs anything. Filtering on the verifier's gap rather
        than the truth's is deliberate: it is the one the gate can see.
        """
        flips = sum(1 for f in self.flips if abs(f.verifier_gap) >= gap)
        decided = sum(
            1 for a, b in itertools.combinations(sorted(self.by_artifact), 2)
            if abs(self.by_artifact[b]["verifier"]
                   - self.by_artifact[a]["verifier"]) >= gap
            and self.by_artifact[b]["oracle"] != self.by_artifact[a]["oracle"]
            and self.by_artifact[b]["verifier"] != self.by_artifact[a]["verifier"])
        return decided - flips, flips

    @property
    def best_by_verifier(self) -> Optional[str]:
        if not self.by_artifact:
            return None
        return max(self.by_artifact, key=lambda s: self.by_artifact[s]["verifier"])

    @property
    def best_by_oracle(self) -> Optional[str]:
        if not self.by_artifact:
            return None
        return max(self.by_artifact, key=lambda s: self.by_artifact[s]["oracle"])

    @property
    def picks_the_same_winner(self) -> bool:
        return self.best_by_verifier == self.best_by_oracle

    def to_markdown(self) -> str:
        rows = [f"# Ordering -- {self.n_artifacts} artifacts, {self.n_units} units",
                ""]
        if self.n_pairs:
            rows += [
                f"- artifact pairs ordered the same way: **{self.agree}/"
                f"{self.agree + len(self.flips)}** "
                f"({self.agreement:.1%})"
                + (f", {self.ties} tied" if self.ties else ""),
                f"- the verifier's favourite is "
                + ("also the truth's" if self.picks_the_same_winner
                   else f"**not** the truth's (`{self.best_by_verifier[:12]}` "
                        f"vs `{self.best_by_oracle[:12]}`)"),
            ]
        rows.append(f"- unit-level Kendall tau-b: {self.tau_b:.4f} "
                    f"({self.concordant} concordant, {self.discordant} discordant)")
        if self.one_directional:
            rows.append(
                "    - every error runs the same way, so **no unit pair can be "
                "discordant** and tau says nothing about ordering here. A "
                "verifier that answered 1.0 to everything would score the same.")
        if self.by_artifact:
            rows += ["", "| artifact | n | mean f | mean Y |", "|---|---|---|---|"]
            for sig in sorted(self.by_artifact,
                              key=lambda s: -self.by_artifact[s]["verifier"]):
                row = self.by_artifact[sig]
                rows.append(f"| `{sig[:16]}` | {int(row['n'])} | "
                            f"{row['verifier']:.3f} | {row['oracle']:.3f} |")
        if self.flips:
            rows += ["", "## Reversed", ""]
            rows += [f"- {f}" for f in self.flips]
            rows += ["", "A reversal on a gap the gate would have refused costs "
                     "nothing. One on a gap it would have committed is a commit "
                     "that made the artifact worse."]
        rows += ["", f"These artifacts come from one run -- a lineage, not "
                 f"independent draws -- and {self.n_pairs} pairs is not a sample "
                 f"to put an interval around. Read a flip as a reason to look at "
                 f"it, not as a rate."]
        return "\n".join(rows)


def rank_agreement(records: Iterable[AuditRecord], *,
                   pairs: Optional[Sequence[Tuple[str, str]]] = None,
                   min_units: int = 1) -> RankReport:
    """Does the verifier order units, and artifacts, the way ground truth does?

    ``pairs`` are ``(base, candidate)`` artifact signatures as the run actually
    compared them, when the caller kept them; without it every artifact is
    compared against every other, which counts comparisons no merge ever made.

    ``min_units`` drops artifacts with too few audited units to have a mean
    worth ordering -- one unit gives a rate of 0 or 1 and would reverse
    orderings on nothing.
    """
    rows = [r for r in records if r.oracle_score is not None]
    if not rows:
        nan = float("nan")
        return RankReport(0, nan, 0, 0, False, 0)

    f = [r.verifier_score for r in rows]
    y = [r.oracle_score for r in rows]
    tau, conc, disc = kendall_tau_b(f, y)
    residuals = {(a - b) > 0 for a, b in zip(f, y) if a != b}
    one_directional = len(residuals) <= 1 and any(a != b for a, b in zip(f, y))

    grouped: Dict[str, List[AuditRecord]] = {}
    for rec in rows:
        grouped.setdefault(rec.artifact_signature, []).append(rec)
    by_artifact = {}
    for sig, rs in grouped.items():
        if len(rs) < min_units:
            continue
        probs = [r.inclusion_prob for r in rs]
        by_artifact[sig] = {
            "n": float(len(rs)),
            "verifier": hajek_mean([r.verifier_score for r in rs], probs),
            "oracle": hajek_mean([r.oracle_score for r in rs], probs),
        }

    if pairs is None:
        names = sorted(by_artifact)
        pairs = list(itertools.combinations(names, 2))
    agree = ties = 0
    flips: List[Flip] = []
    n_pairs = 0
    for base, cand in pairs:
        if base not in by_artifact or cand not in by_artifact:
            continue
        n_pairs += 1
        lo, hi = by_artifact[base], by_artifact[cand]
        df, dy = hi["verifier"] - lo["verifier"], hi["oracle"] - lo["oracle"]
        if df == 0.0 or dy == 0.0:
            ties += 1
        elif (df > 0.0) == (dy > 0.0):
            agree += 1
        else:
            flips.append(Flip(base, cand, df, dy, int(lo["n"]), int(hi["n"])))

    return RankReport(
        n_units=len(rows), tau_b=tau, concordant=conc, discordant=disc,
        one_directional=one_directional, n_artifacts=len(by_artifact),
        by_artifact=by_artifact, n_pairs=n_pairs, agree=agree, ties=ties,
        flips=flips)
