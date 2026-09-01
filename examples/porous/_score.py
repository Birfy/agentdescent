"""The rubric: five criteria, one number, and the gate in front of both.

The brief this implements, in the order the criteria were given:

1. **Rigidity / conformational lock.** Fully rigid is best. It keeps crystal
   structure prediction tractable and stops a molecule collapsing its own voids.
2. **High symmetry.** Fewer distinct environments means fewer distinct couplings
   to run and a smaller CSP search.
3. **Specific, directional interaction sites** where they help -- halogen bonds,
   hydrogen bonds, positioned pi-pi contacts.
4. **An open structure that is still competitive on lattice energy.** Not "the
   biggest void": a void nothing pays for is a structure nobody makes.
5. **Plausible synthetic accessibility.**

Two things about the arithmetic are deliberate.

**The packing term is a geometric mean, not a sum.** ``sqrt(awkwardness x
cohesion)`` is zero when either factor is zero, which is the whole point of
criterion 4: a shape that cannot pack densely but has nothing holding a crystal
together scores nothing, and so does a strongly cohesive flat disc that packs
into a herringbone with no space left in it. A weighted sum would let a
candidate buy one with the other.

**The weights are a shard, not a constant.** The nominal weights below are one
opinion. A molecule that only wins under exactly those numbers has not been
shown to be good, so :func:`weight_profiles` draws perturbed weightings, the
search sees some of them, and the held-back ones are what the final number is
reported on. That is what the engine's held-out split is doing in a problem with
no data in it.

None of this is a lattice energy. The honest version of criterion 4 is a
crystal structure prediction run -- generate packings, minimise with a real
force field, look at the lattice-energy landscape -- and it costs hours per
molecule. This is a topological proxy, it is named as one everywhere it is
reported, and the search built on it is a way of *proposing* candidates to send
to that calculation, not a replacement for it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from examples.porous._descriptors import Descriptors, describe
from examples.porous._smiles import DEFAULT_ELEMENTS, ValidationReport, validate

__all__ = [
    "DEFAULT_WEIGHTS",
    "TERMS",
    "ScoreReport",
    "Weights",
    "evaluate_smiles",
    "parse_weights",
    "weight_profiles",
]

#: The five criteria, in the order they were briefed.
TERMS = ("rigidity", "symmetry", "interactions", "packing", "synthesizability")


@dataclass(frozen=True)
class Weights:
    """How much each criterion is worth. Normalised before use."""

    rigidity: float = 0.25
    symmetry: float = 0.15
    interactions: float = 0.20
    packing: float = 0.25
    synthesizability: float = 0.15

    def normalized(self) -> "Weights":
        total = sum(getattr(self, term) for term in TERMS)
        if total <= 0:
            raise ValueError("weights must sum to something positive")
        return Weights(**{term: getattr(self, term) / total for term in TERMS})

    def as_dict(self) -> Dict[str, float]:
        return {term: round(getattr(self, term), 4) for term in TERMS}


DEFAULT_WEIGHTS = Weights()


def parse_weights(text: str, base: Weights = DEFAULT_WEIGHTS) -> Weights:
    """``"rigidity=0.4,packing=0.3"`` -> a :class:`Weights`, the rest unchanged."""
    values = {term: getattr(base, term) for term in TERMS}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in TERMS:
            raise ValueError(f"unknown criterion {name!r}; expected one of {TERMS}")
        values[name] = float(raw)
    return Weights(**values).normalized()


def weight_profiles(count: int, *, seed: int = 0, jitter: float = 0.45,
                    base: Weights = DEFAULT_WEIGHTS) -> List[Weights]:
    """``count`` weightings of the same five criteria; the first is the nominal one.

    Each other profile multiplies every weight by ``exp(N(0, jitter))`` and
    renormalises, so the criteria keep their identity and their relative
    importance moves. A candidate scored well across a set of these is one whose
    advantage does not depend on the exact numbers at the top of this file --
    and profiles the search never sees are what the run finally reports on.
    """
    if count < 1:
        raise ValueError("need at least one weight profile")
    out = [base.normalized()]
    rng = random.Random(seed * 7919 + 13)
    while len(out) < count:
        values = {
            term: max(1e-3, getattr(base, term) * math.exp(rng.gauss(0.0, jitter)))
            for term in TERMS
        }
        out.append(Weights(**values).normalized())
    return out


# ---------------------------------------------------------------------------
# The five criteria
# ---------------------------------------------------------------------------


def _saturating(value: float, full: float) -> float:
    """``value`` mapped into [0, 1], reaching 1 at ``full``."""
    if full <= 0:
        return 0.0
    return max(0.0, min(1.0, value / full))


def rigidity_term(d: Descriptors) -> float:
    """Torsional freedom, ring content, and how much of it is fused.

    ``1 / (1 + 8 * torsions per heavy atom)`` rather than a linear penalty: the
    first rotatable bond in an otherwise rigid frame costs far more than the
    ninth in a chain, and a linear term would let a large molecule carry rotors
    for free.
    """
    heavy = max(1, d.heavy_atoms)
    core = 1.0 / (1.0 + 8.0 * (d.torsion_cost / heavy))
    ring = d.ring_atoms / heavy
    fused = d.fused_atoms / heavy
    return round(0.5 * core + 0.3 * ring + 0.2 * fused, 4)


def symmetry_term(d: Descriptors) -> float:
    """How few distinct atom environments there are, and how many-fold they are.

    The orbit ratio alone would rate a C2-symmetric chain as highly as a
    tetrahedral core, so a quarter of the term is the size of the largest orbit:
    four equivalent arms is a different synthesis, and a different CSP problem,
    from two.
    """
    heavy = max(1, d.heavy_atoms)
    orbit_ratio = 1.0 - (d.orbit_count - 1) / max(1, heavy - 1)
    fold = _saturating(d.largest_orbit - 1, 3.0)
    return round(max(0.0, 0.75 * orbit_ratio + 0.25 * fold), 4)


def _interaction_strength(d: Descriptors) -> float:
    """Weighted site count: halogen bonds, complementary H-bonds, aromatic rings."""
    pairs = min(d.hbond_donors, d.hbond_acceptors)
    # An acceptor with no donor is not worthless -- aromatic C-H donors are
    # everywhere -- but it is not a designed pair either, so it counts at a
    # quarter.
    lone = max(0, d.hbond_acceptors - d.hbond_donors)
    return d.halogen_sites + 0.8 * pairs + 0.2 * lone + 0.5 * d.aromatic_rings


def interactions_term(d: Descriptors) -> float:
    """Directional sites: how many, how varied, and whether they point anywhere.

    ``rigid_mount`` is the third of the term that stops this from being a
    hydroxyl-counting contest: a hydrogen-bond donor on the end of a propyl
    chain can point wherever the chain lets it, which is the opposite of the
    directional lock a porous packing needs.
    """
    heavy = max(1, d.heavy_atoms)
    strength = _interaction_strength(d)
    density = _saturating(strength / heavy, 0.20)
    kinds = sum([
        d.halogen_sites > 0,
        d.hbond_donors > 0 or d.hbond_acceptors > 0,
        d.aromatic_rings > 0,
    ])
    diversity = kinds / 3.0
    if d.directional_total:
        mount = d.directional_on_rigid / d.directional_total
    else:
        mount = 1.0 if d.aromatic_rings else 0.0
    return round(0.45 * density + 0.25 * diversity + 0.30 * mount, 4)


def packing_terms(d: Descriptors) -> Tuple[float, float]:
    """``(awkwardness, cohesion)`` -- the two factors criterion 4 multiplies.

    *Awkwardness* is how badly the shape fills space: branch points, a small
    coplanar fraction, quaternary and bridgehead centres. *Cohesion* is what
    would pay for the void: aromatic surface for dispersion, directional sites,
    a size that has enough contacts to matter, and few floppy bonds to eat the
    space back.
    """
    heavy = max(1, d.heavy_atoms)
    branch = _saturating(d.branch_atoms, heavy / 8.0)
    # Scaled by how much of the molecule is a ring framework. Without that, a
    # glycerol -- no aromatic system, so nothing coplanar to measure -- came out
    # maximally "not a flat disc", when what it actually is is a floppy chain
    # that packs into whatever shape the lattice wants.
    ring_fraction = d.ring_atoms / heavy
    nonplanar = (1.0 - d.largest_planar_fragment / heavy) * (0.5 + 0.5 * ring_fraction)
    # Non-aromatic bridgeheads and quaternary centres only. An aromatic fusion
    # carbon is a *flat* junction: counting it here rated naphthalene as three-
    # dimensional, and a rubric that rewards fused flat aromatics for porosity
    # has the sign of criterion 4 backwards.
    three_d = _saturating(
        d.quaternary_atoms + d.spiro_atoms + d.bridge_atoms / 2.0, heavy / 12.0)
    awkwardness = 0.25 * branch + 0.50 * nonplanar + 0.25 * three_d

    aromatic = d.aromatic_atoms / heavy
    sites = _saturating(_interaction_strength(d) / heavy, 0.20)
    # Below ~12 heavy atoms there is too little surface for a void to survive
    # sublimation; above ~55 the molecule stops being a small-molecule synthesis
    # problem. Both ends taper rather than cut off.
    if d.heavy_atoms < 12:
        size = _saturating(d.heavy_atoms, 12.0)
    elif d.heavy_atoms > 55:
        size = max(0.0, 1.0 - (d.heavy_atoms - 55) / 45.0)
    else:
        size = 1.0
    flex = 1.0 / (1.0 + 3.0 * (d.torsion_cost / heavy))
    cohesion = 0.3 * aromatic + 0.3 * sites + 0.2 * size + 0.2 * flex
    return round(awkwardness, 4), round(cohesion, 4)


def packing_term(d: Descriptors) -> float:
    awkwardness, cohesion = packing_terms(d)
    return round(math.sqrt(max(0.0, awkwardness) * max(0.0, cohesion)), 4)


def synthesizability_term(d: Descriptors) -> float:
    """A penalty-and-bonus proxy for how hard this would be to actually make.

    Not the published SA_Score: that is a fragment-frequency model fitted to a
    million PubChem molecules, and shipping its fragment table would be shipping
    a dependency. What is here instead is the part of that judgement that is
    structural -- strain, macrocycles, stereocentres, unstable heteroatom
    chains, exotic elements, sheer size -- plus the two things that make a
    porous-cage synthesis *easier*: symmetry, which turns n couplings into one,
    and a skeleton built from aryl, alkyne, nitrile, imine and aryl-halide
    chemistry, which is the toolbox this field actually uses.
    """
    heavy = max(1, d.heavy_atoms)
    score = 1.0
    score -= 0.06 * max(0, d.heavy_atoms - 30) / 10.0
    score -= 0.05 * d.spiro_atoms
    score -= 0.10 * d.macrocycles
    score -= 0.12 * d.strained_rings
    score -= 0.06 * d.stereocentres
    score -= 0.04 * d.rare_elements
    score -= 0.15 * d.groups.get("peroxide", 0)
    score -= 0.10 * d.consecutive_heteroatoms
    score -= 0.05 * d.groups.get("isonitrile", 0)
    familiar = (d.aromatic_atoms + d.sp_atoms
                + sum(v for k, v in d.groups.items() if k.startswith("halogen_aryl"))
                + d.groups.get("nitrile", 0) + d.groups.get("imine", 0))
    score += 0.10 * _saturating(familiar / heavy, 0.7)
    score += 0.15 * symmetry_term(d)
    return round(max(0.0, min(1.0, score)), 4)


@dataclass
class ScoreReport:
    """One candidate, scored -- or refused, with the reason a model can act on."""

    smiles: str
    ok: bool
    reason: str = ""
    canonical: str = ""
    formula: str = ""
    total: float = 0.0
    terms: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    descriptors: Optional[Descriptors] = None
    validation: Optional[ValidationReport] = None
    weights: Weights = DEFAULT_WEIGHTS

    def score_with(self, weights: Weights) -> float:
        """Re-weight without re-deriving anything -- one weight-profile shard."""
        if not self.ok:
            return 0.0
        w = weights.normalized()
        return round(sum(getattr(w, term) * self.terms[term] for term in TERMS), 6)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "ok": self.ok,
            "reason": self.reason,
            "canonical": self.canonical,
            "formula": self.formula,
            "total": self.total,
            "terms": dict(self.terms),
            "details": dict(self.details),
            "weights": self.weights.as_dict(),
            "descriptors": self.descriptors.as_dict() if self.descriptors else {},
        }

    def explain(self) -> str:
        """The score breakdown, as the mutation prompt shows it to the model."""
        if not self.ok:
            return f"INVALID: {self.reason}"
        lines = [f"total {self.total:.3f}  ({self.formula}, "
                 f"{self.descriptors.atom_count if self.descriptors else 0} atoms)"]
        for term in TERMS:
            lines.append(f"  {term:<17s} {self.terms[term]:.3f}"
                         f"  (weight {getattr(self.weights.normalized(), term):.2f})")
        d = self.descriptors
        if d is not None:
            lines.append(
                f"  rotatable bonds {d.rotatable} (torsion cost {d.torsion_cost}), "
                f"rings {d.rings} ({d.aromatic_rings} aromatic), "
                f"atom orbits {d.orbit_count}/{d.heavy_atoms} heavy atoms, "
                f"largest coplanar fragment {d.largest_planar_fragment}")
            lines.append(
                f"  H-bond donors/acceptors {d.hbond_donors}/{d.hbond_acceptors}, "
                f"halogen-bond strength {d.halogen_sites}, "
                f"groups {d.groups or '{}'}")
            awkward, cohesion = packing_terms(d)
            lines.append(
                f"  packing = sqrt(awkwardness {awkward:.3f} x cohesion "
                f"{cohesion:.3f})")
        return "\n".join(lines)


def evaluate_smiles(smiles: str, *, weights: Weights = DEFAULT_WEIGHTS,
                    max_atoms: int = 100,
                    elements: Iterable[str] = DEFAULT_ELEMENTS) -> ScoreReport:
    """Validate, describe and score one SMILES string. Never raises.

    Invalid input is the normal case, not an error: it is what a language model
    produces several times in every search, and the reason string is fed back to
    it on the next expansion.
    """
    report = validate(smiles, max_atoms=max_atoms, elements=elements)
    if not report.ok or report.molecule is None:
        return ScoreReport(smiles=smiles, ok=False, reason=report.reason,
                           validation=report, weights=weights)
    d = describe(report.molecule)
    terms = {
        "rigidity": rigidity_term(d),
        "symmetry": symmetry_term(d),
        "interactions": interactions_term(d),
        "packing": packing_term(d),
        "synthesizability": synthesizability_term(d),
    }
    w = weights.normalized()
    total = round(sum(getattr(w, term) * terms[term] for term in TERMS), 6)
    awkwardness, cohesion = packing_terms(d)
    return ScoreReport(
        smiles=smiles,
        ok=True,
        canonical=report.canonical,
        formula=report.formula,
        total=total,
        terms=terms,
        details={
            "awkwardness": awkwardness,
            "cohesion": cohesion,
            "warnings": list(report.warnings),
            "atom_count": report.atom_count,
            "heavy_atoms": report.heavy_atoms,
        },
        descriptors=d,
        validation=report,
        weights=weights,
    )
