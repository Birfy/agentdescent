"""``P(s, a)``: which node is worth *expanding*, which is not which node is best.

`FlatPuct` ranks nodes by score for exploitation and spreads the exploration
budget by a prior. Upstream ERA has no prior and spends that budget uniformly --
every node in the tree, dead ends included, gets ``1/N``. Here there is
something much better to spend it on, because a molecule carries visible
evidence about whether it has anywhere left to go:

* a candidate five atoms under the size cap cannot grow into anything;
* a candidate with no free aromatic CH has nowhere to put a new group;
* a candidate with no halogen bond and no hydrogen bond has a whole interaction
  mode still unused, and adding one is a known, large gain;
* a flat candidate has the entire third dimension unexplored.

That is **headroom**, and it is deliberately not the score. A rigid symmetric
flat aromatic scores well and is finished; a lopsided first-draft cage scores
badly and is one substitution away from being good. A search that spends its
exploration on the first and not the second is the failure mode this module
exists to stop.

The second half of the prior is the model's own rating of the direction it just
proposed, read out of the reply the search was already paying for. It costs no
extra call, it is absent whenever the model does not give one, and an absent
rating falls back to the headroom alone rather than to zero -- a missing number
must never be the reason a direction is never tried again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from examples.porous._descriptors import Descriptors

__all__ = ["Headroom", "expansion_prior", "structural_headroom"]


@dataclass(frozen=True)
class Headroom:
    """How much room a candidate still has, by kind. Each component is in [0, 1]."""

    size: float
    sites: float
    interactions: float
    shape: float

    @property
    def total(self) -> float:
        return round(0.25 * self.size + 0.25 * self.sites
                     + 0.25 * self.interactions + 0.25 * self.shape, 4)

    def as_dict(self) -> Dict[str, float]:
        return {"size": self.size, "sites": self.sites,
                "interactions": self.interactions, "shape": self.shape,
                "total": self.total}


def structural_headroom(d: Descriptors, *, max_atoms: int = 100) -> Headroom:
    """What this molecule could still become, from its graph alone.

    ``sites`` counts free attachment points against a target of one per eight
    heavy atoms -- enough to place a new group somewhere that matters, rather
    than "has any hydrogen at all".
    """
    remaining = max(0, max_atoms - d.atom_count)
    size = min(1.0, remaining / max(1.0, 0.35 * max_atoms))

    free = max(0, d.heavy_atoms - d.branch_atoms - d.heteroatoms)
    sites = min(1.0, free / max(1.0, d.heavy_atoms / 8.0) / 8.0)

    kinds_used = sum([d.halogen_sites > 0,
                      d.hbond_donors > 0 or d.hbond_acceptors > 0,
                      d.aromatic_rings > 0])
    interactions = (3 - kinds_used) / 3.0

    planar = d.largest_planar_fragment / max(1, d.heavy_atoms)
    three_d = (d.quaternary_atoms + d.spiro_atoms + d.bridge_atoms) > 0
    shape = min(1.0, 0.6 * planar + (0.0 if three_d else 0.4))

    return Headroom(round(size, 4), round(sites, 4), round(interactions, 4),
                    round(shape, 4))


def expansion_prior(headroom: Optional[Headroom],
                    promise: Optional[float] = None,
                    *, promise_scale: float = 10.0) -> float:
    """Blend headroom and the model's rating into the number `FlatPuct` uses.

    Bounded to ``[0.25, 1.0]`` rather than ``[0, 1]``. The floor is the point: a
    prior of zero would bar a node from ever being selected again on the
    strength of an estimate, and these are estimates. Four-to-one between the
    most and least promising node is enough to aim the exploration budget and
    not enough to turn the tree into a greedy walk -- and `--prior-exponent`
    sharpens it further when a caller wants that.
    """
    parts = []
    if headroom is not None:
        parts.append(max(0.0, min(1.0, headroom.total)))
    if promise is not None and promise == promise and promise > 0:
        parts.append(max(0.0, min(1.0, promise / promise_scale)))
    if not parts:
        return 1.0
    blend = sum(parts) / len(parts)
    return round(0.25 + 0.75 * blend, 4)
