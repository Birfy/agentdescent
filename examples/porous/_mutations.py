"""Rule-based molecular edits, so the search runs with no API key at all.

Every other moving part of this example is real -- the gate, the descriptors,
the rubric, the PUCT tree, the engine. The only thing a language model supplies
is *what to try next*, and a small set of chemically-literate edit operators can
supply that too, badly but honestly. That matters for three reasons:

* `python -m examples.porous.porous_tree_search --offline` runs the whole search
  on a bare interpreter, which is how this repository's examples are testable;
* it is the control arm. A model-driven run that does not beat these operators
  on the same budget has not shown it is doing chemistry;
* it makes the search loop debuggable without spending tokens on it.

The operators are the moves a chemist would sketch on the back of a napkin: hang
a halogen or a nitrile off a ring, swap a CH for N, fuse a benzo ring, extend
with a rigid phenylethynyl arm, or cut a substituent back off. Each one can be
applied at a single site or at **every symmetry-equivalent site at once** --
which is the operator that serves criterion 2 directly, and the one a model has
to be told to use.

Nothing here scores anything. The operators propose; the tree decides.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from examples.porous._smiles import (
    Atom,
    DEFAULT_ELEMENTS,
    Bond,
    Molecule,
    SmilesError,
    kekulize,
    orbits,
    parse_smiles,
    ring_bonds,
    ring_sizes,
    validate,
    write_smiles,
)

__all__ = ["Mutation", "FRAGMENTS", "enumerate_mutations", "propose_offline"]


#: ``(fragment SMILES, human name)``. The attachment point is the fragment's
#: first atom. The list is the porous-crystal toolbox: halogen-bond donors,
#: linear acceptors, rigid aryl and aryl-ethynyl extensions, and the two
#: hydrogen-bonding groups that are worth having on a rigid frame.
FRAGMENTS: Tuple[Tuple[str, str], ...] = (
    ("I", "iodo (strong sigma-hole halogen-bond donor)"),
    ("Br", "bromo (halogen-bond donor)"),
    ("Cl", "chloro (weak halogen-bond donor)"),
    ("F", "fluoro (tunes the ring quadrupole for pi-stacking)"),
    ("C#N", "nitrile (linear, directional acceptor)"),
    ("c1ccccc1", "phenyl (rigid aryl arm)"),
    ("C#Cc1ccccc1", "phenylethynyl (rigid linear extension)"),
    ("c1ccncc1", "pyridyl (aromatic-N acceptor on a rigid arm)"),
    ("C#C", "ethynyl (linear rigid stub)"),
    ("O", "hydroxyl (hydrogen-bond donor)"),
    ("N", "amino (hydrogen-bond donor)"),
    ("C(F)(F)F", "trifluoromethyl"),
)


@dataclass(frozen=True)
class Mutation:
    """One proposed child: the SMILES, what was done, and by which operator."""

    smiles: str
    summary: str
    operator: str


def _finish(mol: Molecule, summary: str, operator: str, *,
            max_atoms: int,
            elements: Optional[Iterable[str]] = None) -> Optional[Tuple[Mutation, str]]:
    """Kekulise, serialise and validate an edited graph; drop it if it is not one.

    Edits are made on a molecule that was already kekulised, and
    :func:`kekulize` re-derives the whole assignment from the aromatic flags, so
    running it again is what makes an edit like "this CH is now an N" come out
    with the right alternation -- or be refused, which for a 5-ring it should be.
    """
    try:
        kekulize(mol)
        smiles = write_smiles(mol)
    except (SmilesError, RecursionError):
        return None
    report = validate(smiles, max_atoms=max_atoms,
                      elements=elements or DEFAULT_ELEMENTS)
    if not report.ok:
        return None
    # The canonical key travels with the mutation: `offer` de-duplicates on it,
    # and validating a second time to recover it doubled the cost of the whole
    # enumeration -- which is most of the wall-clock of an `--offline` run.
    return Mutation(smiles, summary, operator), report.canonical


def _attachment_sites(mol: Molecule) -> List[int]:
    """Atoms with a hydrogen to give up, heaviest-framework first."""
    return [i for i, atom in enumerate(mol.atoms)
            if atom.element in ("C", "N") and mol.h_count(i) > 0]


def _attach(mol: Molecule, sites: Sequence[int], fragment: str) -> Molecule:
    edited = copy.deepcopy(mol)
    for site in sites:
        piece = parse_smiles(fragment)
        offset = len(edited.atoms)
        edited.atoms.extend(copy.deepcopy(piece.atoms))
        for bond in piece.bonds:
            edited.bonds.append(
                Bond(bond.a + offset, bond.b + offset, bond.order, bond.aromatic))
        edited.bonds.append(Bond(site, offset, 1, False))
    return edited


def _orbit_sites(mol: Molecule, site: int, limit: int = 8) -> List[int]:
    colors = orbits(mol)
    same = [i for i in _attachment_sites(mol) if colors[i] == colors[site]]
    return same[:limit]


def _fuse_benzo(mol: Molecule, bond: Bond) -> Molecule:
    """Add four aromatic carbons across an aromatic C-C bond: benzene -> naphthalene."""
    edited = copy.deepcopy(mol)
    chain = []
    for _ in range(4):
        edited.atoms.append(Atom(element="C", aromatic=True))
        chain.append(len(edited.atoms) - 1)
    edited.bonds.append(Bond(bond.a, chain[0], 1, True))
    for left, right in zip(chain, chain[1:]):
        edited.bonds.append(Bond(left, right, 1, True))
    edited.bonds.append(Bond(chain[-1], bond.b, 1, True))
    return edited


def _delete_branch(mol: Molecule, bond: Bond) -> Optional[Molecule]:
    """Cut an acyclic bond and keep the larger side. The only shrinking move.

    Without it the search can only grow, and a tree whose every move adds atoms
    walks straight into the size cap and then has nothing left to expand.
    """
    edited = copy.deepcopy(mol)
    edited.bonds = [b for b in edited.bonds
                    if (b.a, b.b) != (bond.a, bond.b)]
    parts = edited.fragments()
    if len(parts) != 2:
        return None
    keep, drop = sorted(parts, key=len, reverse=True)
    if len(drop) > 6 or len(keep) < 4:
        return None
    index_map = {old: new for new, old in enumerate(sorted(keep))}
    trimmed = Molecule()
    trimmed.atoms = [copy.deepcopy(edited.atoms[i]) for i in sorted(keep)]
    for b in edited.bonds:
        if b.a in index_map and b.b in index_map:
            trimmed.bonds.append(
                Bond(index_map[b.a], index_map[b.b], b.order, b.aromatic))
    return trimmed


def enumerate_mutations(smiles: str, *, max_atoms: int = 100,
                        elements: Optional[Iterable[str]] = None,
                        limit: int = 400) -> List[Mutation]:
    """Every distinct valid single edit of ``smiles``, de-duplicated.

    Deterministic and order-stable, so a seeded run reproduces. Candidates that
    do not survive :func:`~examples.porous._smiles.validate` -- an aza-swap that
    breaks a five-ring, an attachment that busts the atom cap -- are simply not
    in the list.
    """
    # Normalised once here rather than threaded down as None: every `_finish`
    # below then gets a real set, and `validate` never sees a None to choke on.
    elements = frozenset(elements) if elements else DEFAULT_ELEMENTS
    report = validate(smiles, max_atoms=max_atoms, elements=elements)
    if not report.ok or report.molecule is None:
        return []
    parent = report.molecule
    parent_key = report.canonical
    seen = {parent_key}
    out: List[Mutation] = []

    def offer(candidate: Optional[Tuple[Mutation, str]]) -> None:
        if candidate is None or len(out) >= limit:
            return
        mutation, key = candidate
        if key in seen:
            return
        seen.add(key)
        out.append(mutation)

    sites = _attachment_sites(parent)
    colors = orbits(parent)
    # One representative per orbit: attaching at two equivalent carbons produces
    # the same molecule, and enumerating both doubles the work to find that out.
    representatives: Dict[int, int] = {}
    for site in sites:
        representatives.setdefault(colors[site], site)

    for fragment, name in FRAGMENTS:
        for site in representatives.values():
            where = "aromatic C" if parent.atoms[site].aromatic else "sp3 C"
            offer(_finish(_attach(parent, [site], fragment),
                          f"add {name} at one {where}", "substitute",
                          max_atoms=max_atoms, elements=elements))
            group = _orbit_sites(parent, site)
            if len(group) > 1:
                offer(_finish(
                    _attach(parent, group, fragment),
                    f"add {name} at all {len(group)} symmetry-equivalent "
                    f"{where} sites", "symmetrise", max_atoms=max_atoms, elements=elements))

    for site in representatives.values():
        atom = parent.atoms[site]
        if atom.aromatic and atom.element == "C" and parent.degree(site) == 2:
            edited = copy.deepcopy(parent)
            edited.atoms[site] = Atom(element="N", aromatic=True)
            offer(_finish(edited, "swap one aromatic CH for N (pyridine-type "
                                  "acceptor, no added bulk)", "aza",
                          max_atoms=max_atoms, elements=elements))
            group = [i for i in _orbit_sites(parent, site) if
                     parent.atoms[i].aromatic and parent.degree(i) == 2]
            if len(group) > 1:
                edited = copy.deepcopy(parent)
                for index in group:
                    edited.atoms[index] = Atom(element="N", aromatic=True)
                offer(_finish(edited,
                              f"swap all {len(group)} symmetry-equivalent "
                              "aromatic CH for N", "symmetrise",
                              max_atoms=max_atoms, elements=elements))

    sizes = ring_sizes(parent)
    for index, bond in enumerate(parent.bonds):
        if not bond.aromatic or sizes.get(index) != 6:
            continue
        if parent.h_count(bond.a) < 1 or parent.h_count(bond.b) < 1:
            continue
        offer(_finish(_fuse_benzo(parent, bond),
                      "fuse a benzo ring onto an aromatic edge (rigidifies, "
                      "adds pi surface)", "fuse", max_atoms=max_atoms, elements=elements))

    on_ring = {id(b) for b in ring_bonds(parent)}
    for bond in parent.bonds:
        if id(bond) in on_ring or bond.order != 1:
            continue
        trimmed = _delete_branch(parent, bond)
        if trimmed is not None:
            offer(_finish(trimmed, "cut a substituent back off", "trim",
                          max_atoms=max_atoms, elements=elements))
    return out


def propose_offline(smiles: str, rng: random.Random, *,
                    max_atoms: int = 100,
                    elements: Optional[Iterable[str]] = None) -> Optional[Mutation]:
    """One edit, drawn uniformly from the valid ones. No scoring happens here.

    Uniform on purpose: an operator that picked the best-scoring child would be
    doing the search's job, and a tree search evaluated against a proposer that
    already hill-climbs measures nothing.
    """
    options = enumerate_mutations(smiles, max_atoms=max_atoms, elements=elements)
    if not options:
        return None
    return options[rng.randrange(len(options))]
