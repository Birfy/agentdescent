"""Graph descriptors for the five things this search is looking for.

Everything here is topology. There is no conformer generation, no force field
and no crystal structure prediction in this file, and pretending otherwise would
be the easiest way to make the search look more principled than it is. What a
graph *can* say is quite a lot: how many bonds can rotate, how many atoms sit in
fused rings, which orbits the atoms fall into, where a halogen sigma-hole points,
whether a shape is a flat disc or an awkward tetrahedral prow.

Each descriptor is defined so that "more is better" and the composition into a
single number lives in :mod:`examples.porous._score`, not here -- so a caller can
read the raw counts, disagree with the weighting, and re-weight without touching
any chemistry.

Where a descriptor is a proxy for something a real workflow would compute
(lattice energy, synthetic accessibility), it says so in its docstring and names
what the honest version would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from examples.porous._smiles import (
    Bond,
    Molecule,
    orbits,
    ring_bonds,
    ring_sizes,
    ring_systems,
)

__all__ = [
    "Descriptors",
    "describe",
    "rotatable_bonds",
    "functional_groups",
]

#: Halogen-bond donor strength, by the size of the sigma hole. Fluorine has
#: essentially none -- it is in the table at zero rather than absent so that a
#: candidate that fluorinates everything is not credited with halogen bonding.
HALOGEN_STRENGTH = {"I": 1.0, "Br": 0.75, "Cl": 0.45, "F": 0.0}


def _heavy_neighbors(mol: Molecule, index: int) -> List[int]:
    return [n for n in mol.neighbors(index) if mol.atoms[n].element != "H"]


def _has_order(mol: Molecule, index: int, order: int) -> bool:
    return any(b.order == order and not b.aromatic for b in mol.incident(index))


def _is_sp(mol: Molecule, index: int) -> bool:
    """Linear carbon: a triple bond, or two cumulated double bonds."""
    if mol.atoms[index].element != "C":
        return False
    doubles = sum(1 for b in mol.incident(index) if b.order == 2 and not b.aromatic)
    return _has_order(mol, index, 3) or doubles >= 2


def _is_sp2(mol: Molecule, index: int) -> bool:
    atom = mol.atoms[index]
    return atom.aromatic or _has_order(mol, index, 2)


def _is_amide_bond(mol: Molecule, bond: Bond) -> bool:
    """A C-N bond whose carbon carries a double-bonded O or S.

    Excluded from the rotatable count for the usual reason: the amide bond has
    partial double-bond character and a rotation barrier of ~20 kcal/mol, so
    counting it as a degree of freedom overstates the flexibility of every
    imide-linked cage.
    """
    for carbon, nitrogen in ((bond.a, bond.b), (bond.b, bond.a)):
        if mol.atoms[carbon].element != "C" or mol.atoms[nitrogen].element != "N":
            continue
        for other in mol.incident(carbon):
            partner = other.other(carbon)
            if other.order == 2 and mol.atoms[partner].element in ("O", "S"):
                return True
    return False


def rotatable_bonds(mol: Molecule) -> List[Bond]:
    """Acyclic single bonds between two non-terminal heavy atoms.

    The usual definition (Veber's, minus the amide exception being optional):
    ring bonds are not rotatable, bonds to a terminal atom change nothing but the
    position of hydrogens, triple bonds are linear, and amides barely rotate.
    This count is the single biggest term in the rigidity score, because a
    flexible molecule both raises the cost of crystal structure prediction and
    tends to collapse its own voids.
    """
    on_ring = {id(b) for b in ring_bonds(mol)}
    out: List[Bond] = []
    for bond in mol.bonds:
        if bond.aromatic or bond.order != 1 or id(bond) in on_ring:
            continue
        if any(mol.atoms[i].element == "H" for i in (bond.a, bond.b)):
            continue
        if any(len(_heavy_neighbors(mol, i)) < 2 for i in (bond.a, bond.b)):
            continue
        if _is_sp(mol, bond.a) or _is_sp(mol, bond.b):
            continue                       # rotation about a linear axis is free
        if _is_amide_bond(mol, bond):
            continue
        out.append(bond)
    return out


def functional_groups(mol: Molecule) -> Dict[str, int]:
    """Counts of the motifs the rubric prices, found by local environment.

    Not SMARTS: a pattern matcher would be a second dependency-free project.
    Every entry here is decidable from an atom, its bonds and its neighbours,
    which is what keeps this readable and is also its limit -- it recognises a
    carbonyl, it does not recognise a beta-diketone.
    """
    groups: Dict[str, int] = {}

    def bump(name: str, n: int = 1) -> None:
        groups[name] = groups.get(name, 0) + n

    for index, atom in enumerate(mol.atoms):
        element = atom.element
        heavy = _heavy_neighbors(mol, index)
        hydrogens = mol.h_count(index)
        if element in HALOGEN_STRENGTH:
            host = heavy[0] if heavy else None
            if host is None:
                continue
            if mol.atoms[host].aromatic:
                bump(f"halogen_aryl_{element}")
            elif _is_sp(mol, host):
                bump(f"halogen_sp_{element}")
            else:
                bump(f"halogen_alkyl_{element}")
        elif element == "O":
            if _has_order(mol, index, 2):
                bump("carbonyl")
            elif hydrogens:
                bump("hydroxyl")
            elif len(heavy) == 2:
                bump("ether")
            if any(mol.atoms[n].element == "O" for n in heavy):
                bump("peroxide")
        elif element == "N":
            if _has_order(mol, index, 3):
                bump("nitrile" if len(heavy) == 1 else "isonitrile")
            elif atom.aromatic:
                bump("aromatic_n_donor" if hydrogens else "aromatic_n_acceptor")
            elif _has_order(mol, index, 2):
                bump("imine")
            elif hydrogens:
                bump("amine_nh")
            else:
                bump("amine_tertiary")
            if sum(1 for n in heavy if mol.atoms[n].element == "N") >= 1:
                bump("n_n_bond")
        elif element == "C":
            if _has_order(mol, index, 3) and any(
                    mol.atoms[n].element == "C" and _is_sp(mol, n) for n in heavy):
                bump("alkyne_carbon")
            if any(b.order == 2 and mol.atoms[b.other(index)].element == "O"
                   for b in mol.incident(index)):
                if any(mol.atoms[n].element == "N" for n in heavy):
                    bump("amide")
        elif element == "S":
            if any(mol.atoms[n].element == "S" for n in heavy):
                bump("disulfide")
        if element in ("O", "N", "S") and hydrogens:
            bump("hbond_donor_h", hydrogens)

    groups["alkyne"] = groups.pop("alkyne_carbon", 0) // 2
    if not groups["alkyne"]:
        groups.pop("alkyne")
    groups["peroxide"] = groups.get("peroxide", 0) // 2
    if not groups["peroxide"]:
        groups.pop("peroxide", None)
    groups["disulfide"] = groups.get("disulfide", 0) // 2
    if not groups["disulfide"]:
        groups.pop("disulfide", None)
    return {k: v for k, v in sorted(groups.items()) if v}


def _consecutive_heteroatoms(mol: Molecule, run: int = 3) -> int:
    """Chains of ``run`` bonded N/O atoms -- azides, triazenes, peroxy chains.

    Not a validity failure (azides are molecules) and a real problem for anyone
    who has to make and then heat one, so it is priced in the synthesizability
    term rather than gated.
    """
    hetero = {i for i, a in enumerate(mol.atoms) if a.element in ("N", "O")}
    found = 0
    for start in hetero:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == run:
                found += 1
                continue
            for nxt in mol.neighbors(node):
                if nxt in hetero and nxt not in path:
                    stack.append((nxt, path + [nxt]))
    return found // 2                      # each chain is walked from both ends


@dataclass
class Descriptors:
    """Every number the rubric reads, computed once per candidate."""

    heavy_atoms: int = 0
    atom_count: int = 0
    formula: str = ""
    rotatable: int = 0
    torsion_cost: float = 0.0
    ring_atoms: int = 0
    rings: int = 0
    ring_systems: int = 0
    fused_atoms: int = 0
    aromatic_atoms: int = 0
    aromatic_rings: int = 0
    macrocycles: int = 0
    strained_rings: int = 0
    spiro_atoms: int = 0
    fusion_atoms: int = 0
    quaternary_atoms: int = 0
    bridge_atoms: int = 0
    branch_atoms: int = 0
    sp_atoms: int = 0
    stereocentres: int = 0
    orbit_count: int = 0
    largest_orbit: int = 0
    largest_planar_fragment: int = 0
    longest_path: int = 0
    heteroatoms: int = 0
    rare_elements: int = 0
    hbond_donors: int = 0
    hbond_acceptors: int = 0
    halogen_sites: float = 0.0
    directional_on_rigid: int = 0
    directional_total: int = 0
    consecutive_heteroatoms: int = 0
    groups: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        out = dict(self.__dict__)
        out["groups"] = dict(self.groups)
        return out


def _longest_path(mol: Molecule) -> int:
    """Topological diameter, by BFS from every atom. Cheap at this size."""
    adjacency = mol.adjacency()
    best = 0
    for start in range(len(mol.atoms)):
        distance = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nxt in adjacency[node]:
                if nxt not in distance:
                    distance[nxt] = distance[node] + 1
                    queue.append(nxt)
        best = max(best, max(distance.values(), default=0))
    return best


def _planar_fragment(mol: Molecule) -> int:
    """The largest set of atoms that must be coplanar, as an upper bound.

    One aromatic ring system, plus the substituents that lie in its plane: a
    halogen, a nitrile, an alkyne, a carbonyl. Two rings joined by a single bond
    are *not* in one fragment -- biphenyl twists about 40 degrees, and treating
    it as one flat disc is exactly the mistake that makes a search prefer
    graphite-like packers.

    Why an upper bound matters here: a large flat fragment is the signature of a
    molecule that packs densely, which is the opposite of what a porous crystal
    needs, so over-estimating it is the conservative direction.
    """
    best = 0
    for system in ring_systems(mol):
        if not all(mol.atoms[i].aromatic for i in system):
            continue
        size = len(system)
        for atom in system:
            for neighbor in mol.neighbors(atom):
                if neighbor in system:
                    continue
                element = mol.atoms[neighbor].element
                if element in HALOGEN_STRENGTH or _is_sp(mol, neighbor) or (
                        element == "O" and _has_order(mol, neighbor, 2)):
                    size += 1
        best = max(best, size)
    return best


def _torsion_cost(mol: Molecule, ring_atoms: Set[int]) -> float:
    """Rotatable bonds, weighted by how much the rotation moves the molecule.

    A phenyl spinning on its stalk sweeps almost the same envelope at every
    angle; a bond in an -O-CH2-CH2- chain moves everything past it. Both are
    rotatable bonds by the count above, and treating them as the same degree of
    freedom is what makes a tetraphenylmethane -- four rotors, one rigid shape --
    look as floppy as a hexane. So a bond into a ring atom whose only exocyclic
    neighbour is that bond counts half, and everything else counts one.
    """
    on_ring = {id(b) for b in ring_bonds(mol)}

    def exocyclic(atom: int) -> int:
        return sum(1 for b in mol.incident(atom)
                   if id(b) not in on_ring
                   and mol.atoms[b.other(atom)].element != "H")

    total = 0.0
    for bond in rotatable_bonds(mol):
        # "Its only exocyclic bond is this one" -- counted over bonds rather
        # than neighbours, so biphenyl's pivot (both ends ring atoms) is still
        # recognised as a spinner instead of falling through to full weight.
        spinner = any(end in ring_atoms and exocyclic(end) == 1
                      for end in (bond.a, bond.b))
        total += 0.5 if spinner else 1.0
    return total


def _branch_atoms(mol: Molecule, ring_atoms: Set[int],
                  systems: Sequence[Set[int]]) -> int:
    """Branch points that change the *shape*, not merely the connectivity.

    Naphthalene's two fusion carbons have three heavy neighbours each and
    branch nothing: the molecule is a flat plate, which is the densest-packing
    shape there is. Counting them made a naphthalene score as awkwardly shaped
    as a tetrahedral core, and the search followed that straight into fused flat
    aromatics.

    So a branch point is either a non-aromatic atom with three or more heavy
    neighbours, or an aromatic atom bonded out to a *different* ring system --
    biphenyl's pivot, where the two planes twist apart.
    """
    system_of: Dict[int, int] = {}
    for index, system in enumerate(systems):
        for atom in system:
            system_of[atom] = index
    count = 0
    for index, atom in enumerate(mol.atoms):
        if atom.element == "H":
            continue
        neighbors = _heavy_neighbors(mol, index)
        if not atom.aromatic:
            if len(neighbors) >= 3:
                count += 1
            continue
        home = system_of.get(index)
        if any(system_of.get(n) is not None and system_of.get(n) != home
               for n in neighbors):
            count += 1
    return count


def describe(mol: Molecule) -> Descriptors:
    """Compute every descriptor for one (already validated) molecule."""
    heavy = [i for i, a in enumerate(mol.atoms) if a.element != "H"]
    n_heavy = len(heavy)
    colors = orbits(mol)
    heavy_colors = [colors[i] for i in heavy]
    sizes = ring_sizes(mol)
    on_ring_bonds = ring_bonds(mol)
    ring_atom_set = {i for bond in on_ring_bonds for i in (bond.a, bond.b)}
    systems = ring_systems(mol)

    ring_count = len(on_ring_bonds) - len(ring_atom_set) + len(systems)
    fused_atoms = 0
    for system in systems:
        bonds_in = [b for b in on_ring_bonds if b.a in system and b.b in system]
        if len(bonds_in) - len(system) + 1 > 1:
            fused_atoms += len(system)

    ring_bond_count: Dict[int, int] = {}
    for bond in on_ring_bonds:
        for i in (bond.a, bond.b):
            ring_bond_count[i] = ring_bond_count.get(i, 0) + 1

    # Rings, not ring systems: a naphthalene is two aromatic rings, and pi-pi
    # stacking scales with rings rather than with how many systems they fall in.
    aromatic_rings = 0
    for system in systems:
        bonds_in = [b for b in on_ring_bonds if b.a in system and b.b in system]
        if all(mol.atoms[i].aromatic for i in system):
            aromatic_rings += len(bonds_in) - len(system) + 1

    stereocentres = 0
    for index in heavy:
        atom = mol.atoms[index]
        if atom.chiral:
            stereocentres += 1
            continue
        if atom.element != "C" or atom.aromatic or _is_sp2(mol, index):
            continue
        neighbors = _heavy_neighbors(mol, index)
        if len(neighbors) + mol.h_count(index) != 4 or mol.h_count(index) > 1:
            continue
        keys = [colors[n] for n in neighbors]
        if len(set(keys)) == len(keys) and len(neighbors) >= 3:
            stereocentres += 1

    groups = functional_groups(mol)
    donors = sum(v for k, v in groups.items() if k == "hbond_donor_h")
    acceptors = sum(
        v for k, v in groups.items()
        if k in ("nitrile", "carbonyl", "ether", "imine", "aromatic_n_acceptor",
                 "amine_tertiary", "hydroxyl"))
    halogens = 0.0
    directional_total = 0
    directional_rigid = 0
    for index in heavy:
        element = mol.atoms[index].element
        if element in HALOGEN_STRENGTH:
            host = _heavy_neighbors(mol, index)
            strength = HALOGEN_STRENGTH[element]
            if host and (mol.atoms[host[0]].aromatic or _is_sp(mol, host[0])):
                halogens += strength
                directional_total += 1
                directional_rigid += 1
            elif host:
                halogens += strength * 0.4
                directional_total += 1
        elif element in ("N", "O", "S"):
            directional_total += 1
            if index in ring_atom_set or _is_sp(mol, index) or any(
                    n in ring_atom_set or _is_sp(mol, n)
                    for n in _heavy_neighbors(mol, index)):
                directional_rigid += 1

    return Descriptors(
        heavy_atoms=n_heavy,
        atom_count=mol.atom_count(),
        formula=mol.formula(),
        rotatable=len(rotatable_bonds(mol)),
        torsion_cost=round(_torsion_cost(mol, ring_atom_set), 3),
        ring_atoms=len(ring_atom_set),
        rings=ring_count,
        ring_systems=len(systems),
        fused_atoms=fused_atoms,
        aromatic_atoms=sum(1 for i in heavy if mol.atoms[i].aromatic),
        aromatic_rings=aromatic_rings,
        macrocycles=sum(1 for size in set(sizes.values()) if size >= 9),
        strained_rings=sum(1 for size in sizes.values() if size <= 4) // 2,
        spiro_atoms=sum(1 for i, n in ring_bond_count.items() if n >= 4),
        fusion_atoms=sum(1 for i, n in ring_bond_count.items() if n == 3),
        quaternary_atoms=sum(
            1 for i in heavy
            if mol.atoms[i].element == "C" and not mol.atoms[i].aromatic
            and len(_heavy_neighbors(mol, i)) == 4),
        bridge_atoms=sum(1 for i, n in ring_bond_count.items()
                         if n >= 3 and not mol.atoms[i].aromatic),
        branch_atoms=_branch_atoms(mol, ring_atom_set, systems),
        sp_atoms=sum(1 for i in heavy if _is_sp(mol, i)),
        stereocentres=stereocentres,
        orbit_count=len(set(heavy_colors)),
        largest_orbit=max((heavy_colors.count(c) for c in set(heavy_colors)),
                          default=0),
        largest_planar_fragment=_planar_fragment(mol),
        longest_path=_longest_path(mol),
        heteroatoms=sum(1 for i in heavy if mol.atoms[i].element not in ("C",)),
        rare_elements=sum(1 for i in heavy
                          if mol.atoms[i].element in ("Si", "Se", "B", "P")),
        hbond_donors=donors,
        hbond_acceptors=acceptors,
        halogen_sites=round(halogens, 3),
        directional_on_rigid=directional_rigid,
        directional_total=directional_total,
        consecutive_heteroatoms=_consecutive_heteroatoms(mol),
        groups=groups,
    )
