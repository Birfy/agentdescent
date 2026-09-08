"""A SMILES parser, kekuliser, writer and validity gate with no dependencies.

Why this exists rather than `import rdkit`
------------------------------------------
The engine has zero required dependencies and every example in this repository
runs on a bare interpreter. RDKit is a 100 MB wheel that CI does not install, so
a molecule search built on it would be untestable here -- and the one thing this
search *must* do on every expansion is decide whether the string a model just
wrote is a molecule at all.

So the gate is written out. It is deliberately narrower than RDKit: the organic
subset plus bracket atoms, no stereochemistry beyond counting stereocentres, no
tautomer perception, no metals. Within that range it is strict, and the strictness
is the point -- an LLM asked for a rigid cage answers with `c1ccc1`, a five-bonded
carbon, or a methyl radical often enough that a search which accepted them would
spend its budget scoring things that cannot exist.

What the gate actually checks, in order:

1. **Syntax** -- brackets, branches and ring-closure digits all balanced.
2. **Elements** -- an allowlist. `[Fe]` is a fine molecule and not one this search
   is about.
3. **One fragment** -- `A.B` is a salt or a solvate, not a candidate molecule.
4. **Kekulisation** -- every aromatic ring system must admit an alternating
   assignment of double bonds. This is what rejects `c1ccc1` and `c1cccc1`, which
   parse perfectly and are not molecules.
5. **Valence** -- against a per-element, per-charge table, after kekulisation.
6. **Radicals** -- twice over, because the two checks catch different things: a
   bracket atom whose electron count does not fill a normal valence, and an odd
   total electron count for the molecule as a whole.
7. **Neutrality** -- net formal charge zero.
8. **Size** -- heavy atoms plus hydrogens against a cap.

`validate()` returns a report rather than raising: an invalid candidate is
ordinary output from a language model, not an exception, and the search records
it as a dead-end node with the reason attached.

One limitation to state plainly: **stereochemistry is parsed and then dropped.**
A chirality marker is read, counted towards the synthesizability penalty and
reported as a warning, but never re-emitted -- its meaning is "the neighbours in
the order they were written", and the canonical writer reorders neighbours. Two
stereoisomers are therefore one molecule to everything here, which is consistent
with the rest of the example: every descriptor is constitutional.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Atom",
    "Bond",
    "Molecule",
    "SmilesError",
    "ValidationReport",
    "ATOMIC_NUMBER",
    "aromatize",
    "DEFAULT_ELEMENTS",
    "canonical_key",
    "fingerprint",
    "orbits",
    "parse_smiles",
    "ring_bonds",
    "ring_sizes",
    "ring_systems",
    "similarity",
    "smallest_cycles",
    "validate",
    "write_smiles",
]


class SmilesError(ValueError):
    """The string is not parseable SMILES, or is not a molecule."""


#: Enough of the periodic table for organic crystal engineering, and no more.
#: The number is used for the electron-parity radical check, so it has to be the
#: real atomic number rather than an index.
ATOMIC_NUMBER = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}

#: What a candidate for a porous molecular crystal may be built from. Se and Si
#: are in because thiophene-like and silane-linked frameworks are real; anything
#: outside is refused rather than scored, because this scoring rubric has nothing
#: sensible to say about a ferrocene.
DEFAULT_ELEMENTS = frozenset({"H", "B", "C", "N", "O", "F", "Si", "P", "S",
                              "Cl", "Se", "Br", "I"})

#: Written without brackets in SMILES, with hydrogens filled implicitly.
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

#: Allowed total valences by ``(element, charge)``. Consulted lowest-first, so a
#: neutral sulfur with two bonds gets no hydrogens beyond valence 2 and a
#: sulfone's four bonds are matched against 4 rather than being called
#: hypervalent. Missing pairs fall back to the neutral row shifted by the charge,
#: which is right for the halogens and wrong often enough that the common
#: charged species are spelled out.
VALENCES: Dict[Tuple[str, int], Tuple[int, ...]] = {
    ("B", 0): (3,), ("B", -1): (4,),
    ("C", 0): (4,), ("C", -1): (3,), ("C", 1): (3,),
    ("N", 0): (3,), ("N", 1): (4,), ("N", -1): (2,),
    ("O", 0): (2,), ("O", 1): (3,), ("O", -1): (1,),
    ("F", 0): (1,), ("F", -1): (0,),
    ("Si", 0): (4,), ("Si", -1): (3,),
    ("P", 0): (3, 5), ("P", 1): (4,), ("P", -1): (2,),
    ("S", 0): (2, 4, 6), ("S", 1): (3, 5), ("S", -1): (1,),
    ("Cl", 0): (1,), ("Cl", -1): (0,),
    ("Se", 0): (2, 4, 6), ("Se", -1): (1,),
    ("Br", 0): (1,), ("Br", -1): (0,),
    ("I", 0): (1,), ("I", -1): (0,),
    ("H", 0): (1,),
}


def allowed_valences(element: str, charge: int) -> Tuple[int, ...]:
    """The total valences ``element`` may carry at ``charge``, ascending."""
    if (element, charge) in VALENCES:
        return VALENCES[(element, charge)]
    base = VALENCES.get((element, 0))
    if base is None:
        return ()
    # A cation of a lone-pair-bearing element gains a bond, an anion loses one.
    # Approximate, and only ever reached for a charge state not spelled out
    # above -- where refusing outright would reject legitimate chemistry.
    shifted = tuple(sorted({max(0, v + charge) for v in base}))
    return shifted


# ---------------------------------------------------------------------------
# The molecule
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    element: str
    aromatic: bool = False
    charge: int = 0
    #: Hydrogen count written inside brackets. ``None`` means "not a bracket
    #: atom", i.e. fill by the implicit rule. The distinction is load-bearing:
    #: `[CH3]` is a methyl radical and `C` is methane.
    explicit_h: Optional[int] = None
    isotope: Optional[int] = None
    chiral: str = ""
    bracket: bool = False

    @property
    def symbol(self) -> str:
        return self.element


@dataclass
class Bond:
    a: int
    b: int
    #: 1, 2 or 3 after kekulisation. An aromatic bond parses as 1 and is
    #: rewritten by :func:`kekulize`.
    order: int = 1
    aromatic: bool = False

    def other(self, index: int) -> int:
        return self.b if index == self.a else self.a


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    #: Set by :func:`kekulize`; until then aromatic bond orders are placeholders.
    kekulized: bool = False

    # -- topology -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.atoms)

    def neighbors(self, index: int) -> List[int]:
        return [b.other(index) for b in self.bonds if index in (b.a, b.b)]

    def incident(self, index: int) -> List[Bond]:
        return [b for b in self.bonds if index in (b.a, b.b)]

    def degree(self, index: int) -> int:
        return len(self.incident(index))

    def bond_between(self, i: int, j: int) -> Optional[Bond]:
        for b in self.bonds:
            if (b.a, b.b) in ((i, j), (j, i)):
                return b
        return None

    def adjacency(self) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in self.atoms]
        for b in self.bonds:
            adj[b.a].append(b.b)
            adj[b.b].append(b.a)
        return adj

    # -- hydrogens and charge ----------------------------------------------

    def bond_sum(self, index: int) -> int:
        return sum(b.order for b in self.incident(index))

    def h_count(self, index: int) -> int:
        """Hydrogens on this atom -- explicit if bracketed, else implicit.

        The implicit rule is SMILES': fill to the lowest normal valence that is
        at least the sum of the bond orders. It needs kekulised orders, which is
        why nothing calls this before :func:`kekulize` has run.
        """
        atom = self.atoms[index]
        if atom.explicit_h is not None:
            return atom.explicit_h
        if atom.element not in ORGANIC_SUBSET:
            return 0
        total = self.bond_sum(index)
        for valence in allowed_valences(atom.element, atom.charge):
            if valence >= total:
                return valence - total
        return 0

    def total_charge(self) -> int:
        return sum(a.charge for a in self.atoms)

    def heavy_atoms(self) -> int:
        return sum(1 for a in self.atoms if a.element != "H")

    def hydrogens(self) -> int:
        explicit = sum(1 for a in self.atoms if a.element == "H")
        return explicit + sum(self.h_count(i) for i, a in enumerate(self.atoms)
                              if a.element != "H")

    def atom_count(self) -> int:
        """Every atom, hydrogens included -- the number the size cap is on."""
        return self.heavy_atoms() + self.hydrogens()

    def formula(self) -> str:
        """Hill order: C, H, then the rest alphabetically."""
        counts: Dict[str, int] = {}
        for i, atom in enumerate(self.atoms):
            counts[atom.element] = counts.get(atom.element, 0) + 1
            h = self.h_count(i)
            if h:
                counts["H"] = counts.get("H", 0) + h
        order = ([e for e in ("C", "H") if e in counts]
                 + sorted(e for e in counts if e not in ("C", "H")))
        return "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in order)

    def fragments(self) -> List[List[int]]:
        adj = self.adjacency()
        seen: Set[int] = set()
        out: List[List[int]] = []
        for start in range(len(self.atoms)):
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for nxt in adj[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            out.append(sorted(component))
        return out


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

#: Not anchored with `^`: `re.match(text, pos)` anchors at the *string* start,
#: so a `^` here made every bracket atom after position 0 unreadable.
_BRACKET = re.compile(
    r"\[(?P<isotope>\d+)?(?P<element>[A-Za-z][a-z]?|\*)(?P<chiral>@{1,2})?"
    r"(?P<h>H(?P<hcount>\d+)?)?(?P<charge>(?:\+{1,3}|-{1,3}|[+-]\d))?"
    r"(?::\d+)?\]")
_TWO_LETTER = ("Cl", "Br", "Si", "Se")
_BOND_ORDER = {"-": 1, "=": 2, "#": 3, "$": 4, "/": 1, "\\": 1, ":": 1}


def _charge_of(text: Optional[str]) -> int:
    if not text:
        return 0
    if len(text) > 1 and text[1].isdigit():
        return int(text[1]) * (1 if text[0] == "+" else -1)
    return len(text) if text[0] == "+" else -len(text)


def parse_smiles(smiles: str) -> Molecule:
    """Parse the organic subset plus bracket atoms into a :class:`Molecule`.

    Raises :class:`SmilesError` on anything it cannot read. Aromatic bonds come
    out with order 1 and ``aromatic=True``; call :func:`kekulize` (which
    :func:`validate` does) before reading hydrogen counts.
    """
    if not isinstance(smiles, str):
        raise SmilesError("SMILES must be a string")
    text = smiles.strip()
    if not text:
        raise SmilesError("empty SMILES")
    if len(text) > 4000:
        raise SmilesError("SMILES is implausibly long")

    mol = Molecule()
    branches: List[int] = []
    ring_open: Dict[int, Tuple[int, Optional[int], bool]] = {}
    previous: Optional[int] = None
    pending_order: Optional[int] = None
    pending_aromatic = False
    index = 0

    def add_atom(atom: Atom) -> int:
        mol.atoms.append(atom)
        return len(mol.atoms) - 1

    def connect(i: int, j: int, order: Optional[int], aromatic: bool) -> None:
        if i == j:
            raise SmilesError("an atom cannot bond to itself")
        if mol.bond_between(i, j) is not None:
            raise SmilesError(f"duplicate bond between atoms {i} and {j}")
        both_aromatic = mol.atoms[i].aromatic and mol.atoms[j].aromatic
        is_aromatic = aromatic or (order is None and both_aromatic)
        mol.bonds.append(Bond(i, j, 1 if order is None else order, is_aromatic))

    while index < len(text):
        char = text[index]

        if char == "[":
            match = _BRACKET.match(text, index)
            if not match:
                raise SmilesError(f"unreadable bracket atom at position {index}")
            element = match.group("element")
            if element == "*":
                raise SmilesError("wildcard atoms are not molecules")
            aromatic = element[0].islower()
            symbol = element.capitalize()
            hcount = match.group("hcount")
            atom = Atom(
                element=symbol,
                aromatic=aromatic,
                charge=_charge_of(match.group("charge")),
                # `[C]` is not `[CH]`: a bracket atom states its hydrogens
                # exactly, and the absence of an `H` means zero -- which is what
                # makes `[CH3]` a methyl radical the valence check can catch.
                explicit_h=(0 if not match.group("h")
                            else int(hcount) if hcount else 1),
                isotope=int(match.group("isotope")) if match.group("isotope") else None,
                chiral=match.group("chiral") or "",
                bracket=True,
            )
            current = add_atom(atom)
            index = match.end()
        elif char.isalpha():
            symbol = None
            if text[index:index + 2] in _TWO_LETTER:
                symbol = text[index:index + 2]
                index += 2
            else:
                upper = char.upper()
                candidate = char if char.isupper() else upper
                if candidate not in ORGANIC_SUBSET and candidate != "H":
                    raise SmilesError(
                        f"'{char}' at position {index} is not in the organic "
                        "subset; write it in brackets")
                symbol = char
                index += 1
            aromatic = symbol[0].islower()
            current = add_atom(Atom(element=symbol.capitalize(), aromatic=aromatic))
        elif char in "-=#$:/\\":
            if pending_order is not None:
                raise SmilesError(f"two bond symbols in a row at position {index}")
            pending_order = _BOND_ORDER[char]
            pending_aromatic = char == ":"
            index += 1
            continue
        elif char == "(":
            if previous is None:
                raise SmilesError("a branch cannot open before an atom")
            branches.append(previous)
            index += 1
            continue
        elif char == ")":
            if not branches:
                raise SmilesError("unbalanced ')'")
            previous = branches.pop()
            index += 1
            continue
        elif char == ".":
            previous = None
            pending_order, pending_aromatic = None, False
            index += 1
            continue
        elif char.isdigit() or char == "%":
            if previous is None:
                raise SmilesError("a ring-closure digit before any atom")
            if char == "%":
                if not text[index + 1:index + 3].isdigit():
                    raise SmilesError("'%' must be followed by two digits")
                label = int(text[index + 1:index + 3])
                index += 3
            else:
                label = int(char)
                index += 1
            if label in ring_open:
                partner, order, aromatic = ring_open.pop(label)
                merged = pending_order if pending_order is not None else order
                connect(partner, previous, merged, aromatic or pending_aromatic)
            else:
                ring_open[label] = (previous, pending_order, pending_aromatic)
            pending_order, pending_aromatic = None, False
            continue
        else:
            raise SmilesError(f"unexpected character {char!r} at position {index}")

        if previous is not None:
            connect(previous, current, pending_order, pending_aromatic)
        pending_order, pending_aromatic = None, False
        previous = current

    if branches:
        raise SmilesError("unbalanced '('")
    if ring_open:
        raise SmilesError(
            "unclosed ring-closure digit(s): "
            + ", ".join(str(k) for k in sorted(ring_open)))
    if pending_order is not None:
        raise SmilesError("a bond symbol with nothing after it")
    if not mol.atoms:
        raise SmilesError("no atoms")
    return mol


# ---------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------


def ring_bonds(mol: Molecule) -> List[Bond]:
    """Every bond that lies on a cycle -- i.e. every bond that is not a bridge.

    Tarjan's bridge search, iteratively: a molecule from a language model can be
    a 90-atom chain, and Python's recursion limit is not a chemistry argument.
    """
    adj: List[List[Tuple[int, int]]] = [[] for _ in mol.atoms]
    for index, bond in enumerate(mol.bonds):
        adj[bond.a].append((bond.b, index))
        adj[bond.b].append((bond.a, index))

    disc = [-1] * len(mol.atoms)
    low = [0] * len(mol.atoms)
    bridges: Set[int] = set()
    timer = 0
    for root in range(len(mol.atoms)):
        if disc[root] != -1:
            continue
        stack: List[Tuple[int, int, int]] = [(root, -1, 0)]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            node, via, cursor = stack.pop()
            if cursor < len(adj[node]):
                stack.append((node, via, cursor + 1))
                nxt, edge = adj[node][cursor]
                if edge == via:
                    continue
                if disc[nxt] == -1:
                    disc[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append((nxt, edge, 0))
                else:
                    low[node] = min(low[node], disc[nxt])
            elif via != -1:
                parent = mol.bonds[via].other(node)
                low[parent] = min(low[parent], low[node])
                if low[node] > disc[parent]:
                    bridges.add(via)
    return [bond for index, bond in enumerate(mol.bonds) if index not in bridges]


def ring_sizes(mol: Molecule) -> Dict[int, int]:
    """``{bond index: size of the smallest ring through it}`` for ring bonds.

    The smallest cycle through a bond is one plus the shortest path between its
    ends with that bond removed -- a BFS per ring bond. Not an SSSR: it does not
    try to pick a minimal cycle *basis*, it answers "how big is the ring this
    bond sits in", which is what every descriptor here asks.
    """
    on_ring = {id(b) for b in ring_bonds(mol)}
    sizes: Dict[int, int] = {}
    adj = mol.adjacency()
    for index, bond in enumerate(mol.bonds):
        if id(bond) not in on_ring:
            continue
        distance = {bond.a: 0}
        queue = [bond.a]
        while queue:
            node = queue.pop(0)
            if node == bond.b:
                break
            for nxt in adj[node]:
                if (node, nxt) in ((bond.a, bond.b), (bond.b, bond.a)):
                    continue           # the bond itself is removed
                if nxt not in distance:
                    distance[nxt] = distance[node] + 1
                    queue.append(nxt)
        if bond.b in distance:
            sizes[index] = distance[bond.b] + 1
    return sizes


def ring_systems(mol: Molecule) -> List[Set[int]]:
    """Fused ring systems: connected components of the ring-bond subgraph.

    A biphenyl is two systems (the bond between the rings is a bridge), a
    naphthalene is one. That difference is the whole rigidity story, which is why
    this is separate from :func:`ring_sizes`.
    """
    on_ring = ring_bonds(mol)
    parent = list(range(len(mol.atoms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bond in on_ring:
        ra, rb = find(bond.a), find(bond.b)
        if ra != rb:
            parent[ra] = rb
    groups: Dict[int, Set[int]] = {}
    members = {i for bond in on_ring for i in (bond.a, bond.b)}
    for atom in sorted(members):
        groups.setdefault(find(atom), set()).add(atom)
    return [group for _, group in sorted(groups.items())]


# ---------------------------------------------------------------------------
# Kekulisation
# ---------------------------------------------------------------------------


def smallest_cycles(mol: Molecule) -> List[List[int]]:
    """The smallest cycle through each ring bond, as an atom list, de-duplicated.

    :func:`ring_sizes` answers "how big"; aromaticity perception needs to know
    *which* atoms, so this returns the paths themselves.
    """
    on_ring = {id(b) for b in ring_bonds(mol)}
    adjacency = mol.adjacency()
    seen: Set[frozenset] = set()
    out: List[List[int]] = []
    for bond in mol.bonds:
        if id(bond) not in on_ring:
            continue
        previous = {bond.a: None}
        queue = [bond.a]
        while queue:
            node = queue.pop(0)
            if node == bond.b:
                break
            for nxt in adjacency[node]:
                if (node, nxt) in ((bond.a, bond.b), (bond.b, bond.a)):
                    continue
                if nxt not in previous:
                    previous[nxt] = node
                    queue.append(nxt)
        if bond.b not in previous:
            continue
        cycle, cursor = [], bond.b
        while cursor is not None:
            cycle.append(cursor)
            cursor = previous[cursor]
        key = frozenset(cycle)
        if key not in seen:
            seen.add(key)
            out.append(cycle)
    return out


#: Elements that can carry an aromatic pi system in the chemistry this search
#: builds from. Boron is here for borazine-like rings; metals are not.
_AROMATIC_ELEMENTS = {"C", "N", "O", "S", "Se", "P", "B"}


def aromatize(mol: Molecule) -> None:
    """Mark Kekule-written aromatic rings as aromatic, in place.

    A model asked for a molecule writes benzene as `c1ccccc1` about as often as
    `C1=CC=CC=C1`, and until this ran the two were *different molecules* to
    everything downstream: different symmetry orbits, a different coplanar
    fragment, and -- worst -- different de-duplication keys, so a search would
    happily re-derive a molecule it already had under the other spelling.

    Huckel, applied to the smallest ring through each ring bond: five- to
    seven-membered, every atom sp2 or a lone-pair-donating heteroatom, and a pi
    count of 4n+2. A ring atom with a double bond inside the ring system donates
    one electron, a saturated heteroatom donates its lone pair, and a carbon
    with an exocyclic double bond (a quinone carbonyl) donates none -- which is
    what keeps a quinone out of this and leaves it written the way it came in.

    Deliberately conservative: a ring this cannot prove aromatic keeps its
    Kekule bonds and stays a perfectly valid molecule. The failure mode of being
    too eager -- declaring an aromatic ring that then will not kekulise -- would
    reject real chemistry.
    """
    ring_bond_ids = {id(b) for b in ring_bonds(mol)}
    ring_atoms = {i for b in ring_bonds(mol) for i in (b.a, b.b)}
    for cycle in smallest_cycles(mol):
        if len(cycle) not in (5, 6, 7):
            continue
        members = set(cycle)
        if all(mol.atoms[i].aromatic for i in cycle):
            continue
        pi = 0
        donors: List[int] = []
        usable = True
        for index in cycle:
            atom = mol.atoms[index]
            if atom.element not in _AROMATIC_ELEMENTS or atom.charge:
                usable = False
                break
            inside = [b for b in mol.incident(index) if b.other(index) in members]
            outside = [b for b in mol.incident(index) if b.other(index) not in members]
            if any(b.order >= 3 for b in mol.incident(index)):
                usable = False
                break
            if any(b.order == 2 for b in inside):
                pi += 1
            elif any(b.order == 2 and b.other(index) in ring_atoms
                     and id(b) in ring_bond_ids for b in outside):
                # A fusion atom whose double bond points into the *other* ring
                # of the system still donates its electron to this one. Without
                # this, only one ring of a Kekule-written naphthalene comes out
                # aromatic and the molecule is half-perceived.
                pi += 1
            elif any(b.order == 2 for b in outside):
                pi += 0                    # an exocyclic carbonyl donates nothing
            elif atom.element in ("N", "O", "S", "Se", "P"):
                pi += 2
                donors.append(index)
            else:
                usable = False             # an sp3 carbon breaks the ring system
                break
        if not usable or pi % 4 != 2:
            continue
        for index in cycle:
            mol.atoms[index].aromatic = True
        for index in donors:
            # Pin the hydrogens a lone-pair donor is carrying *now*. Without it,
            # a Kekule-written pyrrole becomes an aromatic N with two
            # connections, `_needs_double` reads it as pyridine-type, and the
            # ring it was just declared part of no longer kekulises.
            atom = mol.atoms[index]
            hydrogens = mol.h_count(index)
            atom.explicit_h = hydrogens
            atom.bracket = atom.bracket or hydrogens > 0
        for bond in mol.bonds:
            if id(bond) in ring_bond_ids and bond.a in members and bond.b in members:
                bond.aromatic = True


def _needs_double(mol: Molecule, index: int, aromatic_bonds: Sequence[Bond]) -> bool:
    """Does this aromatic atom still owe the ring a double bond?

    The pi-electron bookkeeping in one function, per element:

    * `c` owes one, unless it is charged (cyclopentadienyl, tropylium) or already
      carries an exocyclic double bond (a pyridinone carbonyl).
    * `n`/`p` owe one only at two connections -- pyridine. `[nH]` and an
      N-substituted pyrrole donate a lone pair instead and owe nothing.
    * `o`, `s`, `se`, `b` never owe one.

    Getting this wrong in either direction is visible: too eager and benzene has
    no perfect matching, too lax and `c1ccc1` is accepted.
    """
    atom = mol.atoms[index]
    if any(bond.order >= 2 and not bond.aromatic for bond in mol.incident(index)):
        return False
    if atom.charge:
        return False
    if atom.element in ("O", "S", "Se", "B"):
        return False
    if atom.element in ("N", "P"):
        if atom.explicit_h:
            return False
        return mol.degree(index) == 2
    if atom.element == "C":
        # Three sigma bonds plus a double is four; a carbon with three ring or
        # substituent connections still owes one, a carbon with four does not.
        return mol.degree(index) <= 3
    return False


def kekulize(mol: Molecule) -> None:
    """Assign alternating double bonds to every aromatic system, in place.

    Raises :class:`SmilesError` when no assignment exists, which is the check
    that rejects an aromatic ring a model wrote because lowercase letters looked
    like the right answer. The matching is exhaustive backtracking over the
    aromatic subgraph: molecules here are under a hundred atoms and the
    most-constrained-first order makes the search trivial in practice.
    """
    aromatize(mol)
    on_ring = {id(bond) for bond in ring_bonds(mol)}
    for bond in mol.bonds:
        # An aromatic-flavoured bond that is not on a ring is the bond *between*
        # two rings -- biphenyl's, written without a `-`. It is a single bond,
        # and leaving it in the matching lets a search find "solutions" that
        # double-bond two rings together.
        if bond.aromatic and id(bond) not in on_ring:
            bond.aromatic = False
            bond.order = 1

    aromatic_atoms = [i for i, atom in enumerate(mol.atoms) if atom.aromatic]
    aromatic_bonds = [b for b in mol.bonds if b.aromatic]
    for index in aromatic_atoms:
        if sum(1 for b in mol.incident(index) if b.aromatic) < 2:
            raise SmilesError(
                f"atom {index} ({mol.atoms[index].element.lower()}) is aromatic "
                "but is not inside an aromatic ring")
    for bond in aromatic_bonds:
        if not (mol.atoms[bond.a].aromatic and mol.atoms[bond.b].aromatic):
            raise SmilesError("an aromatic bond joins a non-aromatic atom")

    need = [i for i in aromatic_atoms if _needs_double(mol, i, aromatic_bonds)]
    need_set = set(need)
    options: Dict[int, List[Bond]] = {
        i: [b for b in mol.incident(i) if b.aromatic and b.other(i) in need_set]
        for i in need
    }
    if any(not options[i] for i in need):
        stuck = next(i for i in need if not options[i])
        raise SmilesError(
            f"the aromatic system around atom {stuck} cannot be kekulised "
            "(no alternating double-bond assignment exists)")

    matched: Dict[int, int] = {}
    chosen: Set[int] = set()

    def solve(remaining: Set[int]) -> bool:
        if not remaining:
            return True
        # Most constrained atom first: with it, benzene and a fused triptycene
        # both settle without backtracking at all.
        pivot = min(remaining,
                    key=lambda i: sum(1 for b in options[i]
                                      if b.other(i) in remaining))
        for bond in options[pivot]:
            partner = bond.other(pivot)
            if partner not in remaining:
                continue
            matched[pivot], matched[partner] = partner, pivot
            chosen.add(id(bond))
            if solve(remaining - {pivot, partner}):
                return True
            chosen.discard(id(bond))
            matched.pop(pivot, None)
            matched.pop(partner, None)
        return False

    # Huckel, coarsely: an aromatic bond must sit in a five-, six- or
    # seven-membered ring. Without this, `c1ccc1` -- antiaromatic cyclobutadiene
    # -- has a perfect matching and would be waved through as a molecule.
    sizes = ring_sizes(mol)
    for index, bond in enumerate(mol.bonds):
        if bond.aromatic and sizes.get(index, 0) not in (5, 6, 7):
            raise SmilesError(
                f"an aromatic bond lies in a {sizes.get(index, 0)}-membered "
                "ring; aromatic rings here are 5-, 6- or 7-membered")

    if not solve(set(need)):
        raise SmilesError(
            "the aromatic ring system cannot be kekulised: no alternating "
            "double-bond assignment covers every aromatic atom")

    for bond in aromatic_bonds:
        bond.order = 2 if id(bond) in chosen else 1
    mol.kekulized = True


# ---------------------------------------------------------------------------
# Symmetry, canonical form
# ---------------------------------------------------------------------------


def _stable_hash(value: object) -> int:
    """A hash that does not move between processes.

    Python's `hash()` is salted per process for strings, so a canonical form
    built on it is canonical only within one run -- and two runs of a seeded
    search would then disagree about which molecules they had already seen.
    """
    return int.from_bytes(
        hashlib.blake2b(repr(value).encode("utf-8"), digest_size=8).digest(),
        "big")


def _wl_colors(mol: Molecule, rounds: int) -> List[List[int]]:
    """Weisfeiler-Lehman colours after each round, comparable *across* molecules.

    :func:`orbits` re-ranks its colours to small integers, which makes them
    meaningless outside the molecule they came from. These are raw stable
    hashes, so the same substructure hashes to the same number in two different
    molecules -- which is what :func:`fingerprint` needs.
    """
    def weight(bond: Bond) -> float:
        return 1.5 if bond.aromatic else float(bond.order)

    colors = [
        _stable_hash((atom.element, atom.aromatic, atom.charge, mol.degree(i),
                      mol.h_count(i),
                      tuple(sorted(weight(b) for b in mol.incident(i)))))
        for i, atom in enumerate(mol.atoms)
    ]
    history = [list(colors)]
    for _ in range(rounds):
        colors = [
            _stable_hash((colors[i],
                          tuple(sorted((weight(b), colors[b.other(i)])
                                       for b in mol.incident(i)))))
            for i in range(len(mol.atoms))
        ]
        history.append(list(colors))
    return history


def fingerprint(mol: Molecule, radius: int = 2) -> "Counter":
    """Every atom environment up to ``radius`` bonds, as a multiset.

    A Morgan/ECFP-shaped fingerprint built out of the refinement this module
    already computes. Used for one thing: deciding whether a proposed child is
    actually a *modification of its parent* rather than an unrelated molecule
    the model preferred to write.
    """
    counts: "Counter" = Counter()
    for layer in _wl_colors(mol, radius):
        counts.update(layer)
    return counts


def similarity(left: Molecule, right: Molecule, radius: int = 2) -> float:
    """Tanimoto over :func:`fingerprint` multisets, in ``[0, 1]``.

    Multiset Tanimoto -- ``sum(min) / sum(max)`` -- rather than the set form, so
    substituting four of a molecule's six positions is measurably a bigger
    change than substituting one.
    """
    a, b = fingerprint(left, radius), fingerprint(right, radius)
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return round(intersection / union, 4) if union else 0.0


def orbits(mol: Molecule) -> List[int]:
    """Weisfeiler-Lehman refinement colours -- the topological symmetry classes.

    Two atoms that an automorphism maps onto each other always end up with the
    same colour, so the number of colours is a **lower bound** on the number of
    orbits: the partition can be coarser than the true orbits (WL-1 cannot
    separate every pair of graphs) but never finer. That direction is the safe
    one for a symmetry *reward* -- it can flatter a molecule, it cannot punish a
    genuinely symmetric one -- and it is stated here rather than left for a
    reader to assume the number is an orbit count.
    """
    def weight(bond: Bond) -> float:
        # 1.5 for an aromatic bond, not its Kekule order. A Kekule structure
        # picks one of two equivalent alternations, and refining on it makes
        # pyridine's two ortho carbons *different* -- the molecule would be
        # scored as less symmetric than it is because of how it happened to be
        # written down.
        return 1.5 if bond.aromatic else float(bond.order)

    colors = [
        _stable_hash((atom.element, atom.aromatic, atom.charge, mol.degree(i),
                      mol.h_count(i),
                      tuple(sorted(weight(b) for b in mol.incident(i)))))
        for i, atom in enumerate(mol.atoms)
    ]
    for _ in range(len(mol.atoms)):
        signature = [
            (colors[i], tuple(sorted((weight(b), colors[b.other(i)])
                                     for b in mol.incident(i))))
            for i in range(len(mol.atoms))
        ]
        table = {sig: rank for rank, sig in enumerate(sorted(set(signature),
                                                             key=repr))}
        updated = [table[sig] for sig in signature]
        if len(set(updated)) == len(set(colors)):
            colors = updated
            break
        colors = updated
    ranks = {c: r for r, c in enumerate(sorted(set(colors), key=repr))}
    return [ranks[c] for c in colors]


def _atom_token(mol: Molecule, index: int) -> str:
    atom = mol.atoms[index]
    symbol = atom.element.lower() if atom.aromatic else atom.element
    hydrogens = mol.h_count(index)
    implicit = None
    if atom.element in ORGANIC_SUBSET and atom.charge == 0:
        total = mol.bond_sum(index)
        for valence in allowed_valences(atom.element, atom.charge):
            if valence >= total:
                implicit = valence - total
                break
    bracket = (
        atom.element not in ORGANIC_SUBSET
        or atom.charge != 0
        or atom.isotope is not None
        # An aromatic heteroatom carrying hydrogen has to keep its brackets:
        # written bare, `n1cccc1` says pyridine-with-a-missing-bond and no
        # longer kekulises. `[nH]1cccc1` is pyrrole.
        or (atom.aromatic and atom.element != "C" and hydrogens > 0)
        or (not atom.aromatic and implicit != hydrogens)
    )
    if not bracket:
        return symbol
    inside = ""
    if atom.isotope is not None:
        inside += str(atom.isotope)
    # The chirality marker is parsed, counted and then *dropped*. Its meaning
    # is "the neighbours in the order they were written", and this writer
    # reorders neighbours by refinement colour -- so re-emitting it would state
    # a configuration the string no longer means. Everything downstream is
    # constitutional (the descriptors, the rubric, the canonical key), so
    # stereoisomers are one molecule here, which `validate` warns about.
    inside += symbol
    if hydrogens == 1:
        inside += "H"
    elif hydrogens > 1:
        inside += f"H{hydrogens}"
    if atom.charge:
        sign = "+" if atom.charge > 0 else "-"
        inside += sign if abs(atom.charge) == 1 else f"{sign}{abs(atom.charge)}"
    return f"[{inside}]"


def _bond_token(mol: Molecule, bond: Bond) -> str:
    if bond.aromatic:
        return ""
    if bond.order == 2:
        return "="
    if bond.order == 3:
        return "#"
    if mol.atoms[bond.a].aromatic and mol.atoms[bond.b].aromatic:
        # Two aromatic atoms joined by a genuine single bond -- biphenyl's
        # pivot. Without the explicit `-` a re-parse calls it aromatic and the
        # molecule stops kekulising.
        return "-"
    return ""


def write_smiles(mol: Molecule, root: int = 0,
                 ordering: Optional[Sequence[int]] = None) -> str:
    """Serialise back to SMILES, depth-first from ``root``.

    ``ordering`` gives each atom a sort key for choosing which neighbour to walk
    into first; :func:`canonical_key` passes the refinement colours so that the
    same molecule always produces the same string.
    """
    if not mol.atoms:
        return ""
    keys = list(ordering) if ordering is not None else list(range(len(mol.atoms)))
    incident: List[List[Bond]] = [[] for _ in mol.atoms]
    for bond in mol.bonds:
        incident[bond.a].append(bond)
        incident[bond.b].append(bond)

    visited: Set[int] = set()
    ring_labels: Dict[int, int] = {}
    used_labels: Set[int] = set()
    closures: Dict[int, List[Tuple[int, Bond]]] = {}

    # Pass one: a DFS that decides which bonds are tree bonds and which close a
    # ring, so pass two can emit a label at *both* ends.
    tree: Dict[int, List[Tuple[int, Bond]]] = {}
    order_seen: List[int] = []

    def walk(node: int, incoming: Optional[Bond]) -> None:
        visited.add(node)
        order_seen.append(node)
        tree[node] = []
        for bond in sorted(incident[node],
                           key=lambda b: (keys[b.other(node)], b.order,
                                          b.other(node))):
            if bond is incoming:
                continue
            nxt = bond.other(node)
            if nxt in visited:
                if id(bond) not in ring_labels:
                    free = [n for n in range(1, 100) if n not in used_labels]
                    if not free:
                        raise SmilesError(
                            "more than 99 open ring closures; this writer does "
                            "not recycle labels")
                    label = free[0]
                    used_labels.add(label)
                    ring_labels[id(bond)] = label
                    closures.setdefault(nxt, []).append((label, bond))
                    closures.setdefault(node, []).append((label, bond))
                continue
            if nxt not in visited:
                tree[node].append((nxt, bond))
                walk(nxt, bond)

    roots = [root]
    walk(root, None)
    for start in range(len(mol.atoms)):
        if start not in visited:
            # A second fragment. `validate` refuses these by default, but the
            # writer must not silently drop half a molecule when something else
            # -- an edit operator mid-cut, a caller with its own gate -- hands
            # one over.
            roots.append(start)
            walk(start, None)

    emitted_closures: Set[int] = set()

    def emit(node: int, incoming: Optional[Bond]) -> str:
        out = _atom_token(mol, node)
        for label, bond in sorted(closures.get(node, []), key=lambda item: item[0]):
            token = "" if id(bond) in emitted_closures else _bond_token(mol, bond)
            emitted_closures.add(id(bond))
            out += token + (str(label) if label < 10 else f"%{label}")
        children = tree.get(node, [])
        for position, (child, bond) in enumerate(children):
            branch = _bond_token(mol, bond) + emit(child, bond)
            out += f"({branch})" if position < len(children) - 1 else branch
        return out

    return ".".join(emit(start, None) for start in roots)


def canonical_key(mol: Molecule) -> str:
    """A deterministic string identifying the molecule, for de-duplication.

    Written from every atom in the lowest refinement colour class and the
    smallest string kept, with neighbours ordered by colour -- so two SMILES for
    the same molecule give the same key. It is canonical *up to WL-1
    refinement*: two different molecules that WL-1 cannot tell apart would
    collide, which for the organic graphs this search builds costs at worst a
    duplicate node that is never added.
    """
    colors = orbits(mol)
    if not colors:
        return ""
    best_color = min(colors)
    roots = [i for i, c in enumerate(colors) if c == best_color]
    return min(write_smiles(mol, root=r, ordering=colors) for r in roots)


# ---------------------------------------------------------------------------
# The validity gate
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Whether a string is a molecule this search may score, and why not.

    ``ok`` is the gate. ``reason`` is what the search feeds back to the model on
    the next expansion, so it is written to be actionable: which atom, which
    rule. ``warnings`` are things that do not disqualify a candidate but that the
    rubric prices in -- a zwitterion, a stereocentre, a strained ring.
    """

    ok: bool
    reason: str = ""
    smiles: str = ""
    canonical: str = ""
    formula: str = ""
    atom_count: int = 0
    heavy_atoms: int = 0
    molecule: Optional[Molecule] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "smiles": self.smiles,
            "canonical": self.canonical,
            "formula": self.formula,
            "atom_count": self.atom_count,
            "heavy_atoms": self.heavy_atoms,
            "warnings": list(self.warnings),
        }


def validate(smiles: str, *, max_atoms: int = 100,
             elements: Iterable[str] = DEFAULT_ELEMENTS,
             require_single_fragment: bool = True) -> ValidationReport:
    """Parse, kekulise and check ``smiles``; never raises on bad input.

    The order matters: syntax before elements before kekulisation before
    valence, so the message a model gets back names the *first* thing that is
    wrong rather than a downstream consequence of it.
    """
    allowed = frozenset(elements)
    try:
        mol = parse_smiles(smiles)
    except SmilesError as exc:
        return ValidationReport(False, f"unparseable SMILES: {exc}", smiles=smiles)

    unknown = sorted({a.element for a in mol.atoms} - allowed)
    if unknown:
        return ValidationReport(
            False, f"element(s) outside the allowed set: {', '.join(unknown)}",
            smiles=smiles)

    fragments = mol.fragments()
    if require_single_fragment and len(fragments) > 1:
        return ValidationReport(
            False,
            f"{len(fragments)} disconnected fragments; a candidate must be one "
            "molecule, not a salt or a solvate",
            smiles=smiles)

    try:
        kekulize(mol)
    except SmilesError as exc:
        return ValidationReport(False, f"invalid aromatic system: {exc}",
                                smiles=smiles)

    warnings: List[str] = []
    for index, atom in enumerate(mol.atoms):
        total = mol.bond_sum(index) + mol.h_count(index)
        permitted = allowed_valences(atom.element, atom.charge)
        if not permitted:
            return ValidationReport(
                False, f"no valence model for {atom.element}{atom.charge:+d}",
                smiles=smiles)
        if total > max(permitted):
            return ValidationReport(
                False,
                f"atom {index} ({atom.element}) has valence {total}, above the "
                f"maximum {max(permitted)} for {atom.element}"
                + (f" at charge {atom.charge:+d}" if atom.charge else ""),
                smiles=smiles)
        if atom.bracket and total not in permitted:
            # Only bracket atoms can land here: an unbracketed atom had its
            # hydrogens filled to a legal valence by definition. `[CH3]` is the
            # case this catches -- a methyl radical, written by a model that
            # meant methyl.
            return ValidationReport(
                False,
                f"atom {index} ([{atom.element}]) has valence {total}, which is "
                f"not one of {permitted} -- an open shell (radical), not a "
                "closed-shell molecule",
                smiles=smiles)
        if atom.charge:
            warnings.append(f"formal charge {atom.charge:+d} on atom {index}")
        if atom.chiral:
            warnings.append(
                f"stereocentre at atom {index}; stereochemistry is not tracked "
                "-- this search treats stereoisomers as one molecule")

    charge = mol.total_charge()
    if charge != 0:
        return ValidationReport(
            False, f"net formal charge {charge:+d}; candidates must be neutral",
            smiles=smiles)

    electrons = sum(ATOMIC_NUMBER[a.element] for a in mol.atoms) + mol.hydrogens() \
        - sum(1 for a in mol.atoms if a.element == "H") - charge
    if electrons % 2:
        return ValidationReport(
            False,
            f"{electrons} electrons in total -- an odd count is an open shell "
            "(radical), not a closed-shell molecule",
            smiles=smiles)

    count = mol.atom_count()
    if count > max_atoms:
        return ValidationReport(
            False, f"{count} atoms (hydrogens included), above the cap of "
                   f"{max_atoms}",
            smiles=smiles)

    canonical = canonical_key(mol)
    return ValidationReport(
        True, "", smiles=smiles, canonical=canonical, formula=mol.formula(),
        atom_count=count, heavy_atoms=mol.heavy_atoms(), molecule=mol,
        warnings=sorted(set(warnings)),
    )
