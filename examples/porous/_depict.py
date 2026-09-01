"""2D coordinates and an SVG drawing for a molecule, with no dependencies.

A report about molecules that prints only SMILES strings is a report nobody
reads: `c1(c(c(c(c(c1-c2ccc(cc2)O)-c3ccc(cc3)O)...` is a correct description of
a shape and conveys none of it. This module turns the same graph the rubric
scores into a picture of it.

It is a *depiction*, not a conformation. The coordinates are laid out to be
readable -- regular polygons for rings, 120-degree angles for chains, a
relaxation pass to pull overlaps apart -- and they say nothing about geometry:
the molecule is drawn flat because structural formulae are drawn flat, and the
search itself never claims to know a three-dimensional structure.

Three steps:

1. **Seed placement.** The largest ring system's first ring becomes a regular
   polygon; every other ring is built onto the bond or atom it shares with
   something already placed; acyclic atoms take the widest free angle at their
   neighbour.
2. **Relaxation.** Bonds pull to unit length, atoms two apart inside a ring pull
   to that polygon's diagonal (which is what keeps rings from collapsing), and
   any two atoms that are not bonded push apart.
3. **Rendering.** Kekule double bonds as a shortened inner line, heteroatoms
   labelled with their hydrogens, carbons left as vertices -- the conventions a
   chemist expects, so the picture reads without a legend.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from examples.porous._smiles import (
    Molecule,
    ring_bonds,
    smallest_cycles,
)

__all__ = ["coordinates", "svg"]

Point = Tuple[float, float]

#: Colours by element, in the convention every chemist reads without a legend.
ELEMENT_COLORS = {
    "N": "#2f5fd0", "O": "#c0392b", "S": "#b7950b", "F": "#1e8449",
    "Cl": "#1e8449", "Br": "#8e44ad", "I": "#6c3483", "P": "#d35400",
    "Si": "#7f8c8d", "Se": "#7d6608", "B": "#7d3c98",
}


def _polygon(n: int, first: Point, second: Point) -> List[Point]:
    """The regular ``n``-gon whose first edge is ``first -> second``, turning left."""
    points = [first, second]
    angle = math.pi - 2 * math.pi / n
    for _ in range(n - 2):
        ax, ay = points[-2]
        bx, by = points[-1]
        dx, dy = ax - bx, ay - by
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        points.append((bx + dx * cos_a - dy * sin_a, by + dx * sin_a + dy * cos_a))
    return points


def _ring_order(cycle: Sequence[int], mol: Molecule) -> List[int]:
    """Walk a cycle's atoms in connection order rather than index order."""
    members = set(cycle)
    start = cycle[0]
    order, seen = [start], {start}
    while len(order) < len(cycle):
        for neighbor in mol.neighbors(order[-1]):
            if neighbor in members and neighbor not in seen:
                order.append(neighbor)
                seen.add(neighbor)
                break
        else:
            break                          # not a simple cycle; draw what we have
    return order


def _all_adjacent(cycle: Sequence[int], known: Sequence[int]) -> bool:
    """Are the already-placed atoms of this ring one contiguous run?"""
    positions = sorted(cycle.index(a) for a in known)
    n = len(cycle)
    if len(positions) < 2:
        return True
    gaps = [(positions[(i + 1) % len(positions)] - positions[i]) % n
            for i in range(len(positions))]
    # One big gap and the rest adjacent means the placed atoms are contiguous.
    return sorted(gaps)[:-1] == [1] * (len(gaps) - 1)


def _arc_points(start: Point, end: Point, count: int, bulge: int) -> List[Point]:
    """``count`` evenly spaced points on a unit-bond arc from ``start`` to ``end``.

    Solves for the arc whose chord is ``|start-end|`` and whose ``count + 1``
    equal chords are each one bond long, then walks it. ``bulge`` picks the side.
    Falls back to a straight interpolation when the endpoints are already too
    far apart for an arc to reach.
    """
    ax, ay = start
    bx, by = end
    chord = math.hypot(bx - ax, by - ay)
    segments = count + 1
    if chord >= segments - 1e-9 or count <= 0:
        return [(ax + (bx - ax) * (i + 1) / segments,
                 ay + (by - ay) * (i + 1) / segments) for i in range(count)]

    def spread(phi: float) -> float:
        return math.sin(phi / 2) / math.sin(phi / (2 * segments))

    low, high = 1e-6, 2 * math.pi - 1e-6
    for _ in range(80):
        mid = (low + high) / 2
        if spread(mid) > chord:
            high = mid
        else:
            low = mid
    phi = (low + high) / 2
    radius = chord / (2 * math.sin(phi / 2))
    mx, my = (ax + bx) / 2, (ay + by) / 2
    ux, uy = (bx - ax) / chord, (by - ay) / chord
    height = math.sqrt(max(0.0, radius * radius - (chord / 2) ** 2))
    sign = 1 if bulge >= 0 else -1
    cx, cy = mx - uy * height * sign, my + ux * height * sign
    start_angle = math.atan2(ay - cy, ax - cx)
    end_angle = math.atan2(by - cy, bx - cx)
    delta = (end_angle - start_angle) % (2 * math.pi)
    if sign < 0:
        delta -= 2 * math.pi
    return [(cx + radius * math.cos(start_angle + delta * (i + 1) / segments),
             cy + radius * math.sin(start_angle + delta * (i + 1) / segments))
            for i in range(count)]


def _place_arcs(cycle: Sequence[int], pos: Dict[int, Point],
                mol: Molecule) -> None:
    """Fill every unplaced run of a ring with an arc between its placed ends."""
    n = len(cycle)
    index = 0
    while index < n:
        if cycle[index] in pos:
            index += 1
            continue
        run = []
        cursor = index
        while cycle[cursor % n] not in pos and len(run) < n:
            run.append(cycle[cursor % n])
            cursor += 1
        before = cycle[(index - 1) % n]
        after = cycle[cursor % n]
        if before not in pos or after not in pos:
            index = cursor
            continue
        best, best_clearance = None, -1.0
        for bulge in (1, -1):
            points = _arc_points(pos[before], pos[after], len(run), bulge)
            clearance = min(
                (math.hypot(px - qx, py - qy)
                 for px, py in points for qx, qy in pos.values()),
                default=9.9)
            if clearance > best_clearance:
                best, best_clearance = points, clearance
        for atom, point in zip(run, best or []):
            pos[atom] = point
        index = cursor


def coordinates(mol: Molecule) -> Dict[int, Point]:
    """2D positions for every heavy atom, in units of one bond length."""
    heavy = [i for i, a in enumerate(mol.atoms) if a.element != "H"]
    if not heavy:
        return {}
    cycles = [_ring_order(c, mol) for c in smallest_cycles(mol)]
    cycles.sort(key=len)
    rings_of: Dict[int, List[int]] = {}
    for index, cycle in enumerate(cycles):
        for atom in cycle:
            rings_of.setdefault(atom, []).append(index)

    pos: Dict[int, Point] = {}
    placed_rings: Set[int] = set()

    def place_ring(index: int) -> None:
        cycle = cycles[index]
        known = [a for a in cycle if a in pos]
        n = len(cycle)
        if len(known) >= 2 and not _all_adjacent(cycle, known):
            # A *bridged* ring: it shares atoms with something already placed
            # that are not a single edge -- triptycene's three benzo rings all
            # hang off the same two bridgehead carbons. Building a polygon on an
            # edge that is not there stacks the rings on top of each other, so
            # each run of unplaced atoms is drawn as an arc between the placed
            # ones instead, bulging into whichever side is emptier.
            _place_arcs(cycle, pos, mol)
            placed_rings.add(index)
            return
        if not known:
            points = _polygon(n, (0.0, 0.0), (1.0, 0.0))
            for atom, point in zip(cycle, points):
                pos.setdefault(atom, point)
            placed_rings.add(index)
            return
        # Rotate the cycle so it starts at a placed atom, then build the polygon
        # from the shared edge when there is one -- that is what makes a fused
        # ring share its bond instead of landing on top of its neighbour.
        offset = cycle.index(known[0])
        walk = cycle[offset:] + cycle[:offset]
        if len(known) >= 2 and walk[-1] in pos:
            walk = list(reversed(walk))
            walk = walk[-1:] + walk[:-1]
        first = pos[walk[0]]
        if walk[1] in pos:
            second = pos[walk[1]]
        else:
            second = _free_direction(walk[0], first)
        points = _polygon(n, first, second)
        for atom, point in zip(walk, points):
            pos.setdefault(atom, point)
        placed_rings.add(index)

    def _free_direction(atom: int, origin: Point) -> Point:
        """A unit step from ``atom`` into its widest unoccupied angle."""
        taken = [math.atan2(pos[n][1] - origin[1], pos[n][0] - origin[0])
                 for n in mol.neighbors(atom) if n in pos]
        if not taken:
            return (origin[0] + 1.0, origin[1])
        taken.sort()
        best, span = taken[0] + math.pi, -1.0
        for i, angle in enumerate(taken):
            nxt = taken[(i + 1) % len(taken)] + (2 * math.pi if i + 1 == len(taken) else 0)
            gap = nxt - angle
            if gap > span:
                span, best = gap, angle + gap / 2
        return (origin[0] + math.cos(best), origin[1] + math.sin(best))

    # Seed with the largest ring system, or with atom 0 when there are no rings.
    if cycles:
        biggest = max(range(len(cycles)), key=lambda i: len(cycles[i]))
        place_ring(biggest)
    else:
        pos[heavy[0]] = (0.0, 0.0)

    queue = list(pos)
    while queue:
        atom = queue.pop(0)
        for index in rings_of.get(atom, []):
            if index not in placed_rings:
                place_ring(index)
                queue.extend(a for a in cycles[index] if a in pos)
        for neighbor in mol.neighbors(atom):
            if neighbor in pos or mol.atoms[neighbor].element == "H":
                continue
            pos[neighbor] = _free_direction(atom, pos[atom])
            queue.append(neighbor)
    for atom in heavy:                     # disconnected leftovers
        pos.setdefault(atom, (len(pos) * 1.5, -3.0))
    return _relax(mol, pos, cycles)


def _relax(mol: Molecule, pos: Dict[int, Point], cycles: Sequence[Sequence[int]],
           rounds: int = 600) -> Dict[int, Point]:
    """Pull bonds to unit length, keep rings open, push non-bonded atoms apart."""
    atoms = list(pos)
    bonded = {(min(b.a, b.b), max(b.a, b.b)) for b in mol.bonds}
    # 1-3 pairs inside a ring, with the diagonal that ring's polygon would have.
    diagonals: List[Tuple[int, int, float]] = []
    for cycle in cycles:
        n = len(cycle)
        if n < 3:
            continue
        target = 2 * math.sin(math.pi / n) * (1.0 / (2 * math.sin(math.pi / n))) * \
            2 * math.sin(2 * math.pi / n) / (2 * math.sin(math.pi / n))
        for i, atom in enumerate(cycle):
            diagonals.append((atom, cycle[(i + 2) % n], target))
    for _ in range(rounds):
        shift = {a: [0.0, 0.0] for a in atoms}
        for a, b in bonded:
            if a not in pos or b not in pos:
                continue
            _spring(pos, shift, a, b, 1.0, 0.35)
        for a, b, target in diagonals:
            if a in pos and b in pos:
                # Weaker than the bond springs: a bridged ring cannot keep a
                # regular polygon *and* stay clear of its neighbours, and a
                # rigid diagonal makes it choose the overlap. These constants
                # are the best of a twelve-point sweep scored on bond-length
                # spread, close contacts and bond crossings over six molecules
                # (benzene through triptycene).
                _spring(pos, shift, a, b, target, 0.3)
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                if (min(a, b), max(a, b)) in bonded:
                    continue
                ax, ay = pos[a]
                bx, by = pos[b]
                dx, dy = bx - ax, by - ay
                d2 = dx * dx + dy * dy
                if d2 > 2.25 or d2 < 1e-9:
                    continue
                d = math.sqrt(d2)
                push = (1.5 - d) * 0.35 / d
                shift[a][0] -= dx * push
                shift[a][1] -= dy * push
                shift[b][0] += dx * push
                shift[b][1] += dy * push
        for atom in atoms:
            dx, dy = shift[atom]
            pos[atom] = (pos[atom][0] + max(-0.2, min(0.2, dx)),
                         pos[atom][1] + max(-0.2, min(0.2, dy)))
    return pos


def _spring(pos: Dict[int, Point], shift: Dict[int, List[float]], a: int, b: int,
            target: float, k: float) -> None:
    ax, ay = pos[a]
    bx, by = pos[b]
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy) or 1e-6
    pull = (d - target) * k / d
    shift[a][0] += dx * pull
    shift[a][1] += dy * pull
    shift[b][0] -= dx * pull
    shift[b][1] -= dy * pull


def _label(mol: Molecule, index: int) -> str:
    """What to write at an atom: nothing for carbon, element plus H otherwise."""
    atom = mol.atoms[index]
    if atom.element == "C" and not atom.charge:
        return ""
    text = atom.element
    hydrogens = mol.h_count(index)
    if hydrogens == 1:
        text += "H"
    elif hydrogens > 1:
        text += f"H{hydrogens}"
    if atom.charge:
        text += "+" if atom.charge > 0 else "-"
    return text


def svg(mol: Molecule, *, width: int = 420, height: int = 300,
        stroke: str = "#22303f", font: int = 13, margin: int = 24,
        title: str = "") -> str:
    """One molecule as an inline SVG string, scaled to fit ``width x height``."""
    pos = coordinates(mol)
    if not pos:
        return f'<svg width="{width}" height="{height}"></svg>'
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(1e-6, max(xs) - min(xs))
    span_y = max(1e-6, max(ys) - min(ys))
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    scale = min(scale, 46.0)

    def screen(atom: int) -> Point:
        x, y = pos[atom]
        return (margin + (x - min(xs)) * scale
                + (width - 2 * margin - span_x * scale) / 2,
                height - margin - (y - min(ys)) * scale
                - (height - 2 * margin - span_y * scale) / 2)

    labels = {i: _label(mol, i) for i in pos}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">']
    if title:
        parts.append(f'<title>{title}</title>')

    for bond in mol.bonds:
        if bond.a not in pos or bond.b not in pos:
            continue
        (x1, y1), (x2, y2) = screen(bond.a), screen(bond.b)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / length, dy / length
        # Clear the line where a label sits, so text is never struck through.
        start = 9.0 if labels[bond.a] else 0.0
        end = 9.0 if labels[bond.b] else 0.0
        ax, ay = x1 + ux * start, y1 + uy * start
        bx, by = x2 - ux * end, y2 - uy * end
        if bond.order >= 2:
            px, py = -uy, ux
            gap = 2.6 if bond.order == 2 else 3.2
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" '
                         f'y2="{by:.1f}" stroke="{stroke}" stroke-width="1.5"/>')
            inner = 0.14
            ix1, iy1 = ax + ux * length * inner, ay + uy * length * inner
            ix2, iy2 = bx - ux * length * inner, by - uy * length * inner
            parts.append(
                f'<line x1="{ix1 + px * gap:.1f}" y1="{iy1 + py * gap:.1f}" '
                f'x2="{ix2 + px * gap:.1f}" y2="{iy2 + py * gap:.1f}" '
                f'stroke="{stroke}" stroke-width="1.5"/>')
            if bond.order == 3:
                parts.append(
                    f'<line x1="{ix1 - px * gap:.1f}" y1="{iy1 - py * gap:.1f}" '
                    f'x2="{ix2 - px * gap:.1f}" y2="{iy2 - py * gap:.1f}" '
                    f'stroke="{stroke}" stroke-width="1.5"/>')
        else:
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" '
                         f'y2="{by:.1f}" stroke="{stroke}" stroke-width="1.5"/>')

    for atom, text in labels.items():
        if not text:
            continue
        x, y = screen(atom)
        color = ELEMENT_COLORS.get(mol.atoms[atom].element, stroke)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.5" fill="#ffffff"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + font * 0.35:.1f}" font-size="{font}" '
            f'font-family="Helvetica, Arial, sans-serif" font-weight="600" '
            f'text-anchor="middle" fill="{color}">{text}</text>')
    parts.append("</svg>")
    return "".join(parts)
