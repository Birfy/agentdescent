"""What the model is told, and how its reply is read back.

The brief is the user's, kept verbatim in substance: rigidity, symmetry,
directional interaction sites, an open structure that is still competitive on
lattice energy, and something a chemist could actually make -- on a neutral,
closed-shell, sub-100-atom molecule with a valid SMILES.

Two details in here matter more than they look.

**The prompt shows the parent's score breakdown, not just its score.** A node
told only "you scored 0.61" can improve by luck; a node told "packing 0.00
because the largest coplanar fragment is your whole molecule" is being handed
the failing constraint. That is the difference between a search that walks and
one that climbs.

**The reply protocol asks for a rating (`PROMISE`).** It is the prior -- see
:mod:`examples.porous._prior` -- and it is read out of the same reply the search
was already paying for, so it costs nothing. The prompt is explicit that it is a
rating of the *direction after further work*, not of the molecule as written,
because those are different claims about a first draft.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence, Tuple

__all__ = ["DESIGN_BRIEF", "HINTS", "REPLY_PROTOCOL", "extract_molecule",
           "mutation_prompt", "repair_prompt", "sample_hints"]


#: Crystal-engineering ideas a chemist would bring to this problem and a scoring
#: rubric cannot express. They are shown a couple at a time, drawn at random per
#: expansion, rather than all at once and every time: the brief is what every
#: candidate must satisfy, while these are *ways of thinking* about it, and a
#: model handed the same four paragraphs on every call optimises against the
#: paragraphs. Rotating them is also what keeps a long run from collapsing onto
#: one idea -- different expansions get pushed by different considerations.
HINTS: Tuple[str, ...] = (
    """CSP ENERGY-DENSITY LANDSCAPE AND "SPIKES". Think of the crystal structure prediction
landscape: density on the x-axis, lattice energy on the y-axis. Most molecules crowd into a
dense, low-energy basin -- close-packed structures with no space in them. A porous crystal
appears as a "spike": an isolated low-energy minimum sitting at LOW density, far to the left
of that basin. Aim for a molecule whose plausible packings include such a spike. Low density
alone is not the target -- a low-density structure that is also high in energy is one nobody
will ever crystallise.""",
    """THREE-DIMENSIONALLY AWKWARD RIGID SKELETONS. A rigid skeleton that protrudes in three
dimensions -- triptycene is the archetype, and so are spiro, adamantane-like and propeller
shapes -- cannot mate closely with its neighbours in every direction at once. The geometry
itself obstructs dense packing and leaves voids, without needing any particular interaction
to hold them open. Prefer shapes whose bulk points in several directions over shapes that
are flat or rod-like.""",
    """HYDROGEN-BOND DONOR/ACCEPTOR BALANCE AND Z'. Count donors and acceptors. When the two
are badly mismatched, the surplus groups are frustrated: they cannot all be satisfied by one
symmetric arrangement, symmetry breaks, and the number of independent molecules in the
asymmetric unit (Z') rises -- which makes the structure complicated, hard to predict, and
usually denser. A matched donor-to-acceptor ratio lets a stable, symmetric hydrogen-bond
network form instead, which is what templates an open framework.""",
    """EXTRINSIC POROSITY. The pore does not have to be a cavity inside one molecule.
Extrinsically porous molecular crystals get their channels from how the molecules pack
against each other -- the space is between molecules, not within them -- and those channels
are typically templated and propped open by guest molecules such as solvent. So a molecule
with no internal cavity at all can still give a porous crystal, provided its shape and its
directional contacts template channels rather than close-pack.""",
)


def sample_hints(rng: "random.Random", count: int = 2) -> Sequence[str]:
    """``count`` of :data:`HINTS`, drawn without replacement."""
    if count >= len(HINTS):
        return list(HINTS)
    return rng.sample(list(HINTS), max(0, count))


DESIGN_BRIEF = """You are designing molecules that are likely to crystallise into POROUS
molecular crystals -- crystals held together only by intermolecular forces, whose packing
still leaves permanent, accessible void space.

A candidate should pursue all of these at once:

1. HIGH RIGIDITY / CONFORMATIONAL LOCK. A fully rigid skeleton is best: fused or bridged
   rings, aryl and alkynyl linkers, no long flexible chains. Flexibility both inflates the
   cost of crystal structure prediction and lets a molecule collapse its own voids.
2. HIGH SYMMETRY. Prefer scaffolds and substitution patterns with as much symmetry as the
   chemistry allows -- substituting every symmetry-equivalent position at once rather than
   one of them. Symmetry lowers the synthetic burden and shrinks the CSP search.
3. SPECIFIC, DIRECTIONAL INTERACTION SITES where they earn their place: halogen bonds
   (I > Br > Cl, on aryl or alkynyl carbon), hydrogen-bond donor/acceptor pairs, or
   deliberately positioned pi-pi contacts. Directionality is what templates an open
   packing instead of a dense one.
4. AN OPEN STRUCTURE THAT IS STILL COMPETITIVE ON LATTICE ENERGY. Do not simply chase a
   bigger void: a structure has to be low enough in energy to be the one that actually
   crystallises. Awkward, non-planar, branched shapes that still have plenty of aromatic
   surface and directional contacts are the target. Flat fused aromatics pack densely and
   are the classic failure.
5. PLAUSIBLE SYNTHETIC ACCESSIBILITY. Aryl-aryl couplings, Sonogashira alkynylation,
   aromatic halogenation, imine condensation and nitrile chemistry are cheap; macrocycles,
   many stereocentres, strained rings and exotic heteroatom chains are not.

Hard requirements for every candidate:
- a syntactically valid SMILES string for ONE molecule (no salts, no dot-separated parts);
- unambiguous, sensible valences; neutral overall; closed shell, no radicals;
- at most {max_atoms} atoms including hydrogens -- prefer to stay well under it;
- elements limited to {elements}."""


REPLY_PROTOCOL = """Reply with exactly these three lines and nothing else:

SMILES: <the complete SMILES of the new molecule>
CHANGE: <one sentence: what you changed and which criterion it serves>
PROMISE: <a number from 0 to 10>

PROMISE rates how far this DIRECTION could go after further edits -- not how good the
molecule is right now. A promising first draft of the right scaffold rates high even if it
currently scores poorly; a finished molecule with nothing left to improve rates low."""


_MUTATION_TMPL = """{brief}
{hints}
CURRENT MOLECULE (the parent you must modify):
  SMILES: {parent_smiles}
  {parent_explain}

{siblings}{best}
Propose ONE modification of the parent above. Change one thing, deliberately, and aim it
at the weakest criterion in the breakdown -- or at a criterion you can raise a lot without
losing another. Keep the parent's skeleton unless the breakdown says the skeleton is the
problem. Do not repeat a modification already listed above.

{protocol}"""


_REPAIR_TMPL = """Your previous answer was not a usable molecule.

You wrote:
  SMILES: {bad_smiles}

It was rejected: {error}

Fix it. The parent you were modifying is still:
  SMILES: {parent_smiles}

Keep the chemical intent of what you tried, but write a SMILES that parses, has sensible
valences, is neutral and closed-shell, is one molecule, and stays under {max_atoms} atoms.

{protocol}"""


def mutation_prompt(parent_smiles: str, parent_explain: str, *,
                    max_atoms: int = 100, elements: str = "C, H, N, O, F, Si, P, S, Cl, Se, Br, I",
                    siblings: Sequence[str] = (), best: str = "",
                    hints: Sequence[str] = ()) -> str:
    """The expansion prompt: the brief, the parent with its breakdown, the protocol.

    ``hints`` are the rotating crystal-engineering ideas from :data:`HINTS`; an
    empty sequence leaves the prompt exactly as it was before they existed.
    """
    tried = ""
    if siblings:
        listed = "\n".join(f"  - {item}" for item in siblings[:8])
        tried = ("ALREADY TRIED FROM THIS PARENT (do not repeat):\n" + listed + "\n\n")
    context = f"BEST MOLECULE ANYWHERE IN THE SEARCH SO FAR:\n  {best}\n\n" if best else ""
    hint_block = ""
    if hints:
        hint_block = ("\nWORTH THINKING ABOUT ON THIS EXPANSION:\n\n"
                      + "\n\n".join(hints) + "\n")
    return _MUTATION_TMPL.format(
        brief=DESIGN_BRIEF.format(max_atoms=max_atoms, elements=elements),
        hints=hint_block,
        parent_smiles=parent_smiles,
        parent_explain=parent_explain.replace("\n", "\n  "),
        siblings=tried,
        best=context,
        protocol=REPLY_PROTOCOL,
    )


def repair_prompt(parent_smiles: str, bad_smiles: str, error: str,
                  *, max_atoms: int = 100) -> str:
    """Handed back on a rejected candidate, with the gate's own reason attached."""
    return _REPAIR_TMPL.format(
        bad_smiles=bad_smiles or "(nothing that could be read as a SMILES)",
        error=error, parent_smiles=parent_smiles, max_atoms=max_atoms,
        protocol=REPLY_PROTOCOL)


_SMILES_LINE = re.compile(r"^\s*SMILES\s*[:=]\s*(?P<smiles>\S+)", re.MULTILINE | re.IGNORECASE)
_CHANGE_LINE = re.compile(r"^\s*CHANGE\s*[:=]\s*(?P<text>.+)$", re.MULTILINE | re.IGNORECASE)
_PROMISE_LINE = re.compile(r"PROMISE\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(?P<body>.*?)```", re.DOTALL)


def extract_molecule(reply: str) -> Tuple[str, str, Optional[float]]:
    """``(smiles, change summary, promise)`` from a model reply.

    Lenient about the wrapper and strict about nothing: validation is the gate's
    job, and a reply this cannot read at all comes back as an empty SMILES,
    which becomes a dead-end node exactly as ERA's unparseable program does.
    Reading a fenced block as a fallback is not politeness -- models fence code
    by habit, and refusing the fence would throw away a perfectly good molecule
    over punctuation.
    """
    if not isinstance(reply, str) or not reply.strip():
        return "", "", None
    text = reply.strip()
    promise = None
    match = _PROMISE_LINE.search(text)
    if match:
        try:
            value = float(match.group("value"))
            promise = value if value == value and value > 0 else None
        except ValueError:
            promise = None
    summary = ""
    change = _CHANGE_LINE.search(text)
    if change:
        summary = change.group("text").strip()[:300]

    line = _SMILES_LINE.search(text)
    if line:
        return line.group("smiles").strip().strip("`"), summary, promise

    fence = _FENCE.search(text)
    if fence:
        body = fence.group("body").strip().splitlines()
        if body:
            return body[0].strip(), summary, promise
    # Last resort: a lone line that looks like a SMILES and nothing else.
    for candidate in reversed(text.splitlines()):
        token = candidate.strip().strip("`")
        if token and " " not in token and re.fullmatch(r"[A-Za-z0-9@+\-\[\]()=#$%./\\]+", token):
            return token, summary, promise
    return "", summary, promise
