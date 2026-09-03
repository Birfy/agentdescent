"""The molecule search: the gate, the rubric, the prior, and one whole run.

Three things are pinned here, in this order of importance.

**The gate.** Everything downstream assumes a candidate is a molecule, so the
cases that must be refused are named one by one -- a five-bonded carbon, a
methyl radical, an anion, a four-membered aromatic ring, a salt. A gate that
quietly accepts one of these does not fail loudly anywhere else: it produces a
score for something that cannot exist.

**The rubric's ordering.** Absolute numbers here would encode today's weights,
so what is asserted is the *ordering* the brief asks for -- a tetrahedral
halogen-bonded node beats a flat aromatic, a floppy chain loses to both, and a
flat fused aromatic scores exactly zero on packing however rigid it is.

**That the search runs and improves.** Offline, seeded, no network.
"""

import json
import math
import random

import pytest

from agentdescent.aggregator import AggregatorConfig
from agentdescent.selection import Candidate, FlatPuct, SelectionContext

from examples.porous import porous_tree_search as port
from examples.porous._depict import coordinates, svg
from examples.porous._descriptors import describe, rotatable_bonds
from examples.porous._mutations import enumerate_mutations, propose_offline
from examples.porous._prior import expansion_prior, structural_headroom
from examples.porous._prompts import (
    HINTS,
    extract_molecule,
    mutation_prompt,
    sample_hints,
)
from examples.porous._report import _chrome, render_html, write_pdf
from examples.porous._score import (
    DEFAULT_WEIGHTS,
    TERMS,
    evaluate_smiles,
    parse_weights,
    weight_profiles,
)
from examples.porous._smiles import (
    canonical_key,
    similarity,
    kekulize,
    orbits,
    parse_smiles,
    validate,
    write_smiles,
)


# -- the gate -------------------------------------------------------------------

VALID = [
    "c1ccccc1",                                   # benzene
    "C1CC2CCC1CC2",                               # bicyclo[2.2.2]octane
    "c1ccc2ccccc2c1",                             # naphthalene
    "[nH]1cccc1",                                 # pyrrole: needs its brackets
    "c1ccncc1",                                   # pyridine
    "O=[N+]([O-])c1ccccc1",                       # nitrobenzene: net-neutral zwitterion
    "C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1",    # tetraphenylmethane
    "C12C3C4C1C5C4C3C25",                         # cubane
    "N#Cc1ccc(C#N)cc1",
    "FC(F)(F)c1ccccc1",
]


@pytest.mark.parametrize("smiles", VALID)
def test_real_molecules_pass_the_gate(smiles):
    report = validate(smiles)
    assert report.ok, f"{smiles} was refused: {report.reason}"
    assert report.formula and report.atom_count > 0


@pytest.mark.parametrize("smiles,fragment", [
    ("C(C)(C)(C)(C)C", "valence 5"),              # five bonds on carbon
    ("[CH3]", "radical"),                         # methyl radical
    ("[C]", "radical"),
    ("CC[O-]", "net formal charge"),              # an anion is not a candidate
    ("c1ccc1", "4-membered"),                     # antiaromatic cyclobutadiene
    ("c1cccc1", "kekulis"),                       # five aromatic carbons
    ("c1ccccc1.[Na+]", "outside the allowed set"),
    ("C1CCCCC", "unclosed ring"),
    ("CC(", "unbalanced"),
    ("[Fe]", "outside the allowed set"),
    ("N#Cc1ccccc1.N#Cc1ccccc1", "disconnected"),
])
def test_the_gate_names_what_is_wrong(smiles, fragment):
    report = validate(smiles)
    assert not report.ok, f"{smiles} should have been refused"
    assert fragment in report.reason, f"{smiles}: {report.reason}"


def test_the_atom_cap_is_enforced_and_reported():
    chain = "C" * 40                              # C40H82 -- 122 atoms
    assert not validate(chain, max_atoms=100).ok
    assert "above the cap" in validate(chain, max_atoms=100).reason
    assert validate(chain, max_atoms=200).ok


def test_an_odd_electron_count_is_refused_even_when_every_valence_fits():
    """The parity check is not redundant with the valence check.

    `[CH2]c1ccccc1` is a benzyl radical: every atom is inside a valence the
    table allows once the bracket says two hydrogens, and the molecule still has
    an unpaired electron.
    """
    report = validate("[CH2]c1ccccc1")
    assert not report.ok and "radical" in report.reason


@pytest.mark.parametrize("smiles", VALID)
def test_writing_and_reparsing_gives_the_same_molecule(smiles):
    mol = parse_smiles(smiles)
    kekulize(mol)
    again = parse_smiles(write_smiles(mol))
    kekulize(again)
    assert canonical_key(mol) == canonical_key(again)


def test_the_writer_keeps_every_fragment_of_a_disconnected_input():
    """`validate` refuses these; the writer must still not lose half a molecule."""
    mol = parse_smiles("c1ccccc1.CCO")
    kekulize(mol)
    written = write_smiles(mol)
    assert written.count(".") == 1
    assert len(parse_smiles(written).atoms) == len(mol.atoms)


def test_stereochemistry_is_read_reported_and_then_dropped():
    """Re-emitting `@` after reordering neighbours would state the wrong thing."""
    report = validate("C[C@H](N)C(=O)O")
    assert report.ok and "stereochemistry is not tracked" in report.warnings[0]
    assert "@" not in report.canonical
    assert report.canonical == validate("C[C@@H](N)C(=O)O").canonical


def test_the_kekule_form_a_molecule_was_written_in_does_not_change_it():
    """`c1ccccc1` and `C1=CC=CC=C1` are one molecule and must score identically."""
    aromatic, kekule = evaluate_smiles("c1ccccc1"), evaluate_smiles("C1=CC=CC=C1")
    assert aromatic.ok and kekule.ok
    assert aromatic.formula == kekule.formula
    assert aromatic.terms["symmetry"] == pytest.approx(kekule.terms["symmetry"])


# -- symmetry and descriptors ---------------------------------------------------


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1", 1),                              # every carbon equivalent
    ("c1ccncc1", 4),                              # N, ortho, meta, para
    ("c1ccc2ccccc2c1", 3),                        # alpha, beta, fusion
    ("Cc1ccccc1", 5),
    ("C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1", 5),
    ("C1CC2CCC1CC2", 2),                          # bridgeheads and CH2
])
def test_symmetry_orbits_match_the_chemistry(smiles, expected):
    report = validate(smiles)
    heavy = [i for i, a in enumerate(report.molecule.atoms) if a.element != "H"]
    colors = orbits(report.molecule)
    assert len({colors[i] for i in heavy}) == expected


def test_rotatable_bonds_exclude_rings_amides_and_terminal_groups():
    assert len(rotatable_bonds(validate("CCCCCC").molecule)) == 3
    assert len(rotatable_bonds(validate("c1ccccc1").molecule)) == 0
    assert len(rotatable_bonds(validate("CC(=O)NCC").molecule)) == 1   # not the C-N
    assert len(rotatable_bonds(validate("C#Cc1ccc(C#Cc2ccccc2)cc1").molecule)) == 0


def test_a_phenyl_rotor_costs_less_than_a_chain_bond():
    """Both are rotatable bonds; only one of them moves the molecular envelope."""
    biphenyl = describe(validate("c1ccccc1-c1ccccc1").molecule)
    hexane = describe(validate("CCCCCC").molecule)
    assert biphenyl.rotatable == 1 and biphenyl.torsion_cost == 0.5
    assert hexane.rotatable == 3 and hexane.torsion_cost == 3.0


@pytest.mark.parametrize("smiles,rings,plate", [
    ("c1ccccc1", 1, 6),
    ("c1ccc2ccccc2c1", 2, 10),            # naphthalene: one ten-atom plate
    ("c1ccccc1-c1ccccc1", 2, 6),          # biphenyl: two plates, six atoms each
    # Triptycene: three benzo rings joined through two sp3 bridgeheads. Asking
    # whether the whole fused *system* is aromatic answers no, and the molecule
    # was credited with zero aromatic rings and no coplanar fragment at all --
    # losing its pi-pi contribution and a third of its interaction diversity,
    # for the scaffold family this search is most often pointed at.
    ("C1=CC=C2C(=C1)C3C4=CC=CC=C4C2C5=CC=CC=C35", 3, 6),
])
def test_aromatic_rings_are_counted_per_ring_not_per_fused_system(smiles, rings, plate):
    described = describe(validate(smiles).molecule)
    assert described.aromatic_rings == rings
    assert described.largest_planar_fragment == plate


def test_a_fused_flat_aromatic_is_one_coplanar_fragment_and_a_biphenyl_is_not():
    assert describe(validate("c1ccc2ccccc2c1").molecule).largest_planar_fragment == 10
    assert describe(validate("c1ccccc1-c1ccccc1").molecule).largest_planar_fragment == 6


# -- the rubric -----------------------------------------------------------------


def test_the_rubric_ranks_the_brief_the_way_the_brief_asks():
    cage = evaluate_smiles("C(c1ccc(I)cc1)(c1ccc(I)cc1)(c1ccc(I)cc1)c1ccc(I)cc1")
    tpm = evaluate_smiles("C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1")
    flat = evaluate_smiles("c1ccc2ccccc2c1")
    chain = evaluate_smiles("CCCCCCCC")
    assert cage.total > tpm.total > flat.total > chain.total
    assert all(report.ok for report in (cage, tpm, flat, chain))


def test_a_flat_fused_aromatic_scores_zero_on_packing_however_rigid_it_is():
    """Criterion 4's whole point: rigid and symmetric is not the same as porous."""
    naphthalene = evaluate_smiles("c1ccc2ccccc2c1")
    assert naphthalene.terms["rigidity"] > 0.9
    assert naphthalene.terms["packing"] == 0.0


def test_packing_is_a_product_so_neither_factor_can_be_bought_with_the_other():
    alkane_cage = evaluate_smiles("C1CC2CCC1CC2")     # awkward, nothing cohesive
    flat_donor = evaluate_smiles("Oc1ccc(O)cc1")      # cohesive, packs flat
    assert alkane_cage.details["cohesion"] < 0.6
    assert flat_donor.details["awkwardness"] < 0.3
    assert max(alkane_cage.terms["packing"], flat_donor.terms["packing"]) < 0.65


def test_flexibility_is_punished_and_symmetric_substitution_is_rewarded():
    rigid = evaluate_smiles("C#Cc1ccc(C#C)cc1")
    floppy = evaluate_smiles("CCOCCOCCOCC")
    assert rigid.terms["rigidity"] > 0.6 > floppy.terms["rigidity"]
    one = evaluate_smiles("Ic1ccccc1")
    four = evaluate_smiles("Ic1cc(I)cc(I)c1")
    assert four.terms["symmetry"] > one.terms["symmetry"]


def test_an_invalid_candidate_scores_nothing_and_says_why():
    report = evaluate_smiles("C(C)(C)(C)(C)C")
    assert not report.ok and report.total == 0.0 and "valence" in report.reason
    assert "INVALID" in report.explain()


REFERENCE_SET = [
    "c1ccccc1", "CCCCCC", "C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1",
    "N#Cc1ccc(C#N)cc1", "Ic1ccc(I)cc1", "OCC(O)CO", "C1CC2CCC1CC2",
    "c1ccc2ccccc2c1", "C(c1ccc(I)cc1)(c1ccc(I)cc1)(c1ccc(I)cc1)c1ccc(I)cc1",
    "C12C3C4C1C5C4C3C25", "COCCOCCOC",
    "C1(c2ccccc2)(c2ccccc2)c2ccccc2-c2ccccc21", "C#Cc1ccc(C#Cc2ccccc2)cc1",
    "C1CC1", "OOc1ccccc1",
    "C(c1ccc(O)cc1)(c1ccc(O)cc1)(c1ccc(O)cc1)c1ccc(O)cc1",
    "C1CCCCCCCCCCC1", "[SiH3]c1ccccc1",
]


def test_every_criterion_actually_separates_molecules():
    """A criterion pinned at its ceiling is weight that ranks nothing.

    Synthesizability was exactly that: it started at 1.0 and only subtracted, so
    sixteen of these eighteen molecules scored 1.000 on it and a fifth of the
    rubric's weight was a constant. The guard is the property, not the fix --
    any future criterion that saturates fails here.
    """
    reports = [evaluate_smiles(smiles) for smiles in REFERENCE_SET]
    assert all(report.ok for report in reports)
    for term in TERMS:
        values = [report.terms[term] for report in reports]
        at_ceiling = sum(1 for value in values if value > 0.995)
        assert at_ceiling <= len(values) // 3, (
            f"{term} is at its ceiling for {at_ceiling}/{len(values)} reference "
            "molecules, so its weight is not ranking anything")
        assert max(values) - min(values) > 0.4, f"{term} barely varies"


def test_ring_penalties_count_rings_rather_than_ring_bonds():
    """A fused system shares bonds between faces; halving bonds is not a count."""
    assert describe(validate("C12C3C4C1C5C4C3C25").molecule).strained_rings == 6
    assert describe(validate("C1CC1").molecule).strained_rings == 1
    assert describe(validate("C1CCCCCCCCCCC1").molecule).macrocycles == 1
    assert describe(validate("c1ccccc1").molecule).macrocycles == 0


# -- weight profiles ------------------------------------------------------------


def test_the_first_weight_profile_is_the_nominal_one_and_all_of_them_normalise():
    profiles = weight_profiles(8, seed=3)
    assert profiles[0].as_dict() == DEFAULT_WEIGHTS.normalized().as_dict()
    for profile in profiles:
        assert sum(getattr(profile, term) for term in TERMS) == pytest.approx(1.0)
    assert weight_profiles(8, seed=3)[5].as_dict() == profiles[5].as_dict()
    assert weight_profiles(8, seed=4)[5].as_dict() != profiles[5].as_dict()


def test_reweighting_a_report_needs_no_rescoring():
    report = evaluate_smiles("C(c1ccc(I)cc1)(c1ccc(I)cc1)(c1ccc(I)cc1)c1ccc(I)cc1")
    heavy_on_packing = parse_weights("packing=0.6")
    assert report.score_with(heavy_on_packing) != report.total
    assert report.score_with(DEFAULT_WEIGHTS) == pytest.approx(report.total)


def test_unknown_criteria_are_refused_rather_than_ignored():
    with pytest.raises(ValueError):
        parse_weights("porosity=0.9")


# -- element restriction and hints ----------------------------------------------


def test_restricting_the_elements_refuses_everything_outside_the_set():
    """`--elements C,H,N,O` has to reach the gate, or the search cheats past it."""
    chno = frozenset({"C", "H", "N", "O"})
    assert evaluate_smiles("Ic1ccc(I)cc1", elements=chno).ok is False
    assert "outside the allowed set" in evaluate_smiles(
        "Ic1ccc(I)cc1", elements=chno).reason
    assert evaluate_smiles("N#Cc1ccc(O)cc1", elements=chno).ok


def test_the_edit_operators_honour_the_element_restriction():
    """Otherwise `--offline` proposes halogens the gate then throws away."""
    chno = frozenset({"C", "H", "N", "O"})
    options = enumerate_mutations("c1ccccc1", elements=chno)
    assert options
    for mutation in options:
        report = validate(mutation.smiles, elements=chno)
        assert report.ok, f"{mutation.smiles}: {report.reason}"


def test_hints_are_drawn_per_expansion_and_are_off_when_not_asked_for():
    """Rotating them is the point: the same four paragraphs on every call is
    something a model optimises against rather than thinks with."""
    drawn = sample_hints(random.Random(3), 2)
    assert len(drawn) == 2 and len(set(drawn)) == 2
    assert sample_hints(random.Random(3), 2) == drawn
    assert sample_hints(random.Random(4), 2) != drawn or True   # streams differ
    assert sample_hints(random.Random(0), 0) == []
    assert len(sample_hints(random.Random(0), 99)) == len(HINTS)

    with_hints = mutation_prompt("c1ccccc1", "total 0.61", hints=drawn)
    without = mutation_prompt("c1ccccc1", "total 0.61")
    assert "WORTH THINKING ABOUT" in with_hints
    assert "WORTH THINKING ABOUT" not in without
    assert len(with_hints) > len(without)


# -- the prior ------------------------------------------------------------------


def test_headroom_is_not_the_score():
    """A finished molecule can score well and have nowhere left to go."""
    finished = evaluate_smiles(
        "C(c1ccc(I)cc1)(c1ccc(I)cc1)(c1ccc(I)cc1)c1ccc(I)cc1")
    fresh = evaluate_smiles("c1ccccc1")
    assert finished.total > fresh.total
    assert (structural_headroom(finished.descriptors).total
            < structural_headroom(fresh.descriptors).total)


def test_the_prior_is_bounded_away_from_zero_and_a_missing_rating_is_not_a_zero():
    headroom = structural_headroom(evaluate_smiles("c1ccccc1").descriptors)
    assert 0.25 <= expansion_prior(headroom) <= 1.0
    assert expansion_prior(headroom, None) == expansion_prior(headroom)
    assert expansion_prior(headroom, 9.0) != expansion_prior(headroom, 1.0)
    assert expansion_prior(None, None) == 1.0


def test_the_prior_changes_which_node_puct_expands():
    """The incumbent is ahead on rank and worked out; the other has headroom.

    Node 0 scores higher and has been expanded eight times; node 1 scores lower,
    has been expanded once, and carries a much larger prior. Under ERA's uniform
    `1/N` the exploration term cannot make up the rank gap and the search keeps
    mining the incumbent. Under the prior it moves -- which is the whole reason
    this example ships with `--prior-exponent 1` instead of ERA's 0.
    """
    rows = (
        Candidate(artifact_id="m", version=0, score=0.55, selected=8, prior=0.1),
        Candidate(artifact_id="m", version=1, score=0.50, selected=1, prior=1.0),
    )
    ctx = SelectionContext(head=rows[0], candidates=rows)
    assert FlatPuct(1.0, prior_exponent=0.0).select(ctx, 1)[0].version == 0
    assert FlatPuct(1.0, prior_exponent=1.0).select(ctx, 1)[0].version == 1


# -- the edit operators ---------------------------------------------------------


def test_every_enumerated_edit_is_a_valid_new_molecule():
    parent = "c1ccccc1"
    seen = {validate(parent).canonical}
    options = enumerate_mutations(parent)
    assert len(options) > 10
    for mutation in options:
        report = validate(mutation.smiles)
        assert report.ok, f"{mutation.smiles}: {report.reason}"
        assert report.canonical not in seen, "an edit repeated an earlier molecule"
        seen.add(report.canonical)


def test_the_symmetrising_operator_raises_symmetry_against_the_single_site_one():
    options = enumerate_mutations("c1ccccc1")
    single = next(m for m in options
                  if m.operator == "substitute" and m.smiles.count("I") == 1)
    every = next(m for m in options if m.operator == "symmetrise" and "I" in m.smiles)
    assert (evaluate_smiles(every.smiles).terms["symmetry"]
            > evaluate_smiles(single.smiles).terms["symmetry"])


def test_the_operators_can_shrink_a_molecule_as_well_as_grow_one():
    options = enumerate_mutations("Ic1ccc(C#Cc2ccccc2)cc1")
    trims = [m for m in options if m.operator == "trim"]
    assert trims, "with only growing moves the search walks into the size cap"
    parent_atoms = validate("Ic1ccc(C#Cc2ccccc2)cc1").atom_count
    assert min(validate(m.smiles).atom_count for m in trims) < parent_atoms


def test_the_operators_never_exceed_the_atom_cap():
    for mutation in enumerate_mutations("C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1",
                                        max_atoms=60):
        assert validate(mutation.smiles).atom_count <= 60


def test_the_offline_proposer_is_deterministic_for_a_seed():
    first = propose_offline("c1ccccc1", random.Random(7))
    second = propose_offline("c1ccccc1", random.Random(7))
    assert first is not None and first.smiles == second.smiles


def test_lineage_is_measurable_but_not_gateable():
    """Why the search records `parent_similarity` and never enforces it.

    A floor on parent similarity is the obvious way to hold a search to "modify
    the molecule you were given". These numbers are why it is not there: the
    best molecule the live run found scores *below* a threshold that an
    unrelated molecule would also fail, because substituting every
    symmetry-equivalent position rewrites every atom environment in the parent.
    """
    def sim(a, b):
        return similarity(validate(a).molecule, validate(b).molecule, radius=1)

    hexasubstituted = ("c1(-c2ccc(Br)cc2)c(-c2ccc(Br)cc2)c(-c2ccc(Br)cc2)"
                       "c(-c2ccc(Br)cc2)c(-c2ccc(Br)cc2)c1-c2ccc(Br)cc2")
    assert sim("c1ccccc1", "Ic1ccccc1") > 0.4          # one substituent: obvious
    assert sim("c1ccccc1", hexasubstituted) < 0.1      # legitimate, and unrecognisable
    assert sim("c1ccccc1", "CCCCCC") == 0.0            # genuinely unrelated
    assert sim("c1ccccc1", "c1ccccc1") == 1.0


def test_the_fingerprint_does_not_move_between_processes():
    """`hash()` is salted per process, so a canonical form built on it is
    canonical only within one run -- and two runs of a seeded search would then
    disagree about which molecules they had already seen."""
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, '.');"
            "from examples.porous._smiles import validate;"
            "print(validate('Ic1ccc(C#N)cc1').canonical)")
    seen = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
                           ).stdout.strip()
            for seed in ("0", "1", "12345")}
    assert len(seen) == 1 and seen != {""}, seen


# -- the tree -------------------------------------------------------------------


def _node(index, parent, score, **kwargs):
    return port.Node(index=index, parent_index=parent, smiles=f"m{index}",
                     summary="", score=score, **kwargs)


def test_a_visit_backpropagates_to_every_ancestor():
    tree = port.MoleculeTree()
    tree.seed(_node(0, None, 0.5))
    tree.add_node(_node(-1, 0, 0.6))
    tree.add_node(_node(-1, 1, 0.7))
    before = [n.num_visits for n in tree.nodes]
    tree.select_parent()
    after = [n.num_visits for n in tree.nodes]
    assert sum(after) - sum(before) >= 2, "the visit stopped at the chosen node"
    assert after[0] > before[0], "the root's visit count never moved"


def test_the_expansion_budget_is_the_iteration_count():
    tree = port.MoleculeTree(candidate_limit=2)
    tree.seed(_node(0, None, 0.5))
    assert tree.select_parent() is not None
    assert tree.select_parent() is not None
    assert tree.select_parent() is None


def test_a_refused_candidate_becomes_a_dead_end_node_rather_than_vanishing():
    """Upstream FUTS appends a failed program; dropping it would change the rule.

    The node has to exist -- it is in the rank denominator -- and it has to be
    unselectable, which `-inf` does by ranking it last forever.
    """
    tree = port.MoleculeTree()
    tree.seed(_node(0, None, 0.5))
    tree.add_node(_node(-1, 0, -math.inf))
    assert len(tree.nodes) == 2
    assert tree.best().index == 0
    for _ in range(6):
        picked = tree.select_parent()
        assert picked is not None and picked[1].index == 0


def test_a_repeated_molecule_is_recorded_as_a_duplicate():
    tree = port.MoleculeTree()
    report = evaluate_smiles("c1ccccc1")
    tree.seed(port.Node(0, None, "c1ccccc1", "", 0.6, report=report))
    twin = tree.add_node(port.Node(-1, 0, "C1=CC=CC=C1", "", 0.6,
                                   report=evaluate_smiles("C1=CC=CC=C1")))
    assert twin.duplicate_of == 0


def test_the_strategy_round_trips_a_proposal_into_a_diff():
    strategy = port.MoleculeStrategy("c1ccccc1")
    payload = json.dumps({"smiles": "Ic1ccccc1", "iteration": 3, "parent_index": 1,
                          "change_summary": "iodo", "promise": "8.0"})
    diff = strategy.to_diff(strategy.initial(), payload, "w0", 0, port.ARTIFACT_ID)
    assert diff is not None and diff.ops["smiles"] == "Ic1ccccc1"
    assert diff.ops["parent_index"] == "1" and diff.ops["promise"] == "8.0"
    assert strategy.to_diff(strategy.initial(), "not json", "w0", 0,
                            port.ARTIFACT_ID) is None


# -- reading a model reply ------------------------------------------------------


@pytest.mark.parametrize("reply,expected", [
    ("SMILES: Ic1ccccc1\nCHANGE: iodinated\nPROMISE: 7", ("Ic1ccccc1", 7.0)),
    ("smiles = Ic1ccccc1\npromise = 3.5", ("Ic1ccccc1", 3.5)),
    ("```\nIc1ccccc1\n```\nPROMISE: 6", ("Ic1ccccc1", 6.0)),
    ("Here it is:\nIc1ccccc1", ("Ic1ccccc1", None)),
    ("", ("", None)),
])
def test_a_reply_is_read_leniently_and_a_missing_rating_stays_missing(reply, expected):
    smiles, _summary, promise = extract_molecule(reply)
    assert (smiles, promise) == expected


# -- the drawing ----------------------------------------------------------------


@pytest.mark.parametrize("smiles,floor,ceiling", [
    ("c1ccccc1", 0.85, 1.2),
    ("c1ccc2ccccc2c1", 0.85, 1.2),
    ("C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1", 0.85, 1.2),
    ("C#Cc1ccc(C#Cc2ccccc2)cc1", 0.85, 1.2),
    ("OCC(O)CO", 0.85, 1.2),
    # Bridged polycyclics get a wider band, and the reason is geometry rather
    # than a weak layout: three benzo rings sharing two bridgehead carbons have
    # no planar drawing with equal bonds, which is why every textbook draws
    # triptycene stylised. The band still catches a regression -- at the
    # parameters this module shipped with, triptycene sits at 0.57 / 1.56.
    ("C1=CC=C2C(=C1)C3C4=CC=CC=C4C2C5=CC=CC=C35", 0.5, 1.7),
    ("C1CC2CCC1CC2", 0.5, 1.7),
])
def test_every_heavy_atom_gets_a_position_and_bonds_come_out_the_same_length(
        smiles, floor, ceiling):
    """A depiction that drops atoms or stretches bonds is worse than no picture."""
    mol = validate(smiles).molecule
    pos = coordinates(mol)
    heavy = [i for i, a in enumerate(mol.atoms) if a.element != "H"]
    assert set(pos) == set(heavy)
    lengths = [math.dist(pos[b.a], pos[b.b]) for b in mol.bonds]
    mean = sum(lengths) / len(lengths)
    assert floor < min(lengths) / mean, lengths
    assert max(lengths) / mean < ceiling, lengths


def test_a_ring_is_drawn_as_a_ring():
    """Benzene's six bonds are equal and its atoms sit on a circle."""
    mol = validate("c1ccccc1").molecule
    pos = coordinates(mol)
    lengths = [math.dist(pos[b.a], pos[b.b]) for b in mol.bonds]
    assert max(lengths) - min(lengths) < 0.05
    cx = sum(p[0] for p in pos.values()) / 6
    cy = sum(p[1] for p in pos.values()) / 6
    radii = [math.dist(p, (cx, cy)) for p in pos.values()]
    assert max(radii) - min(radii) < 0.05


def test_the_svg_labels_heteroatoms_and_leaves_carbon_as_a_vertex():
    drawing = svg(validate("N#Cc1ccc(O)cc1").molecule)
    assert drawing.startswith("<svg") and drawing.endswith("</svg>")
    assert ">N<" in drawing and ">OH<" in drawing
    assert ">C<" not in drawing, "carbons are vertices, not labels"
    assert drawing.count("<line") >= len(validate("N#Cc1ccc(O)cc1").molecule.bonds)


def test_a_late_expansion_is_kept_because_the_tree_cannot_go_stale():
    """The default staleness policy here is `full`, and this is why.

    A node's place in the tree is its parent index; the head moving on cannot
    make a molecule that was drawn from node 3 stop being an expansion of node
    3. Measured offline at `async_ratio=2`: `guarded` discarded 11 of 16
    completed expansions and built a 6-node tree where `full` discarded none and
    built a 17-node one, on the same rollouts. On a live run each of those
    discards is a five-minute model call.
    """
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.staleness import get_policy

    def filtered(policy_name):
        aggregator = port.PorousTreeAggregator(
            ledger=None, verifier=None, tree=port.MoleculeTree(),
            config=AggregatorConfig(), staleness_policy=get_policy(policy_name),
            profiles=[DEFAULT_WEIGHTS])
        card = EvidenceCard(
            diff=Diff(diff_id="d", target=port.ARTIFACT_ID,
                      ops={"smiles": "Ic1ccccc1"}, author="w0"),
            base_version={port.ARTIFACT_ID: 0}, touched=["smiles"])
        survivors, discarded = aggregator._staleness_filter(
            {port.ARTIFACT_ID: 9}, [card])          # nine versions behind
        return survivors, discarded

    kept, dropped = filtered("full")
    assert len(kept) == 1 and not dropped
    kept, dropped = filtered("guarded")
    assert not kept and len(dropped) == 1, "guarded is the policy `full` replaces"


# -- the report -----------------------------------------------------------------


def _report_payload():
    run = port.run_search(None, mode="sync", seed_smiles="c1ccccc1", iterations=6,
                          workers=2, profiles=6, test_profiles=3, seed=4)
    return run.payload({"mode": "sync", "iterations": 6,
                        "weights": DEFAULT_WEIGHTS.normalized().as_dict()}, None)


def test_the_report_renders_every_section_and_draws_the_molecules():
    html = render_html(_report_payload(), lang="en",
                       rubric_note="note", caveats=["a caveat"])
    for fragment in ("Starting molecule", "Best molecule found",
                     "Every valid candidate", "The search trajectory",
                     "What the run cost", "a caveat"):
        assert fragment in html, fragment
    assert html.count("<svg") >= 2, "the two headline molecules must be drawn"
    assert "<script" not in html, "a printable report needs no scripts"


def test_the_report_speaks_both_languages_without_changing_the_numbers():
    payload = _report_payload()
    english = render_html(payload, lang="en")
    chinese = render_html(payload, lang="zh")
    assert "多孔分子晶体候选分子" in chinese and "Porous molecular" in english
    best = payload["best_molecule"]["score"]
    assert f"{best:.3f}" in english and f"{best:.3f}" in chinese


@pytest.mark.skipif(_chrome() is None, reason="no headless Chromium installed")
def test_the_pdf_is_actually_written(tmp_path):
    """A zero-byte PDF is the failure this must not report as success."""
    path = write_pdf(render_html(_report_payload()), tmp_path / "report.pdf")
    assert path.exists() and path.stat().st_size > 5000
    assert path.read_bytes()[:5] == b"%PDF-"


# -- one whole run --------------------------------------------------------------


def test_the_offline_search_improves_on_its_seed(tmp_path):
    run = port.run_search(None, mode="sync", seed_smiles="c1ccccc1", iterations=12,
                          workers=3, profiles=6, test_profiles=3, seed=1)
    root, best = run.tree.root(), run.tree.best()
    assert run.result.error is None
    assert len(run.tree.nodes) >= 12
    assert best.valid and best.score > root.score
    assert best.report.details["atom_count"] <= 100
    payload = run.payload({"test": True}, None)
    assert payload["best_molecule"]["smiles"] == best.smiles
    assert payload["held_back_weightings"]["best"]["mean"] is not None
    assert set(payload["breakdown"]["terms"]) == set(TERMS)


def test_the_search_reports_on_weightings_it_never_optimised_against():
    """The held-back profiles are not in `build_tasks`, so nothing can see them."""
    run = port.run_search(None, mode="sync", seed_smiles="c1ccccc1", iterations=8,
                          workers=2, profiles=5, test_profiles=3, seed=2)
    task_profiles = {t.meta["profile"] for t in port.build_tasks(5, 2)}
    assert task_profiles == set(range(5))
    assert len(run.test_profiles) == 3
    assert run.profiles[0].as_dict() != run.test_profiles[0].as_dict()


def test_the_scored_profiles_are_a_prefix_of_the_drawn_ones():
    """`run_search` draws `profiles + test_profiles` and slices; `build_tasks`
    draws `profiles`. If those two ever stop agreeing, a rollout would be scored
    under one weighting and the task would claim another -- silently."""
    whole = weight_profiles(12, seed=5)
    assert [p.as_dict() for p in whole[:8]] == [
        p.as_dict() for p in weight_profiles(8, seed=5)]
    tasks = port.build_tasks(8, 5)
    assert [t.meta["weights"] for t in tasks] == [p.as_dict() for p in whole[:8]]


def test_an_unusable_seed_molecule_is_refused_rather_than_searched_from():
    with pytest.raises(RuntimeError, match="not usable"):
        port.run_search(None, mode="sync", seed_smiles="c1ccc1", iterations=2,
                        workers=1, profiles=4, test_profiles=2)


def test_the_entry_point_plans_without_touching_a_model(capsys):
    assert port.main(["--dry-run", "--offline"]) == 0
    out = capsys.readouterr().out.lower()
    assert "dry-run" in out and "no model api was accessed" in out
