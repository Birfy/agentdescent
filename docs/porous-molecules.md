# Molecule search — porous molecular crystals

**Start from one molecule, search the edits to it with flat PUCT, and score every
candidate against five criteria for porous molecular crystals.** The engine is
the same one every other example on this site runs; what is new is the domain —
and that the domain has something to put in AlphaZero's `P(s,a)` slot, which ERA
leaves uniform.

- Entry point: [`examples/porous/porous_tree_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/porous/porous_tree_search.py)
- Folder README: [`examples/porous/`](https://github.com/Birfy/agentdescent/blob/main/examples/porous/README.md)
- Selection rule: [`FlatPuct`](selection.md) — the ERA rule, with a real prior
- Aggregator seam: [`aggregator_factory`](aggregator-factory.md)

```bash
python -m examples.porous.porous_tree_search --dry-run
python -m examples.porous.porous_tree_search --offline --iterations 24 --workers 4

# any OpenAI-compatible endpoint, through agentdescent.agents.openai_compatible
export OPENAI_BASE_URL=...      # e.g. https://ark.cn-beijing.volces.com/api/coding/v1
export OPENAI_API_KEY=...
python -m examples.porous.porous_tree_search --provider openai \
    --model deepseek-v4-pro --iterations 24 --workers 4 --yes
```

`--offline` runs the whole search with the rule-based edit operators instead of a
model: no API key, no network, and a control arm a model-driven run has to beat.

## The problem

A *porous molecular crystal* is held together only by intermolecular forces —
no covalent framework, no coordination bonds — and still packs with permanent,
accessible void space. Tetraphenylmethane derivatives, triptycenes and
tetrahedral halogen-bond donors are the classic scaffolds. Whether a given
molecule does it is settled by a crystal structure prediction run: generate
packings, minimise them, look at the lattice-energy landscape. That costs hours
per molecule, which is exactly the situation a search wants a cheap proxy for.

So this example searches on a **topological proxy** and says so everywhere it
reports a number. What it produces is a ranked shortlist to send to the real
calculation, not a replacement for it.

## The loop

```
seed molecule (default benzene — rigid, symmetric, and packs densely)
  └─ PUCT picks any node in the tree:  rank + c_puct · P(s,a) · √N / (1 + n)
       └─ one deliberate modification, which must come back as a SMILES
            └─ the gate: parses, sensible valences, neutral, closed shell,
               one molecule, at most --max-atoms atoms
                 └─ valid   → scored on the held-out weight profiles
                    invalid → the gate's reason goes back to the model for one
                              retry, then the node is appended scoring −inf
                      └─ visits backpropagate up the parent chain
```

Every part of that except the rubric and the gate is shared code:
`FlatPuct` selects, `evolve()` supplies workers and the ledger, and the tree is
installed as the merge optimizer through `aggregator_factory=`. The tree is
append-only and a node's place in it is its parent index, so a late-arriving
proposal is still a legitimate expansion of the parent it was drawn from —
which is why the staleness policy may rebase rather than discard.

## The gate — is this a molecule at all?

An LLM asked for a rigid cage will answer with a five-bonded carbon, an
unclosed ring digit, a methyl radical or `c1ccc1` often enough that accepting
them would spend the budget scoring things that cannot exist. There is no RDKit
here — the repository has zero required dependencies — so
[`_smiles.py`](https://github.com/Birfy/agentdescent/blob/main/examples/porous/_smiles.py)
implements the parts that decide validity:

| Check | What it rejects |
|---|---|
| Syntax | unbalanced branches, unclosed ring-closure digits, stray bond symbols |
| Elements | anything outside `C H N O F Si P S Cl Se Br I` |
| One fragment | `A.B` — a salt or solvate is not a candidate molecule |
| Kekulisation | an aromatic ring system with no alternating double-bond assignment (`c1cccc1`), and aromatic rings that are not 5-, 6- or 7-membered (`c1ccc1`) |
| Valence | a per-element, per-charge table, applied after kekulisation |
| Radicals | a bracket atom whose electron count fills no normal valence (`[CH3]`), **and** an odd total electron count for the molecule |
| Neutrality | any net formal charge |
| Size | heavy atoms plus hydrogens over `--max-atoms` (default 100) |

Aromaticity is **perceived**, not assumed: a Kekulé-written benzene is
recognised as aromatic (Hückel over the smallest ring through each ring bond,
including fusion atoms whose double bond points into the neighbouring ring). Two
spellings of one molecule therefore de-duplicate against each other and get the
same symmetry orbits, which they did not before that step existed.

### "Modify the molecule you were given" is recorded, not enforced

Every expansion is asked for **one deliberate modification of the parent**, and
each node records `parent_similarity` — a multiset Tanimoto over atom
environments at radius 1 — so a reader of the result file can check that it got
one. There is no floor on it, and the measurement is why:

| parent → child | similarity |
|---|---|
| benzene → iodobenzene | 0.44 |
| benzene → hexakis(4-bromophenyl)benzene *(the best molecule the live run found)* | **0.06** |
| benzene → hexane *(shares nothing)* | 0.00 |

There is no threshold between "bold but legitimate" and "unrelated", because the
move this rubric most wants — substituting **every** symmetry-equivalent
position at once — rewrites every atom environment in the parent. A gate at any
value that rejects the third row also rejects the second.
`tests/test_porous_molecules.py::test_lineage_is_measurable_but_not_gateable`
pins those numbers, so the decision fails loudly if the metric ever changes.

## The rubric

Five criteria, each in `[0, 1]`, combined by weights that default to
`rigidity 0.25, symmetry 0.15, interactions 0.20, packing 0.25,
synthesizability 0.15` and are overridable with `--weights`.

**Rigidity.** `0.5 · 1/(1 + 8·torsions per heavy atom) + 0.3 · ring fraction +
0.2 · fused fraction`. Torsions are weighted: a phenyl spinning on its stalk
sweeps nearly the same envelope at every angle and counts half, a bond in an
`-O-CH₂-CH₂-` chain moves everything past it and counts one. Without that
weighting a tetraphenylmethane — four rotors, one rigid shape — scored as floppy
as a hexane.

**Symmetry.** Weisfeiler-Lehman refinement colours over the heavy atoms. Atoms
an automorphism maps onto each other always share a colour, so the colour count
is a *lower bound* on the orbit count — the direction that can flatter a
molecule but never punish a genuinely symmetric one. A quarter of the term is
the size of the largest orbit, because four equivalent arms is a different
synthesis, and a different CSP problem, from two.

**Directional interaction sites.** Halogen-bond σ-hole strength (I 1.0, Br 0.75,
Cl 0.45, **F 0.0** — fluorine has essentially no σ-hole and a candidate that
fluorinates everything must not be credited with halogen bonding), complementary
hydrogen-bond pairs, and aromatic rings for π-stacking. Three components:
saturating density, how many *kinds* of interaction are present, and — the third
that stops this being a hydroxyl-counting contest — what fraction of the sites
are mounted on something rigid. A donor on the end of a propyl chain points
wherever the chain lets it.

**Open packing that is still competitive on lattice energy.**

```
packing = sqrt(awkwardness × cohesion)
```

A **product**, not a sum, and that is the whole of criterion 4. *Awkwardness* is
how badly the shape fills space: branch points that actually branch, a small
coplanar fraction, non-aromatic bridgeheads and quaternary centres. *Cohesion*
is what would pay for the void: aromatic surface, directional sites, a size with
enough contacts to matter, few floppy bonds to eat the space back. A shape that
cannot pack densely but has nothing holding a crystal together scores zero; so
does a strongly cohesive flat disc. A weighted sum would let a candidate buy one
with the other, which is the failure mode "chase the biggest void" names.

Naphthalene is the check on that term: rigid (1.00), symmetric (0.83), and
**packing exactly 0.00** — a flat plate is the densest-packing shape there is.

**Synthesizability.** Not the published SA_Score, whose fragment-frequency table
would be a dependency. Structural strain, macrocycles, stereocentres, unstable
heteroatom chains, exotic elements and sheer size, against the two things that
make a cage synthesis *easier*: symmetry, which turns *n* couplings into one,
and a skeleton built from aryl, alkyne, nitrile, imine and aryl-halide
chemistry.

### Weight profiles are the held-out split

The rubric is deterministic, so there is no data to hold out — and a molecule
that only wins under one exact set of five weights has not been shown to be
good. So the *weighting* is the shard: `--profiles 8` perturbed weightings are
the tasks `evolve()` splits into a train half and a gate half, and
`--test-profiles 4` more never enter the task list at all. The number a run
finally reports is on those.

## The prior — `P(s,a)`

`FlatPuct`'s exploration term is `c_puct · P(s,a) · √N / (1 + n)`. Upstream ERA
has no prior: `futs.py` uses `1/len(nodes)` for every node, so the budget is
spread evenly over a tree in which most nodes are finished or dead. A molecule
carries evidence about whether it has anywhere left to go, and
[`_prior.py`](https://github.com/Birfy/agentdescent/blob/main/examples/porous/_prior.py)
spends the budget on that instead:

| Component | Reads |
|---|---|
| `size` | atoms left under the cap — a candidate five atoms short of it cannot grow |
| `sites` | free attachment points, against one per eight heavy atoms |
| `interactions` | interaction modes not yet used: no halogen bond and no hydrogen bond is a whole mode still available |
| `shape` | how flat it still is, and whether it has any three-dimensional junction at all |

blended with the model's own `PROMISE` rating, read out of the same reply the
expansion was already paying for. The result is clamped to `[0.25, 1.0]`: the
floor matters, because a prior of zero would bar a node from ever being selected
again on the strength of an estimate, and these are estimates.

**Headroom is deliberately not the score.** A rigid symmetric flat aromatic
scores well and is finished; a lopsided first-draft cage scores badly and is one
substitution from being good. `--prior-exponent 0` restores ERA's uniform prior
exactly, `1` (the default here) uses the blend, `2` squares it.

## Measured — the prior, offline, is not worth anything yet

Six seeds per row, benzene as the starting molecule, 24 expansions over 4
workers, proposals from the rule-based operators — so the *only* difference
between rows is `--prior-exponent`. Gain is on the four weightings held back
entirely, mean ± population standard deviation over the six seeds.

| `--prior-exponent` | mean gain | best molecule seen |
|---|---|---|
| `0` — ERA's uniform `1/N` | **+0.163** ± 0.045 | 0.823 |
| `1` — the headroom prior (this example's default) | **+0.153** ± 0.043 | 0.823 |
| `2` — squared | **+0.153** ± 0.043 | 0.823 |

**There is no effect here, and the honest reading is that this experiment could
not have found one.** The difference between the first row and the other two is
a fifth of one standard deviation; rows two and three are identical to four
decimal places, because in a 24-node tree the rank term moves by `1/23` per
place and the prior moves the exploration term by less than two places, so the
argmax rarely changes. All three rows converge on the same best molecule.

Two things that experiment is *not* evidence about:

* **Half the prior was missing.** Offline, nothing supplies a `PROMISE` rating,
  so the prior was structural headroom alone. The premise the design rests on —
  that a model's own rating of a direction, which costs no extra call, is worth
  spending exploration on — is untested by this table.
* **The proposer is uniform-random.** The rule-based operators draw uniformly
  from the valid single edits and do no hill-climbing, on purpose: a tree search
  evaluated against a proposer that already climbs measures nothing. That also
  means the tree is what limits the result here, not the chemistry.

The mechanism itself is pinned by
`tests/test_porous_molecules.py::test_the_prior_changes_which_node_puct_expands`,
which constructs the case the prior exists for — an incumbent that is ahead on
rank and worked out, against a fresher node with headroom — and asserts that the
uniform prior keeps mining the incumbent while the headroom prior moves.

## Measured — with a model

`deepseek-v4-pro` (a reasoning model, behind an OpenAI-compatible endpoint),
benzene as the seed, 4 expansions over 2 workers, `--prior-exponent 1`:

| | molecule | score | held-back weightings |
|---|---|---|---|
| seed | `c1ccccc1` | 0.614 | 0.541 |
| best | `c1(-c2ccc(Br)cc2)c(...)...` — hexakis(4-bromophenyl)benzene | **0.795** | **0.801** |

Four expansions, and the search left a flat dense packer for a propeller-shaped
hexasubstituted benzene with six halogen-bond donors — a scaffold family that
really does form clathrates. The reply that produced it named the criteria it
was aiming at:

> *"I substituted all six benzene positions with 4-iodophenyl groups, turning the
> flat, dense parent into a non-planar, high-symmetry propeller with six
> directional I halogen-bond donors to target the weak packing and interaction
> scores."*

which is the design of the prompt working: the model is shown the parent's
**breakdown**, not its score, so it can aim at the failing criterion.

Two costs worth recording. A reasoning model spends around 200 s and 14 000
tokens on one expansion — the thinking is billed and timed like output, even
though the reply is three lines — so a 4-expansion run took 13 minutes of
wall-clock for 20 calls, and at the old 180 s timeout three of four workers lost
their first round to `TimeoutError` before the adapter's retries recovered them.
That is why `--api-timeout` defaults to 300 s and `--max-tokens` to 4096 here: a
reasoning model starved of either returns an empty reply, which this search
would record as a node the gate refused, indistinguishable from bad chemistry. And two workers in the same round proposed the same molecule under
different spellings — the aromaticity perception caught it and the second node
is flagged `duplicate_of`, but the "already tried" list in the prompt can only
show children that have already been merged, so same-round twins are a real cost
of parallel expansion here.

## What this is not

* **Not a lattice energy.** No conformers, no force field, no CSP. Criterion 4
  is a shape-and-cohesion proxy; the honest version is a structure-prediction
  run, and the output of this search is its input.
* **Not 3D.** Every descriptor is computed on the molecular graph. Two molecules
  with the same graph and different accessible conformations are the same
  molecule here.
* **Not a synthesis planner.** The synthesizability term is a structural
  heuristic; a retrosynthesis tool would disagree with it, sometimes sharply.
* **Not a substitute for RDKit.** The gate covers the organic subset plus
  bracket atoms. It has no tautomer perception and no stereochemistry beyond
  counting stereocentres.

## Replacing the scorer

The rubric is a pure function of a SMILES string, which is the seam:

```python
from examples.porous import porous_tree_search as search

run = search.run_search(my_completion, seed_smiles="C(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1",
                        iterations=48, workers=4, prior_exponent=1.0)
print(run.tree.best().smiles, run.tree.best().score)
```

To score with something real — a CSP pipeline, a trained property model, a
docking run — replace `evaluate_smiles` in
[`_score.py`](https://github.com/Birfy/agentdescent/blob/main/examples/porous/_score.py).
Nothing else in the search knows what the number means; `PorousTreeAggregator`
only requires that larger is better and that an unusable candidate comes back as
`-inf`. If the new scorer is expensive, raise `--workers` and run `--async`: the
tree is append-only, so an expansion that lands late is still valid.
