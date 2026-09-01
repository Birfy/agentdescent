# Porous molecular crystals — flat-PUCT tree search over molecules

Start from one molecule. Search the space of edits to it with the same flat-PUCT
tree the [ERA port](../era/) runs, scoring each candidate against a five-criterion
rubric for **porous molecular crystals** — crystals held together only by
intermolecular forces whose packing still leaves permanent, accessible voids.

| | |
|---|---|
| Kind | Molecular design search (a domain application, not an algorithm port) |
| Governance layer | L2 (`blast_radius=0.2`) — scoring is a pure function, nothing executes |
| Selection rule | `agentdescent.selection.FlatPuct`, with a **non-uniform `P(s,a)`** |
| Search engine | `evolve()` / `async_evolve()` with `aggregator_factory=` |
| Dependencies | none — the SMILES parser, kekuliser and validity gate are in this folder |

```bash
python -m examples.porous.porous_tree_search --dry-run          # plan only
python -m examples.porous.porous_tree_search --offline \
       --iterations 24 --workers 4                              # no API key needed
python -m examples.porous.porous_tree_search --provider openai \
       --model deepseek-v4-pro --iterations 24 --workers 4 --yes
```

## The loop

```
seed molecule (default: benzene, whose packing term is exactly 0.00)
  └─ PUCT picks any node in the tree:  rank + c_puct · P(s,a) · √N / (1 + n)
       └─ one deliberate modification is proposed, and must come back as a SMILES
            └─ the SMILES is validated: parses, sensible valences, neutral,
               closed shell, one molecule, under the atom cap
                 └─ valid → scored on held-out weight profiles
                    invalid → the gate's reason goes back for one retry, then the
                              node is appended scoring −inf, a permanent dead end
                      └─ visits backpropagate up the parent chain
```

## The five criteria

| Criterion | What the descriptor actually measures |
|---|---|
| **Rigidity** | torsion cost per heavy atom (aryl rotors count half, chain bonds full), ring-atom fraction, fused-atom fraction |
| **Symmetry** | Weisfeiler-Lehman orbit count over heavy atoms, plus the size of the largest orbit — 4 equivalent arms is not the same as 2 |
| **Directional sites** | halogen-bond σ-hole strength (I > Br > Cl, F = 0), complementary H-bond donor/acceptor pairs, aromatic rings — weighted by whether the site is mounted on something rigid |
| **Open packing** | `sqrt(awkwardness × cohesion)` — a **product**, so a shape that cannot pack densely but has nothing holding a crystal together scores nothing, and neither does a cohesive flat disc |
| **Synthesizability** | strain, macrocycles, stereocentres, unstable heteroatom chains and exotic elements against symmetry and a skeleton built from aryl / alkyne / nitrile / imine / aryl-halide chemistry |

Weights are `--weights rigidity=0.3,packing=0.3,...`; all five are renormalised.

## The prior

ERA has no prior: `futs.py` gives every node `1/N`, so the exploration budget is
spread evenly over a tree in which most nodes are finished or dead. A molecule
carries visible evidence about whether it has anywhere left to go, so this
example spends that budget on **headroom** instead:

* atoms left under the cap, free attachment sites,
* interaction modes not yet used (no halogen bond? no hydrogen bond?),
* how flat it still is.

blended with the model's own `PROMISE` rating of the direction — read out of the
same reply the expansion was already paying for, so it costs no extra call.
`--prior-exponent 0` restores ERA's uniform prior exactly.

**Headroom is not the score.** A rigid symmetric flat aromatic scores well and is
finished; a lopsided first-draft cage scores badly and is one substitution from
being good.

## What the score is not

It is a **topological proxy**, not a lattice energy. There is no conformer
generation, no force field and no crystal structure prediction anywhere in this
folder. Criterion 4 in its honest form is a CSP run costing hours per molecule;
what this search produces is a ranked list of candidates *to send to* that
calculation. [`docs/porous-molecules.md`](../../docs/porous-molecules.md) states
where the proxy is weakest and how to replace the scorer with a real one.

## Files

| File | What is in it |
|---|---|
| `porous_tree_search.py` | the entry point: the tree, the strategy, the aggregator, the CLI |
| `_smiles.py` | SMILES parser, aromaticity perception, kekuliser, writer, canonical key, and the validity gate |
| `_descriptors.py` | rings, rotatable bonds, orbits, planar fragments, functional groups |
| `_score.py` | the five criteria, the weighting, and the perturbed weight profiles |
| `_prior.py` | structural headroom and `P(s,a)` |
| `_mutations.py` | rule-based edit operators, for `--offline` and as the control arm |
| `_prompts.py` | the design brief, the reply protocol, and the reply parser |
