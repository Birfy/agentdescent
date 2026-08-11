# Candidate selection — where the next batch starts

!!! note "One field of the bundle"
    This is the `selection` field of the [Policies bundle](policies.md); where a keyword argument exists it is a shortcut onto that field, and an explicit argument wins over a bundle default.


`evolve()` has a replaceable rule for almost every decision it makes: which task
a worker rolls out ([sampling](sampling.md)), what a stale diff is worth
([staleness](staleness.md)), which diffs contradict, whether to fuse, whether to
commit, when to promote ([aggregator](aggregator.md)). It was missing one:

> **Which candidate does the next batch of workers start from?**

The engine has one `dev` head and starts every worker there. `TaskScheduler`'s
UCB looks like the missing piece and is not — it chooses a *task*, not a
*candidate*.

## Why this was a real gap, not tidiness

Look at the ports. GEPA's Pareto frontier, EvoSkill's top-K aggregate frontier,
DGM's archive and ADAS's archive are each a candidate-selection rule, and each is
written out by hand inside its own example, because the engine had nowhere to put
one.

1. **"We did not change the semantics" could not be checked.** The
   [parallelisation matrix](results.md) claims each port's published selection
   rule is untouched. While that rule lives in the example, the claim rests on a
   human reading the file.
2. **Tree search could not be expressed at all.** With one head there is nowhere
   for beam search or MCTS to keep the frontier they are made of.
3. **Pareto was implemented twice** — GEPA's per-instance version and EvoSkill's
   top-K aggregate — and the difference between them is a fidelity detail this
   repository documents in prose. It should be an argument.

## Selection and merging are not alternatives

This is the part worth being explicit about, because "pick the best candidate"
and "merge every candidate" sound like competing answers. They are different
layers:

```
SelectionPolicy picks k starting points
  └─ N/k workers under each, each proposing a diff
       └─ the aggregator merges them into that starting point
```

One selected starting point still has N/k workers under it whose diffs are merged
back into it. The merge layer sits *under* any search strategy rather than
competing with one.

## The policies

```python
from agentdescent import Policies, evolve
from agentdescent.selection import Archive, Beam, MCTS, ParetoFrontier, SingleHead

evolve(tasks, reward, agent=agent,
       policies=Policies(selection=Beam(4)))
```

| policy | corresponds to | note |
|---|---|---|
| `SingleHead()` | today's engine | the default; every worker starts from the head |
| `Beam(k)` | classic beam search | the walk over the `k` best is continuous across calls, so asking one at a time expands each in turn; `Beam(1)` computes `SingleHead`'s answer by another route, and the tests assert they agree |
| `ParetoFrontier(mode=...)` | GEPA / EvoSkill | `win_frequency` is GEPA's Algorithm 2 exactly; `per_instance` is plain Pareto walked round-robin; `topk_aggregate` is EvoSkill's — three published rules, one argument |
| `Archive(sampling=...)` | DGM / ADAS / SICA | `sigmoid_novelty` is DGM's `choose_selfimproves`; `performance`, `novelty` (softmax ÷ `1 + selected`), `best` (SICA's `idxmax`), `uniform` as the ablation |
| `MCTS(exploration=...)` | tree search | UCT over the candidate tree; one evolve step is one rollout, value is held-out reward, backup runs up `Candidate.parent` |

Three details that are decisions rather than defaults:

**An unscored candidate sorts first, not last.** `Candidate.score is None` means
*unmeasured*. Ranking it as the worst is how a beam collapses onto a single line
of descent and stops being a beam, and how an archive stops exploring.

**Per-instance Pareto refuses to fall back.** Given candidates with no
`per_task` scores, `win_frequency` and `per_instance` raise rather than quietly
ranking on the aggregate — which would be running *EvoSkill's* rule and
reporting it under GEPA's name.

**`per_instance` is not GEPA, and used to claim it was.** Plain Pareto keeps
every candidate nothing dominates, including ones that are best at nothing, and
walks them round-robin. Algorithm 2 admits only the per-instance winners and
draws in proportion to how many instances each still wins. `win_frequency` is
what the old name meant.

**The walk over a pool is continuous, not restarted.** `Beam` and
`ParetoFrontier` offset their round-robin by `SelectionContext.round`. It
matters because the population layer asks for **one** starting point per merge —
the ledger holds one live head — and a policy that answered "the best" every
time made `k` inert: `Beam(4)` was `Beam(1)`, and `ParetoFrontier` sat on
whichever front member was admitted first, usually the seed, while candidates
scoring far higher arrived and were never expanded. Rotating expands each slot
in turn: serial where textbook beam search is parallel, same frontier. At
`round == 0` it is exactly the old per-call round-robin, which is what makes the
change checkable — the tests pin that every policy's round-0 answer is the
answer it gave before.

**`Archive` is deterministic given its seed.** An archive that samples differently
on a re-run makes a seeded comparison meaningless. Pass `rng=` instead when the
caller owns the stream: a port migrating off a hand-written rule has to keep
drawing from *its* rng in *its* order, or every number it published moves.

## How a policy takes effect: serialised heads

Declaring a policy installs the [population layer](api.md#the-population-layer),
and that is the whole mechanism:

```python
evolve(tasks, reward, agent=agent,
       policies=Policies(selection=Beam(4)))    # installs PopulationAggregator
```

`PopulationAggregator` subclasses the shipped aggregator — staleness, conflict,
fusion, acceptance and promotion all run unchanged — and wraps three things
around it. It archives every distinct committed head with its held-out score. It
asks the policy which archived candidate the next batch should mutate. It
commits that candidate back to `dev`, so the next round's workers start from it.
`finalize` commits the archive's best scorer, so a run ends on its best
candidate rather than on whatever it was exploring when the budget ran out.

The heads are **serialised, not concurrent** — one at a time on one branch — so
the search is real but a wide beam does not run wide in wall-clock. Both drivers
get it from the same place: `Policies(selection=…)` reaches
`_build_engine`, and the layer is installed there.

`Policies(selection=…)` and `aggregator_factory=` are refused together. They
configure the same seat, and choosing one silently would leave a caller who
passed both with no way to read which one ran.

## What is deliberately not here yet

Multiple **live** heads. The ledger holds one `dev` branch, staleness is defined
as `η = max(head − base)`, and promotion compares `dev` against `stable`. The
population layer sidesteps that by taking turns rather than by making `head`
plural; making the ledger hold concurrent branches is separate work.

The refusal that remains is narrower and is about the *menu*: a policy chooses
among `SelectionContext.candidates`, and one that returns something else raises
`MultiHeadUnsupported`.

```
Beam.select() returned a candidate that is not in the archive it was given (4
entries). A selection policy chooses among the candidates in
SelectionContext.candidates; it cannot invent one, because a state that was
never a committed head has never been scored by the gate.
```

The type carries two bases on purpose. `NotImplementedError` is what callers
already catch. `ContractError` is how it gets out of the barrier-free loop's
merger thread instead of being absorbed there as a provider failure and retried
until the sweep budget runs out.

!!! note "`Beam(1)` is no longer the same run as `SingleHead`"
    It is still the same *answer* on the pool `SingleHead` sees — one candidate,
    and `tests/test_selection.py` pins that. But `Beam(1)` over an archive
    restarts from the best scorer, which differs from "continue from the head"
    the moment the head is not the best. That is beam search with width one, and
    it is what the policy always meant; before the population layer it had
    nowhere to show.

## Examples-level policies, and how they actually run

The MethodPolicy ports add two paper rules as ~15-line policies:

| Policy | Rule | Port |
|---|---|---|
| `BinaryTournament` | sample two candidates, breed the winner (unscored wins, Beam's optimism) | [PromptBreeder](algo-promptbreeder.md) |
| `SoftMixed` | `λ·uniform + (1−λ)·softmax(α·(s−s_max))` over top-k, seed always included | [AFlow](algo-aflow.md) |

These run on the same population layer as the shipped policies, and declaring
one is all it takes — the method runner no longer routes anything, because the
engine does it.

A port only reaches for `aggregator_factory=` when its rule is not expressible
as a `SelectionPolicy` at all. PromptBreeder's is the case: Algorithm 1's
tournament *evaluates* both sampled units and *replaces* the loser, and a policy
is handed candidates with cached scores and returns one. So
[`PromptBreederPopulation`](algo-promptbreeder.md) subclasses
`PopulationAggregator` and keeps `BinaryTournament` beside it as the declared,
equivalent policy — the two cannot disagree about who wins.

## Legacy-port policies

The mechanism-heavy ports express their parent rules at this seam. Two of them
used to keep a *local* class because the shipped policy could not express the
published rule; both said so on the class, in the same words — "close enough to
look right and wrong enough to change a measured run". Those two rules are now
shipped modes, so the difference a result carries is an argument rather than a
file a reader has to find:

| Port | Rule | Now |
|---|---|---|
| [GEPA](algo-gepa.md) | per-instance frontier, sampled by unique wins | `ParetoFrontier(mode="win_frequency")` |
| [DGM](algo-dgm.md) | `sigmoid(10·(s−0.5)) × 1/(1+children)` sampling | `Archive(sampling="sigmoid_novelty")` |
| [ADAS](algo-adas.md) | best of the keep-all archive | shipped `Beam(1)` (always was) |
| [EvoSkill](algo-evoskill.md) | best member of the bounded top-K frontier | `FrontierBest`, local |
| [OpenEvolve](algo-openevolve.md) | exploit best with probability ε, else uniform | `EpsilonGreedy`, local |

Migrating a rule must not move a number, and "must not" is checked rather than
intended: `tests/test_port_selection_equivalence.py` steps the shipped mode and
the rule it replaced through **one shared rng in lockstep**, 200 consecutive
draws, rather than comparing their distributions. A policy that drew the right
parent from the wrong stream offset would pass a distribution test and change
every seeded run.

ADAS is the reason `sigmoid_novelty_weights` is a function and not only a
sampling mode: ADAS uses the same weights for a different *draw* — up to five
archive entries without replacement, to condition the meta-agent, rather than
one parent. Shared formula, unshared draw. It had a byte-identical copy of the
formula until this landed.
