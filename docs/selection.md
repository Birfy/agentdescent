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
| `Beam(k)` | classic beam search | `Beam(1)` computes `SingleHead`'s answer by another route, and the tests assert they agree |
| `ParetoFrontier(mode=...)` | GEPA / EvoSkill | `per_instance` and `topk_aggregate` — the fidelity difference as a parameter |
| `Archive(sampling=...)` | DGM / ADAS | `performance`, `novelty` (performance ÷ `1 + selected`), `uniform` as the ablation |
| `MCTS(exploration=...)` | tree search | UCT over the candidate tree; one evolve step is one rollout, value is held-out reward, backup runs up `Candidate.parent` |

Three details that are decisions rather than defaults:

**An unscored candidate sorts first, not last.** `Candidate.score is None` means
*unmeasured*. Ranking it as the worst is how a beam collapses onto a single line
of descent and stops being a beam, and how an archive stops exploring.

**`per_instance` Pareto refuses to fall back.** Given candidates with no
`per_task` scores it raises rather than quietly ranking on the aggregate — which
would be running *EvoSkill's* rule and reporting it under GEPA's name.

**`Archive` is deterministic given its seed.** An archive that samples differently
on a re-run makes a seeded comparison meaningless.

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

The mechanism-heavy ports express their parent rules as policy classes at this
seam (local where the upstream rule differs from a shipped policy — the
difference is always documented on the class):

| Policy | Rule | Port |
|---|---|---|
| `DGMParentSelection` | `sigmoid(10·(s−0.5)) × 1/(1+children)` sampling | [DGM](algo-dgm.md) |
| `ParetoWinFrequency` | per-instance Pareto frontier, sampled by unique wins | [GEPA](algo-gepa.md) |
| `FrontierBest` | best member of the bounded top-K frontier | [EvoSkill](algo-evoskill.md) |
| shipped `Beam(1)` | best of the keep-all archive (exact match, no subclass) | [ADAS](algo-adas.md) |
| `EpsilonGreedy` | exploit best with probability ε, else uniform | [OpenEvolve](algo-openevolve.md) |
