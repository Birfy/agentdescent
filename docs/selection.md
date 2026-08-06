# Candidate selection — where the next batch starts

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

## What is deliberately not here yet

Multiple **live** heads. The ledger holds one `dev` branch, staleness is defined
as `η = max(head − base)`, and promotion compares `dev` against `stable` — all
three assume "head" names one thing. So a policy that returns a starting point
other than the head raises `NotImplementedError`:

```
Beam.select() asked to start from a candidate other than the current head, and
the ledger holds one live branch: `dev`, with staleness defined as
eta = max(head - base). Multi-head support is a separate change; until then a
selection policy may only return the head it was given.
```

It raises rather than being collapsed to the head, because a caller who passes a
beam and watches a run finish has every reason to believe the beam ran — the same
rule [`Policies.require_supported`](api.md) enforces for the bundle as a whole.

Every policy above is therefore usable *today* in the shape a run actually
starts in: an archive of one, a beam over one candidate. That is what makes the
seam checkable now instead of after the ledger changes. Making the ledger hold
concurrent branches, and redefining `η` when `head` is plural, is separate work.
