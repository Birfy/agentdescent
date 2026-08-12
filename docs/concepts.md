# Concepts

The ideas behind AgentDescent, in the order you need them. This is the *why*; for
the *what fits where* see [architecture.md](architecture.md), and for the *how
to run* see [install and first run](install.md).

---

## 1. The problem: RSI throughput

Existing recursive-self-improvement systems share one serial template:

```
while True:
    τ  = rollout(agent, task)      # run the task, collect a trajectory
    d  = propose(τ)                # reflect, propose an improvement diff
    ok = evaluate(agent + d)       # score the variant
    if ok: agent = merge(agent, d) # adopt
```

For agentic tasks, `rollout` and `evaluate` are expensive (tool calls, HPC
queues, eval suites). The improvement rate is capped at **one diff per
iteration**. AgentDescent's bet: the binding constraint is often *throughput*, not
proposal quality — so parallelize.

!!! note "This premise is a hypothesis, not consensus"
    Some RSI surveys argue the real bottleneck is verifier reliability and human
    direction, not parallel throughput. AgentDescent treats throughput as a
    *testable engineering assumption*: the win only materializes if (a) high-value
    diffs are genuinely being wasted in a serial queue, and (b) verifier
    reliability doesn't collapse first — which is what the L0 frozen layer and the
    AuditScheduler exist to protect.

---

## 2. The central analogy — and where it breaks

Model training solved the identical "single-GPU SGD is too slow" problem a
decade ago, not with a better single-step optimizer but with **parallelism +
asynchrony + a theory of the staleness / long-tail / scheduling problems that
creates**. AgentDescent transplants that theory:

| Model training | AgentDescent |
|---|---|
| parameter tensor θ | library of `Evolvable` artifacts |
| gradient *g* | `Diff` + `EvidenceCard` |
| parameter server | git-backed `Ledger` (version-vectored) |
| optimizer step | `Aggregator` merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | per-diff η + rebase re-verify |
| gradient clipping / trust region | diff-size cap + verifier audit |
| EMA | `stable`/`dev` dual branch |
| training code (immutable) | L0 frozen layer |

Every row above lands on code that runs by default. Three further rows do not,
and are kept separate rather than folded in — see
[Three rules borrowed from PPO and GRPO](#three-rules-borrowed-from-ppo-and-grpo).

The analogy **necessarily fails in one place**, and that failure *is* the
framework's core technical problem:

!!! danger "Gradients add. Diffs do not."
    You cannot average two diffs. Aggregation is **conflict resolution +
    statistical acceptance + transactional commit** — a discrete-space
    optimizer, not a vector sum.

### Three precision boundaries (don't over-claim the isomorphism)

- **Parameter server → Ledger.** A classic parameter server is mutable shared
  state with *no commit history*. AgentDescent's git-backed Ledger adds version
  vectors and history — and that "extra" is precisely what makes rebase /
  staleness handling possible. It's a feature, claimed as a *difference*, not an
  equivalence.
- **model soup → candidate fusion.** Model soups work because of linear mode
  connectivity (shared pretrain init). The diff analogue requires a **shared
  `base_version`**; cross-base fusion must rebase to a common head first.
- **EMA → dual branch.** The slow branch is an *exponential* confirmation of the
  fast branch (EMA), not a uniform trajectory average (SWA).

### Three rules borrowed from PPO and GRPO

!!! danger "Status: implemented, **not validated**. None is on by default."
    Each of these is an *option* until an A/B on a real dataset says otherwise.
    An analogy without an A/B is decoration, and a `PPOAggregator` that computed
    no importance ratio and had no policy distribution would make the table above
    stop being a map of the code — which is the only thing it is good for. So
    these live in [`agentdescent.advantage`](api.md), off by default, and a rule
    that fails its A/B should be **deleted** with the negative result recorded
    here.

There is no PPO here and no GRPO: no parameters, no log-probabilities, no policy
distribution, and therefore no importance ratio and no clipping objective. What
transfers is three *decision rules*.

| RL | here | before | status |
|---|---|---|---|
| group-relative advantage (GRPO) | a proposal's reward against its group's | only the zero-advantage *filter* — `DifficultyWeighted` drops all-pass/all-fail groups; the relative value never reached acceptance | signal recorded on every `EvidenceCard`; `AdvantageAcceptance` / `AdvantageConflict` opt-in |
| trust region / clipping (PPO) | op and character caps on one diff | constants `6` / `32_000`, chosen by guess | `AdaptiveTrustRegion` opt-in via `AggregatorConfig(trust_region_policy=)` |
| KL to a reference policy | distance from the `stable` branch | did not exist | `MergeContext.stable_distance` measured; `StableDistanceAcceptance` opt-in |

**A measured limit on the first one, before anyone tries to use it.** A group is
one base version and one task cluster, and the base version moves on every
commit — so a group is at most *one round of workers*, split across however many
clusters they landed in. With four workers over four clusters the largest
possible group is one, and the default `min_group=4` records nothing at all.

Clusters are the visible half. The half that bites the A/B is arithmetic, and it
applies even when every task is in one cluster — which is the case for the
HotpotQA workload `bench/ab_run.py` actually runs, since those tasks carry no
`cluster` at all. `GroupAdvantage` returns `None` until the group *reaches*
`min_group`, so the first `min_group - 1` rollouts of every round can never
carry a value however the rewards fall. At the harness's own default of four
workers that is three in four:

| `--workers` | rollouts per round that can carry an advantage | measured |
|---|---|---|
| 2 | 0 of 2 — the arm is the control | 0% |
| 4 | 1 of 4 | 25% |
| 8 | 5 of 8 | 62% |
| 16 | 13 of 16 | 81% |

The measured column is with rewards that have real spread. With flat rewards
inside a round it is 0% at every width — that is GRPO's zero-advantage filter
working, not a defect.

Neither limit is a bug to tune away: together they say where this signal can
exist — wide rounds, few clusters, a head that has stopped moving, and rewards
that disagree with each other. An A/B has to be run somewhere it does, so
`bench/ab_run.py` refuses `--rule advantage` below `min_group` and prints the
ceiling before it spends anything. `tests/test_advantage.py` and
`tests/test_ab_run.py` pin it.

Two design choices that are load-bearing:

* **`None` is not zero.** An unknown advantage and an advantage of exactly zero
  ("did as well as its peers") are different facts, and a consumer that conflates
  them puts noise into the gate wearing the costume of a measurement.
* **The stable-distance penalty refuses, it does not adjust the score.** Scores
  here feed a Beta posterior; silently lowering one corrupts every downstream
  number that reads it. A candidate far from `stable` clears a proportionally
  higher bar instead, and the refusal says so.

---

## 3. Staleness

The heart of the async story. A worker proposes a diff against the version it
*read*; by the time the aggregator sees it, head may have moved.

### 3.1 Per-diff η

```
η(d) = max over touched artifacts (head_version − base_version)
```

η is measured **per diff** (not a batch average), because a diff is a discrete
individual. η = 0 means "proposed against the current head".

### 3.2 The tolerance α

α is how much staleness the aggregator will try to salvage before giving up:

- α is larger for **hot** artifacts (they iterate fast, so lag is expected):
  `alpha_head=5` versus `alpha_tail=1`. The selector is `classify(artifact)`, so
  in practice *hot* means [L1](governance.md) — the boundary lives in exactly one
  place, and re-deriving it elsewhere is a bug the aggregator has already had.
- α is `0` for **contract-breaking** diffs (a cross-contract rebase costs more
  than re-proposing).

Full detail: [staleness policies](staleness.md).

### 3.3 Staleness policies (FlashEvolve: Full / Guarded / Reflective)

The policy maps `(η, α, contract_breaking)` to one of three actions. It is the
**only** thing that changes between async regimes — the merge pipeline is
untouched.

| Policy | Rule | Character |
|---|---|---|
| **Full** | accept regardless of η | max throughput, min safety |
| **Guarded** | η=0 → accept · η≤α → rebase · else → discard | AReaL bounded-staleness |
| **Reflective** | η=0 → accept · else → always rebase + re-verify | recovers wasted proposals |

**REBASE** = apply the diff to the current head and cheaply re-verify the
improvement still holds; keep it only if it does. This is the discrete analogue
of AReaL's *decoupled PPO objective*: "data sampled by an old policy, corrected
onto the current policy, then reused."

!!! tip "Why artifacts tolerate staleness better than parameters"
    A stale gradient is simply lost. A stale *diff* can be rebased and its
    evidence card re-verified. That structural difference is the framework's
    claimed advantage over parameter-space RL (design spec, RQ2), and the rebase
    half of it is what `evolve()` actually implements.

!!! warning "Settled evidence is retained, not yet reused"
    A discarded card is `settle()`d rather than dropped, and the design calls for
    re-filing it into the trajectory pool. **Nothing in the library reads the pool
    back** — so treat it as a diagnostic ring of recent rejections, not a queue
    that feeds later rounds.

    It is *bounded* (`SETTLED_MAX_CARDS=256`, `SETTLED_MAX_CHARS=2M`, oldest
    evicted), so a long run cannot accumulate the oversized diffs the trust region
    rejects.

### 3.4 async_ratio (ROLL Flash) — the global lag budget

In the async runtime, a worker refreshes its snapshot only once head has drifted
more than `async_ratio` versions ahead of it.

- **small ratio** → near-synchronous, few stale diffs, more sync overhead.
- **large ratio** → highly asynchronous, high throughput, many stale diffs the
  policy must handle.

A **backpressure** guard forces a global sync if the pipeline stalls (evidence
keeps arriving but nothing commits) — otherwise a mismatched `async_ratio > α`
would livelock under the Guarded policy: workers keep proposing against a
snapshot too old for the policy to accept, every card is discarded, head never
moves, and so the lag budget never triggers a refresh either. `stall_patience=`
on both async paths; `result.forced_refreshes` counts how often it fired.

Observed trade-off at `async_ratio=4` (all three converge to 1.000):

| policy | rollouts | stale discarded | wall-clock |
|---|---|---|---|
| Full | ~8k | 0 | ~3.2s |
| Reflective | ~7.8k | ~0.7k | ~3.3s |
| Guarded | ~20k | ~17k | ~5.1s |

The ratios between the rows are the result; the absolute counts scale with the
machine, since each run is bounded by wall-clock. Rerun
`python -m examples.run_async` rather than quoting these — the same caveat
[the efficiency page](efficiency.md) attaches to its own numbers.

---

## 4. The aggregator: a discrete-space optimizer

One "optimizer step" per artifact bucket. Cards are bucketed **by artifact**;
a bucket fires when it reaches batch size `B` or a `T_max` timeout (so cold
artifacts don't starve). Then, in order:

1. **Staleness filter** (§3) — split cards into survivors / discarded.
2. **Conflict resolution** — detect *semantic contradiction* (same key, different
   proposed value). Mere key overlap is not a conflict: two workers proposing the
   same value are duplicates, and collapsing them is what content-addressing is
   for. Contradictions are projected out PCGrad-style — keep the better of the
   pair on a shared verification subset — iterating until no surviving pair
   contradicts.
3. **Fusion** — complementary diffs are fused into one union candidate
   (model-soup analogy) and handed to the gate; ranking the union against the
   singles is opt-in (`fusion_tournament=True`, a per-workload diagnostic).
4. **Audit gate** — the candidate is submitted to the AuditScheduler, and a
   high-blast-radius or low-trust one is forced through the oracle, which can
   **veto it here** (`oracle-rejected`) before the acceptance test runs. **The
   optimizer audits itself** — as a blocking gate on the accept path, not a
   post-commit spot-check.
5. **Statistical acceptance** — commit only if `P(Δ > 0) > 1 − δ` under a Beta
   posterior comparison of candidate vs base, **not** a point threshold. `δ`
   anneals with version (LR decay); a trust-region caps diff size.
6. **Commit** — compare-and-swap on `dev`, one artifact per merge. (`Ledger`
   also implements `commit_atomic`, a 2PC across artifacts for a
   contract-breaking diff that must land with its adapters, but the reference
   aggregator buckets per artifact and no engine path uses it.)
7. **Dual-branch promotion** — `dev → stable` after *K* **regression-free
   rounds** on dev (EMA-style confirmation). A commit restarts the clock, so the
   artifact most likely to be promoted is the one that has *stopped* changing
   because nothing beats it. Production workers ride `stable`; explorers ride
   `dev`.

!!! info "Why a statistical test, not a threshold"
    A Beta posterior gives evidence-starved *tail* artifacts an automatically
    more conservative effective update — the discrete analogue of Adam's
    per-parameter adaptive step size. One noisy reflection can't drag a
    rarely-seen artifact off course.

---

## 5. The three long tails

"The long tail" is really three independent problems, each with its own
mechanism:

- **L-traj (system layer)** — heavy-tailed rollout durations. Handled today by
  the duration estimator + LPT dispatch and by **detecting** stragglers: a
  rollout that overruns its prediction is counted. Reachable from
  [`async_evolve`](async.md) as `duration_estimator=` → `result.stragglers`; it
  used to exist only in the reference runtime, which accepts nothing but the
  synthetic domain, so the mechanism was unreachable from the API a real workload
  uses. It is **not** a parameter of `evolve()` — not even with
  `asynchronous=True`, which forwards only the knobs it declares — so reach for
  `async_evolve` directly when you want it. The synchronous path bounds a slow
  rollout with `round_timeout=` instead, which abandons the straggler rather than
  measuring it.
  **The resume half is not implemented** — nothing pops that queue, and the
  recorded item carries no continuation state, because a resumable rollout would
  have to expose its turns and `run(rendered, task) -> output` is opaque. The
  design's "resume on the latest Ledger for a free cross-version A/B signal"
  therefore remains a design note, not behaviour. Removing the barrier
  ([async](evolution.md#the-barrier-free-runtime-async_evolve)) is what keeps one
  slow rollout from setting the pace in practice.
- **L-task (data layer)** — Zipfian artifact triggering (head skills flooded,
  tail skills starved — a problem parameter space doesn't have, since gradients
  flow to all parameters but diffs only to triggered artifacts). Handled by
  **UCB over task clusters** and a difficulty filter (GRPO zero-advantage
  groups) — both implemented, the latter also reachable from
  `evolve()` as [`DifficultyWeighted`](sampling.md)
  task sampling. The design's cross-product **(task-cluster × artifact)** is
  **not implemented**: `TaskCluster` has no artifact dimension, and both reference
  runtimes register exactly one artifact, as does `evolve()` — so the second axis
  has nowhere to live yet, and the mechanism operates on clusters while the
  problem statement is about artifacts. The design also calls for a **tail canary
  set** inside held-out eval; that is **not implemented** — held-out is one
  undifferentiated split.
- **L-value (signal layer)** — most diffs are marginal, a few are high-value
  refactors. The **AuditScheduler** spends the scarce oracle budget by
  `blast_radius × uncertainty / trust`, where *trust* is how often the cheap layer
  has agreed with the full held-out set. That agreement is measured on **every**
  merge and costs nothing (the acceptance test already scores both on the full
  set), which is what makes the gate reachable: trust used to be written only
  *inside* the branch it gates, so an artifact below `blast_radius 0.5` could
  never earn an audit — measured at the default 0.2, `oracle_calls_used` was 0 for
  a whole run.

    !!! warning "The priority queue has no consumer — and is off by default"
        `AuditScheduler.submit` computes `Ĝ` for every merge, but with the default
        `collect=False` it does not queue anything: nothing in the engines would
        pop that heap, so building it would be work done on the merge path for a
        reader who never arrives. Pass `collect=True` to keep it for an
        out-of-band audit process. The audit that actually runs either way is the
        inline `force_oracle` gate in the merge pipeline, which reads trust but
        not the queue. Treat the ranking as a priority *model* — the ordering a
        fuller system would spend a background budget against — not as work in
        flight.

---

## 6. Governance: blast radius decides parallelism

Every artifact is sorted — *automatically, by its `blast_radius`* — into a layer
that decides its aggregation protocol and update cadence (a two-timescale
system):

| Layer | Content | Cadence | Protocol |
|---|---|---|---|
| **L2 fast** | local skills, prompts, few-shot | hours | full async merge (the whole §4 pipeline) |
| **L1 slow** | harness, context policy, tool routing, learned verifier | days/weeks | serialized eval + staged rollout |
| **L0 frozen** | oracle interface, audit budget, merge permissions, safety | humans only | read-only to the loop |

- **L2** is naturally isolated — only tasks that trigger an artifact are affected
  — so it merges fully async.
- **L1** has no such isolation, so *at most one L1 diff is in evaluation at a
  time*. That holds today **by construction, not by the gate**: every merge
  decision runs on one thread — the round barrier in `evolve()`, the single merger
  in `async_evolve` and `AsyncAgentDescent` — so at most one diff of any layer is
  ever in evaluation. `L1SerialGate` is the primitive that would enforce it once
  merges run concurrently (a process or host pool); it is tested in isolation and
  is not in the path. The design's staged rollout beyond that (offline
  counterfactual replay → canary → full) is **not implemented**.
- The L1/L2 boundary is `FAST_MAX = 0.30`, and `governance.classify` is the only
  place it is defined. It used to be re-derived twice more from raw floats — the
  aggregator's staleness tolerance at `> 0.5` and the audit gate at `>= 0.5` — so
  an artifact at 0.4 was L1 by governance and treated as L2 by both mechanisms
  that decide what being L1 *means*.
- **L0** must be frozen: without it, the self-referential loop eventually
  pollutes itself (a verifier that learns to pass itself is undetectable).
  AgentDescent minimizes the frozen set to "audit + permissions", leaving the
  verifier's learnable part in L1 under audit. Unlike L1/L2, **L0 is reached by
  name, not by blast radius** — no measurement can tell you an artifact *is* the
  oracle, and an estimated layer is exactly what would fail to catch it. The
  reserved names are ordinary words (`oracle`, `audit_budget`,
  `merge_permissions`, `safety_constraints`), so `evolve(artifact_id="oracle")` is
  refused up front with a message saying to rename it.

---

## 7. Parallel paradigms (DP / TP / PP)

All three ride the same async runtime and aggregator; they differ only in how
work is partitioned and recombined.

- **DP (data parallel)** — same snapshot, tasks sharded across workers, diffs
  merged. The default.
- **TP (tensor parallel)** — split one *hot* artifact along its internal
  structure into disjoint sections; each worker owns a section, so edits are
  conflict-free **by construction** and the merge degrades to concatenation plus
  a lightweight consistency reviewer (the all-reduce analogue). For a few
  super-hot artifacts only. The sections partition the **artifact's** key space,
  which is a different thing from the tasks a worker rolls out — conflating the
  two is what made TP silently discard most of its proposals; `route=` maps one
  onto the other. See [Parallelism](parallelism.md).
- **PP (pipeline parallel)** — artifacts form a dependency chain
  (`lit-review → mol-engine → hpc-submit`); each stage has its own worker, and a
  downstream failure back-propagates blame to the earliest failing upstream
  stage (shared with the counterfactual-replay attribution). **Not an `evolve()`
  mode**: the engine evolves one artifact and PP needs one per stage, so
  `evolve(parallel=PipelineParallel(...))` raises. `PipelineChain` provides the
  stage ordering and blame attribution as a standalone primitive.

## 8. A directory as a parameter

The analogy has one more consequence that is easy to miss. The aggregator never
interprets a state key — it only asks whether two diffs touch the same one. So
the **choice of key space is the choice of what can be improved in parallel**:

| keys are | two workers collide when | parallelism |
|---|---|---|
| a content hash (`AppendRules`) | they learn the identical lesson | maximal — almost everything fuses |
| one slot (`SingleSlot`) | always | none — every round is a tournament |
| named categories (`KeyedRules`) | they edit the same category | bounded by category count |
| **file paths** (`FileTree`) | they edit the same file | bounded by file count |

The last row is what makes a *directory* an ordinary parameter: a skill folder,
a folder of subagent definitions, or a small codebase. Nothing in the optimizer
changes; the artifact is simply materialised onto disk for each rollout so a real
agent can read it with its own tools. See
[Directory evolution](directory-evolution.md), and
[Strategies](strategies.md#the-key-space-is-the-design-decision) for how to pick.

It also puts a hole in governance in plain sight. `FROZEN_IDS` freezes an
*artifact*; it cannot say "this skill may evolve, but not its test suite". For a
directory that distinction is the difference between a measured improvement and a
self-graded one, which is why frozen **paths** exist and are enforced both at
proposal time and at run time — see
[Governance](governance.md#freezing-paths-not-just-artifacts).
