# Concepts

The ideas behind Concordia, in the order you need them. This is the *why*; for
the *what fits where* see [architecture.md](architecture.md), and for the *how
to run* see [usage.md](usage.md).

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
iteration**. Concordia's bet: the binding constraint is often *throughput*, not
proposal quality — so parallelize.

!!! note "This premise is a hypothesis, not consensus"
    Some RSI surveys argue the real bottleneck is verifier reliability and human
    direction, not parallel throughput. Concordia treats throughput as a
    *testable engineering assumption*: the win only materializes if (a) high-value
    diffs are genuinely being wasted in a serial queue, and (b) verifier
    reliability doesn't collapse first — which is what the L0 frozen layer and the
    AuditScheduler exist to protect.

---

## 2. The central analogy — and where it breaks

Model training solved the identical "single-GPU SGD is too slow" problem a
decade ago, not with a better single-step optimizer but with **parallelism +
asynchrony + a theory of the staleness / long-tail / scheduling problems that
creates**. Concordia transplants that theory:

| Model training | Concordia |
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

The analogy **necessarily fails in one place**, and that failure *is* the
framework's core technical problem:

!!! danger "Gradients add. Diffs do not."
    You cannot average two diffs. Aggregation is **conflict resolution +
    statistical acceptance + transactional commit** — a discrete-space
    optimizer, not a vector sum.

### Three precision boundaries (don't over-claim the isomorphism)

- **Parameter server → Ledger.** A classic parameter server is mutable shared
  state with *no commit history*. Concordia's git-backed Ledger adds version
  vectors and history — and that "extra" is precisely what makes rebase /
  staleness handling possible. It's a feature, claimed as a *difference*, not an
  equivalence.
- **model soup → candidate fusion.** Model soups work because of linear mode
  connectivity (shared pretrain init). The diff analogue requires a **shared
  `base_version`**; cross-base fusion must rebase to a common head first.
- **EMA → dual branch.** The slow branch is an *exponential* confirmation of the
  fast branch (EMA), not a uniform trajectory average (SWA).

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

- α is larger for **hot** artifacts (they iterate fast, so lag is expected).
- α is `0` for **contract-breaking** diffs (a cross-contract rebase costs more
  than re-proposing).

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
    evidence card re-verified — and if discarded, the evidence settles back into
    the pool for reuse. That structural difference is the framework's claimed
    advantage over parameter-space RL (design spec, RQ2).

### 3.4 async_ratio (ROLL Flash) — the global lag budget

In the async runtime, a worker refreshes its snapshot only once head has drifted
more than `async_ratio` versions ahead of it.

- **small ratio** → near-synchronous, few stale diffs, more sync overhead.
- **large ratio** → highly asynchronous, high throughput, many stale diffs the
  policy must handle.

A **backpressure** guard forces a global sync if the pipeline stalls (evidence
keeps arriving but nothing commits) — otherwise a mismatched `async_ratio > α`
would livelock under the Guarded policy.

Observed trade-off at `async_ratio=4` (all three converge to 1.000):

| policy | rollouts | stale discarded | wall-clock |
|---|---|---|---|
| Full | ~8k | 0 | ~3.2s |
| Reflective | ~7.8k | ~0.7k | ~3.3s |
| Guarded | ~20k | ~17k | ~5.1s |

---

## 4. The aggregator: a discrete-space optimizer

One "optimizer step" per artifact bucket. Cards are bucketed **by artifact**;
a bucket fires when it reaches batch size `B` or a `T_max` timeout (so cold
artifacts don't starve). Then, in order:

1. **Staleness filter** (§3) — split cards into survivors / discarded.
2. **Conflict resolution** — syntactic (hunk overlap) and semantic
   (contradictory edits) detection. Contradictions are projected out
   PCGrad-style: keep the better of the pair on a shared verification subset.
3. **Fusion tournament** — complementary diffs are fused (model-soup analogy)
   and run against the individual candidates on held-out data; the best wins.
4. **Statistical acceptance** — commit only if `P(Δ > 0) > 1 − δ` under a Beta
   posterior comparison of candidate vs base, **not** a point threshold. `δ`
   anneals with version (LR decay); a trust-region caps diff size.
5. **Commit** — compare-and-swap on `dev` (2PC for contract-breaking
   multi-artifact diffs).
6. **Dual-branch promotion** — `dev → stable` after *K* regression-free rounds
   (EMA-style confirmation). Production workers ride `stable`; explorers ride
   `dev`.
7. **Audit** — the merge decision is itself submitted to the AuditScheduler;
   high-blast-radius or low-trust merges are forced through the oracle. **The
   optimizer audits itself.**

!!! info "Why a statistical test, not a threshold"
    A Beta posterior gives evidence-starved *tail* artifacts an automatically
    more conservative effective update — the discrete analogue of Adam's
    per-parameter adaptive step size. One noisy reflection can't drag a
    rarely-seen artifact off course.

---

## 5. The three long tails

"The long tail" is really three independent problems, each with its own
mechanism:

- **L-traj (system layer)** — heavy-tailed rollout durations. Turn-level
  checkpoint + `ResumeQueue`, resumed against the *latest* Ledger. The resumed
  trajectory (first half on V_k, second half on V_{k+1}) is a **free A/B
  signal** across versions.
- **L-task (data layer)** — Zipfian artifact triggering (head skills flooded,
  tail skills starved — a problem parameter space doesn't have, since gradients
  flow to all parameters but diffs only to triggered artifacts). Handled by
  **UCB over (task-cluster × artifact)**, a difficulty filter (GRPO
  zero-advantage groups), and a **tail canary set** in held-out eval.
- **L-value (signal layer)** — most diffs are marginal, a few are high-value
  refactors. The **AuditScheduler** spends the scarce oracle budget by
  `blast_radius × uncertainty / trust`.

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
  time* (the serial gate), plus offline counterfactual replay → canary → full.
- **L0** must be frozen: without it, the self-referential loop eventually
  pollutes itself (a verifier that learns to pass itself is undetectable).
  Concordia minimizes the frozen set to "audit + permissions", leaving the
  verifier's learnable part in L1 under audit.

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
  super-hot artifacts only.
- **PP (pipeline parallel)** — artifacts form a dependency chain
  (`lit-review → mol-engine → hpc-submit`); each stage has its own worker, and a
  downstream failure back-propagates blame to the earliest failing upstream
  stage (shared with the counterfactual-replay attribution).
