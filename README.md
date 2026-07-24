# Concordia

**A parallel, self-evolving framework for accelerating recursive self-improvement (RSI).**

Concordia is a reference implementation of the design in
[`docs/concordia_design.md`](docs/concordia_design.md). It ports the
parallel-training playbook — data/tensor/pipeline parallelism, parameter
servers, decoupled/asynchronous RL, partial rollout — onto RSI, where the
"parameters" are a **library of evolvable artifacts** (skills, prompts, harness
modules, verifiers) and the "gradients" are **diffs carrying evidence cards**.

The core observation (design §1): serial RSI is bounded at **1 diff / T_iter**.
Concordia runs *N* workers in parallel and merges their diffs into a shared,
versioned artifact library, targeting **O(N / T_iter)** improvement throughput.

> The one place the analogy *must* break defines the whole system: **gradients
> add, diffs do not.** Aggregation is therefore not averaging but
> **conflict resolution + statistical acceptance + transactional commit**.

## The central analogy

| Model training | Concordia (parallel RSI) |
|---|---|
| parameter tensor θ | library of `Evolvable` artifacts |
| gradient *g* | `Diff` + `EvidenceCard` |
| parameter server | git-backed, version-vectored `Ledger` |
| optimizer step | `Aggregator` merge decision |
| per-param adaptive LR (Adam) | per-artifact Beta-posterior test |
| staleness / decoupled PPO | per-diff η + rebase re-verify |
| partial rollout | turn-level checkpoint / `ResumeQueue` |
| EMA (weight averaging) | stable/dev dual branch |
| training code (not self-modifiable) | L0 frozen layer |

## Install & run

```bash
pip install -e ".[dev]"

# RQ1 — merge vs fork, end to end
python -m examples.run_demo

# RQ2 — staleness tolerance sweep (alpha in {0,1,5,inf})
python -m examples.rq2_staleness

# tests
pytest
```

No external services or model APIs are required: the reference domain
([`concordia/domains/router.py`](concordia/domains/router.py)) is a fully
deterministic keyword-router skill, so the entire parallel loop runs in-process
and is unit-tested — while still producing genuine diffs that measurably improve
a held-out metric.

## Architecture → code map

Every module cites the design section it implements.

| Component | Module | Design § |
|---|---|---|
| `Evolvable` unit, `Diff`, `EvidenceCard`, version vectors | [`evolvable.py`](concordia/evolvable.py) | 3.2, 3.3 |
| Git-backed Ledger: version vectors, CAS, 2PC, dual branch | [`ledger.py`](concordia/ledger.py) | 3.1, 4.5 |
| Aggregator: staleness → conflict → fusion → Beta accept → commit | [`aggregator.py`](concordia/aggregator.py) | 4 |
| Statistics: Beta posterior, `P(Δ>0)`, annealed δ, UCB | [`stats.py`](concordia/stats.py) | 4.4, 5.2 |
| Three schedulers: UCB task / audit / resume queue | [`scheduler.py`](concordia/scheduler.py) | 5 |
| Three-layer verifier (rule / learned / oracle) | [`verifier.py`](concordia/verifier.py) | 3.1, 5.3 |
| Layered governance by blast radius (L0/L1/L2) | [`governance.py`](concordia/governance.py) | 6 |
| Worker: rollout + propose | [`worker.py`](concordia/worker.py) | 3.1 |
| Orchestrator + fork baseline | [`orchestrator.py`](concordia/orchestrator.py) | 3.1, RQ1 |

## How aggregation works (the `Aggregator` pipeline)

Cards are bucketed **by artifact**. When a bucket triggers (batch size `B`, or a
`T_max` timeout so cold artifacts don't starve), the aggregator runs one
optimizer step:

1. **Staleness filter (§4.2)** — per-diff `η = max(head − base)` over touched
   artifacts. `η = 0` proceeds; `0 < η ≤ α` is **rebased and cheaply
   re-verified** (does the delta still hold on the new head?); `η > α` is
   discarded and its evidence *settled back* into the pool. `α` adapts to
   artifact heat; contract-breaking diffs force `α = 0`.
2. **Conflict resolution (§4.3)** — syntactic (hunk overlap) and semantic
   (contradictory ops) detection; contradictions are projected out PCGrad-style,
   keeping the better of the pair on a shared subset.
3. **Fusion tournament (§4.3)** — complementary diffs are fused (model-soup
   analogy) and run against the individual candidates on held-out data.
4. **Statistical acceptance (§4.4)** — commit only if
   `P(Δ > 0) > 1 − δ` under a per-artifact Beta posterior, not a point threshold.
   `δ` anneals with version (LR decay); a trust-region caps diff size.
5. **Commit (§4.1)** — compare-and-swap on `dev` (2PC for contract-breaking
   multi-artifact diffs).
6. **Dual-branch promotion (§4.5)** — `dev → stable` after *K* regression-free
   rounds (EMA-style confirmation).
7. **Audit (§5.3)** — the merge decision is itself submitted to the
   `AuditScheduler`; high-blast-radius / low-trust merges are forced through the
   oracle. The optimizer audits itself.

## The three long tails (§5)

Concordia treats "the long tail" as three separate problems:

- **L-traj** (system): heavy-tailed rollout durations → turn-level checkpoint +
  `ResumeQueue`, resumed against the latest ledger (a free cross-version A/B
  signal).
- **L-task** (data): Zipfian artifact triggering → **UCB over
  (cluster × artifact)** so starved tail artifacts get an exploration bonus,
  plus a difficulty filter and a tail canary set.
- **L-value** (signal): most diffs are marginal → `AuditScheduler` spends the
  scarce oracle budget on `blast_radius × uncertainty / trust`.

## Governance (§6)

Artifacts sort into layers automatically by `blast_radius`:

- **L2 fast** — local skills/prompts → full async merge.
- **L1 slow** — harness/verifier → serialized in-flight changes + staged rollout.
- **L0 frozen** — oracle, audit budget, merge permissions, safety constraints →
  read-only to the loop. Without a frozen layer, the self-referential loop
  eventually pollutes itself (a verifier that learns to pass itself).

## Scope & honesty

This is a **research reference implementation**, not a production system. It is
faithful to the design's *mechanisms* and runs end-to-end on a synthetic domain
so the mechanisms are observable and testable. The design doc (§2) is explicit
that Concordia's novelty is a **narrow, defensible engineering synthesis** —
concurrent, staleness-bounded, conflict-resolved **diff-level merge** over a
git-backed versioned ledger — and that its throughput premise is a *testable
engineering hypothesis*, not community consensus. See `docs/concordia_design.md`
for the full related-work discussion (FlashEvolve / SkillClaw / CoEvoSkills) and
the RQ1–RQ3 experiment plan.
