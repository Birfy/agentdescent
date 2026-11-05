# The main path — which modules are the product, which are advanced

The package has more modules than the product uses. This page is the map a new
contributor or integrator should read before adding another one. It is a
**design contract**, not an API list: three paths are supported, a few layers
are advanced, and everything else is experimental or a compatibility seam.

Use `python -m tools.module_audit` to see the current static picture. The audit
classifies modules as `cli/mcp`, `exported-only` or `isolated`: `cli/mcp` is the
product path, `exported-only` is supported Python API without a product surface,
and `isolated` is a candidate for productization, an experimental namespace, or
removal.

## Path A — evolve one artifact

```text
tasks + reward + agent + Strategy
    -> evolve() / async_evolve()
    -> Policies (sampler/selection/proposal/conflict/fusion/acceptance/staleness)
    -> verifier / ledger / runstore
```

| concern | canonical modules |
|---|---|
| artifact | `strategies`, `treestrategy`, `filetree`, `evolvable` |
| loop | `evolution`, `async_evolve`, `aggregator`, `policies`, `sampling`, `selection`, `fusion`, `staleness`, `governance` |
| execution | `agents`, `runners`, `executor`, `parallel`, `scheduler`, `sandbox`, `sandbox_container`, `sandbox_shared` |
| evidence | `verifier`, `verifier_ensemble`, `audit/`, `stats`, `stats_eval`, `conformal`, `sequential` |
| state | `ledger`, `runstore`, `checkpoint`, `manifest`, `trajectory`, `decisions` |

This is the path every CLI/MCP `plan/evolve/status/show/apply` verb uses.

## Path B — governed release

```text
paired scores -> TrustEvidence -> PromotionDecision -> signed ReleaseRecord
```

| step | canonical module |
|---|---|
| build evidence | `trust_evidence.build_trust_evidence` |
| decide | `promotion.promote` + `PromotionRules` |
| sign + lineage | `promotion_ledger.PromotionLedger.record` |
| one-call surface | `release_pipeline.release_candidate/release_evidence` |
| admission (optional) | `policy_release` (signed manifest + admission, not a second decision engine) |

`agentdescent release` and the MCP `release` tool are the product surface.
`agentdescent release verify` (and MCP `release_verify`) checks the signed
receipt independently: it recomputes the signature over the embedded manifest
and can pin the expected key id and lineage node. Do not hand-wire the three
steps in application code; use `release_pipeline` or the CLI.

## Path C — continuous platform

```text
LoopKernel (durable) -> RSIPlatform (five layers) -> PlatformCampaign (multi-cycle)
```

| concern | canonical module |
|---|---|
| durable loop | `loop_kernel` + `durable_runs` |
| one cycle | `platform.RSIPlatform` |
| multi-cycle | `platform_campaign.PlatformCampaign` |
| stop policy | `rsi_control.RSIController` |
| compute | `compute_policy` + `model_router` |
| experience | `experience.ExperienceEngine` |
| product surface | `agentdescent platform` / MCP `platform` |

`run_guarded` (`runtime`) is one guarded session, not a second platform. Use it
when you need a single session with acceptance/decision logging; use
`RSIPlatform` when the five layers must hand off to each other.

`agentdescent platform status` (and MCP `platform_status`) reads a finished
campaign's run database without the compute, foundry or experience callbacks;
it only inspects the durable run store.

## Advanced layers (supported, narrower)

| layer | canonical modules | surface |
|---|---|---|
| capability growth | `task_pipeline`, `capability_scout`, `capability_foundry`, `capability_archive`, `capability_loop` | `agentdescent capability` |
| tenancy | `workspaces`, `tenant_platform` | `agentdescent tenant` (`plan`, `run`) |
| operations | `ops_fabric` (write), `ops_console` (read), `rsi_metrics`, `rsi_telemetry`, `campaign_ops` | `agentdescent ops` (`plan`, `publish`, `history`) |
| meta-learning | `meta` (including `continuous_meta_run`), `meta_curriculum`, `meta_population`, `meta_campaign`, `meta_audit`, `meta_exam`, `verified_seed` | `agentdescent meta-campaign`, `exam`, `audit` |
| governed harness | `safe_selfmod`, `proofs`, `approval`, `safety_controller`, `governed_harness` | `agentdescent governed-harness` (`plan`, `run`, `approvals`, `approve`, `reject`) |

These are not “extra algorithms”; they are the control, evidence and memory
layers around the three paths.

## One concept, one owner

The package grew by adding a module per capability. That left several near
duplicates. The canonical/alias split is:

| concept | canonical | what it is not |
|---|---|---|
| session stop | `rsi_control.RSIController` | `controller.run_controller` is a generic wrapper; do not add a third stop rule |
| durable run | `durable_runs` + `loop_kernel` | `runtime.run_guarded` is one session; `campaign` is spec sessions |
| promotion decision | `promotion.promote` | `policy_release` is admission, not a second decision |
| signed release | `promotion_ledger` | `manifest` is a file-input manifest, not a release receipt |
| ops metrics | `rsi_metrics` | `metrics` is the per-run meter |
| traces | `rsi_telemetry` | `ops_console` is the read model, not the writer |
| experience | `experience.ExperienceEngine` | `adaptive_experience` is a surface, not a second learner |
| platform campaign | `platform_campaign` | `campaign` is spec-file sessions; `meta_campaign` is meta sessions |

When two modules disagree, the canonical one wins. Until the alias PRs land this
is documentation/behavior intent, not a runtime switch: both code paths still
exist, so new code should follow the canonical column.

## Rules for the next module

1. Name the product consumer before writing the module: a CLI/MCP verb, an
   existing canonical module, or a documented advanced path.
2. Reuse `surface_common`, `trust_evidence`, `promotion`, `experience` and
   `loop_kernel`; do not reimplement evidence, stops, quotas or replay.
3. Every new command returns `schema_version: 1`, fails closed with
   `{"ok": false, "error": ...}` under `--json`, and confines MCP refs/writes
   with `AGENTDESCENT_SURFACE_ALLOW` / `AGENTDESCENT_SURFACE_WRITE_DIRS`.
4. If it is research, do not export it from `agentdescent/__init__`; describe it
   as experimental in `modules.md`. The `agentdescent/experimental/` namespace is
   the next step for it once that move lands.
5. If it duplicates an existing concept, extend the canonical module or add a
   documented alias; do not ship a second engine.

## Related pages

- [Module audit](module-audit.md) — the current static reachability report.
- The experimental status page (once it lands) — research and recently
  productized modules.
- [Architecture](architecture.md) — the five layers and their data flow.
- [Module map](modules.md) — the catalog.
