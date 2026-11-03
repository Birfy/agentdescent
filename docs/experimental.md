# Experimental modules — status, not a promise

The package contains more modules than the CLI/MCP product path uses. Some are
research prototypes, some have recently been productized, and one is a
contributor template. This page marks them so a reader does not mistake
"importable" for "supported".

The authoritative list is the output of the read-only audit:

```bash
python -m tools.module_audit
python -m tools.module_audit --json
```

`isolated` in that report means "not statically referenced from the CLI/MCP path
or the package `__init__`". It does **not** mean "dead": a module may be loaded by
a string ref, an entry point or a copied template. The audit's
[limitations](module-audit.md#limitations) are deliberate.

## Research prototypes

These are usable from Python and covered by tests/examples, but they are not on
the product path. Treat their APIs as unstable.

| module | what it explores | status |
|---|---|---|
| `agentdescent.society_runtime` | proposer/critic/verifier/historian blackboard runtime | research |
| `agentdescent.openended` | POET-style task/solver co-evolution on a novelty/difficulty frontier | research |
| `agentdescent.scaling` | score-vs-compute saturation and budget back-solving | research / bench |
| `agentdescent.calibrated_deferral` + `agentdescent.deferral` | learned auto-approve / defer / deny thresholds | research pair; merge candidate |

## Contributor template

`agentdescent.algorithms._TEMPLATE.py` is a copy source for a new algorithm
card in `docs/algorithm-authoring.md`; it is not imported at runtime and is not
a candidate for deletion.

## Recently productized

These are no longer isolated; keep them out of the "unwired" list.

| module | consumer |
|---|---|
| `agentdescent.adaptive_experience` | `experience cycle` / MCP `experience_cycle` (#711) |
| `agentdescent.budgeted_loop` | `platform` `step.budgeted` (#712) |
| `agentdescent.tenant_platform` + `agentdescent.workspaces` | `tenant` one quota-checked platform cycle (#715) |

## What this means

- Do not add an experimental module to `agentdescent/__init__` just to make it
  importable; it is already importable by path.
- Prefer the product path in `docs/index.md` and the CLI/MCP verbs for supported
  work.
- Before quarantining or deleting an `isolated` module, check its tests, docs,
  string refs and entry points; the audit is a map, not a verdict.

The cleanup order is: measure (module audit) → document status (this page) →
move genuinely experimental code into a namespace with compatibility shims →
merge duplicate layers. Each step is a separate, reversible PR.
