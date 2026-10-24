# The meta stack on the command line

The self-evolution machinery is operator-facing through three verbs, in the
same words the MCP server exposes:

```
agentdescent portfolio [archive]                          rank layers, reallocate budget
agentdescent exam     [archive] --domain module:builder    re-judge adoptions with an oracle
agentdescent audit    <run_id> --archive PATH              trace an adoption's evidence rings
```

`archive` defaults to `$AGENTDESCENT_HOME/archive` (or `~/.agentdescent/archive`).
Every command accepts `--json` for machine-readable output. An `archive` that
does not exist is an error (`error: archive '<path>' does not exist`, exit `2`),
never silently created, so a typo'd path cannot pass for an empty archive.

## `portfolio` — rank the layers, reallocate the next window

Reads the consolidation archive, builds one `LayerBudget` per decision slot
(`layers_from_archive`), runs `portfolio_report`, and prints each layer's
return-per-cost, its reallocated budget, and why:

```bash
agentdescent portfolio ~/.agentdescent/archive
# meta:task_sampler: return-per-cost 0.077 (return 0.850 / spend 11.0, 2 measurements) -> budget 11.0
#     held: 2/3 measurements, not enough to move budget on
```

An existing but empty archive exits `1` with a pointer to run a meta session first — the
portfolio needs rule records to rank.

## `exam` — re-judge adoptions with an oracle

`--domain module:builder` resolves a `module:attribute` ref whose zero-arg
builder returns the domain: `oracle`, `tasks_for`, `run`, `reward`, `propose`,
`strategy`, and optionally `spec` (the SlotSpec the rules were written by —
omitted, it falls back to the shipped spec for the `--slot`):

```bash
agentdescent exam ~/.agentdescent/archive --domain mypkg.domain:build --slot task_sampler
# 2 rule(s) examined: 1 passed, 1 flagged, 0 demoted
#   run-1: passed  (oracle gain +0.100 at or above min_gain +0.000)
#   run-2: flagged  (oracle gain -0.050 below min_gain +0.000 (flag 1/2))
```

## `audit` — trace one adoption through its evidence rings

`chain_audit` walked per ring — governance, decision, meta-gain, self-exam,
inner-reward, harness — with the optional stores you hand it:

```bash
agentdescent audit run-1 --archive ~/.agentdescent/archive \
    --meta-audit ~/.agentdescent/audit.jsonl --inner-audit runs/audit.jsonl
# run run-1: NOT VERIFIED
#   rings not verified: decision, meta-gain, self-exam, inner-reward, harness
#   governance: verified  slot 'task_sampler' is decision-plane (L1 harness)
#   ...
```

An unknown run, a `--meta-audit` / `--inner-audit` file that does not exist, or
a bad `--domain` / `--harness` ref fails with a one-line error
and a non-zero exit, never a traceback.
