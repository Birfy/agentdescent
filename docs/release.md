# Release a candidate — evidence, decision, signed receipt

`agentdescent release` is the product seam for the trusted-release stack. It
turns paired held-out scores into the same evidence `evolve` can produce, runs
the fail-closed promotion decision, and signs an accepted decision into an
`ArtifactManifest` with a lineage node.

It does **not** build, deploy or merge anything. It answers one question:
*does this candidate have enough independent evidence to be released, and can
we produce a receipt that says exactly what was accepted and why?*

```text
paired scores -> trust evidence -> promotion decision -> signed manifest + lineage
```

## Quickstart

```bash
export AGENTDESCENT_RELEASE_KEY="$(openssl rand -hex 32)"

agentdescent release \
  --before  evidence/before.json \
  --after   evidence/after.json \
  --artifact-id    skill-csv-total \
  --artifact-hash  sha256:41d8... \
  --key-id         ci-release \
  --lineage        evidence/lineage.json \
  --clean --risk 0.05 \
  --min-effect 0.01
```

`before.json` and `after.json` are one object each, keyed by the same held-out
task ids:

```json
{"t1": 0.42, "t2": 0.71, "t3": 0.55}
```

The command prints a short human summary. `--json` prints the full object; use
it in CI:

```bash
agentdescent release ... --json --out evidence/release.json
```

Exit codes:

| code | meaning |
|---:|---|
| 0 | released (signed manifest written) |
| 1 | a normal reject/defer; the lineage was not touched |
| 2 | invalid input (scores, rules, lineage, signing material) |

## What is required to release

An accepted promotion is signed only when all of these exist:

- a non-empty `--artifact-id`;
- a non-empty `--artifact-hash` (this is the manifest's content identity);
- a non-empty `--key-id`;
- a non-empty secret from `--secret-env` (default
  `AGENTDESCENT_RELEASE_KEY`) or `--secret-file`;
- enough paired evidence to clear the `PromotionRules` in force.

The secret is deliberately **never** accepted as a command-line argument: argv
is visible in process lists and shell history.

## The rules

Without `--rules`, the surface uses `PromotionRules` with `min_effect` from
`--min-effect` and `max_risk` from `--max-risk`. Every other rule keeps its
fail-closed library default: contamination, canary, eval and risk are required.

For a reproducible policy, pass a JSON object:

```json
{
  "require_eval": true,
  "require_contamination": true,
  "require_canary": true,
  "require_conformal": true,
  "require_sequential": true,
  "require_risk": true,
  "require_verifier": true,
  "require_calibration": false,
  "min_effect": 0.01,
  "min_tasks": 20,
  "max_risk": 0.1,
  "allow_contaminated": false
}
```

Unknown keys are refused: a typo must not silently relax a gate.

`--contamination` and `--approval` default to `unknown`, which is not the same
as "clean" or "approved". An unknown value leaves the corresponding evidence
missing, and the promotion decision defers. Pass `--clean` / `--contaminated`
and `--approved` / `--rejected` only when you actually know the answer.

## Evidence beyond the default

The surface exposes the most common side inputs:

- `--proxy-gains` / `--oracle-gains`: JSON arrays of paired gains for the
  Goodhart proxy-vs-oracle check;
- `--alpha`: statistical alpha (default 0.05);
- `--seed`: bootstrap seed;
- `--risk`: the candidate's risk in `[0, 1]`.

For metamorphic, verifier-panel or calibration evidence, use
`release_candidate` / `release_evidence` from Python: the CLI intentionally
keeps the flag surface small and the library remains the complete API.

## Lineage

`--lineage evidence/lineage.json` loads the existing graph (JSON or JSONL) and
writes it back only when a release happens. Use `--parents a,b` to link the new
artifact to its predecessors:

```bash
agentdescent release ... --lineage evidence/lineage.json --parents skill-csv-total
```

The file is written atomically. A malformed lineage file is an error, not a
reason to start a new graph silently. The read-extend-write is not a lock: do
not run two releases against the same lineage file concurrently; serialize them
in CI or give each release job its own file and merge the graphs later.

## From an MCP host

The same operation is a tool:

```json
{
  "before": {"t1": 0.42, "t2": 0.71},
  "after":  {"t1": 0.55, "t2": 0.83},
  "artifact_id": "skill-csv-total",
  "artifact_hash": "sha256:41d8...",
  "key_id": "ci-release",
  "lineage": "evidence/lineage.json",
  "contamination": "clean",
  "approval": "approved",
  "risk": 0.05
}
```

The server reads the signing secret only from its own environment
(`AGENTDESCENT_RELEASE_KEY`); the MCP tool has no secret argument, so a host
agent cannot use the tool to read an arbitrary file as the key. `lineage`,
`rules`, `proxy_gains` and `oracle_gains` remain operator-supplied paths, so
run the server only for hosts that are trusted with its filesystem. A
reject/defer returns `released: false` and the decision's `reasons` / `missing`.

## Why the pipeline is shaped this way

- **Evidence is computed, not accepted from the caller.** The caller supplies
  task scores; `build_trust_evidence` computes the bootstrap interval, canary,
  conformal and sequential reports.
- **The decision is fail-closed.** Missing evidence defers; failing evidence
  rejects; an explicit `approved: false` rejects even when approval is not
  required. When the rules do not require conformal or sequential evidence, a
  non-decision (`insufficient` / `continue`) is omitted from the receipt, but a
  **reject is never dropped**: it still fails the promotion.
- **The receipt is content-addressed.** The signed manifest carries the
  evidence hash, decision hash, artifact hash, key id, parents and lineage
  node, so the release can be audited without the original process.
- **Reject/defer is free of side effects.** The lineage graph is not mutated,
  so a failed gate never leaves a half-release behind.
