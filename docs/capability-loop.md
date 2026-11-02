# Capability loop — failures become an admitted, archived capability

`agentdescent capability` is the product surface for the existing
`TaskPipeline` -> `CapabilityScout` -> `CapabilityArchive` chain:

```text
production traces -> task pipeline -> tool synthesis -> capability foundry -> QD archive
```

It does not write tools itself. It runs the tested seams and reports the
pipeline verdict, the scout outcome and the archive result.

## Quickstart

```bash
agentdescent capability init --out capability.json --demo
agentdescent capability plan --spec capability.json
agentdescent capability run  --spec capability.json --json --out report.json
```

`init --demo` writes a runnable offline spec and its `capability-traces.jsonl`.
`run` should end with `verdict: "admitted"` and an `archive.artifact_id`.

## Spec

```json
{
  "traces": "traces.jsonl",
  "goal": "parse a CSV line",
  "descriptor": [0, 0],
  "pipeline": "mypkg.wiring:build_pipeline",
  "scout": "mypkg.wiring:build_scout",
  "archive": "mypkg.wiring:build_archive",
  "make_request": "mypkg.wiring:make_request",
  "kind": "tool",
  "tags": ["csv"],
  "parents": [],
  "metadata": {}
}
```

- `pipeline`, `scout`, `archive` are zero-argument builders (or ready objects)
  for `TaskPipeline`, `CapabilityScout` and `CapabilityArchive`.
- `make_request` is `(ToolCandidate) -> CapabilityRequest | None`.
- `traces` is JSONL, a JSON array, or an inline list. Each row needs
  `trace_id` and `task` (or `prompt`); `reward` defaults to `0.0`.
- `descriptor` must match the archive's bins.

## Verdicts

| verdict | meaning |
|---|---|
| `admitted` | the archive registered the capability |
| `duplicate` | a near-duplicate already exists |
| `not-elite` | the cell's current elite scored higher |
| `pipeline-*` | the task pipeline could not produce usable tasks |
| `capability-*` | the scout/foundry did not admit a capability |

A `pipeline-*` / `capability-*` result is a normal report, not an exception;
the archive is untouched. The spec executes the Python builders it names. For
MCP, the operator must set `AGENTDESCENT_SURFACE_ALLOW` to the allowed module
prefixes (empty refuses every ref) and `AGENTDESCENT_SURFACE_WRITE_DIRS` for
report writes. The MCP `capability` tool takes the same object with
`dry_run=true` for the plan step; plan resolves refs and loads traces without
calling the builders.
