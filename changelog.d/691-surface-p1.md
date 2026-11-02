Added
Product surfaces for the four P1 capabilities, each with a CLI verb and an MCP
tool: `platform` runs a durable five-layer campaign from a JSON spec with a
runnable `--demo`; `capability` turns failure traces into an admitted, archived
capability; `ops` publishes terminal runs to the console/lineage/telemetry/
metrics with alerts; and `experience` reports, ingests, recalls, updates and
preference-replays an ExperienceStore. All returns carry `schema_version: 1`,
invalid input fails closed with a JSON `{"ok": false, "error": ...}` under
`--json`, and each command has a docs page and an offline demo or sample.
