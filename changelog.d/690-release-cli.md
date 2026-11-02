Added
`agentdescent release` and the MCP `release` tool: one evidence-gated release
seam that builds trusted evidence from paired held-out scores, runs the
fail-closed promotion decision, and only on an accepted promotion signs an
`ArtifactManifest` and writes its lineage node. Signing material comes from an
environment variable or file (never argv); reject/defer leaves the lineage
untouched; the JSON result carries `schema_version: 1`. Use `--lineage` to
persist and extend the receipt graph and `--out` to write the full report.
