Fixed
`algorithms run ace` / `dgm` no longer drop or comma-shred the user's existing text prompt: ace seeds the playbook from the target and only appends lessons under `# Playbook (ACE)`, dgm keeps the prose intact and lists admitted capabilities under `## Capabilities`, and an unchanged run reports `apply` noop (#731).
