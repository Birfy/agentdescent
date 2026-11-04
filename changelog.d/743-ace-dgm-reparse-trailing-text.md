Fixed
`algorithms` ace / dgm: re-opening an applied target no longer drops user text written after the `# Playbook (ACE)` / `## Capabilities` block; it is re-attached after the block, an unchanged run stays a byte-identical `apply` noop, and multi-line ACE lessons round-trip (#743).
