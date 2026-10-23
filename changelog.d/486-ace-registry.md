Added
ACE's grow-and-refine playbook is now a named algorithm (`"algorithm": "ace"`,
`kind: text`) in the in-package registry (#384). `ACEPlaybook`, its
near-duplicate de-dup and the `GrowAndRefine` Curator moved from `examples/ace`
into `agentdescent/algorithms/ace.py`; the example keeps its FiNER domain,
prompts and CLI and imports the one implementation.
