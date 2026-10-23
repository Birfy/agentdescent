Added
EvoSkill's bounded skill-library frontier is now a named algorithm
(`"algorithm": "evoskill"`, `kind: agent_dir`) in the in-package
registry (#384). The two-role Proposer/Generator induction, the top-K frontier
and the best-member parent rule moved from `examples/evoskill` into
`agentdescent/algorithms/evoskill.py`; the example keeps its OfficeQA domain,
prompts and CLI and imports the one implementation.
