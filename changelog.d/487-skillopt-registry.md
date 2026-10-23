Added
SkillOpt's skill-document training is now a named algorithm
(`"algorithm": "skillopt"`, `kind: text`) in the in-package registry (#384).
The bounded edit patch, strict held-out gate, rejected-edit buffer and annealed
textual learning rate moved from `examples/skillopt` into
`agentdescent/algorithms/skillopt.py`; the example keeps its SearchQA domain,
prompts and CLI and imports the one implementation.
