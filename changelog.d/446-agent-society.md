Added
Agent society (`agentdescent/agent_society.py`): the framework evolves a
cooperating topology, not just one artifact. `AgentNode` (name, instruction,
budget) + `SocietyGraph` (nodes, delegation edges, root) are the container;
`society_spec` is the evolvable artifact — a gated `society()` source compiled
into a `SocietyGraph`, with `SOCIETY_SEED` the single-agent baseline;
`society_problem` is a multi-step task routed through the graph, rewarded as
correctness minus delegation cost; and `society_evolve` runs the ordinary
`evolve()` over those tasks, so the search finds the topology that solves the
task with the least hand-off. `SocietyReport` compares the evolved graph with
the seed on validation tasks — the delegation dimension (Anthropic's
multi-agent research system; AgentEvolve) measured, not assumed.
