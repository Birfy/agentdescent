# Use it from your agent — the plugin

> Work in progress: this page fills in as the plugin lands. The design record is
> [Plugin for DSH, Claude Code, Codex and other agents](plugin-design.md).

The pieces that exist so far:

* `agentdescent.evolvespec` — an `evolve()` call as data (`EvolveSpec`,
  `compose`, `run_spec`, `load_spec`).
* `agentdescent.rewards.command_scorer` — grade with any program.
* `agentdescent.agents.dsh()` — DeepSeek Harness's headless profile as a worker.
