# dsh-agentdescent

Evolve skills, agent definitions, prompts, code and host plugins from inside
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), using
[AgentDescent](https://github.com/Birfy/agentdescent).

```bash
pip install "agentdescent[mcp]"            # the tools this plugin exposes
dsh plugin --profile web add link:/path/to/dsh-agentdescent
```

The plugin registers the `agentdescent` skill at runtime -- no file to install,
and it cannot drift from the package -- and its `cordis.patch.yml` adds the
`agentdescent mcp` server, so the model gets `mcp__agentdescent__plan`,
`__start`, `__status`, `__show`, `__apply` and the rest.

Check it composed:

```bash
dsh --profile web --dump-config | grep -n agentdescent
```

`agentdescent install dsh` is the alternative that needs no npm install: it
writes the same skill as a file into `$DSH_HOME/skills/` and the same MCP row
into `$DSH_HOME/cordis.patch.yml`. Use one or the other, not both.

Generated from the Python package; see `agentdescent/integrations/__init__.py`.
