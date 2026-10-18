Changed
Install docs no longer say "once published" / "before publication" for 0.6.0:
the release is on PyPI. README installs it with `pip install "agentdescent>=0.6.0"`
(core package; the offline demo needs no extra), `docs/plugins.md` and
`docs/plugin-quickstart.md` use `pip install "agentdescent[mcp]>=0.6.0"` for the
CLI and MCP tools, and `docs/install.md` answers "Can I install from PyPI?" with
that command while keeping the installer / current-main flow first; a source
checkout is the alternative (#165).
