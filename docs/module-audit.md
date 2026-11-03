# Module audit — what is on the product path?

The package has three kinds of modules. This read-only audit separates them
without changing an import:

```bash
python -m tools.module_audit
python -m tools.module_audit --json
python -m tools.module_audit --limit 15 --fail-on-isolated
```

| category | meaning |
|---|---|
| `cli/mcp` | reachable from `cli.py`, `mcp.py`, `evolvespec.py`, `demo.py` or `__main__.py` through intra-package imports or `agentdescent.module:attribute` string refs |
| `exported-only` | not on the CLI/MCP path, but reachable from the package `__init__` (public Python surface) |
| `isolated` | only its own tests/docs/examples reference it |

The JSON report also carries module/LOC totals, the top-level `__all__` length,
the top-level CLI command count, the MCP tool count, and per-subpackage totals.

It is **report-only** by default: it never moves a file, changes an export or
fails a build. `--fail-on-isolated` is an opt-in local check for a cleanup
branch, not a repository policy. Use the output to decide what to quarantine,
merge or productize; the audit does not make that decision.

## Limitations

- It follows **static** imports and `agentdescent.module:attribute` string refs.
  Modules resolved only by a runtime `importlib.import_module`, an entry-point
  plugin, or a copied source template can appear `isolated` without being dead.
  Read `isolated` as "not statically referenced", not "safe to delete".
- `cli/mcp` vs `exported-only` is a **static dependency** split. Importing any
  submodule also executes the package `__init__`, which is why ancestors of a
  reached leaf are counted as reached.
- Use `--package <name>` when auditing a package directory whose import name is
  not `agentdescent`.
