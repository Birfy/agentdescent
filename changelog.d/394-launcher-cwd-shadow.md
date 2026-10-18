Fixed
`agentdescent install` no longer writes an MCP launcher that breaks when the host
is started in a directory containing an `agentdescent/` folder (#394). `python -m`
puts the host's cwd first on `sys.path`, so that folder shadowed the installed
package and the server died on import, which the host shows only as `Connection
closed`. The pinned launcher is now `<python> -P -m agentdescent.cli mcp` on
Python >= 3.11 and the absolute path of the `agentdescent` console script on 3.9
and 3.10 (only a script that actually runs this interpreter; plain `-m` otherwise, and `install` then prints a WARNING that the launcher still honours the host's cwd); `-I` is not used because it would
hide `pip install --user` packages. `install` recognises the old, `-P` and
console-script shapes as its own, so reinstalling upgrades an existing config in
place without duplicating it, and `uninstall` removes any of them. When the server
cannot be imported, `agentdescent mcp` now prints one line (naming the directory
that shadows the package, when that is the cause) and exits 3, instead of a second
traceback from `mcp_unavailable()`.
