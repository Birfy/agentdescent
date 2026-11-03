Changed
`agentdescent install` / `uninstall` accept common host aliases, case-insensitive
(`claude`, `grok`, `grokbot`, `deepseek-harness`, `open-code`, …), and a bare
`install` / `uninstall` lists every host with its state and the command to run
instead of argparse's "the following arguments are required" (exit 2; `--json`
returns the same data) (#721).
