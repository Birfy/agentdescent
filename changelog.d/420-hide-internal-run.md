Fixed
`agentdescent --help` no longer prints `run  ==SUPPRESS==`, and neither the help
nor the usage line shown on every argparse error lists the internal `run`
worker (what `evolve --detach` spawns) next to `evolve`; argparse does not
honour `help=SUPPRESS` on a sub-parser. `agentdescent run --run-dir ...` still
works, and `run --help` now says it is internal (#420).
