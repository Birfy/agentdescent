Removed
`integrations/dsh-agentdescent/node_modules` (210 tracked files of a pnpm peer
install) is no longer in git (#165). The package's `files` list never shipped
it, the tests build their own tmp `node_modules`, and the npm publish workflow
uses only the declared `files`. A root `node_modules/` entry in `.gitignore`
keeps it that way; `tests/test_repo_hygiene.py` asserts none are tracked.
