Added
Genesis (`examples/genesis`): a blind mode in which the acceptance suite lives
outside the repository and the agents see the requirement and the assertion's
name rather than its source, `own_review` / `own_test_failures` so a child's own
tests are run and implementation code with no test under its node is sent back
(a test file is `test_*.py` or `*_test.py`; package markers, `conftest.py` and
generated modules owe none), the `fly` domain (only `REQUIREMENTS.md` is frozen),
a containerised session sandbox that forwards credentials by name rather than by
value, and the `claude-cli` provider with `--max-tokens` / `--timeout` in
`examples/_common.py`. `agentdescent/` itself is untouched (#188).
