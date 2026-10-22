Fixed
`init` and `doctor` now flag a bare isolated `opencode` worker when only a
gateway key such as `ARK_API_KEY` is set: with no `-m` OpenCode picks its own
default model for that provider, which a Coding-only account may not have
activated. Both point at the documented `extra_args` + `env.OPENCODE_CONFIG`
recipe in `docs/plugins.md`; a worker already configured with a pinned model
and provider-only config stays quiet.
