Fixed
`plan` / `evolve` no longer call a provider key "untested" for an isolated
OpenCode worker. With no `-m` and only a gateway key such as `ARK_API_KEY` set,
the warning now says OpenCode would fall back to its own default model for that
provider (measured: an unactivated Volcengine default, every call failed) and
gives the documented fix, `-m ark/deepseek-v4-flash` plus
`env.OPENCODE_CONFIG` pointing at a provider-only file. That documented recipe
itself (a pinned `provider/model` and `OPENCODE_CONFIG` or
`OPENCODE_CONFIG_CONTENT` in the agent's `env`) no longer gets the "a login does
not carry over" warning at all.
