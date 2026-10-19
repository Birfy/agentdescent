Fixed
`doctor` (CLI and MCP) now reports every provider key the installers forward --
the key names of the shared `FORWARDED_ENV` list, `ARK_API_KEY` included -- in
`provider_keys_present`, and raises "no provider key in the environment" only when
none of them is set. Before, it checked three hard-coded names, so a machine with
only `ARK_API_KEY` exited 1 and told the host there was no key. When the only key
present is not OpenAI's or Anthropic's, `doctor` adds a hint to use it through
`openai_compatible` with `api_key_env` and `default_base_url` / `base_url_env`
(endpoint and model from the user). The shipped `SKILL.md` gains an "Other
providers" paragraph saying the same, so a host never has to read the package
source (#423).
