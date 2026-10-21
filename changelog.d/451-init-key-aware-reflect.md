Fixed
`agentdescent init` no longer guesses a model: the `openai_compatible` reflector
(and the `kind: text` / `policy_slot` agent) is written with the key that is set
here (`"api_key_env": "ARK_API_KEY"` when only a non-OpenAI key is present) and a
`<your-model>` / `<your-endpoint-url>` placeholder, or the values from the new
`--model`, `--base-url` and `--key-env` flags. Its warning names that key, `plan`
and `evolve` refuse a spec that still holds a placeholder, and `plan`'s
missing-OPENAI_API_KEY warning lists the provider keys that are set (#451).
