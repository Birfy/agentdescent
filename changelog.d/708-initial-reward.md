Added
`evolve` / `async_evolve` score the seed artifact once on the held-out set
before round 0 and record it as `initial_reward` on `EvolutionResult`,
`result.json`, and run-store `status`. Foreground `evolve` / `demo` print
`held-out reward: <initial> -> <final>`; `show` and `status --json` / MCP
carry the field. Seed scoring never aborts the run (`null` on failure).
`plan`'s estimate adds `seed_score_calls` (#708).
