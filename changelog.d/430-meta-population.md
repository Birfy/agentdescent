Added
Harness composition (`agentdescent/harness.py`): combine algorithm ports, learn
which wins where. A `Harness` is a named `(tasks, repo) -> EvolutionResult`
callable; the consolidation archive becomes the leaderboard — one
`ConsolidationRecord` per `(config, harness)` under `artifact_id="harness:<name>"`
with the domain's `config` and the measured `final_reward`, written atomically
and idempotently. `record_harness` / `harness_score` / `harness_leaderboard`
read and write that evidence; `select_harness` routes a new domain to the
harness the recorded rewards say wins there (same-config fingerprint first, best
reward), falling back to round-robin or a caller default when no evidence exists
— a harness with no runs on a domain can never beat one that was measured there.
`compose_run` runs the selected harness (`mode="best"`) or every harness
(`mode="all"`, the leaderboard's training mode), each on a fresh ledger repo,
records the outcomes, and returns a `HarnessReport` (per-harness reward /
time-to-quality, winner, selection reason). Every algorithm port binds into a
`Harness` unchanged; the archive's shape and `config_fingerprint` domain
separation are reused as-is.
