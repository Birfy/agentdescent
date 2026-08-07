# Porting checklist

Use this before calling an algorithm example a **benchmark-faithful** port;
microports/analogues follow the [MethodPolicy checklist](porting-methodpolicy.md).

- [ ] Give the port its own directory: `examples/<name>/` with `__init__.py`, a `README.md`, the entry point, and any helper only this port uses.
- [ ] Start from **released code**, not only the paper; follow code on conflicts and document each conflict in `docs/algo-<name>.md`.
- [ ] Use the original repository's dataset rather than substituting an easier benchmark.
- [ ] Build the CLI with `examples._common.add_standard_args`; keep the upstream iteration term (`rounds`, `generations`, `iterations`, or `steps`).
- [ ] Honour the shared flags through `_common.confirm` (`--yes`), `_common.completion_for` (`--provider`/`--model`) and `_common.worker_count` (`--serial`), and add the module to `PORTS` in `tests/test_example_entrypoints.py`.
- [ ] Make `--serial` visible in the plan the port prints, not just inside `evolve()` — a port that reports the parallel plan while running one worker is describing a different run from the one it performed.
- [ ] Load data through `agentdescent.dataloader`, never port-specific HTTP.
- [ ] Choose an explicit `Strategy` (`AppendRules`, `KeyedRules`, `SingleSlot`, `FileTree`, or a justified custom strategy).
- [ ] Prefer an existing scorer from `agentdescent.rewards` when it matches the benchmark.
- [ ] Make `--dry-run` return before data/model setup: zero network and zero API key.
- [ ] Add `tests/test_<name>_example.py`; all tests must run offline.
- [ ] Add `docs/algo-<name>.md`: algorithm summary, every deviation, runnable command, and measured result (or why none exists).
- [ ] Add one row to both the README and `docs/self-evolution-examples.md`.
- [ ] State heavy-infrastructure boundaries such as Docker or gated data instead of hiding them.
- [ ] Record the port author/maintainer so later fidelity decisions have an owner.
