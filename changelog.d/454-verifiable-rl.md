Added
Verifiable-reward policy improvement (`agentdescent/verifiable_rl.py`): the
framework trains on truth it can check. `verifiable_problem` wraps a
`(value, task) -> output` problem so its outcome carries the output and the
gold in `detail`; `verifiable_reward` scores exact match (1.0 / 0.0 — a rule
that cannot be checked is not rewarded, the DeepSeek-R1 verifiability
contract); and `policy_improve` evolves a decision slot with that reward via
`meta_evolve`, reporting the per-round verifiable reward curve (the
trajectory-level signal RL trains on) and the before/after on held-out
verifiable facts (`VerifiableReport`).
