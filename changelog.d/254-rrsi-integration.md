Added
RRSI wired into `evolve()` (`examples/rrsi/rrsi_policies.py`, #254 / #221):
`build_rrsi_bundle()` assembles the three regularizers -- `RRsiCritic`
(pre-evaluation screen: annealed edit-budget gate + deterministic leakage
screening + optional model reviewer), `RRsiProposalPolicy` (injects budget,
measured per-edit history, pruning targets and stall exploration into the
reflection), and `RRsiAcceptance` (the pinned Algorithm-2 floor + both
cost-rule branches + novelty, with monotone `S*`) -- into one `Policies`
bundle sharing a checkpointable `RRsiState`. New engine seam
`Policies(screen=...)` (`ScreenPolicy`) runs **before measurement**, so a
leaking or over-budget card is settled and reported as `critic-rejected`
without a held-out sweep; new outcomes `noise-floor` / `cost-rule`; new
counters `critic_checked` / `critic_rejected`; the aggregator's checkpoint
now carries an acceptance policy's optional state, so a resumed run does not
lose `S*`. The selection core and schedule fixtures are reused, not
rewritten; declared deviations (one-candidate-per-round judging, per-merge
anneal on the barrier-free path, configured `noise_band`) are recorded in
`docs/algo-rrsi.md`. Default behaviour unchanged -- a run without the bundle
reports zeros.
