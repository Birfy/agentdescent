"""The RRSI integration has to be right in the exact places issue #221 calls out:

* leakage and edit-budget screening must run **before** evaluation -- a leaking
  or over-budget card must never buy a held-out sweep, never mind commit;
* the selector must reuse the pinned Algorithm-2 rule (historical-best floor,
  both cost-rule branches, novelty) and keep ``S*`` monotone and resumable;
* the proposer must actually condition on the measured history, the pruning
  targets and the stall-exploration note -- not just carry the knobs;
* the untouched path stays untouched (a run without the bundle reports zeros).

Every fixture here is offline and deterministic.
"""

import pytest

from agentdescent import Contract, Invariant
from agentdescent.aggregator import MergeOutcome
from agentdescent.evolution import SingleSlot, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.policies import MergeContext, Policies

from examples.rrsi.rrsi_policies import (
    RRsiAcceptance,
    RRsiConfig,
    RRsiCritic,
    RRsiProposalPolicy,
    RRsiState,
    build_rrsi_bundle,
)


def _task(n):
    return Task(id=f"t{n}", prompt=f"item {n}", meta={"gold": "42"})


def _diff(ops, diff_id="d"):
    return Diff(diff_id=diff_id, target="artifact", ops=ops)


def _card(ops, cost_tokens=0):
    return EvidenceCard(diff=_diff(ops), base_version={}, touched=["artifact"],
                        trajectory_refs=[_task(0)], cost_tokens=cost_tokens)


def _context(state, *, base=(5, 5), cand=(5, 5), ops=("value",), costs=0):
    from agentdescent.evolution import EvolvingArtifact
    art = EvolvingArtifact("artifact", {"value": "seed"}, strategy=SingleSlot())
    cand_art = EvolvingArtifact("artifact", {"value": "new"}, strategy=SingleSlot())
    cards = [_card({k: "x" for k in ops}, cost_tokens=costs)]
    return MergeContext(
        artifact=art, candidate=cand_art, cards=cards,
        base_counts=base, cand_counts=cand, diff=cards[0].diff)


# -- the critic: budget + leakage before evaluation -----------------------------


def test_the_budget_anneals_and_the_critic_enforces_it():
    state = RRsiState(RRsiConfig(b_min=1, b_max=5, T=4))
    critic = RRsiCritic(state)
    assert state.budget() == 5              # round 0: early, wide
    assert critic.screen(_card({"a": "x", "b": "y", "c": "z"})) is None
    # round 3 of T=4 has annealed to 2: a 3-op diff now exceeds it
    state.round = 3
    assert state.budget() == 2
    assert critic.screen(_card({"a": "x", "b": "y"})) is None
    reason = critic.screen(_card({"a": "x", "b": "y", "c": "z"}))
    assert reason is not None and "budget" in reason


def test_the_critic_rejects_evolve_set_leakage():
    state = RRsiState()
    critic = RRsiCritic(state, protected=("t3", "42"))
    assert critic.screen(_card({"value": "generic rule"})) is None
    reason = critic.screen(_card({"value": "always answer t3 with 42"}))
    assert reason is not None and "leaks evolve-set content" in reason


def test_the_model_reviewer_is_only_asked_when_deterministic_checks_pass():
    seen = {"n": 0}

    def reviewer(prompt):
        seen["n"] += 1
        return "leak"

    state = RRsiState()
    critic = RRsiCritic(state, protected=("t3",), reviewer=reviewer)
    # deterministic leak -> the reviewer is never asked
    critic.screen(_card({"value": "mention t3"}))
    assert seen["n"] == 0
    # clean diff -> the reviewer is asked and can veto
    reason = critic.screen(_card({"value": "generic rule"}))
    assert reason == "critic: leak"
    assert seen["n"] == 1


def test_protected_strings_never_come_from_the_holdout():
    # The discipline issue #221 demands: protected strings are derived from the
    # evolve set only. A holdout answer must not be in the critic's set.
    train_gold = {"t0": "42", "t1": "17"}
    holdout_gold = "SECRET-FINAL-HOLDOUT-ANSWER"
    protected = [f"{k}={v}" for k, v in train_gold.items()]
    assert all(holdout_gold not in p for p in protected)
    assert "SECRET-FINAL-HOLDOUT-ANSWER" not in " ".join(protected)


# -- the state: history, stall, pruning ----------------------------------------


def test_record_measurement_builds_the_history_ledger():
    state = RRsiState()
    # the diffs carry upstream-visible content signals: `skills/` -> skill,
    # `Memory(` -> memory (not the path-segment heuristic)
    state.record_measurement(
        _diff({"skills/x/SKILL.md": "always call the SkillRegistry"}),
        delta_score=0.1, delta_cost=0.0, accepted=True)
    state.record_measurement(
        _diff({"memory/y.md": "Memory().remember(...)"}),
        delta_score=-0.1, delta_cost=0.0, accepted=False)
    assert state.history.attempted()["skill"] == 1
    assert state.history.attempted()["memory"] == 1
    recs = state.history.records()
    assert len(recs) == 2
    assert recs[0]["accepted"] is True
    assert recs[1]["accepted"] is False


def test_stall_detection_and_pruning_targets():
    state = RRsiState(RRsiConfig(noise_band=0.05, prune_window=2,
                                 stall_window=3))
    # a component that never yields positive gain is a pruning target
    state.record_measurement(
        _diff({"tools/a.md": "ToolRegistry.register_tool('a')"}),
        delta_score=0.0, delta_cost=0.0, accepted=False)
    state.record_measurement(
        _diff({"tools/b.md": "ToolRegistry.register_tool('b')"}),
        delta_score=-0.05, delta_cost=0.0, accepted=False)
    assert "client_tool" in state.pruning_targets()
    # a healthy component is not
    state2 = RRsiState(RRsiConfig(prune_window=2))
    state2.record_measurement(
        _diff({"skill/a.md": "skills/SKILL.md: use the registry"}),
        delta_score=0.3, delta_cost=0.0, accepted=True)
    assert state2.pruning_targets() == []
    # a flat S* trajectory beyond the noise band is a stall (the pinned
    # stall_flag needs a window of points; progress[i] is S* after merge i+1)
    state3 = RRsiState(RRsiConfig(noise_band=0.05, stall_window=3))
    state3.progress = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert state3.stalled() is True
    state3.progress = [0.5, 0.5, 0.6, 0.7, 0.8]
    assert state3.stalled() is False


def test_the_proposer_prompt_carries_budget_history_pruning_and_stall():
    from agentdescent.policies import ProposalContext

    calls = {"prompts": []}

    def complete(prompt):
        calls["prompts"].append(prompt)
        return "always state the year"

    state = RRsiState(RRsiConfig(T=10, prune_window=2, stall_window=3))
    state.record_measurement(
        _diff({"tools/a.md": "ToolRegistry.register_tool('a')"}),
        delta_score=0.0, delta_cost=0.0, accepted=False)
    state.progress = [0.5, 0.5, 0.5, 0.5, 0.5]     # stalled
    policy = RRsiProposalPolicy(complete, state)
    out = policy.propose(ProposalContext(
        rendered="seed", task=_task(0), output="answer", reward=0.0))
    prompt = calls["prompts"][0]
    assert out == ["always state the year"]
    assert f"Round {state.round}" in prompt
    assert "[RRSI history]" in prompt
    assert "[RRSI pruning]" in prompt and "client_tool" in prompt
    assert "[RRSI exploration]" in prompt


def test_state_checkpoint_round_trip(tmp_path):
    # the search state (round / S* / cost / progress) rides the aggregator
    # checkpoint; the per-edit ledger is the durable History at history_path,
    # so a reopened state reads the same evidence the loop conditioned on
    hist = str(tmp_path / "rrsi-history.jsonl")
    state = RRsiState(history_path=hist)
    state.best_score = 0.9
    state.round = 3
    state.record_measurement(
        _diff({"skill/a.md": "skills/SKILL.md: use the registry"}),
        delta_score=0.4, delta_cost=0.0, accepted=True)
    fresh = RRsiState(history_path=hist)
    fresh.restore(state.checkpoint())
    assert fresh.best_score == 0.9
    assert fresh.round == 3
    assert len(fresh.history.records()) == 1
    assert fresh.history.records()[0]["component"] == "skill"
    assert fresh.history.attempted()["skill"] == 1


# -- the selector: pinned Algorithm 2 via the engine's acceptance slot ----------


def test_acceptance_applies_the_historical_best_floor():
    state = RRsiState(base_score=0.9)
    policy = RRsiAcceptance(state)
    # candidate 0.8 < S* (0.9) with delta 0 -> noise floor
    d = policy.accept(_context(state, base=(5, 5), cand=(8, 2)))
    assert d.accept is False
    assert d.category == MergeOutcome.NOISE_FLOOR.value
    assert state.best_score == 0.9           # S* unchanged


def test_acceptance_large_gain_cost_rule():
    state = RRsiState(base_score=0.5)
    state.incumbent_cost = 100
    policy = RRsiAcceptance(state)
    # deltaS = 0.7 - 0.5 = 0.2 > noise_band(0); deltaC = (300-100)/100 = 2.0
    # beta0=0.1, beta1=1.0 -> budget 0.1 + 0.2 = 0.3 < 2.0 -> refuse
    d = policy.accept(_context(state, base=(5, 5), cand=(7, 3), costs=300))
    assert d.accept is False
    assert d.category == MergeOutcome.COST_RULE.value


def test_a_streak_of_rejections_anneals_the_budget_and_trips_stall():
    """The annealed round must advance on *every* merge decision, not just
    accepts: a search that keeps being rejected is spending rounds stuck, and
    the sparsity anneal plus the stall note are exactly the mechanisms that
    exist to get it unstuck. Before the fix, ten straight rejections left
    `round` at 0, the budget at b_max, and `stalled()` False (review)."""
    st = RRsiState(RRsiConfig(b_min=1, b_max=5, T=4, stall_window=3),
                   base_score=0.5)
    acc = RRsiAcceptance(st)
    for _ in range(10):
        d = acc.accept(_context(st, base=(5, 5), cand=(3, 7)))
        assert d.accept is False
    assert st.round == 10, (
        f"the round must advance per merge, not per accept; round={st.round}")
    assert st.budget() <= 2, (
        "a streak of rejections must anneal the edit budget, not pin it at "
        f"b_max; budget={st.budget()}")
    assert st.stalled() is True, (
        "a flat progress must trip the stall note while the search is stuck")


def test_an_improvement_after_a_rejection_streak_is_not_a_stall():
    """The accept must append the *post-decision* S*, so an improvement that
    breaks a rejection streak is not reported as a stall. Before the fix,
    `record_merge()` ran before `best_score` was updated, so an accepted jump
    left progress flat (`[0.5, 0.5, 0.5]`) and `stalled()` True even though S*
    just moved (review)."""
    st = RRsiState(RRsiConfig(stall_window=3), base_score=0.5)
    acc = RRsiAcceptance(st)
    # two rejections (base 0.5 -> cand 0.3): progress stays at 0.5
    assert acc.accept(_context(st, base=(5, 5), cand=(3, 7))).accept is False
    assert acc.accept(_context(st, base=(5, 5), cand=(3, 7))).accept is False
    # one acceptance (base 0.5 -> cand 0.8): S* moves to 0.8
    assert acc.accept(_context(st, base=(5, 5), cand=(8, 2))).accept is True
    assert st.best_score == pytest.approx(0.8)
    assert st.progress == [0.5, 0.5, 0.8], (
        "the accept must append the post-decision S*, not the pre-decision one; "
        f"got {st.progress}")
    assert st.stalled() is False, (
        "an improvement that broke the streak must not read as a stall")
    assert st.round == 3


def test_an_improvement_after_a_rejection_streak_survives_checkpoint():
    """The corrected progress must round-trip through the acceptance state's
    checkpoint: a resumed run must not inherit a stale flat history that reads
    as a stall when S* actually moved."""
    st = RRsiState(RRsiConfig(stall_window=3), base_score=0.5)
    acc = RRsiAcceptance(st)
    acc.accept(_context(st, base=(5, 5), cand=(3, 7)))     # reject
    acc.accept(_context(st, base=(5, 5), cand=(3, 7)))     # reject
    acc.accept(_context(st, base=(5, 5), cand=(8, 2)))     # accept -> 0.8
    fresh = RRsiState(RRsiConfig(stall_window=3), base_score=0.5)
    fresh.restore(st.checkpoint())
    assert fresh.best_score == pytest.approx(0.8)
    assert fresh.progress == [0.5, 0.5, 0.8]
    assert fresh.stalled() is False, (
        "restored history must not report a stall for an S* that moved")


def test_screen_rejections_advance_the_schedule():
    """The aggregator's screen-rejected path (critic) must advance the anneal
    too -- it is a merge decision, and the RRSI acceptance policy opts in via
    on_screen_reject (review: the round advanced only on accepts)."""
    state = RRsiState(RRsiConfig(b_min=1, b_max=5, T=4))
    policy = RRsiAcceptance(state)
    for _ in range(5):
        policy.on_screen_reject()
    assert state.round == 5
    assert state.budget() <= 4   # annealed below b_max after rejections


def test_card_cost_is_a_per_rollout_mean_not_a_batch_sum():
    """The cost rule must not depend on how many cards happened to land in the
    bucket: `EvidenceCard.cost_tokens` is one proposing rollout's usage, so
    `_card_cost` is a per-rollout mean, not a sum. Three cards at 100 tokens
    each must read the same as one card at 100 (review: the sum gave dC=+2.0
    and rejected on batch size)."""
    from examples.rrsi.rrsi_policies import _card_cost

    single = _card_cost([_card({"a": "x"}, cost_tokens=100)])
    batch = _card_cost([_card({"a": "x"}, cost_tokens=100),
                        _card({"b": "y"}, cost_tokens=100),
                        _card({"c": "z"}, cost_tokens=100)])
    assert single == batch == 100.0, (
        f"per-rollout mean must be batch-size independent; "
        f"single={single}, batch={batch}")
    # missing/zero telemetry -> None (dC = 0, the documented behaviour)
    assert _card_cost([_card({"a": "x"})]) is None
    assert _card_cost([]) is None


def test_acceptance_accepts_an_improvement_and_advances_s_star():
    state = RRsiState(base_score=0.5)
    policy = RRsiAcceptance(state)
    d = policy.accept(_context(state, base=(5, 5), cand=(7, 3)))
    assert d.accept is True
    assert state.best_score == pytest.approx(0.7)   # monotone advance
    assert state.round == 1
    # a later candidate below the new S* is refused by the floor
    d2 = policy.accept(_context(state, base=(7, 3), cand=(5, 5)))
    assert d2.accept is False
    assert d2.category == MergeOutcome.NOISE_FLOOR.value


def test_acceptance_within_band_rule_counts_novelty():
    # within the noise band, novelty can carry an otherwise-flat candidate
    cfg = RRsiConfig(noise_band=0.05, w_s=1.0, w_c=1.0, w_n=1.0)
    state = RRsiState(cfg, base_score=0.5)
    policy = RRsiAcceptance(state)
    # deltaS = 0.1 <= band(0.05)? no -> 0.1 > 0.05 is large-gain. Use cand=0.52
    # deltaS=0.02 <= band: within-band; novel structural skill (unseen) -> w_n*1
    d = policy.accept(_context(state, base=(50, 50), cand=(52, 48),
                               ops=("skills/x/SKILL.md",)))
    assert d.accept is True
    assert state.best_score == pytest.approx(0.52)


# -- end to end ----------------------------------------------------------------


def _offline_actors():
    def run(rendered, task):
        return "answer" + (" 2026" if "year" in rendered else "")

    def reward(task, output):
        return 1.0 if "2026" in output else 0.0

    return run, reward


def _tasks(n=8):
    return [Task(id=f"t{i}", prompt=f"item {i}") for i in range(n)]


def _complete_factory(leak=False):
    def complete(prompt):
        return "always mention t3" if leak else "always state the year"
    return complete


def test_end_to_end_leaking_candidate_is_screened_before_measurement(tmp_path):
    run, reward = _offline_actors()
    bundle, state = build_rrsi_bundle(_complete_factory(leak=True),
                                      protected=("t3",))
    result = evolve(_tasks(), reward, run=run,
                    strategy=SingleSlot(initial_value="seed"), rounds=2,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    policies=bundle)
    assert result.critic_rejected >= 1
    assert result.critic_checked >= 1
    assert result.outcomes().get("critic-rejected", 0) >= 1
    assert result.outcomes().get("committed", 0) == 0


def test_end_to_end_a_clean_run_commits_under_the_bundle(tmp_path):
    run, reward = _offline_actors()
    bundle, state = build_rrsi_bundle(_complete_factory(),
                                      protected=("t3",))
    result = evolve(_tasks(), reward, run=run,
                    strategy=SingleSlot(initial_value="seed"), rounds=2,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    policies=bundle)
    assert result.outcomes().get("committed", 0) >= 1
    assert result.outcomes().get("critic-rejected", 0) == 0
    assert result.critic_rejected == 0
    assert state.best_score > 0.0            # S* advanced


def test_untouched_path_reports_zeros(tmp_path):
    run, reward = _offline_actors()
    result = evolve(_tasks(), reward, run=run,
                    strategy=SingleSlot(initial_value="seed"), rounds=1,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    propose=lambda r, t, o, re: "always state the year")
    assert result.critic_checked == 0
    assert result.critic_rejected == 0
    assert result.outcomes().get("critic-rejected", 0) == 0


def test_resume_preserves_the_historical_best_floor(tmp_path):
    run, reward = _offline_actors()
    repo = str(tmp_path / "repo")
    bundle1, state1 = build_rrsi_bundle(_complete_factory())
    evolve(_tasks(), reward, run=run, strategy=SingleSlot(initial_value="seed"),
           rounds=1, n_workers=2, repo_path=repo, policies=bundle1,
           checkpointing=True)
    assert state1.best_score > 0.0
    # a fresh bundle on the same repo: the aggregator restores the acceptance
    # state, so the floor is preserved, not reset
    bundle2, state2 = build_rrsi_bundle(_complete_factory())
    evolve(_tasks(), reward, run=run, strategy=SingleSlot(initial_value="seed"),
           rounds=1, n_workers=2, repo_path=repo, policies=bundle2,
           checkpointing=True)
    assert state2.best_score == pytest.approx(state1.best_score)


def test_invariant_rejections_advance_the_rrsi_schedule(tmp_path):
    """A violates-invariant refusal is still a merge decision: the RRSI anneal
    and stall detector must advance once per rejection (review #254 P2). A
    one-op SingleSlot proposal passes the critic but fails mutable_keys that
    only permit an unrelated key; after a streak, round/progress move and
    stall trips, without inventing reward/cost measurements."""
    run, reward = _offline_actors()
    cfg = RRsiConfig(b_min=1, b_max=5, T=4, stall_window=3)
    bundle, state = build_rrsi_bundle(_complete_factory(), protected=("t3",),
                                      config=cfg)
    contract = Contract(invariants=(Invariant("mutable_keys", ("other",)),))
    result = evolve(_tasks(), reward, run=run,
                    strategy=SingleSlot(initial_value="seed"), rounds=4,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    policies=bundle, contract=contract)
    assert result.outcomes().get("violates-invariant", 0) >= 1
    assert result.outcomes().get("committed", 0) == 0
    n_rej = result.outcomes().get("violates-invariant", 0)
    assert state.round == n_rej, (
        f"exactly one schedule advance per invariant refusal; "
        f"round={state.round} rejections={n_rej}")
    assert state.best_score == 0.0, "S* must stay at base (no invented measurement)"
    assert len(state.progress) == n_rej
    assert all(p == 0.0 for p in state.progress)
    if n_rej >= 3:
        assert state.stalled() is True
    assert state.budget() < 5, "rejection streak must anneal the edit budget"


def test_resume_preserves_default_in_memory_history(tmp_path):
    """#337: a fresh default bundle resumed from a checkpoint keeps the per-edit
    evidence (accepted counts, attempts), not just S*/round/progress."""
    run, reward = _offline_actors()
    repo = str(tmp_path / "repo")
    bundle1, state1 = build_rrsi_bundle(_complete_factory())
    evolve(_tasks(), reward, run=run, strategy=SingleSlot(initial_value="seed"),
           rounds=2, n_workers=2, repo_path=repo, policies=bundle1,
           checkpointing=True)
    assert state1.history.records(), "the first run measured nothing"
    bundle2, state2 = build_rrsi_bundle(_complete_factory())
    assert state2.history.records() == []
    evolve(_tasks(), reward, run=run, strategy=SingleSlot(initial_value="seed"),
           rounds=1, n_workers=2, repo_path=repo, policies=bundle2,
           checkpointing=True)
    before = state1.history.records()
    assert state2.history.records()[:len(before)] == before
