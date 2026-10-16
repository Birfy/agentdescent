"""The learned fusion decision has to be right in the places a learner is
dangerous and a merge is load-bearing.

A policy that learns "when does merging win" from the tournament's own trials
must keep the default as the default (no data -> tournament verbatim), must
only skip the ranking sweep when the prediction is confident, must observe its
mistakes through counterfactual exploration, and must retire itself when it
starts being wrong. Every fixture here pins one of those.
"""

from agentdescent import Diff, EvolvingArtifact, SingleSlot
from agentdescent.fusion import LearnableFusionPolicy


def _setup():
    base = EvolvingArtifact("artifact", {"k": "v"}, strategy=SingleSlot())
    d1 = Diff(diff_id="1", target="artifact", ops={"a": "x"})
    d2 = Diff(diff_id="2", target="artifact", ops={"b": "y"})
    return base, [d1, d2]


class _ScoreVerifier:
    """A cheap layer whose score is a pure function of the candidate's state."""

    def __init__(self, fn):
        self.fn = fn

    def cheap_eval(self, artifact):
        return self.fn(artifact)


def _union_wins(artifact):
    return 0.5 + 0.1 * len(artifact.state)   # union has the most keys


def _single_wins(artifact):
    return 0.9 - 0.1 * len(artifact.state)   # singles have fewer keys


def _tie(artifact):
    return 0.6                                # everyone equal


def test_default_is_the_tournament_verbatim():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0)
    policy.bind(_ScoreVerifier(_union_wins))
    out = policy.select(base, diffs)
    assert out[2] is True                    # the union was ranked and won
    assert policy.tournament_runs == 1
    assert policy.skipped == 0
    assert len(policy.trials) == 1           # a real measured trial


def test_features_are_diff_only_and_zero_verifier_calls():
    base, diffs = _setup()
    union, _ = LearnableFusionPolicy._union_of(diffs)
    f = LearnableFusionPolicy._features(diffs, union)
    assert len(f) == 5
    assert f[0] > 0                          # log1p(n_candidates)
    assert 0.0 <= f[1] <= 1.0                # overlap fraction
    assert f[4] == 1.0                       # a union exists
    none, _ = LearnableFusionPolicy._union_of(diffs[:1])
    f2 = LearnableFusionPolicy._features(diffs[:1], none)
    assert f2[4] == 0.0


def test_the_policy_learns_to_skip_the_ranking_sweep():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0, epsilon=0.0, min_trials=20)
    policy.bind(_ScoreVerifier(_union_wins))
    for _ in range(60):
        policy.select(base, diffs)
    assert policy.skipped >= 1               # once warm, the union wins -> skip
    assert policy.status()["skips_per_merge"] > 0
    # a skipped merge returns the union without ranking it
    out = policy.select(base, diffs)
    assert out[2] is True
    assert set(out[0].ops) == {"a", "b"}


def test_the_policy_does_not_skip_when_the_union_loses():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0, epsilon=0.0, min_trials=20)
    policy.bind(_ScoreVerifier(_single_wins))
    for _ in range(60):
        policy.select(base, diffs)
    assert policy.skipped == 0               # a single wins -> never confident
    assert policy.status()["skip_enabled"] is True


def test_a_tie_is_read_conservatively_and_never_skipped():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0, epsilon=0.0, min_trials=20)
    policy.bind(_ScoreVerifier(_tie))
    for _ in range(60):
        policy.select(base, diffs)
    assert policy.skipped == 0
    # the conservative reading: a fusion that merely ties does not win
    assert all(t.winner != "fused" for t in policy.trials)


def test_the_rollback_guard_retires_a_wrong_learner():
    # Regime 1: the union always wins (the model learns to skip). Regime 2: a
    # single starts winning. The counterfactual exploration proves the model
    # confident-and-wrong, and the guard retires skipping permanently.
    base, diffs = _setup()
    mode = {"single_wins": False}

    def two_regime(artifact):
        return _single_wins(artifact) if mode["single_wins"] else _union_wins(artifact)

    policy = LearnableFusionPolicy(seed=0, epsilon=1.0, min_trials=15)
    policy.bind(_ScoreVerifier(two_regime))
    for _ in range(25):
        policy.select(base, diffs)            # warm-up on union-wins
    assert policy.status()["skip_enabled"] is True
    mode["single_wins"] = True
    for _ in range(10):
        policy.select(base, diffs)            # regime flips; guard must fire
    assert policy.status()["skip_enabled"] is False


def test_turning_the_tournament_off_degrades_to_the_union_path():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0, tournament=True)
    policy.bind(_ScoreVerifier(_union_wins))
    policy.tournament = False                  # the budget governor's flip
    out = policy.select(base, diffs)
    assert out[2] is True                      # the plain union
    assert policy.skipped == 0
    assert policy.tournament_runs == 0
    assert policy.trials == []


def test_bind_forwards_the_verifier_to_the_inner_tournament():
    base, diffs = _setup()
    policy = LearnableFusionPolicy(seed=0)
    verifier = _ScoreVerifier(_union_wins)
    policy.bind(verifier)
    assert policy.inner.verifier is verifier
    assert policy._union_policy.verifier is verifier


def test_bad_knobs_are_refused():
    import pytest
    with pytest.raises(ValueError):
        LearnableFusionPolicy(uncertainty_threshold=0.0)
    with pytest.raises(ValueError):
        LearnableFusionPolicy(epsilon=1.5)


def test_the_policy_runs_end_to_end_through_evolve(tmp_path):
    # Installed as the engine's fusion slot, the learner ranks every merge
    # (collecting labels) and reports fusion_stats -- the pipeline the
    # tournament already provides, with the learner in the seat.
    from agentdescent import Policies, evolve
    from agentdescent.evolution import SingleSlot, Task

    tasks = [Task(id=f"t{i}", prompt=f"item {i}") for i in range(8)]

    def run(rendered, task):
        return "answer" + (" 2026" if "year" in rendered else "")

    def reward(task, output):
        return 1.0 if "2026" in output else 0.0

    policy = LearnableFusionPolicy(seed=0)
    result = evolve(
        tasks, reward, run=run,
        propose=lambda r, t, o, re: "always state the year",
        strategy=SingleSlot(initial_value="seed"), rounds=2, n_workers=2,
        repo_path=str(tmp_path / "repo"),
        policies=Policies(fusion=policy))
    assert result.outcomes().get("committed", 0) >= 1
    assert policy.tournament_runs >= 1
    stats = result.fusion_stats()
    assert stats.trials >= 1
