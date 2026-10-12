"""Value-directed budget allocation: learn which candidates are worth the
expensive evaluation, with counterfactual exploration to correct the learner.

``BudgetGovernor`` degrades optional spend (fusion tournaments, self-verify
rollouts) against global floors: once ``spent / max_tokens`` crosses a threshold,
*every* candidate stops getting the expensive part. The EvoAlloc line of work
measures the cost of that fixed allocation when evaluation is the scarce
resource. These tests pin the replacement: a per-candidate decision learned from
the run's own merge history, plus an epsilon-greedy counterfactual term that
re-measures candidates the model judged unworthy so a biased value model is
corrected rather than reinforced.
"""

from __future__ import annotations

import random

from agentdescent.allocator import (
    AllocatorContext,
    LearningValueModel,
    ValueBudgetAllocator,
)
from agentdescent.budget import BudgetGovernor
from agentdescent.evolution import Task, evolve


def _high():
    return AllocatorContext(advantage=1.0, size=2)


def _low():
    return AllocatorContext(advantage=-0.5, size=2)


#: A candidate *before* the self-verify ran -- the pre-spend features only
#: (advantage, size, blast radius), which is what the worker actually decides
#: on. The post-outcome signals (delta, P(improve), stable distance) never
#: enter the model's feature vector.
def _unmeasured(advantage: float, size: int = 2):
    return AllocatorContext(advantage=advantage, size=size)


# ---------------------------------------------------------------------------
# the value model
# ---------------------------------------------------------------------------


def test_value_model_learns_high_advantage_predicts_commit():
    m = LearningValueModel(lr=0.3, reg=0.05)
    for _ in range(30):
        m.update(_high().features(), True)
        m.update(_low().features(), False)
    p_high = m.predict(_high().features())
    p_low = m.predict(_low().features())
    assert p_high > p_low, "the model must learn that a positive advantage commits"


def test_value_model_stays_calm_on_a_single_outlier():
    """One extreme candidate must not blow the weights -- the logit clip."""
    m = LearningValueModel(lr=0.3, reg=0.05)
    m.update((0.999999, 64, 1.0), True)   # extreme pre-spend vector
    for _ in range(10):
        m.update(_low().features(), False)
    assert all(abs(w) < 1e6 for w in m._w), "weights must stay finite"


def test_value_model_bias_learns_on_all_zero_features():
    """The bias weight must move even when every feature is zero -- otherwise
    an all-zero candidate (nothing measured on this workload) can never learn a
    non-neutral prior. This pins the logistic model's arithmetic: the bias is
    the last weight, matched against ``1.0`` in the gradient, and never double-
    counted in the logit."""
    m = LearningValueModel(lr=0.3, reg=0.05)
    zeros = (0.0, 0.0, 0.0)   # advantage, size, blast_radius
    for _ in range(50):
        m.update(zeros, True)          # every all-zero candidate commits
    assert m.predict(zeros) > 0.6, "the bias must learn a non-neutral prior"
    assert m._w[-1] != 0.0, "the bias weight must move"


def test_value_model_blast_radius_moves_its_own_weight():
    """blast_radius must be a *feature* with its own learned weight, not folded
    into the bias. Under a single data point the two are confounded (many
    ``(w_blast, bias)`` pairs give the same logit), so this asserts the feature
    actually moves prediction: a candidate with high blast_radius but nothing
    else measured must be scored differently from the all-zero baseline."""
    m = LearningValueModel(lr=0.3, reg=0.05)
    zeros = (0.0, 0.0, 0.0)          # nothing measured
    blast = (0.0, 0.0, 1.0)          # only blast_radius set
    # All-zero candidates commit; blast-only candidates do not.
    for _ in range(30):
        m.update(zeros, True)
        m.update(blast, False)
    p_zero = m.predict(zeros)
    p_blast = m.predict(blast)
    assert p_blast < p_zero, (
        "blast_radius must carry independent signal, not be folded into bias")


# ---------------------------------------------------------------------------
# the allocator decision
# ---------------------------------------------------------------------------


def test_allocator_skips_low_value_candidate_after_learning():
    alloc = ValueBudgetAllocator(min_observations=3,
                                 epsilon_start=0.0, epsilon_end=0.0)
    for _ in range(5):
        alloc.observe(_high(), True)      # feedback: real delta
        alloc.observe(_low(), False)
    # Decision side sees the *unmeasured* candidate (delta unknown), and must
    # still tell high-advantage from low-advantage on the up-front signals.
    assert alloc.decide(_unmeasured(1.0)).deep_eval, "high value keeps the deep eval"
    assert not alloc.decide(_unmeasured(-0.5)).deep_eval, "low value loses it"


def test_allocator_deep_evals_everything_before_it_has_an_opinion():
    """Skipping on no evidence is the bias the exploration exists to avoid."""
    alloc = ValueBudgetAllocator(min_observations=5)
    assert alloc.decide(_low()).deep_eval, "no opinion yet -> spend, don't skip"
    assert alloc.n_skipped == 0


def test_allocator_counterfactual_exploration_re_measures_skipped():
    """A candidate the model judged unworthy is still evaluated with epsilon
    probability, so the model sees the ground truth it was wrong about."""
    alloc = ValueBudgetAllocator(min_observations=2,
                                 epsilon_start=1.0, epsilon_end=1.0)
    for _ in range(2):
        alloc.observe(_low(), False)
    random.seed(7)
    n_cf = sum(1 for _ in range(100)
               if alloc.decide(_low()).counterfactual)
    assert n_cf > 0, "epsilon>0 must occasionally re-measure a skipped candidate"


def test_counterfactual_outcome_corrects_the_value_model():
    """The model believed `_low` never commits; counterfactual outcomes that
    *do* commit must push its prediction up -- the selection-bias correction."""
    alloc = ValueBudgetAllocator(min_observations=2,
                                 epsilon_start=1.0, epsilon_end=1.0)
    for _ in range(4):
        alloc.observe(_low(), False)
    p_before = alloc.model.predict(_low().features())
    # Now the workload flips: every low candidate actually commits.
    for _ in range(20):
        alloc.observe(_low(), True)
    p_after = alloc.model.predict(_low().features())
    assert p_after > p_before, (
        "counterfactual outcomes must move the model toward the truth")


# ---------------------------------------------------------------------------
# governor integration
# ---------------------------------------------------------------------------


def test_governor_without_allocator_keeps_old_behaviour():
    """No allocator -> the clock thresholds decide, exactly as before."""
    g = BudgetGovernor(max_tokens=1000)
    g.spend(800)                       # 80% spent
    assert g.candidate_worth(_high())
    assert g.candidate_worth(_low())   # old behaviour: no per-candidate skip
    g.spend(920)                       # 92% spent, past the hard floor
    assert not g.candidate_worth(_high())


def test_governor_with_allocator_skips_only_when_budget_is_tight():
    alloc = ValueBudgetAllocator(min_observations=3,
                                 epsilon_start=0.0, epsilon_end=0.0)
    for _ in range(5):
        alloc.observe(_high(), True)
        alloc.observe(_low(), False)
    # Budget not tight: everything deep-evals (old behaviour).
    loose = BudgetGovernor(max_tokens=1000, allocator=alloc)
    loose.spend(100)                  # 90% left, above budget_start
    assert loose.candidate_worth(_low())
    # Budget tight: the allocator decides per candidate.
    tight = BudgetGovernor(max_tokens=1000, allocator=alloc)
    tight.spend(800)                  # 20% left, below budget_start
    assert tight.candidate_worth(_high())
    assert not tight.candidate_worth(_low())


def test_governor_never_spends_past_the_hard_floor_via_allocator():
    """The clock thresholds are the ceiling the allocator may never exceed."""
    alloc = ValueBudgetAllocator(min_observations=0,
                                 epsilon_start=0.0, epsilon_end=0.0)
    g = BudgetGovernor(max_tokens=1000, allocator=alloc)
    g.spend(950)                      # 95% spent, past the hard floor
    assert not g.candidate_worth(_high()), "no allocator may spend past 90%"


def test_observe_allocated_is_a_no_op_without_allocator():
    g = BudgetGovernor(max_tokens=1000)
    g.observe_allocated(_high(), committed=True)   # must not raise
    assert g.allocator is None


# ---------------------------------------------------------------------------
# engine integration
# ---------------------------------------------------------------------------


def _reward(task, output):
    return 1.0 if output == "yes" else 0.0


class _Reflector:
    def solve(self, rendered, task):
        return "yes" if "rule" in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return "a helpful rule"


def test_evolve_feeds_commit_outcomes_to_the_allocator():
    import warnings
    alloc = ValueBudgetAllocator(min_observations=2)
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolve(tasks, _reward, agent=_Reflector(), rounds=4, n_workers=2,
               max_concurrency=2, max_tokens=200_000, allocator=alloc)
    assert alloc.n_observed >= 1, "the merge path must report outcomes"
    assert alloc.n_committed >= 1, "the committed candidate must be observed"


def test_evolve_without_allocator_is_unchanged():
    import warnings
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(tasks, _reward, agent=_Reflector(), rounds=3, n_workers=2,
                     max_concurrency=2, max_tokens=200_000)
    assert res.final_reward == 1.0


def test_async_evolve_accepts_allocator():
    import warnings
    from agentdescent.async_evolve import async_evolve
    from agentdescent.evolution import AppendRules
    alloc = ValueBudgetAllocator(min_observations=2)
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(6)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = async_evolve(
            tasks, _reward, run=_Reflector().solve,
            propose=_Reflector().propose, strategy=AppendRules(),
            n_workers=2, max_seconds=5.0, max_tokens=100_000,
            allocator=alloc, held_out_frac=0.4)
    assert res.final_reward is not None


# ---------------------------------------------------------------------------
# every measured decision reports exactly one outcome (PR #204 review)
# ---------------------------------------------------------------------------


def _candidate(artifact, cand, diff, *, base_counts=(2.0, 0.0),
               cand_counts=(3.0, 0.0), fused=False, base_cheap=0.9,
               cand_cheap=0.95):
    """A measured `_Candidate` for the merge path."""
    from agentdescent.aggregator import _Candidate
    return _Candidate(
        artifact_id=artifact.id, artifact=artifact, candidate=cand, diff=diff,
        cards=[], survivor_cards=[], head={artifact.id: artifact.version},
        fused=fused, considered=1, survived=1, discarded=0, conflicts=0,
        base_counts=base_counts, cand_counts=cand_counts,
        base_cheap=base_cheap, cand_cheap=cand_cheap)


def test_every_merge_outcome_is_observed_exactly_once(tmp_path):
    """Oracle rejection, acceptance rejection, commit and CAS conflict must
    each report exactly one outcome to the allocator -- rejection is a negative
    label, and without it a model fit on commits alone never leaves warm-up
    (PR #204 review)."""
    from agentdescent.aggregator import Aggregator, AggregatorConfig
    from agentdescent.evolvable import Diff
    from agentdescent.evolution import AppendRules, EvolvingArtifact
    from agentdescent.ledger import Ledger
    from agentdescent.scheduler import AuditScheduler
    from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget

    class AlwaysRejectAudit(AuditScheduler):
        """`force_oracle` always opens, and the oracle always vetoes."""

        def force_oracle(self, blast_radius, artifact_id):
            return True

        def full_eval(self, artifact):   # not used; see below
            return 0.0

    class VetoingVerifier(ThreeLayerVerifier):
        def full_eval(self, artifact):
            return 0.0          # never better than base -> oracle veto

    def serialize(a): return {"state": a.state, "blast_radius": a.blast_radius}
    def deserialize(aid, v, s):
        return EvolvingArtifact(aid, s.get("state", {}), v,
                                s.get("blast_radius", 0.2))
    lg = Ledger(str(tmp_path / "repo"), serialize, deserialize)
    lg.register(EvolvingArtifact("a", {"k": "v0"}, blast_radius=0.2))
    base = lg.snapshot(Ledger.DEV).get("a")

    alloc = ValueBudgetAllocator(min_observations=1, epsilon_start=0.0,
                                 epsilon_end=0.0)

    def state_scorer(a, t):
        return 0.9 if a.state.get("k") == "v1" else 0.3

    def make_agg(verifier, audit=None):
        if audit is None:
            audit = AuditScheduler()
        return Aggregator(lg, verifier, audit,
                          AggregatorConfig(batch_trigger=1))
    agg = make_agg(ThreeLayerVerifier(
        eval_fn=state_scorer, held_out=[1, 2, 3],
        budget=VerifierBudget(oracle_calls_remaining=100)))
    agg.allocator = alloc

    before = alloc.n_observed

    # 1. Acceptance rejection: candidate fails the Beta test (cheap and full
    #    layers agree it is worse, so the audit gate stays closed).
    cand_bad = base.apply(Diff(diff_id="bad", target="a", ops={"k": "bad"},
                               author="w"))
    c_bad = _candidate(base, cand_bad,
                       Diff(diff_id="bad", target="a", ops={"k": "bad"},
                            author="w"),
                       base_counts=(2.0, 1.0),      # rate 0.67
                       cand_counts=(1.0, 3.0),      # rate 0.25, clearly worse
                       cand_cheap=0.1)              # cheap agrees it is worse
    r_bad = agg._decide(c_bad)
    assert r_bad.category in ("below-threshold", "cas-conflict"), r_bad.category
    # exactly one observation, and it is a negative label
    assert alloc.n_observed == before + 1, "acceptance rejection must observe once"
    assert alloc.n_committed == 0

    # 2. Commit: candidate beats the base on the full held-out.
    cand_good = base.apply(Diff(diff_id="good", target="a",
                                ops={"k": "v1"}, author="w"))
    c_good = _candidate(base, cand_good,
                        Diff(diff_id="good", target="a", ops={"k": "v1"},
                             author="w"),
                        base_counts=(2.0, 1.0),      # rate 0.67
                        cand_counts=(4.0, 0.0))      # rate 1.0, better
    r_good = agg._decide(c_good)
    assert r_good.category == "committed", r_good.category
    assert alloc.n_observed == before + 2, "a commit must observe once"
    assert alloc.n_committed == 1

    # 3. Oracle rejection: `force_oracle` always opens and the oracle vetoes
    #    -- one more negative observation.
    agg_oracle = make_agg(VetoingVerifier(
        eval_fn=state_scorer, held_out=[1, 2, 3],
        budget=VerifierBudget(oracle_calls_remaining=100)),
        audit=AlwaysRejectAudit())
    agg_oracle.allocator = alloc
    # force_oracle always opens and the oracle (full_eval -> 0.0, or the
    # candidate's sub-base rate) vetoes -- a third, negative observation.
    c_oracle = _candidate(base, cand_bad,
                          Diff(diff_id="oracle", target="a", ops={"k": "bad"},
                               author="w"),
                          base_counts=(2.0, 1.0), cand_counts=(1.0, 3.0),
                          cand_cheap=0.1)
    r_oracle = agg_oracle._decide(c_oracle)
    assert r_oracle.category == "oracle-rejected", r_oracle.category
    assert alloc.n_observed == before + 3, (
        "oracle rejection must observe once")
    assert alloc.n_committed == 1, "a veto is a negative label"


def test_sync_async_pre_spend_features_are_identical():
    """The decision-time features the sync and async workers build must be the
    same vector the aggregator trains on -- advantage, size, blast radius, and
    nothing post-outcome (PR #204 review)."""
    from agentdescent.evolvable import Diff
    diff = Diff(diff_id="d", target="a", ops={"k1": "v", "k2": "w"}, author="w")
    from agentdescent.evolution import _allocator_context
    from agentdescent.async_evolve import _async_allocator_context
    sync = _allocator_context(diff, advantage=0.7, score=0.3, blast_radius=0.4)
    async_ctx = _async_allocator_context(diff, advantage=0.7, blast_radius=0.4)
    assert sync.features() == async_ctx.features(), (
        "sync and async decision features must match")
    # Only the three pre-spend features are model inputs; post-outcome signals
    # are absent even when supplied.
    assert sync.features() == (0.7, 2.0, 0.4)
    # The training-side context (with post-outcome fields filled) must produce
    # the same feature vector as the decision side.
    from agentdescent.aggregator import _allocator_context_from
    from agentdescent.evolution import EvolvingArtifact
    art = EvolvingArtifact("a", {"k": "v0"}, blast_radius=0.4)
    cand = art.apply(diff)
    card = type("Card", (), {"advantage": 0.7,
                             "before_after_delta": 0.9,
                             "diff": diff})()
    c = type("C", (), {
        "artifact": art, "candidate": cand, "diff": diff,
        "cards": [card], "fused": False})()
    trained = _allocator_context_from(c, p_improve=0.99, stable_distance=0.3)
    assert trained.features() == sync.features(), (
        "training features must equal decision features, so the model cannot "
        "fit labels with post-outcome signals the decision never sees")


def test_fused_observe_uses_the_originating_proposal_context():
    """A fused commit's label attaches to the *originating* proposal's pre-spend
    vector, not the fused diff's union size -- the worker decided on the single
    proposal, and training on a union size no decision saw leaks post-merge
    information (PR #204 review, finding 2 remainder)."""
    from agentdescent.aggregator import _allocator_context_from
    from agentdescent.evolvable import Diff
    from agentdescent.evolution import EvolvingArtifact
    art = EvolvingArtifact("a", {"k": "v0"}, blast_radius=0.2)
    proposal = Diff(diff_id="p1", target="a", ops={"k1": "v"}, author="w")
    fused = Diff(diff_id="fused(p1+p2)", target="a",
                 ops={"k1": "v", "k2": "w"}, author="aggregator")   # size 2
    card = type("Card", (), {"advantage": 0.7,
                             "before_after_delta": 0.9,
                             "diff": proposal})()                    # size 1
    cand = art.apply(fused)
    c = type("C", (), {
        "artifact": art, "candidate": cand, "diff": fused,
        "cards": [card], "fused": True})()
    ctx = _allocator_context_from(c, p_improve=0.99, stable_distance=0.3)
    # The decision side saw the proposal (size 1), and the fused label must
    # attach to that same vector -- not the fused size 2.
    assert ctx.features() == (0.7, 1.0, 0.2), (
        f"fused observe must use the originating proposal's size, got "
        f"{ctx.features()}")


class _RecordingAllocator:
    """Records every decide/observe input feature vector, then always deep-evals.

    Lets an end-to-end run prove fused commits observe the *originating*
    proposal's pre-spend vector, not the fused union size."""

    def __init__(self):
        self.decided = []
        self.observed = []
        self.budget_start = 2.0        # budget never tight -> always decide

    def decide(self, ctx):
        self.decided.append(ctx.features())
        from agentdescent.allocator import AllocateDecision
        return AllocateDecision(deep_eval=True)

    def observe(self, ctx, committed):
        self.observed.append((ctx.features(), committed))


def test_fused_commit_observes_originating_proposal_end_to_end():
    """A real `evolve()` run fuses two proposals and commits the union; the
    observe input must be a worker's actual pre-spend vector (the single rule's
    size 1), never the fused size 2 (PR #204 review finding 2 remainder)."""
    import warnings
    from agentdescent.evolution import AppendRules

    rec = _RecordingAllocator()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(8)]

    def run(rendered, task):
        return rendered

    def reward(task, output):
        return min(1.0, output.count("rule") * 0.5)

    propose = iter([f"rule {i}" for i in range(20)])

    def make_propose(rendered, task, output, score):
        return next(propose, "rule x")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolve(tasks, reward, run=run, propose=make_propose,
               strategy=AppendRules(), rounds=1, n_workers=2,
               max_concurrency=1, max_tokens=10_000, allocator=rec,
               held_out_frac=0.5, shuffle=False)

    assert rec.decided, "workers must call decide before spending"
    assert rec.observed, "the merge must call observe"
    # Every observed vector's size must be one the allocator actually decided
    # on -- a single proposal's diff size. The fused union (size >= 2) must not
    # appear, because no worker ever decided about a size-2 candidate.
    decided_sizes = {f[1] for f in rec.decided}
    for feats, committed in rec.observed:
        assert feats[1] in decided_sizes, (
            f"observe fed size {feats[1]}, but decisions only saw "
            f"sizes {decided_sizes} -- the fused union size leaked into training")


def test_async_fused_commit_observes_originating_proposal_end_to_end():
    """Same guarantee on the barrier-free path: the merger's observe for a
    fused commit carries a worker's pre-spend vector, not the union size."""
    import warnings
    from agentdescent.async_evolve import async_evolve
    from agentdescent.evolution import AppendRules

    rec = _RecordingAllocator()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(8)]

    def run(rendered, task):
        return rendered

    def reward(task, output):
        return min(1.0, output.count("rule") * 0.5)

    propose = iter([f"rule {i}" for i in range(20)])

    def make_propose(rendered, task, output, score):
        return next(propose, "rule x")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = async_evolve(
            tasks, reward, run=run, propose=make_propose,
            strategy=AppendRules(), n_workers=2, max_seconds=5.0,
            max_tokens=10_000, allocator=rec, held_out_frac=0.5,
            shuffle=False)
    assert rec.decided, "workers must call decide before spending"
    decided_sizes = {f[1] for f in rec.decided}
    for feats, committed in rec.observed:
        assert feats[1] in decided_sizes, (
            f"async observe fed size {feats[1]}, but decisions only saw "
            f"sizes {decided_sizes}")
