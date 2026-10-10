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
    return AllocatorContext(before_after_delta=0.9, advantage=1.0, size=2)


def _low():
    return AllocatorContext(before_after_delta=0.0, advantage=-0.5, size=2)


#: A candidate *before* the self-verify ran -- delta not yet measured, which is
#: what the worker actually decides on. The value model must distinguish these
#: on the signals that are available up front (advantage, size), not on delta.
def _unmeasured(advantage: float, size: int = 2):
    return AllocatorContext(before_after_delta=None, advantage=advantage, size=size)


# ---------------------------------------------------------------------------
# the value model
# ---------------------------------------------------------------------------


def test_value_model_learns_high_delta_predicts_commit():
    m = LearningValueModel(lr=0.3, reg=0.05)
    for _ in range(30):
        m.update(_high().features(), True)
        m.update(_low().features(), False)
    p_high = m.predict(_high().features())
    p_low = m.predict(_low().features())
    assert p_high > p_low, "the model must learn that a positive delta commits"


def test_value_model_stays_calm_on_a_single_outlier():
    """One extreme candidate must not blow the weights -- the logit clip."""
    m = LearningValueModel(lr=0.3, reg=0.05)
    m.update((0.999999, 1.0, 64, 0.0, 0.0, 0.0, 0.2), True)   # extreme
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
    zeros = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
    zeros = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    blast = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)     # only blast_radius set
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
