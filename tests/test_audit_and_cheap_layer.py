"""The audit gate must be reachable, and the cheap layer must be cheap.

Two halves of one mechanism, and both were inert.

**The gate could not open.** `force_oracle` fires on
`blast_radius >= 0.5 or trust < 0.75`, and the only writer of trust --
`update_trust` -- sat *inside* that branch. So for any artifact below 0.5 the
condition could never become true: measured on the default `blast_radius=0.2`,
`oracle_calls_used` was 0 for a whole run and trust stayed pinned at its initial
1.0. An artifact at 0.4 -- which `governance.classify` calls L1, the *slow,
conservative* layer -- got exactly as much oracle scrutiny as an L2 skill: none.

**The layers all computed the same number.** `evolve()` pinned
`rule_subset=len(held_out)` with zero noise, so rule / learned / oracle were one
full held-out sweep wearing three names. The aggregator therefore paid a full
sweep for every candidate it merely wanted to *rank*, and `oracle_budget` capped
nothing -- its documented fallback (`rule_eval`) returned the very value it was
trying to avoid buying.
"""

import warnings

import pytest

from agentdescent.aggregator import Aggregator
from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.scheduler import AuditScheduler
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget

TASKS = [Task(id=str(i), prompt=f"q{i}", meta={"gold": "y"}) for i in range(20)]


def _probe_factory(store):
    class _Probe(Aggregator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            store["agg"] = self

    def factory(ledger, verifier, audit, config, policy):
        return _Probe(ledger, verifier, audit, config, staleness_policy=policy)
    return factory


def _run(blast_radius=0.2, cheap_eval_tasks=4, rounds=8, repo_path=None, **kw):
    """A workload where candidates genuinely differ, so there is a verdict to audit."""
    store = {}
    n = [0]

    def propose(rendered, task, output, score):
        n[0] += 1
        return "good rule" if n[0] % 2 else f"useless rule {n[0]}"

    def reward(task, output):
        return 1.0 if (int(task.id) % 2 == 0 and "good" in (output or "")) else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(TASKS, reward, run=lambda rendered, t: rendered, propose=propose,
                     strategy=AppendRules(), blast_radius=blast_radius, rounds=rounds,
                     n_workers=4, self_verify=False, oracle_budget=200,
                     cheap_eval_tasks=cheap_eval_tasks, repo_path=repo_path,
                     aggregator_factory=_probe_factory(store), **kw)
    return res, store["agg"]


# -- the gate is reachable ------------------------------------------------------


def test_trust_can_fall_without_spending_oracle_budget():
    """The circularity: trust gated the oracle, and only the oracle wrote trust.

    Agreement between the cheap layer and the full held-out set is free -- the
    Beta test already scores both on the full set -- so it is measured on every
    merge, not only when the gate happens to be open.
    """
    _, agg = _run(blast_radius=0.2)
    assert agg.audit.trust("artifact") != 1.0, (
        "trust never moved from its initial value at blast_radius=0.2, so "
        "force_oracle's trust term is still unreachable")


def test_a_distrusted_artifact_opens_the_gate_below_the_blast_radius_threshold():
    """Once trust is writable, a low-blast-radius artifact can earn an audit."""
    audit = AuditScheduler()
    assert not audit.force_oracle(0.2, "art"), "premise: a trusted L2 artifact is not audited"
    audit.update_trust("art", oracle_agreed=False)
    assert audit.trust("art") < 0.75
    assert audit.force_oracle(0.2, "art"), (
        "a cheap layer that keeps misjudging must be able to open the oracle gate "
        "even below blast_radius 0.5")


def test_a_high_blast_radius_artifact_is_still_audited_unconditionally():
    """The other half of the condition must keep working."""
    _, agg = _run(blast_radius=0.6)
    assert agg.verifier.budget.oracle_calls_used > 0


# -- the cheap layer is actually cheap ------------------------------------------


def test_the_cheap_layer_scores_fewer_tasks_than_the_oracle(tmp_path):
    """Otherwise `oracle_budget` caps nothing: the fallback costs what it saves."""
    _, agg = _run(cheap_eval_tasks=3, repo_path=str(tmp_path))
    v = agg.verifier
    assert len(v._subset(v.rule_subset)) == 3
    assert len(v.held_out) > 3, "premise: the held-out set is bigger than the sample"


def test_the_cheap_sample_is_stable_so_candidates_are_compared_like_for_like():
    """`_subset` drew a fresh random sample per call.

    Safe only while `rule_subset` covered the whole set and the draw was a no-op.
    With a real cheap layer it silently scores candidate A on one set of tasks and
    candidate B on another, then calls the difference a winner -- and defeats the
    evaluation cache, which memoises per (artifact, task).
    """
    v = ThreeLayerVerifier(eval_fn=lambda a, ts: len(ts), held_out=list(range(20)),
                           rule_subset=5)
    first = list(v._subset(5))
    for _ in range(5):
        assert list(v._subset(5)) == first, "the cheap sample must not move between calls"
    # and the rule/learned layers must agree on which items they are scoring
    assert set(v._subset(5)) <= set(v.held_out)


def test_the_full_set_still_decides_whether_to_commit(tmp_path):
    """Sub-sampling trades ranking precision, never commit safety."""
    _, agg = _run(cheap_eval_tasks=2, repo_path=str(tmp_path))
    successes, failures = agg.verifier.eval_counts(
        agg.ledger.snapshot("dev").get("artifact"))
    assert round(successes + failures) == len(agg.verifier.held_out), (
        "the acceptance test must score the whole held-out set, not the cheap sample")


def test_the_default_is_unchanged_so_existing_runs_are_not_silently_resampled(tmp_path):
    """`cheap_eval_tasks=None` keeps exact scoring -- opting in is the caller's call."""
    _, agg = _run(cheap_eval_tasks=None, repo_path=str(tmp_path))
    v = agg.verifier
    assert v.rule_subset == len(v.held_out)
    assert list(v._subset(v.rule_subset)) == list(v.held_out)
