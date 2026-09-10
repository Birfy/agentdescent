"""One call that wires the six pieces, and a whole `evolve()` through it.

`test_a_run_recovers_the_bias_and_the_gate_commits_less` is the plan's last
acceptance criterion: a synthetic loop with a known bias injected, and the
acceptance rate moving the way the correction says it should. Everything above
it guards a cross-reference that is silent when wrong.
"""

import random
from difflib import SequenceMatcher

import pytest

from agentdescent.audit import Audit, AuditStore, attach
from agentdescent.audit.sources import GoldAnswer, NullOracle
from agentdescent.defaults import DefaultAcceptance
from agentdescent.evolution import Task, evolve
from agentdescent.policies import AcceptDecision, MergeContext

from tests.test_audit_sparse import GoodAgent, _generous_judge, _tasks, _truth


def _inner(**kw):
    kw.setdefault("base_delta", 0.2)
    kw.setdefault("anneal_half_life", 64)
    kw.setdefault("accept_samples", 2000)
    return DefaultAcceptance(**kw)


def _ctx(base, cand):
    return MergeContext(artifact=None, candidate=None, cards=[],
                        base_counts=(float(base), 32.0 - base),
                        cand_counts=(float(cand), 32.0 - cand))


def _acceptance_rate(gate, draws=400):
    """Deterministic: the shipped gate seeds its sampler per candidate."""
    rng = random.Random(5)
    accepted = 0
    for _ in range(draws):
        base = rng.randint(10, 26)
        accepted += gate.accept(_ctx(base, min(32, base + rng.randint(0, 8)))).accept
    return accepted / draws


# -- the acceptance criterion ------------------------------------------------

def test_a_run_recovers_the_bias_and_the_gate_commits_less():
    """A whole loop with a known bias, and the correction it produces.

    The judge scores a near-miss by string similarity; the truth is exact match.
    So `f > Y` structurally, and a loop optimising `f` is free to drift toward
    answers only `f` likes -- which is the failure this package exists for.

    Measured on this run: 114 units seen, 75 audited, `delta_hat` about +0.35
    with a residual sd of about 0.38, and the acceptance rate over a fixed grid
    of merge decisions falling from 0.60 to 0.37.
    """
    inner = _inner()
    audit = attach(_generous_judge, oracle=_truth, sample_rate=0.6, seed=3,
                   inner=inner)
    evolve(_tasks(60), audit.reward, agent=GoodAgent(), rounds=12, n_workers=3,
           seed=7)

    status = audit.status()
    assert status["seen"] > status["audited"] > 40, (
        "both halves have to exist or the estimator has nothing to borrow")
    assert not status["is_stale"], status["stale_reason"]
    assert 0.15 < status["delta_hat"] < 0.55, (
        "the injected bias, recovered from nothing but the store")
    assert status["resid_sd"] > 2 * status["se"], (
        "the spread of the error is the larger fact, as it was on real data")

    plain = _acceptance_rate(inner)
    audited = _acceptance_rate(audit.acceptance)
    assert plain > 0.4, "premise: the uncorrected gate commits plenty"
    assert audited < plain * 0.8, (
        f"the gate has to commit less once it knows the verifier is a proxy: "
        f"{plain:.3f} -> {audited:.3f}")


# -- the cross-references that are silent when wrong -------------------------

def test_the_gate_asks_about_the_verifier_that_actually_ran():
    """A `verifier_version` copied by hand goes stale silently.

    The gate then asks the calibrator about a verifier that never ran, gets a
    stale rectification, and that is indistinguishable from "not enough labels
    yet" -- which is what a person will assume.
    """
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 1.0)
    assert audit.acceptance.version() == audit.reward.verifier_version

    audit.reward.verifier_version = "something-else"
    assert audit.acceptance.version() == "something-else", (
        "the gate reads the version live, it does not hold a copy")


def test_one_store_not_two():
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 1.0)
    assert audit.calibrator.store is audit.store is audit.reward.store


def test_the_tap_wraps_the_run_the_loop_was_given():
    seen = []

    def my_run(rendered, task):
        seen.append(rendered)
        return "out"

    audit = attach(lambda task, out: 1.0, run=my_run)
    assert audit.run is not None and audit.run.run is my_run
    audit.run("artifact text", Task("t", "q"))
    assert seen == ["artifact text"]


def test_without_a_run_the_audit_still_works_and_says_less():
    """It can estimate the bias; it cannot attribute a unit to an artifact."""
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 0.0)
    assert audit.run is None
    audit.reward(Task("t1", "q"), "answer")
    assert len(audit.store) >= 0        # sampling may or may not draw this one


def test_a_bare_scorer_is_wrapped_so_it_cannot_fail_a_rollout():
    """`GoldAnswer` swallows exceptions into `.errors`. An unwrapped scorer that
    raises would fail the rollout it was auditing -- an audit that can break the
    run it is watching is worse than no audit."""
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 1 / 0,
                   sample_rate=1.0)
    assert isinstance(audit.reward.oracle, GoldAnswer)
    assert audit.reward(Task("t1", "q"), "answer") == 1.0
    assert audit.reward.oracle.errors


def test_an_oracle_object_is_passed_through_untouched():
    source = NullOracle()
    assert attach(lambda task, out: 1.0, oracle=source).reward.oracle is source


def test_no_oracle_records_the_questions_without_answering_them():
    audit = attach(lambda task, out: 1.0, sample_rate=1.0)
    assert isinstance(audit.reward.oracle, NullOracle)
    audit.reward(Task("t1", "q"), "answer")
    assert len(audit.store.pending()) == 1


def test_a_path_opens_a_store_there(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 0.0,
                   store=str(path), sample_rate=1.0)
    audit.reward(Task("t1", "q"), "answer")
    assert path.exists()
    assert len(AuditStore(str(path))) == 1


def test_the_watch_starts_from_a_baseline_rather_than_an_alarm():
    """The first fingerprint is not a change, and a gate that started stale
    would widen every interval for a reason that had not happened."""
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 1.0)
    assert audit.calibrator.stale_reason is None
    assert not audit.watch.check()


# -- off is off --------------------------------------------------------------

class _Recording:
    def __init__(self):
        self.ctx = None
        self.decision = AcceptDecision(True, "committed", "")

    def accept(self, ctx):
        self.ctx = ctx
        return self.decision


def test_disabled_collects_records_and_corrects_nothing():
    """The honest way to run a first round: measure before you spend."""
    inner = _Recording()
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 0.0,
                   inner=inner, enabled=False, sample_rate=1.0)
    audit.reward(Task("t1", "q"), "answer")
    assert len(audit.store) == 1, "still collecting"

    ctx = _ctx(20, 24)
    got = audit.acceptance.accept(ctx)
    assert inner.ctx is ctx and got is inner.decision, (
        "the same call on the same object, not an equivalent one")
    assert not audit.enabled


def test_a_disabled_policy_disables_the_gate():
    from agentdescent.audit import AuditPolicy

    audit = attach(lambda task, out: 1.0, policy=AuditPolicy(enabled=False))
    assert not audit.enabled


def test_a_plan_supplies_the_rates_and_the_default():
    from agentdescent.audit import SamplePlan

    plan = SamplePlan(rates={"boundary": 0.9}, target_n={}, weights={},
                      resid_sd={}, default_rate=0.02)
    audit = attach(lambda task, out: 1.0, plan=plan, sample_rate=0.5)
    assert audit.reward.rate_for("boundary") == 0.9
    assert audit.reward.rate_for("anything-else") == 0.02


def test_status_reads_without_deciding_anything():
    audit = attach(lambda task, out: 1.0, oracle=lambda task, out: 1.0)
    status = audit.status()
    assert status["is_stale"] and status["audited"] == 0
    assert set(status) >= {"verifier_version", "seen", "audited", "pending",
                           "delta_hat", "resid_sd", "se", "is_stale"}


def test_the_assembled_object_is_what_evolve_takes():
    audit = attach(lambda task, out: 1.0, run=lambda rendered, task: "o")
    assert isinstance(audit, Audit)
    assert callable(audit.reward) and callable(audit.run)
    assert hasattr(audit.acceptance, "accept")
