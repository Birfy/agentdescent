"""`dev -> stable` promotion must reward an artifact for *holding up*.

The counter was bumped on the commit path, so it measured how many times an
artifact had **changed** -- the opposite of what every description of it says
("survival rounds", "regression-free rounds", "EMA confirmation rounds"). The
incentive was inverted: an artifact that converged stopped committing and could
therefore never be promoted, while one that thrashed promoted every K commits.

`python -m examples.run_demo` showed it plainly -- `stable` sat at 0.000 for all
40 rounds while `dev` reached 1.000 by round 3 -- and nothing caught it, because
`run_demo` is one of the examples with no test at all.
"""

import tempfile

import pytest

from agentdescent.aggregator import Aggregator, AggregatorConfig, MergeReport
from agentdescent.domains.router import (
    RouterSkill, deserialize_router, make_task_universe, router_eval,
    serialize_router,
)
from agentdescent.evolvable import Diff
from agentdescent.ledger import Ledger
from agentdescent.orchestrator import AgentDescent
from agentdescent.scheduler import AuditScheduler
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget


def _aggregator(tmp_path, promote_after_k=3):
    universe = make_task_universe(seed=7)
    _, held_out = universe.split()
    ledger = Ledger(str(tmp_path), serialize_router, deserialize_router)
    ledger.register(RouterSkill("art", table={}))
    verifier = ThreeLayerVerifier(eval_fn=router_eval, held_out=held_out,
                                 budget=VerifierBudget(oracle_calls_remaining=50))
    return Aggregator(ledger, verifier, AuditScheduler(),
                      AggregatorConfig(promote_after_k=promote_after_k))


def _stable_version(agg):
    return agg.ledger.head_version(Ledger.STABLE).get("art", 0)


def test_a_converged_artifact_is_promoted(tmp_path):
    """The success state -- nothing left to commit -- must not block promotion.

    This is the whole bug: under the old rule, "no more commits" meant "no more
    promotions", so the better the artifact the less likely it was to reach
    `stable`.
    """
    agg = _aggregator(tmp_path, promote_after_k=3)
    agg._survival["art"] = 0
    before = _stable_version(agg)
    for _ in range(3):
        agg.step()                       # quiet rounds: nothing to merge
    assert _stable_version(agg) > before, \
        "an artifact that stopped changing was never promoted to stable"


def test_a_commit_restarts_the_clock(tmp_path):
    """A freshly committed version has survived nothing yet."""
    agg = _aggregator(tmp_path, promote_after_k=3)
    agg._age_and_promote([_report("below-threshold")])    # the artifact is now tracked
    agg.step()
    assert agg._survival["art"] == 2
    agg._age_and_promote([_report("committed", committed=7)])
    assert agg._survival["art"] == 0, "a commit must restart the survival clock"


def test_a_measured_regression_restarts_the_clock(tmp_path):
    """An oracle rejection is the one signal that the artifact got worse."""
    agg = _aggregator(tmp_path, promote_after_k=3)
    agg._age_and_promote([_report("below-threshold")])
    agg.step()
    assert agg._survival["art"] == 2
    agg._age_and_promote([_report("oracle-rejected")])
    assert agg._survival["art"] == 0, "a regression must restart the survival clock"


def _report(category, committed=None):
    """A MergeReport carrying just the fields promotion looks at."""
    return MergeReport("art", None, False, 1, 1, 0, 0, 0.5, committed,
                       category, category)


def test_a_rejected_candidate_is_not_a_regression(tmp_path):
    """`below-threshold` means the artifact held its ground, which is survival.

    The gate rejecting a challenger is the artifact *winning*; treating that as a
    regression would make a healthy converged run un-promotable again.
    """
    agg = _aggregator(tmp_path, promote_after_k=5)
    agg._age_and_promote([_report("below-threshold")])
    assert agg._survival["art"] == 1, "surviving a challenge is a round survived"


def test_an_unchanged_head_is_promoted_once_not_every_k_rounds(tmp_path):
    """Confirmation is per *version*, not a heartbeat.

    The synchronous driver steps once per round, but the async merger polls every
    couple of milliseconds -- so re-copying an unchanged artifact every K sweeps
    ran 52 promotions in one 6-second run, each a handful of git operations under
    the lock every worker queues behind.
    """
    agg = _aggregator(tmp_path, promote_after_k=3)
    agg._age_and_promote([_report("below-threshold")])     # start tracking
    versions = []
    for _ in range(12):
        agg.step()
        versions.append(_stable_version(agg))
    promotions = sum(1 for a, b in zip(versions, versions[1:]) if b > a)
    assert promotions == 1, (
        f"an unchanged dev head was copied to stable {promotions} times; "
        "promotion should be idempotent per version")


def test_a_new_dev_version_becomes_promotable_again(tmp_path):
    """Idempotence must not mean "promote once and never again"."""
    agg = _aggregator(tmp_path, promote_after_k=2)
    agg._age_and_promote([_report("below-threshold")])
    for _ in range(3):
        agg.step()
    first = _stable_version(agg)
    assert first > 0, "premise: the initial head should have been promoted"

    # dev moves on -- commit a real new version through the ledger
    snap = agg.ledger.snapshot(Ledger.DEV)
    art = snap.get("art")
    agg.ledger.commit(art.apply(Diff("d", "art", {"kw00": "acidbase"})),
                      {"art": snap.version["art"]}, branch=Ledger.DEV)
    agg._age_and_promote([_report("committed", committed=2)])
    for _ in range(3):
        agg.step()
    assert _stable_version(agg) > first, \
        "a newly committed version was never confirmed onto stable"


def test_finalize_publishes_a_head_that_stopped_before_confirming(tmp_path):
    """`target_reward` fires on the very commit that reaches it.

    Confirmation takes K rounds, so without this the artifact the run was *for*
    would never reach the branch production reads.
    """
    agg = _aggregator(tmp_path, promote_after_k=99)        # unreachable in this test
    agg._age_and_promote([_report("committed", committed=1)])
    assert _stable_version(agg) == 1, "premise: nothing confirmed yet"
    agg.finalize()
    assert _stable_version(agg) > 1, "a clean run must publish its head"


# -- end to end: the demo that made this visible -------------------------------


def test_the_reference_run_actually_reaches_the_stable_branch():
    """`run_demo`'s headline table showed `stable` at 0.000 for all 40 rounds.

    The docs published a table where it caught up at round 8 -- output this code
    could not produce. This is the assertion that was missing.
    """
    universe = make_task_universe(seed=7)
    with tempfile.TemporaryDirectory() as repo:
        system = AgentDescent(repo, universe, n_workers=6, noise=0.15,
                              refresh_interval=2, seed=0)
        history = system.run(rounds=20)
        # inside the context: `final_accuracy` reads the ledger, which lives here
        assert history[-1].dev_accuracy > 0.9, "premise: dev should converge"
        assert history[-1].stable_accuracy > 0.9, (
            "the stable branch never caught up with dev; dual-branch promotion "
            f"(design §4.5) is not firing. stable={history[-1].stable_accuracy}")
        assert system.final_accuracy(branch=Ledger.STABLE) > 0.9
