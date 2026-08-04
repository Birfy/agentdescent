"""The reference domain, expressed for the general engine.

`RouterSkill` + `Worker` are an `Evolvable` and a rollout written for the
reference runtime. `RouterStrategy` + `router_run` / `router_reward` /
`router_propose` say the same thing in the vocabulary `evolve()` speaks, so the
domain stops needing a second engine to run it (#93).

These pin the two things that make the translation faithful rather than merely
working, both of which were wrong first and measured wrong before they were
fixed.
"""

import warnings

import pytest

from agentdescent import evolve
from agentdescent.domains.router import (
    RouterStrategy,
    cluster_split_frac,
    cluster_tasks,
    make_task_universe,
    parse_table,
    render_table,
    router_propose,
    router_reward,
    router_run,
)


def _run_ported(universe, *, noise=0.1, rounds=40, seed=1):
    train, held = cluster_tasks(universe, n_clusters=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(
            train + held, router_reward, run=router_run,
            propose=router_propose(universe.gold, noise=noise, seed=seed),
            strategy=RouterStrategy(), artifact_id="mol-router",
            rounds=rounds, n_workers=6, max_concurrency=1,
            held_out_frac=cluster_split_frac(train, held),
            solved_threshold=0.999, seed=seed)


# -- the representation ---------------------------------------------------------


def test_the_table_round_trips_through_render():
    """`render` is the only channel the engine has to the artifact, and it is
    also the evaluation cache key -- a lossy one makes two tables share a score."""
    table = {"kw01": "thermo", "kw02": "kinetics"}
    assert parse_table(render_table(table)) == table
    assert parse_table(render_table({})) == {}


def test_a_proposal_becomes_at_most_max_ops():
    """The reference `Worker.max_ops` bound, preserved: a rollout fixes several
    keywords at once, which is what gives the fusion tournament complementary
    diffs to work with."""
    strategy = RouterStrategy(max_ops=2)
    diff = strategy.to_diff({}, "kw01: a\nkw02: b\nkw03: c", "w0", 1, "art")
    assert diff is not None and len(diff.ops) == 2


def test_a_proposal_that_changes_nothing_makes_no_diff():
    strategy = RouterStrategy()
    assert strategy.to_diff({"kw01": "a"}, "kw01: a", "w0", 1, "art") is None


# -- the split ------------------------------------------------------------------


def test_train_and_held_out_share_their_keywords():
    """Clusters bucket by *keyword* hash, so splitting the clusters positionally
    gives the two sides disjoint keyword sets and a skill learned on one can
    never score on the other.

    Measured before this was fixed: 6 commits, a rising dev accuracy, and a
    held-out reward pinned at 0.000 for the whole run. The train side keeps
    keyword clusters (the long-tail structure cluster leasing needs); the
    held-out side is chunked round-robin over the same keywords.
    """
    train, held = cluster_tasks(make_task_universe(seed=7), n_clusters=6)
    train_kw = {k for t in train for k in t.meta["keywords"]}
    held_kw = {k for t in held for k in t.meta["keywords"]}
    assert held_kw, "premise: the held-out side has keywords at all"
    assert not (held_kw - train_kw), (
        f"held-out keywords the skill can never learn: {sorted(held_kw - train_kw)}")


def test_the_split_fraction_lands_exactly_where_the_clusters_do():
    """`evolve()` splits positionally at `round(n * (1 - frac))`."""
    train, held = cluster_tasks(make_task_universe(seed=7), n_clusters=6)
    total = len(train) + len(held)
    assert round(total * (1 - cluster_split_frac(train, held))) == len(train)


# -- the corrector --------------------------------------------------------------


def test_noise_is_a_rate_not_a_permanent_mask():
    """Seeding the draw per (task, keyword) is deterministic *and* wrong: the
    draw for a keyword never changes, so a keyword the noise corrupts stays
    corrupted, `to_diff` sees the state already equals it, and the entry can
    never be corrected.

    Measured with the per-keyword seed: all 24 keywords learned, three committed
    wrong, proposals stopped entirely, held-out plateaued at 0.853 while the
    reference reached 1.000.
    """
    universe = make_task_universe(seed=7)
    propose = router_propose(universe.gold, noise=0.5, seed=1)
    train, _ = cluster_tasks(universe, n_clusters=6)
    empty = RouterStrategy().render({})

    seen = set()
    for _ in range(40):
        reply = propose(empty, train[0], "", 0.0)
        if reply:
            seen.add(reply)
    assert len(seen) > 1, (
        "the corrector produced one fixed answer for one state, so a noisy "
        "label could never be corrected by a later attempt")


def test_a_clean_corrector_proposes_only_gold():
    universe = make_task_universe(seed=7)
    propose = router_propose(universe.gold, noise=0.0, seed=1)
    train, _ = cluster_tasks(universe, n_clusters=6)
    reply = propose(RouterStrategy().render({}), train[0], "", 0.0)
    for line in reply.splitlines():
        keyword, _, label = line.partition(":")
        assert universe.gold[keyword.strip()] == label.strip()


# -- the loop -------------------------------------------------------------------


def test_the_ported_domain_learns_the_table():
    """The whole point: the general engine drives this domain to the same place
    the reference runtime does."""
    universe = make_task_universe(seed=7)
    result = _run_ported(universe)
    assert result.error is None
    assert result.final_reward > 0.95, (
        f"the ported loop plateaued at {result.final_reward:.3f}")
    assert result.history[0].held_out_reward < result.final_reward


def test_merge_still_beats_fork():
    """RQ1, on the ported path: no single fork sees every keyword, so merging
    complementary diffs has to win."""
    from agentdescent.orchestrator import run_fork_baseline

    universe = make_task_universe(seed=7)
    merged = _run_ported(universe).final_reward
    fork = run_fork_baseline(universe, n_workers=6, noise=0.1, rounds=40, seed=1)
    assert merged > fork, f"merge {merged:.3f} did not beat fork {fork:.3f}"


def test_it_reaches_the_same_place_as_the_reference_runtime():
    """Not a numeric golden -- the two measure at different points, and the port
    changes what a `before_after_delta` is measured over. What has to hold is
    that both converge on the table."""
    import tempfile

    from agentdescent.orchestrator import AgentDescent

    universe = make_task_universe(seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = AgentDescent(tempfile.mkdtemp() + "/repo", universe,
                                 n_workers=6, noise=0.1, seed=1)
        reference.run(rounds=40)
    assert abs(reference.final_accuracy() - _run_ported(universe).final_reward) < 0.05
