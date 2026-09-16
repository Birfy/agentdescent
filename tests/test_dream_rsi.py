"""Offline tests for the Dream-RSI port: the replay simulator and the RSI loop.

No model, no network, no sandbox. The discovery agent is a seeded synthetic
world, the engine is the real ``evolve()`` underneath ``meta_evolve``, and the
policy-development agent is a script.

What is pinned here is the part that has to be exactly right for a replay score
to mean anything: the legal set, the batch constraints, Equation 1, the
prefix-only rule, and the paper's one formal guarantee -- that the selected
policy is never worse than the deployed one on the fixed history.
"""

import json
import math

import pytest

from agentdescent.dream import (PARALLEL_REFINE_SEED, Attempt, DiscoveryTree,
                                ReplayObjective, ReplayWorld, SimulatorPool,
                                dream_rsi, exploration_policy, explore,
                                mean_replay_value, replay_problem,
                                replay_reflector, replay_value)
from agentdescent.meta import compile_policy_source, policy_source
from agentdescent.selection import SingleHead

from examples.dreamrsi import dream_rsi_worlds as port
from examples.dreamrsi._world import SOURCE, TARGET, discovery_agent


# -- fixtures -----------------------------------------------------------------


def seed_policy():
    spec = exploration_policy()
    return spec.compile(spec.render(spec.initial()))


def compiled(source):
    return exploration_policy(source).compile(source)


def hand_world():
    """Two branches of two attempts, one of them a failure. The smallest world
    that has a root to open, a frontier to refine, a branch that ends, and a
    node whose score is ``None``."""
    tree = DiscoveryTree.rooted(0.10, name="hand")
    a = tree.add(0, Attempt(0.40, True, {"fail_class": "ok"}))        # 1
    tree.add(a.index, Attempt(0.55, True, {"fail_class": "ok"}))      # 2
    b = tree.add(0, Attempt(None, False, {"fail_class": "compile_other"}))  # 3
    tree.add(b.index, Attempt(0.30, True, {"fail_class": "ok"}))      # 4
    return ReplayWorld(tree, name="hand")


def recorded(n=4, workers=3, rounds=5, family=SOURCE, seed=0):
    """A pool of worlds the seed policy recorded on the synthetic domain."""
    pool = SimulatorPool()
    policy, attempts = seed_policy(), 0
    for index in range(n):
        tree = explore(discovery_agent(family, seed=seed), policy,
                       n_workers=workers, max_rounds=rounds,
                       root_score=family.root_score, attempt_offset=attempts,
                       name=f"w{index}")
        attempts += len(tree) - 1
        pool.add(tree, name=tree.name)
    return pool


# -- the tree -----------------------------------------------------------------


def test_explore_records_a_paper_shaped_tree():
    """Only the root gets several children: a leaf is continued once and then
    stops being a leaf, which is the shape every replay depends on."""
    tree = explore(discovery_agent(SOURCE, seed=0), seed_policy(), n_workers=3,
                   max_rounds=4, root_score=SOURCE.root_score)
    assert tree.paper_shaped
    assert tree.unreachable == 0
    assert len(tree.children(0)) == 3, "W branches, one per worker, in round one"
    assert tree.decision_rounds == 4
    assert len(tree) == 1 + 3 * 4


def test_a_raising_discovery_agent_becomes_a_failed_attempt_not_a_lost_rollout():
    """The online rollout is the expensive half; one flaky provider call must
    not throw it away. A raising agent is recorded as the node it is."""
    def flaky(tree, parent, attempt):
        if attempt % 2:
            raise RuntimeError("provider flaked")
        return Attempt(0.5, True, {"fail_class": "ok"})

    tree = explore(flaky, seed_policy(), n_workers=2, max_rounds=2, root_score=0.1)
    failed = [n for n in tree.nodes[1:] if n.detail.get("fail_class") == "agent-error"]
    assert len(failed) == 2 and len(tree) == 5
    assert all(n.score is None and not n.valid for n in failed)
    assert "provider flaked" in failed[0].detail["error"]


def test_a_tree_survives_json():
    tree = explore(discovery_agent(SOURCE, seed=1), seed_policy(), n_workers=2,
                   max_rounds=3, root_score=SOURCE.root_score, name="w")
    back = DiscoveryTree.from_json(tree.to_json())
    assert [n.to_payload() for n in back.nodes] == [n.to_payload() for n in tree.nodes]
    assert (back.name, back.decision_rounds) == (tree.name, tree.decision_rounds)


def test_a_pool_survives_json():
    pool = recorded(n=2, workers=2, rounds=3)
    back = SimulatorPool.from_json(pool.to_json())
    assert [w.name for w in back] == [w.name for w in pool]
    assert len(back.worlds[0].tree) == len(pool.worlds[0].tree)


def test_a_re_expanded_interior_node_is_reported_as_unreachable():
    """A flat-PUCT tree is not paper-shaped, and the replay cannot reach the
    second child of an interior node. Saying so beats discarding it silently."""
    tree = DiscoveryTree.rooted(0.1)
    a = tree.add(0, Attempt(0.4))
    tree.add(a.index, Attempt(0.5))
    tree.add(a.index, Attempt(0.6))          # re-expansion of an interior node
    tree.add(3, Attempt(0.7))                # ...and its subtree
    assert not tree.paper_shaped
    assert tree.unreachable == 2


# -- the legal set and the batch constraints ----------------------------------


def test_the_legal_set_is_the_root_and_the_revealed_leaves():
    """``A(T) = {r} u {v in T : v is a leaf}`` -- Section 3, verbatim."""
    world = hand_world()
    seen = []

    class Recorder:
        def select(self, ctx, n):
            legal = {c.version for c in ctx.candidates if c.state["legal"] == "1"}
            shown = {c.version for c in ctx.candidates}
            children = {v: set(world.tree.children(v)) & shown for v in shown}
            expect = {v for v in shown if v == 0 or not children[v]}
            seen.append((legal, expect))
            return [c for c in ctx.candidates if c.version in legal][:n]

    world.replay(Recorder(), ReplayObjective(), n_workers=2, max_rounds=6)
    assert seen and all(got == want for got, want in seen)


def test_the_policy_never_sees_an_unrevealed_node():
    """The prefix-only rule, enforced by construction rather than asked for."""
    world = hand_world()
    revealed = {0}

    class Peeker:
        def select(self, ctx, n):
            assert {c.version for c in ctx.candidates} <= set(range(len(world.tree)))
            assert {c.version for c in ctx.candidates} == revealed
            batch = [c for c in ctx.candidates if c.state["legal"] == "1"][:n]
            for cand in batch:
                for kid in world.tree.children(cand.version):
                    if kid not in revealed:
                        revealed.add(kid)
                        break
            return batch

    world.replay(Peeker(), ReplayObjective(), n_workers=2, max_rounds=6)
    assert revealed == set(range(len(world.tree)))


def test_a_non_root_node_may_appear_in_a_batch_only_once():
    world = hand_world()

    class Doubler:
        def select(self, ctx, n):
            legal = [c for c in ctx.candidates if c.state["legal"] == "1"]
            frontier = [c for c in legal if c.parent is not None]
            if frontier:
                return [frontier[0], frontier[0]]        # the same cell twice
            return [legal[0], legal[0]]                  # two branches from the root

    replay = world.replay(Doubler(), ReplayObjective(), n_workers=4, max_rounds=6)
    opened = replay.rounds[0]
    assert len(opened.revealed) == 2, "the root is several cells, one per branch"
    later = [r for r in replay.rounds[1:] if r.batch]
    assert all(len(r.revealed) <= 1 for r in later)
    assert replay.illegal >= 1, "the duplicate non-root pick is counted"


def test_a_batch_is_capped_at_the_worker_count():
    world = hand_world()

    class Greedy:
        def select(self, ctx, n):
            legal = [c for c in ctx.candidates if c.state["legal"] == "1"]
            return legal * 5                              # far more than n

    replay = world.replay(Greedy(), ReplayObjective(), n_workers=1, max_rounds=6)
    assert all(len(r.batch) <= 1 for r in replay.rounds)
    assert replay.illegal > 0


def test_a_pick_outside_the_candidates_is_counted_not_raised():
    world = hand_world()

    class Inventive:
        def select(self, ctx, n):
            return [999, -1] + [c for c in ctx.candidates if c.state["legal"] == "1"][:1]

    replay = world.replay(Inventive(), ReplayObjective(), n_workers=3, max_rounds=4)
    assert replay.illegal >= 2
    assert replay.nodes >= 1, "the legal part of the batch still ran"


def test_a_barren_pick_reveals_nothing_and_ends_the_replay():
    """A leaf whose branch ended in the record is *legal* -- the paper's ``A(T)``
    does not consult the record -- and picking it is charged as a wasted worker
    rather than as an illegal move."""
    world = hand_world()

    class Stubborn:
        def select(self, ctx, n):
            deepest = [c for c in ctx.candidates if c.state["legal"] == "1"
                       and c.parent is not None]
            return deepest[-1:] if deepest else [ctx.candidates[0]]

    replay = world.replay(Stubborn(), ReplayObjective(), n_workers=1, max_rounds=8)
    assert replay.stop_reason == "exhausted"
    assert replay.barren == 1
    assert replay.illegal == 0


def test_an_empty_batch_stops_the_replay():
    class Quitter:
        def select(self, ctx, n):
            return []

    replay = hand_world().replay(Quitter(), ReplayObjective(), n_workers=2,
                                 max_rounds=8)
    assert (replay.stop_reason, replay.nodes, replay.decision_rounds) == \
        ("empty-batch", 0, 0)
    assert replay.quality == pytest.approx(0.10), "the root is always revealed"


def test_a_policy_that_raises_is_scored_rather_than_crashing_the_run():
    class Broken:
        def select(self, ctx, n):
            raise RuntimeError("boom")

    replay = hand_world().replay(Broken(), ReplayObjective(), n_workers=2,
                                 max_rounds=4)
    assert replay.stop_reason == "error" and "boom" in (replay.error or "")


# -- determinism, and what a seed is worth ------------------------------------


def test_replay_is_deterministic_and_ignores_the_problem_seed():
    """The condition ``docs/meta-evolution.md`` states for a seed not to be a
    replicate: on a deterministic problem, two seeds are one comparison."""
    world = hand_world()
    problem = replay_problem(world, ReplayObjective(), n_workers=2, max_rounds=6)
    policy = seed_policy()
    first, second = problem(policy, 0), problem(policy, 17)
    assert first.to_json() == second.to_json().replace('"seed":17', '"seed":0')


def test_a_policy_cannot_carry_state_from_one_replay_into_the_next():
    """``question.reset()`` in the paper's API; a fresh instance here."""
    counter = '''class Policy:
    def __init__(self):
        self.calls = 0

    def select(self, ctx, n):
        self.calls += 1
        if self.calls > 2:
            return []
        return [c for c in ctx.candidates if c.state.get("legal") == "1"][:n]
'''
    policy = compiled(counter)
    world = hand_world()
    a = world.replay(policy, ReplayObjective(), n_workers=2, max_rounds=6)
    b = world.replay(policy, ReplayObjective(), n_workers=2, max_rounds=6)
    assert a.revealed == b.revealed and a.value == b.value


def test_the_recording_policy_replays_its_own_world_exactly():
    """The tightest check on the whole simulator: replay the policy that built
    the tree and every node comes back, in the order it was created."""
    policy = seed_policy()
    tree = explore(discovery_agent(SOURCE, seed=3), policy, n_workers=3,
                   max_rounds=4, root_score=SOURCE.root_score)
    replay = ReplayWorld(tree).replay(policy, ReplayObjective(), n_workers=3,
                                      max_rounds=8)
    assert list(replay.revealed) == list(range(len(tree)))
    assert replay.quality == pytest.approx(tree.best())
    assert replay.nodes == len(tree) - 1


# -- Equation 1 ---------------------------------------------------------------


def test_equation_one_on_a_hand_checked_world():
    """``V = max score - b1 * N + b2 * N / max(1, k)``."""
    world = hand_world()
    objective = ReplayObjective(beta1=0.01, beta2=0.02)
    replay = world.replay(seed_policy(), objective, n_workers=2, max_rounds=8)
    expected = (replay.quality - 0.01 * replay.nodes
                + 0.02 * replay.nodes / max(1, replay.decision_rounds))
    assert replay.value == pytest.approx(expected)
    assert replay.parallelism == pytest.approx(
        replay.nodes / max(1, replay.decision_rounds))


def test_the_default_objective_is_discovery_quality_alone():
    """Both coefficients default to zero because the paper publishes neither."""
    objective = ReplayObjective()
    replay = hand_world().replay(seed_policy(), objective, n_workers=2, max_rounds=8)
    assert replay.value == pytest.approx(replay.quality)


def test_equal_weights_make_the_objective_blind_to_what_a_rollout_spent():
    """Why ``scaled`` does not default to equal weights.

    The cost and parallelism terms are together ``N * (b2 / k - b1)``, which
    vanishes at ``k = b2 / b1`` for **any** ``N``. Equal weights put that at
    ``max_nodes / n_workers`` -- the round budget -- so every policy that runs
    its rounds out is scored on quality alone however many continuations it
    spent, which is the one thing the objective exists to charge for.
    """
    equal = ReplayObjective.scaled(max_nodes=24, n_workers=4,
                                   cost_weight=0.25, parallel_weight=0.25)
    assert equal.beta2 / equal.beta1 == pytest.approx(24 / 4)
    for nodes in (6, 12, 24):
        assert equal.value(0.5, nodes, 6) == pytest.approx(0.5)
    default = ReplayObjective.scaled(max_nodes=24, n_workers=4)
    assert default.beta2 / default.beta1 == pytest.approx(0.10 * 24 / (0.25 * 4))
    spent = [default.value(0.5, nodes, 6) for nodes in (6, 12, 24)]
    assert spent[0] > spent[1] > spent[2], "the default charges per continuation"


def test_the_documented_weight_comparison_holds_on_the_example_world():
    """The table in `docs/algo-dream-rsi.md`, measured rather than asserted.

    One world, ``max_nodes=24``, ``n_workers=4``. The seed and the example's
    evolved policy reach the *same* quality; the evolved one spends a third
    fewer continuations in the same number of rounds. At equal weights that
    saving is worth exactly nothing -- both score the identical ``V`` -- and at
    the shipped default the cheaper policy is correctly ahead.
    """
    world = recorded(n=1, workers=4, rounds=6).worlds[0]
    seed, evolved = seed_policy(), compiled(port.OFFLINE_PROPOSAL)
    equal = ReplayObjective.scaled(max_nodes=24, n_workers=4,
                                   cost_weight=0.25, parallel_weight=0.25)
    default = ReplayObjective.scaled(max_nodes=24, n_workers=4)

    runs = {name: world.replay(policy, equal, n_workers=4, max_rounds=12)
            for name, policy in (("seed", seed), ("evolved", evolved))}
    assert runs["seed"].quality == pytest.approx(runs["evolved"].quality)
    assert runs["seed"].decision_rounds == runs["evolved"].decision_rounds == 6
    assert (runs["seed"].nodes, runs["evolved"].nodes) == (24, 16)
    assert runs["seed"].value == pytest.approx(runs["evolved"].value), (
        "equal weights make a third of the discovery budget worth nothing")

    cheap = world.replay(evolved, default, n_workers=4, max_rounds=12)
    dear = world.replay(seed, default, n_workers=4, max_rounds=12)
    assert cheap.value > dear.value, "the default charges for what was spent"


def test_the_replay_reward_is_bounded_and_keeps_the_ordering_of_raw_v():
    world = hand_world()
    objective = ReplayObjective.scaled(max_nodes=4, n_workers=2)
    problem = replay_problem(world, objective, n_workers=2, max_rounds=8)
    score = replay_value()
    quitter = compiled("class Policy:\n"
                       "    def select(self, ctx, n):\n"
                       "        return []\n")
    full, stop = problem(seed_policy(), 0), problem(quitter, 0)
    for outcome in (full, stop):
        assert 0.0 <= score(outcome) <= 1.0
    assert (score(full) > score(stop)) == (full.final > stop.final)
    assert full.detail["value_hi"] > full.detail["value_lo"]


def test_the_outcome_curve_is_best_so_far_so_auc_still_means_what_it_meant():
    replay = hand_world().replay(seed_policy(), ReplayObjective(), n_workers=2,
                                 max_rounds=8)
    curve = replay.curve()
    assert curve == sorted(curve), "best-so-far never falls"
    assert curve[-1] == pytest.approx(replay.quality)


# -- the gate -----------------------------------------------------------------


def test_the_seed_policy_is_the_papers_parallel_refining():
    """Round one opens ``W`` branches; every round after refines each of them."""
    world = ReplayWorld(explore(discovery_agent(SOURCE, seed=5), seed_policy(),
                                n_workers=4, max_rounds=4,
                                root_score=SOURCE.root_score))
    replay = world.replay(seed_policy(), ReplayObjective(), n_workers=4,
                          max_rounds=8)
    assert replay.rounds[0].batch == (0, 0, 0, 0), "W workspaces, opened at once"
    for record in replay.rounds[1:]:
        assert 0 not in record.batch, "then it refines, it does not re-open"
        assert len(record.batch) == 4


def test_the_exploration_smoke_permits_the_empty_batch_the_stock_one_forbids():
    """The one reason this slot needs a smoke test of its own: stopping is a
    decision an exploration policy is supposed to be able to make."""
    source = ("class Policy:\n"
              "    def select(self, ctx, n):\n"
              "        return []\n")
    exploration_policy(source).compile(source)                 # accepted
    with pytest.raises(ValueError):
        compile_policy_source("selection", source)             # the stock gate


@pytest.mark.parametrize("source, reason", [
    ("import os\nclass Policy:\n    def select(self, ctx, n):\n        return []\n",
     "import"),
    ("class Policy:\n    def select(self, ctx, n):\n        return open('x')\n",
     "forbidden call"),
    ("class Policy:\n    def select(self, ctx, n):\n        raise ValueError('x')\n",
     "smoke"),
    ("class Policy:\n    def select(self, ctx, n):\n        return list(ctx.candidates) * 9\n",
     "too many picks"),
    ("class Policy:\n    def select(self, ctx, n):\n        return ['not a candidate']\n",
     "not from the candidates"),
])
def test_the_gate_refuses(source, reason):
    with pytest.raises(ValueError):
        exploration_policy(source).compile(source)


def test_the_gate_refuses_a_policy_that_mutates_the_candidates_it_is_given():
    source = ("class Policy:\n"
              "    def select(self, ctx, n):\n"
              "        rows = ctx.candidates\n"
              "        rows.sort(key=lambda c: c.version)\n"
              "        return list(rows)[:n]\n")
    with pytest.raises(ValueError):
        exploration_policy(source).compile(source)


def test_the_engines_own_single_head_still_works_as_an_exploration_policy():
    """The slot is `selection`, so a stock policy has to remain usable in it.

    `SingleHead` is `[ctx.head] * n`, which here means "put every worker on the
    best node you have" -- greedy depth-first refinement, not a crash. Its
    second copy of a non-root head is over that node's multiplicity of one and
    is counted, which is the right answer rather than an error: the policy
    asked for a worker the tree has nowhere to put.
    """
    replay = hand_world().replay(SingleHead(), ReplayObjective(), n_workers=2,
                                 max_rounds=8)
    assert replay.nodes > 0 and replay.error is None
    assert replay.rounds[0].batch == (0, 0), "two branches opened from the root"


def test_an_evolved_policy_is_redeployable_through_the_ordinary_slot():
    """The value compiles into what ``Policies(selection=...)`` takes, which is
    what "redeploy the improved policy online" has to mean."""
    from agentdescent.policies import Policies
    policy = compiled(port.OFFLINE_PROPOSAL)
    assert Policies(selection=policy).selection is policy
    assert policy_source("selection", PARALLEL_REFINE_SEED,
                         smoke=lambda _: None) is not None


# -- the outer loop -----------------------------------------------------------


def test_the_selected_policy_is_never_worse_on_the_fixed_history():
    """The paper's only formal guarantee: the candidate set contains the current
    policy, so ``V(pi_{t+1}) >= V(pi_t)`` on ``H_t``."""
    result = dream_rsi(discovery_agent(SOURCE, seed=0),
                       model=port.offline_model(), rounds=2, n_workers=3,
                       online_rounds=4, replay_rounds=8, online_repeats=6,
                       dream_rounds=1, dream_workers=2,
                       root_score=SOURCE.root_score, held_out_frac=0.4)
    assert result.rounds
    for record in result.rounds:
        assert record.value_after >= record.value_before


#: A plausible-looking proposal that the dreaming gate commits and the paper's
#: argmax refuses. It prunes on the frontier node's own **latest** score, so a
#: branch whose last attempt happened to fail is dropped along with its history
#: -- see `examples/dreamrsi/dream_rsi_worlds.OFFLINE_PROPOSAL`, which ranks on
#: the branch's best instead and is the version that survives both checks.
RANK_ON_LATEST = """class Policy:
    def select(self, ctx, n):
        legal = [c for c in ctx.candidates if c.state.get("legal") == "1"]
        if not legal:
            return []
        roots = [c for c in legal if c.parent is None]
        frontier = [c for c in legal if c.parent is not None]
        if not frontier:
            return ([roots[0]] * n) if roots else []
        frontier.sort(key=lambda c: (c.score is None, -(c.score or 0.0)))
        shallow = min(c.per_task.get("depth", 0.0) for c in frontier)
        if shallow < 2.0 and len(frontier) >= 2:
            return frontier[:n]
        return frontier[:max(1, n // 2)]
"""


def test_a_regression_the_dreaming_gate_admits_is_still_refused():
    """The argmax is not the same thing as the gate, and this is why it is there.

    ``meta_evolve`` commits on ``held_out_frac`` of the pool; the selection step
    re-scores the committed value on **all** of it. This is not hypothetical --
    it is the case `docs/algo-dream-rsi.md` quotes: the gate's three held-out
    worlds commit :data:`RANK_ON_LATEST`, and on the whole pool it scores below
    the policy it would have replaced.
    """
    from agentdescent.meta import meta_evolve

    spec = exploration_policy()
    rendered = spec.render(spec.initial())
    pool = recorded(n=8, workers=4, rounds=6)
    objective = ReplayObjective.scaled(max_nodes=24, n_workers=4)
    base = mean_replay_value(spec.compile(rendered), pool, objective,
                             n_workers=4, max_rounds=12)

    def model(prompt):
        return "```python\n" + RANK_ON_LATEST + "```"

    evolved = meta_evolve(pool.problems(objective, n_workers=4, max_rounds=12),
                          slot="selection", spec=exploration_policy(rendered),
                          propose=replay_reflector(model, spec),
                          meta_reward=replay_value(), seeds=[0], rounds=3,
                          n_workers=4, held_out_frac=0.4)
    assert evolved.rendered.strip() != rendered.strip(), "the gate committed it"
    assert mean_replay_value(spec.compile(evolved.rendered), pool, objective,
                             n_workers=4, max_rounds=12) < base

    result = dream_rsi(discovery_agent(SOURCE, seed=0), model=model, rounds=1,
                       n_workers=4, online_rounds=6, replay_rounds=12,
                       online_repeats=8, dream_rounds=3, dream_workers=4,
                       root_score=SOURCE.root_score, held_out_frac=0.4)
    assert not result.rounds[0].redeployed, "the argmax kept the regression out"
    assert result.rendered.strip() == rendered.strip()


def test_the_loop_redeploys_a_better_policy_and_the_next_rollout_costs_less():
    """End to end: dream, select, redeploy, and the online bill falls."""
    result = dream_rsi(discovery_agent(SOURCE, seed=0),
                       model=port.offline_model(), rounds=2, n_workers=4,
                       online_rounds=6, replay_rounds=12, online_repeats=8,
                       dream_rounds=2, dream_workers=2,
                       root_score=SOURCE.root_score, held_out_frac=0.4)
    first, second = result.rounds[0], result.rounds[1]
    assert first.redeployed, "the scripted policy scores above the seed on the pool"
    assert second.online_nodes < first.online_nodes
    assert result.rendered.strip() == port.OFFLINE_PROPOSAL.strip()
    assert len(result.pool) == 16


def test_dreaming_is_skipped_until_the_pool_can_be_split():
    result = dream_rsi(discovery_agent(SOURCE, seed=0),
                       model=port.offline_model(), rounds=2, n_workers=2,
                       online_rounds=3, replay_rounds=6, online_repeats=1,
                       min_worlds=4, dream_rounds=1, dream_workers=2,
                       root_score=SOURCE.root_score)
    assert [r.stop_reason for r in result.rounds] == ["pool-too-small"] * 2
    assert all(not r.redeployed for r in result.rounds)


def test_the_loop_needs_a_proposer():
    with pytest.raises(ValueError):
        dream_rsi(discovery_agent(SOURCE, seed=0), rounds=1)


def test_the_loop_refuses_to_have_its_own_arguments_passed_through():
    with pytest.raises(TypeError):
        dream_rsi(discovery_agent(SOURCE, seed=0), model=port.offline_model(),
                  rounds=1, slot="acceptance")


# -- the reflector ------------------------------------------------------------


def test_the_reflector_shows_the_trajectory_and_the_decomposition():
    """What separates it from `slot_reflector`: a policy that lost on cost is
    told so, round by round, rather than handed one number."""
    seen = []

    def model(prompt):
        seen.append(prompt)
        return "```python\n" + port.OFFLINE_PROPOSAL + "```"

    spec = exploration_policy()
    propose = replay_reflector(model, spec)
    world = hand_world()
    outcome = replay_problem(world, ReplayObjective.scaled(max_nodes=4, n_workers=2),
                             n_workers=2, max_rounds=6)(seed_policy(), 0)

    class _Task:
        prompt = "hand (seed 0)"

    propose(spec.render(spec.initial()), _Task(), outcome.to_json(), 0.42)
    propose(spec.render(spec.initial()), _Task(), outcome.to_json(), 0.51)
    assert "continuations spent (N)" in seen[0]
    assert "Round by round" in seen[0]
    assert "mean batch size" in seen[0]
    assert "Earlier revisions" in seen[1], "feedback from earlier revisions"
    assert "0.420" in seen[1]


# -- the example --------------------------------------------------------------


def test_the_example_world_saturates_so_a_cheaper_policy_can_exist():
    """The property the whole offline demonstration rests on: a branch stops
    paying. Without it, "reveal everything" is optimal and the recording policy
    wins by construction."""
    tree = explore(discovery_agent(SOURCE, seed=0), seed_policy(), n_workers=1,
                   max_rounds=8, root_score=SOURCE.root_score)
    scores = [n.score for n in tree.nodes[1:] if n.score is not None]
    early = scores[1] - scores[0]
    late = scores[-1] - scores[-2]
    assert early > late, f"the branch never flattened: {scores}"


def test_the_offline_proposal_beats_the_seed_on_worlds_the_seed_recorded():
    pool = recorded(n=8, workers=4, rounds=6)
    objective = ReplayObjective.scaled(max_nodes=24, n_workers=4)
    base = mean_replay_value(seed_policy(), pool, objective, n_workers=4,
                             max_rounds=12)
    better = mean_replay_value(compiled(port.OFFLINE_PROPOSAL), pool, objective,
                               n_workers=4, max_rounds=12)
    assert better > base


def test_the_offline_proposal_trades_quality_for_calls_and_the_report_says_which():
    report = port.validate(port.OFFLINE_PROPOSAL, family_name=TARGET.name,
                           workers=4, online_rounds=6, seeds=list(range(40)),
                           cost_weight=0.25, parallel_weight=0.10)
    assert report["call_ratio"] > 1.0, "it is cheaper"
    assert report["seed_calls"] == 24.0
    assert set(report) == {"seed_best", "seed_calls", "evolved_best",
                           "evolved_calls", "quality_delta", "call_ratio"}
    assert math.isfinite(report["quality_delta"])


def test_the_entry_point_runs_offline_end_to_end(capsys, tmp_path):
    out = tmp_path / "result.json"
    pool_out = tmp_path / "pool.json"
    code = port.main(["--offline", "--rounds", "1", "--workers", "3",
                      "--online-rounds", "4", "--repeats", "6",
                      "--dream-rounds", "1", "--validate-seeds", "4",
                      "--out", str(out), "--pool-out", str(pool_out)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["offline"] is True and payload["worlds"] == 6
    assert SimulatorPool.from_json(pool_out.read_text()).worlds
    assert "agent calls" in capsys.readouterr().out


def test_the_entry_point_dry_run_touches_nothing(capsys):
    assert port.main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "no model API was accessed" in printed
    assert "Parallel refining" in printed
