"""EvoX Genesis port: the mechanism, one component at a time, offline.

Every test here is about a piece of the paper's model, not about the domain --
the domain is a compact stand-in and its numbers prove nothing about upstream.
What the numbers *can* prove is that the mechanism is the one described: the path
coordinate bounds authority, delegation does not advance the version, the parent's
verdict is the parent's, and two agents editing one file both survive.
"""

from __future__ import annotations

import pytest

from agentdescent.aggregator import AggregatorConfig, MergeOutcome, diffs_contradict
from agentdescent.defaults import DefaultAcceptance, DefaultConflict
from agentdescent.evolution import EvolvingArtifact, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.filetree import canonical
from agentdescent.policies import MergeContext, Policies, ProposalContext

from examples.genesis import _domain as domain
from examples.genesis._delegation import (Brief, Delegation, Edit,
                                          RecursiveDelegation, render_edits)
from examples.genesis._judge import ParentJudge
from examples.genesis._octopus import OctopusConflict, git_available, three_way
from examples.genesis._spatial import SpatialContract, parse_situated_edits
from examples.genesis._world import (CONTEXT_FILE, LocalWorld, WorldLog, owns,
                                     parse_routing, routing_entry)


# ---------------------------------------------------------------------------
# w = (v, p), and the two operations on it
# ---------------------------------------------------------------------------

def test_delegation_moves_the_path_and_not_the_version():
    """(v, p) -> (v, q). The paper's whole distinction from a serial loop."""
    parent = LocalWorld(version=7, path="src")
    child = parent.delegate("src/backend")
    assert (child.path, child.version) == ("src/backend", 7)


def test_delegation_cannot_leave_the_subtree():
    with pytest.raises(ValueError):
        LocalWorld(version=1, path="src/frontend").delegate("src/backend")


@pytest.mark.parametrize("raw", ["./", ".", "", "/"])
def test_the_root_is_one_path(raw):
    assert LocalWorld(version=1, path=raw).path == ""
    assert owns(raw, "anything/at/all.py")


def test_situate_inherits_the_context_chain_root_first():
    state = {CONTEXT_FILE: "# root", "src/CONTEXT.md": "# src",
             "src/backend/CONTEXT.md": "# backend", "src/backend/x.py": "x = 1"}
    text = LocalWorld(version=1, path="src/backend").situate(state)
    assert text.index("# root") < text.index("# src") < text.index("# backend")
    assert "src/backend/x.py" in text


# ---------------------------------------------------------------------------
# CONTEXT.md: inherited on the way in, maintained on the way out
# ---------------------------------------------------------------------------

_TABLE = """# src

## Intent
Everything under ./src/backend/ is mentioned here in prose, and that is all.

## Routing Table
- `./src/frontend/` -> tokenizer and parser

## Constraints
- `./spec/` is read-only.
"""


def test_only_the_routing_section_confers_a_route():
    """A path named in prose is a mention; treating it as a route would let any
    sentence create authority."""
    assert parse_routing(_TABLE) == ["src/frontend"]


def test_a_node_delegates_to_what_its_context_md_routes_to():
    state = {"src/CONTEXT.md": _TABLE, "src/backend/evaluator.py": "x"}
    assert LocalWorld(version=1, path="src").routing(state) == ["src/frontend"]


def test_a_node_without_a_table_falls_back_to_the_directories_that_exist():
    """A repository whose CONTEXT.md tree is not written yet stays usable."""
    state = {"src/CONTEXT.md": "# src\n", "src/backend/e.py": "x", "src/frontend/l.py": "y"}
    assert LocalWorld(version=1, path="src").routing(state) == ["src/backend", "src/frontend"]


def test_a_routing_entry_is_written_once_and_only_once():
    """*Current state, not history* -- the table must not become a transcript."""
    once = routing_entry(_TABLE, "src/backend", "evaluation")
    assert once is not None and once.count("`./src/backend/`") == 1
    assert routing_entry(once, "src/backend", "evaluation") is None


def test_a_manager_that_opens_an_unrouted_node_records_it_at_its_own_level():
    """A node its parent does not route to is a node later agents cannot find."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/backend", "evaluate the parse tree")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, f"{b.world.path}/e.py", "x = 1\n")],
        log=log, max_depth=3, root_path="src")
    proposals = policy.propose(_proposal_ctx({"src/CONTEXT.md": _TABLE},
                                             Task(id="t", prompt="x")))
    edits = {e["path"]: e["content"] for e in parse_situated_edits(proposals[0])}
    assert "src/backend/e.py" in edits, "the child's work must survive"
    assert "src/backend" in parse_routing(edits["src/CONTEXT.md"])
    assert policy.routes_opened == 1


def test_a_routing_note_is_trimmed_before_a_source_file():
    """Bookkeeping must never cost the change it is describing."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/a", "x"), Delegation("src/b", "y")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, f"{b.world.path}/f{i}.py", str(i))
                            for i in range(2)],
        log=log, max_depth=3, max_edits=4, root_path="src")
    edits = parse_situated_edits(policy.propose(
        _proposal_ctx({"src/CONTEXT.md": _TABLE}, Task(id="t", prompt="x")))[0])
    assert len(edits) == 4 and policy.truncated == 1
    assert all(not e["path"].endswith(CONTEXT_FILE) for e in edits)


def test_the_offline_manager_takes_its_decomposition_from_context_md():
    """The table is the mechanism: edit it and the delegation follows."""
    state = domain.initial_files()
    brief = Brief(world=LocalWorld(version=1, path=""), objective="o",
                  context="", state=state, task=Task(id="t", prompt="1 + 2"),
                  output="", reward=0.0, depth=0)
    assert [d.path for d in domain.offline_manager(brief)] == ["src"]

    rerouted = dict(state)
    rerouted["CONTEXT.md"] = state["CONTEXT.md"].replace("`./src/`", "`./src/frontend/`")
    brief = Brief(world=LocalWorld(version=1, path=""), objective="o",
                  context="", state=rerouted, task=Task(id="t", prompt="1 + 2"),
                  output="", reward=0.0, depth=0)
    assert [d.path for d in domain.offline_manager(brief)] == ["src/frontend"]


# ---------------------------------------------------------------------------
# The spatial contract
# ---------------------------------------------------------------------------

def _strategy(**kwargs) -> SpatialContract:
    return SpatialContract(initial_files={CONTEXT_FILE: "# root"},
                           frozen=("spec/**",), **kwargs)


def test_an_edit_outside_the_authors_subtree_is_dropped_and_counted():
    log = WorldLog()
    strategy = _strategy(log=log)
    proposal = render_edits([Edit("src/frontend", "src/backend/evaluator.py", "x")], "r")
    assert strategy.to_diff(strategy.initial(), proposal, "w0", 1, "world") is None
    assert log.contract_violations == 1


def test_a_frozen_path_is_refused_even_to_the_root():
    strategy = _strategy()
    proposal = render_edits([Edit("", "spec/CONTEXT.md", "rewritten")], "r")
    assert strategy.to_diff(strategy.initial(), proposal, "w0", 1, "world") is None


def test_a_new_file_inside_the_subtree_is_an_ordinary_edit():
    """The property tensor parallelism cannot give: creation is what formation is.

    Under TP the ownership map is fixed before round 0, so a path that is not in
    it belongs to no section and every creation is a `section-violation`. Here it
    is just an edit.
    """
    strategy = _strategy()
    proposal = render_edits([Edit("src", "src/brand/new/file.py", "print(1)\n")], "r")
    diff = strategy.to_diff(strategy.initial(), proposal, "w0", 1, "world")
    assert diff is not None and diff.ops == {"src/brand/new/file.py": "print(1)\n"}


def test_the_strategy_declares_no_key_space_so_tp_is_refused_rather_than_lossy():
    """`evolve()` refuses TP for a strategy with no `keys()`, which is the point.

    Declaring a partial key space would let TP run and silently discard every
    file the run created -- the failure this port is built to avoid.
    """
    assert not hasattr(_strategy(), "keys")


@pytest.mark.parametrize("existing,attempt", [
    ("src/lexer.py", "src/lexer.py/__init__.py"),   # an ancestor is a file
    ("src/pkg/a.py", "src/pkg"),                     # the target is a directory
])
def test_an_edit_that_would_stop_the_tree_being_a_tree_is_dropped(existing, attempt):
    """`a/b.py` and `a/b.py/c.py` cannot both exist, and the engine never checks.

    Found the expensive way: an agent delegated to `src/frontend/lexer.py` as
    though it were a node, wrote inside it, and `EvolutionResult.write_to` raised
    `FileExistsError` after 40 episodes and 328 model calls -- the whole run lost
    at the one point where it was being saved.
    """
    log = WorldLog()
    strategy = _strategy(log=log)
    state = dict(strategy.initial(), **{existing: "x = 1\n"})
    proposal = render_edits([Edit("src", attempt, "y = 2\n")], "r")
    assert strategy.to_diff(state, proposal, "w0", 1, "world") is None
    assert log.shape_violations == 1


def test_a_delegation_to_a_file_is_refused_because_a_node_is_a_directory():
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: [Delegation("src/lexer.py", "fix the lexer")],
        executor=lambda b: [Edit(b.world.path, f"{b.world.path}/x.py", "1")],
        log=log, max_depth=3, root_path="src")
    state = {"src/CONTEXT.md": "# src\n", "src/lexer.py": "x = 1\n"}
    policy.propose(_proposal_ctx(state, Task(id="t", prompt="x")))
    assert policy.mistaken_nodes == 1
    assert [e.role for e in log.episodes] == ["executor"], \
        "the refused delegation must not open an episode"


def test_a_reply_that_ignores_the_protocol_costs_its_episode_and_does_not_crash():
    assert parse_situated_edits("I would rather not.") == []
    assert _strategy().to_diff({}, "I would rather not.", "w0", 1, "world") is None


# ---------------------------------------------------------------------------
# The octopus merge
# ---------------------------------------------------------------------------

_BASE = "def a():\n    raise NotImplementedError\n\n\ndef b():\n    raise NotImplementedError\n"
_OURS = _BASE.replace("def a():\n    raise NotImplementedError", "def a():\n    return 1")
_THEIRS = _BASE.replace("def b():\n    raise NotImplementedError", "def b():\n    return 2")


@pytest.mark.skipif(not git_available(), reason="git merge-file is the merge engine")
def test_three_way_merges_disjoint_hunks_and_conflicts_on_an_overlap():
    merged = three_way(_BASE, _OURS, _THEIRS)
    assert merged is not None and "return 1" in merged and "return 2" in merged
    rival = _BASE.replace("def a():\n    raise NotImplementedError", "def a():\n    return 9")
    assert three_way(_BASE, _OURS, rival) is None


class _Verifier:
    """Enough of the verifier protocol for `DefaultConflict` to rank two cards."""

    def cheap_eval(self, artifact) -> float:
        return float(len(artifact.state.get("f.py", "")))


def _cards():
    artifact = EvolvingArtifact("world", state={"f.py": _BASE})
    ours = EvidenceCard(diff=Diff("d1", "world", {"f.py": _OURS}), base_version={}, touched=["f.py"])
    theirs = EvidenceCard(diff=Diff("d2", "world", {"f.py": _THEIRS}), base_version={}, touched=["f.py"])
    return artifact, [ours, theirs]


@pytest.mark.skipif(not git_available(), reason="git merge-file is the merge engine")
def test_octopus_keeps_both_edits_where_the_keyed_union_drops_one():
    """The comparison the port exists to make runnable.

    One file, two agents, two different functions. The engine's rule is "same
    key, different value -> contradiction", so one of them is dropped on score.
    Upstream three-way merges them and keeps both.
    """
    artifact, cards = _cards()
    assert diffs_contradict(cards[0].diff, cards[1].diff)

    kept_default, dropped = DefaultConflict(_Verifier()).resolve(artifact, list(cards))
    assert (len(kept_default), dropped) == (1, 1)

    policy = OctopusConflict(inner=DefaultConflict(_Verifier()))
    kept, dropped = policy.resolve(artifact, list(cards))
    assert dropped == 0 and policy.merged == 1
    merged = kept[0].diff.ops["f.py"]
    assert "return 1" in merged and "return 2" in merged


def test_octopus_defers_a_real_overlap_to_the_inner_rule():
    artifact, cards = _cards()
    rival = _BASE.replace("def a():\n    raise NotImplementedError", "def a():\n    return 9")
    cards[1] = EvidenceCard(diff=Diff("d2", "world", {"f.py": rival}),
                            base_version={}, touched=["f.py"])
    policy = OctopusConflict(inner=DefaultConflict(_Verifier()))
    kept, dropped = policy.resolve(artifact, cards)
    assert policy.merged == 0 and policy.conflicted == 1
    assert len(kept) == 1 and dropped == 1


# ---------------------------------------------------------------------------
# The parent's verdict
# ---------------------------------------------------------------------------

def _ctx(base, cand, ops=None):
    artifact = EvolvingArtifact("world", state={})
    return MergeContext(artifact=artifact, candidate=artifact, cards=[],
                        base_counts=base, cand_counts=cand,
                        diff=Diff("d", "world", ops or {"src/a.py": "x"}))


def _configured(policy):
    policy.configure(AggregatorConfig())
    return policy


def test_disabled_it_is_the_engines_rule_exactly():
    """`--engine-gate` has to be the shipped gate, not an imitation of it."""
    judge = _configured(ParentJudge(enabled=False))
    shipped = _configured(DefaultAcceptance())
    for base, cand in (((5, 5), (9, 1)), ((5, 5), (5, 5)), ((5, 5), (1, 9))):
        mine, theirs = judge.accept(_ctx(base, cand)), shipped.accept(_ctx(base, cand))
        assert (mine.accept, mine.category, mine.detail) == \
               (theirs.accept, theirs.category, theirs.detail)


def test_partial_progress_is_accepted_and_asks_for_more_work():
    """Upstream's rule, verbatim: a tie is progress, and the subtree is revisited."""
    log = WorldLog()
    judge = ParentJudge(log=log, enabled=True)
    decision = judge.accept(_ctx((5, 5), (5, 5), {"src/backend/a.py": "x",
                                                  "src/backend/b.py": "y"}))
    assert decision.accept and judge.partial == 1
    assert set(log.pending_rework) == {"src/backend"}


def test_a_regression_is_refused():
    judge = ParentJudge(enabled=True)
    assert judge.accept(_ctx((5, 5), (2, 8))).accept is False
    assert judge.rejected == 1


def test_every_verdict_is_a_category_the_aggregator_can_name():
    """`MergeOutcome` is a closed vocabulary: an invented category raises mid-merge."""
    judge = ParentJudge(log=WorldLog(), enabled=True)
    for base, cand in (((5, 5), (9, 1)), ((5, 5), (5, 5)), ((5, 5), (1, 9))):
        MergeOutcome(judge.accept(_ctx(base, cand)).category)


# ---------------------------------------------------------------------------
# The recursion
# ---------------------------------------------------------------------------

def _proposal_ctx(state, task, reward=0.0):
    return ProposalContext(rendered=canonical(state), task=task, output="",
                           reward=reward, base_version=3)


def test_one_rollout_is_a_tree_of_episodes_at_one_version():
    log = WorldLog()
    policy = RecursiveDelegation(manager=domain.offline_manager,
                                 executor=domain.offline_executor, log=log,
                                 max_depth=3)
    state = domain.initial_files()
    assert policy.propose(_proposal_ctx(state, Task(id="t", prompt="1 + 2")))
    episodes = log.episodes
    assert [e.role for e in episodes][:2] == ["manager", "executor"]
    assert {e.version for e in episodes} == {3}, "delegation must not move the version"
    assert log.observed_depth() >= 1


def test_a_child_that_returns_nothing_is_sent_back_rather_than_annotated():
    """"Try again" is queued; only a refusal is written into the world."""
    log = WorldLog()
    policy = RecursiveDelegation(manager=lambda b: [Delegation("src", "do nothing")],
                                 executor=lambda b: [], log=log, max_depth=2)
    assert policy.propose(_proposal_ctx({CONTEXT_FILE: "# root"},
                                        Task(id="t", prompt="x"))) == []
    assert [e.verdict for e in log.episodes if e.path == "src"] == ["rework"]
    assert set(log.pending_rework) == {"src"}


def test_a_refused_child_leaves_its_reason_in_the_nodes_context_md():
    """Appendix 1.4: the rejected code does not survive, the saved reason can."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src", "write outside your path")]
                           if b.world.path == "" else []),
        executor=lambda b: [Edit(b.world.path, "elsewhere/x.py", "nope")],
        log=log, max_depth=2)
    proposals = policy.propose(_proposal_ctx({CONTEXT_FILE: "# root"},
                                             Task(id="t", prompt="x")))
    edits = parse_situated_edits(proposals[0])
    assert [e["path"] for e in edits] == [f"src/{CONTEXT_FILE}"]
    assert "refused:" in edits[0]["content"]
    assert [e.verdict for e in log.episodes if e.path == "src"] == ["rejected"]


def test_an_outstanding_rework_request_is_taken_before_a_free_choice():
    log = WorldLog()
    log.request_rework("src/backend", "keep going here")
    seen = []
    policy = RecursiveDelegation(manager=lambda b: (seen.append(b.world.path) or []),
                                 executor=lambda b: [], log=log, max_depth=2)
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root"}, Task(id="t", prompt="x")))
    assert seen == ["src/backend"]
    assert log.pending_rework == {}


# ---------------------------------------------------------------------------
# The domain, and one end-to-end formation
# ---------------------------------------------------------------------------

def test_the_frozen_suite_agrees_with_its_own_oracle():
    tasks = domain.build_tasks()
    run = domain.make_runner()
    rendered = canonical(domain._reference_tree())
    assert sum(domain.reward(t, run(rendered, t)) for t in tasks) == len(tasks)


def test_the_repository_starts_with_no_implementation_in_it():
    files = domain.initial_files()
    assert files and all(p.endswith(".md") for p in files)
    tasks = domain.build_tasks()
    run = domain.make_runner()
    rendered = canonical(files)
    assert sum(domain.reward(t, run(rendered, t)) for t in tasks) == 0


def test_formation_grows_a_working_toolchain_from_the_empty_repository():
    """End to end, offline: the parent's rule takes 0.000 to a working world."""
    tasks = domain.build_tasks()
    log = WorldLog()
    strategy = SpatialContract(initial_files=domain.initial_files(),
                               frozen=domain.FROZEN, log=log, max_files_per_diff=6)
    delegation = RecursiveDelegation(manager=domain.offline_manager,
                                     executor=domain.offline_executor, log=log,
                                     max_depth=3)

    def superseded(rendered, task, output, reward):  # replaced by the bundle
        raise AssertionError("the proposal policy was not installed")

    result = evolve(
        tasks, domain.reward, run=domain.make_runner(), propose=superseded,
        strategy=strategy, artifact_id="world", blast_radius=0.2,
        rounds=12, n_workers=4, max_concurrency=4, self_verify=False,
        held_out_frac=0.4, seed=0,
        policies=Policies(proposal=delegation,
                          acceptance=ParentJudge(log=log, enabled=True),
                          conflict=OctopusConflict()))

    assert result.error is None
    assert result.final_reward > 0.9, result.outcomes()
    assert log.observed_depth() >= 2, "the run never delegated past the first level"
    assert log.contract_violations == 0
