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
from agentdescent.filetree import canonical, match_any
from agentdescent.policies import MergeContext, Policies, ProposalContext

from examples.genesis import _domain as domain
from examples.genesis import _jqx as jqx
from examples.genesis import _md as md
from examples.genesis import _stackvm as stackvm
from examples.genesis._delegation import (Brief, Delegation, Edit,
                                          RecursiveDelegation, render_edits)
from examples.genesis._judge import ParentJudge
from examples.genesis._octopus import OctopusConflict, git_available, three_way
from examples.genesis._spatial import SpatialContract, parse_situated_edits
from examples.genesis._world import (CONTEXT_FILE, SKILLS_DIR, TRUNCATED,
                                     LocalWorld, WorldLog, owns, parse_routing,
                                     resolve_edit_path, routing_entry)


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
        log=log, max_depth=3, max_edits=4, root_path="src",
        accountability=False)
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


_TREE = {CONTEXT_FILE: "r", "src/CONTEXT.md": "s", "src/frontend/CONTEXT.md": "f",
         "src/frontend/lexer.py": "old", "spec/CONTEXT.md": "p"}


@pytest.mark.parametrize("owner,written,expected,relative", [
    # the defect this exists for: a node-relative filename read as
    # repository-relative sent real work to the top of the repository
    ("src/frontend", "lexer.py", "src/frontend/lexer.py", True),
    ("src/frontend", "__init__.py", "src/frontend/__init__.py", True),
    ("src", "backend/evaluator.py", "src/backend/evaluator.py", True),
    ("src/frontend", "CONTEXT.md", "src/frontend/CONTEXT.md", True),
    # ...and the case it must never mangle: a genuine request about another node
    ("src/frontend", "src/__init__.py", "src/__init__.py", False),
    ("src/frontend", "src/frontend/parser.py", "src/frontend/parser.py", False),
    ("", "anything.py", "anything.py", False),
])
def test_a_path_is_resolved_against_the_node_the_agent_stands_in(
        owner, written, expected, relative):
    assert resolve_edit_path(_TREE, owner, written) == (expected, relative)


def test_skills_are_inherited_along_the_chain_like_context():
    """The paper lists reusable skills among what an accepted version carries."""
    state = {f"{SKILLS_DIR}/house.md": "a", f"src/{SKILLS_DIR}/modules.md": "b",
             f"src/other/{SKILLS_DIR}/not-mine.md": "c"}
    world = LocalWorld(version=1, path="src")
    assert world.skills(state) == [f"{SKILLS_DIR}/house.md", f"src/{SKILLS_DIR}/modules.md"]
    assert f"{SKILLS_DIR}/modules.md" in world.situate(state)


def test_an_agent_deep_in_the_tree_is_shown_the_contract_it_is_judged_against():
    """Paper 3.1: an agent may inspect the whole project; it only *begins* at p.

    The specification is a sibling of the implementation, so it is on nobody's
    CONTEXT.md chain. Before this, every agent inferred the language from one
    failing input and built a coherent toolchain against the wrong contract.
    """
    brief = LocalWorld(version=1, path="src/frontend").situate(
        domain.initial_files(), contracts=domain.FROZEN)
    assert "src.parse(source)" in brief
    assert '("num", <int>)' in brief
    # ...and without it, invisible -- which is the defect, stated as a test.
    assert "src.parse(source)" not in LocalWorld(version=1, path="src/frontend").situate(
        domain.initial_files())


def test_an_oversized_brief_carries_upstreams_own_truncation_marker():
    """The marker is a signal to prune, not decoration -- so it has to be theirs."""
    state = {CONTEXT_FILE: "x" * 50_000}
    assert LocalWorld(version=1, path="").situate(state).endswith(TRUNCATED)


# ---------------------------------------------------------------------------
# The parent's integration evidence, and the merge it does on its children
# ---------------------------------------------------------------------------

def test_a_parent_refuses_a_child_whose_work_breaks_the_suite():
    """Paper 3.3: the parent decides on tests and integration evidence."""
    review = domain.suite_review(domain.build_tasks(), sample=6)
    state = dict(domain.reference_tree())
    parent = Brief(world=LocalWorld(version=1, path="src"), objective="o", context="",
                   state=state, task=Task(id="t", prompt="1 + 2"), output="",
                   reward=1.0, depth=0)
    broken = [Edit("src/frontend", domain.LEXER, "def tokenize(source):\n    return []\n")]
    verdict = review(parent, broken)
    assert verdict is not None and verdict[0] == "rejected"
    assert "regressed" in verdict[1]


def test_a_parent_accepts_structural_work_that_moves_no_case():
    """A node's first file is structure, and structure moves nothing."""
    review = domain.suite_review(domain.build_tasks(), sample=6)
    parent = Brief(world=LocalWorld(version=1, path="src"), objective="o", context="",
                   state=domain.initial_files(), task=Task(id="t", prompt="1 + 2"),
                   output="", reward=0.0, depth=0)
    assert review(parent, [Edit("src", "src/notes.py", "# nothing yet\n")]) is None


@pytest.mark.skipif(not git_available(), reason="git merge-file is the merge engine")
def test_two_children_writing_one_file_are_merged_rather_than_one_rejected():
    """A child inside another's subtree may legally touch the same path.

    Rejecting on a bare collision discarded a whole contribution for touching a
    file a sibling also touched -- which is the merge the system exists to do.
    """
    # The reachable shape: one child is situated *inside* the other's subtree, so
    # `src/a/b/f.py` is legally within the authority of both.
    log = WorldLog()
    state = {"src/CONTEXT.md": "# src\n", "src/a/b/f.py": _BASE}
    bodies = {"src/a": _OURS, "src/a/b": _THEIRS}
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/a", "x"), Delegation("src/a/b", "y")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "src/a/b/f.py", bodies[b.world.path])],
        log=log, max_depth=3, root_path="src",
        accountability=False)
    edits = parse_situated_edits(policy.propose(
        _proposal_ctx(state, Task(id="t", prompt="x")))[0])
    merged = next(e["content"] for e in edits if e["path"] == "src/a/b/f.py")
    assert "return 1" in merged and "return 2" in merged
    assert policy.sibling_merges == 1 and policy.sibling_conflicts == 0
    assert [e.verdict for e in log.episodes if e.path.startswith("src/")] == \
           ["accepted", "accepted"]


@pytest.mark.skipif(not git_available(), reason="git merge-file is the merge engine")
def test_an_overlapping_sibling_is_sent_back_and_its_other_work_stands():
    log = WorldLog()
    rival = _BASE.replace("def a():\n    raise NotImplementedError",
                          "def a():\n    return 9")
    state = {"src/CONTEXT.md": "# src\n", "src/a/b/f.py": _BASE}
    bodies = {"src/a": _OURS, "src/a/b": rival}
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/a", "x"), Delegation("src/a/b", "y")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "src/a/b/f.py", bodies[b.world.path]),
                            Edit(b.world.path, f"{b.world.path}/own.py", "1\n")],
        log=log, max_depth=3, max_edits=8, root_path="src",
        accountability=False)
    edits = {e["path"] for e in parse_situated_edits(policy.propose(
        _proposal_ctx(state, Task(id="t", prompt="x")))[0])}
    assert policy.sibling_conflicts == 1
    assert "src/a/b/own.py" in edits, "the child's non-overlapping work still stands"
    assert "src/a/b" in log.pending_rework
    assert [e.verdict for e in log.episodes if e.path == "src/a/b"] == ["rework"]


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
    roles = [e.role for e in episodes]
    assert roles[0] == "manager" and "executor" in roles, roles
    assert {e.version for e in episodes} == {3}, "delegation must not move the version"
    assert log.observed_depth() >= 2, "the tree must be a tree"


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
    """Appendix 1.4: the rejected code does not survive, the saved reason can.

    Rejection now comes from the parent's *evidence*, not from scope -- an edit
    outside the child's path is a request, which is the next test.
    """
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src", "do it")] if b.world.path == "" else []),
        executor=lambda b: [Edit(b.world.path, "src/x.py", "nope")],
        review=lambda parent, returned: ("rejected", "it breaks the build"),
        log=log, max_depth=2)
    proposals = policy.propose(_proposal_ctx({CONTEXT_FILE: "# root"},
                                             Task(id="t", prompt="x")))
    edits = parse_situated_edits(proposals[0])
    assert [e["path"] for e in edits] == [f"src/{CONTEXT_FILE}"]
    assert "refused: it breaks the build" in edits[0]["content"]
    assert [e.verdict for e in log.episodes if e.path == "src"] == ["rejected"]


def test_a_node_relative_filename_lands_in_the_node_not_at_the_repository_root():
    """The whole of a real run's work went to the top of the tree over this."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/frontend", "build the lexer")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "lexer.py", "lex\n")],
        log=log, max_depth=3, root_path="src",
        accountability=False)
    routed = {"src/CONTEXT.md": "# src\n\n## Routing Table\n- `./src/frontend/` -> lexer\n"}
    edits = parse_situated_edits(policy.propose(
        _proposal_ctx(routed, Task(id="t", prompt="x")))[0])
    assert [e["path"] for e in edits] == ["src/frontend/lexer.py"]
    assert policy.resolved_relative == 1
    assert policy.requests_raised == 0, "it was never a cross-node request"


def test_a_manager_writes_its_own_nodes_file_after_its_children_return():
    """Upstream's third phase: delegation does not discharge accountability.

    Measured before this existed: a model run produced five correct modules and
    no `src/__init__.py`, so the public surface the specification names did not
    exist and every case scored zero. The manager had delegated, every time.
    """
    log = WorldLog()
    seen = []

    def executor(brief):
        seen.append((brief.world.path, brief.objective.startswith("review")))
        if brief.world.path == "src/frontend":
            return [Edit(brief.world.path, "src/frontend/lexer.py", "lex\n")]
        return [Edit(brief.world.path, "src/__init__.py", "surface\n")]

    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/frontend", "lexer")]
                           if b.world.path == "src" else []),
        executor=executor, log=log, max_depth=3, root_path="src")
    routed = {"src/CONTEXT.md": "# src\n\n## Routing Table\n- `./src/frontend/` -> lexer\n"}
    edits = {e["path"] for e in parse_situated_edits(policy.propose(
        _proposal_ctx(routed, Task(id="t", prompt="x")))[0])}
    assert edits == {"src/frontend/lexer.py", "src/__init__.py"}
    assert ("src", True) in seen, "the manager never got its accountability turn"
    assert policy.accountability_edits == 1


def test_the_accountability_turn_will_not_overwrite_a_childs_file():
    """A manager is accountable for its subtree; a child's files are the child's."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/frontend", "lexer")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "src/frontend/lexer.py",
                                 "mine\n" if b.world.path == "src" else "theirs\n")],
        log=log, max_depth=3, root_path="src")
    routed = {"src/CONTEXT.md": "# src\n\n## Routing Table\n- `./src/frontend/` -> lexer\n"}
    edits = {e["path"]: e["content"] for e in parse_situated_edits(policy.propose(
        _proposal_ctx(routed, Task(id="t", prompt="x")))[0])}
    assert edits == {"src/frontend/lexer.py": "theirs\n"}
    assert policy.accountability_declined == 1


def test_a_change_outside_a_childs_path_is_reported_up_and_handled_there():
    """Upstream: "report the need back up to your parent, which will handle it".

    The child never writes outside its subtree -- the ancestor that has authority
    there makes the change, under its own name.
    """
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/frontend", "build the lexer")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "src/frontend/lexer.py", "lex\n"),
                            Edit(b.world.path, "src/__init__.py", "surface\n")],
        log=log, max_depth=3, root_path="src")
    routed = {"src/CONTEXT.md": "# src\n\n## Routing Table\n- `./src/frontend/` -> lexer\n"}
    edits = {e["owner"]: e["path"] for e in parse_situated_edits(policy.propose(
        _proposal_ctx(routed, Task(id="t", prompt="x")))[0])}
    assert edits == {"src/frontend": "src/frontend/lexer.py", "src": "src/__init__.py"}
    assert (policy.requests_raised, policy.adopted_requests,
            policy.unmet_requests) == (1, 1, 0)


def test_a_need_nobody_in_the_chain_can_meet_is_counted_not_dropped():
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/frontend", "x")]
                           if b.world.path == "src" else []),
        executor=lambda b: [Edit(b.world.path, "src/frontend/a.py", "1\n"),
                            Edit(b.world.path, "spec/extra.md", "2\n")],
        log=log, max_depth=3, root_path="src")
    # `spec/` is a real top-level node, so the path is repository-relative and the
    # request is genuine -- and nobody from `src` down has authority there.
    policy.propose(_proposal_ctx({"src/CONTEXT.md": "# src\n", "spec/CONTEXT.md": "s"},
                                 Task(id="t", prompt="x")))
    assert (policy.requests_raised, policy.adopted_requests,
            policy.unmet_requests) == (1, 0, 1)


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

DOMAINS = [domain, stackvm, jqx, md]
DOMAIN_IDS = ["minilang", "stackvm", "jqx", "md"]


@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_every_formation_domain_has_been_seen_to_pass(spec):
    """Two different claims, and both are worth a test.

    For the three oracle-scored domains: a suite that disagrees with its own oracle
    scores a *correct* candidate wrong, and no run could ever finish.

    For md, where no reference is in the scoring path at all, this is the weaker but
    more important claim -- the suite is passable. Shipping a specification nobody
    has ever seen satisfied is its own kind of dishonesty, so the reference exists
    for this test and for the offline actor, and for nothing else.
    """
    tasks, run = spec.build_tasks(), spec.make_runner()
    rendered = canonical(spec.reference_tree())
    assert sum(spec.reward(t, run(rendered, t)) for t in tasks) == len(tasks)


@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_every_formation_domain_starts_with_no_implementation(spec):
    files = spec.initial_files()
    # Markdown, plus any frozen entry script the human supplies -- a domain whose
    # result is a program needs a shell around the library, and that shell is the
    # human's, not an agent's.
    frozen_scripts = [p for p in files if not p.endswith(".md")]
    assert files and all(match_any(p, spec.FROZEN) for p in frozen_scripts)
    tasks, run = spec.build_tasks(), spec.make_runner()
    assert sum(spec.reward(t, run(canonical(files), t)) for t in tasks) == 0


@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_every_stage_is_reachable_from_the_train_split(spec):
    """A unit the train split cannot reach is a unit the run can never grow.

    Found the expensive way on jqx: the case list was ordered by filter, so two
    builtins appeared only in the held-out tail. No rollout could fail on them, so
    no proposal was ever requested for them, and the run stalled at 0.923 with two
    stubs open and `stop reason: rounds`. `evolve()` splits by position, so this is
    a property of the *order* of the task list, and nothing else checks it.
    """
    tasks = spec.build_tasks()
    cut = max(1, round(len(tasks) * (1 - spec.HELD_OUT_FRAC)))
    missing = sorted(set(t.meta["kind"] for t in tasks if not t.meta.get("audit"))
                     - set(t.meta["kind"] for t in tasks[:cut]))
    assert not missing, f"stages only in the held-out tail: {missing}"


# ---------------------------------------------------------------------------
# md: scored by a frozen test suite, with no oracle in the loop
# ---------------------------------------------------------------------------

def test_an_md_task_is_one_test_and_its_prompt_is_that_tests_source():
    """The executor is shown the assertion it failed, which is the whole design:
    the specification is executable, so the prompt can be the specification.

    With the file's prelude above it, because a test body that reads
    `forces(CONFIG, box=BOX)` says nothing on its own about what `CONFIG` is.
    """
    task = next(t for t in md.build_tasks()
                if t.meta["func"] == "test_forces_sum_to_zero")
    body = md.MD.given[task.meta["file"]]
    assert task.prompt.startswith(f"# {task.meta['file']}\n")
    assert "from src import energy, forces" in task.prompt       # the prelude
    assert "CONFIG = [[0.0, 0.0, 0.0]" in task.prompt            # and its constants
    assert task.prompt.endswith("assert max(abs(c) for c in total) < 1e-9, total")
    assert body.count("def test_forces_sum_to_zero():") == 1
    # Only this test. Showing the whole file would show the agent the tests it has
    # not been asked about, and the episode is bounded at four edits anyway.
    assert sum(1 for line in task.prompt.splitlines()
               if line.startswith("def test_")) == 1


def test_the_md_audit_set_is_not_in_the_repository_the_agents_see():
    """`frozen` stops a file being *written*, not read. An audit test in the tree is
    an audit test the executor can read and write code against."""
    files = md.initial_files()
    assert md._AUDIT and not any(path in files for path in md._AUDIT)
    audited = [t for t in md.build_tasks() if t.meta["audit"]]
    assert audited and all(t.meta["file"] not in files for t in audited)


def test_the_md_audit_set_lands_exactly_in_the_held_out_tail():
    """`HELD_OUT_FRAC` is computed, not chosen: every driven test -- every
    requirement -- has to be on the search's side of the cut."""
    tasks = md.build_tasks()
    cut = max(1, round(len(tasks) * (1 - md.HELD_OUT_FRAC)))   # evolve()'s own split
    assert [t.meta["audit"] for t in tasks[:cut]] == [False] * cut
    assert all(t.meta["audit"] for t in tasks[cut:])


def test_the_md_audit_tests_are_injected_only_while_they_are_scored():
    """They are not in the candidate repository, and they still run against it."""
    run = md.make_runner()
    rendered = canonical(md.reference_tree())
    assert "audit/" not in rendered
    audited = [t for t in md.build_tasks() if t.meta["audit"]]
    assert sum(md.reward(t, run(rendered, t)) for t in audited) == len(audited)


def test_the_md_suite_catches_an_integrator_that_is_only_stable():
    """Euler with one force evaluation per step is stable, plausible, and wrong.

    This is what the invariants are for, and what an oracle over sampled values
    would not reliably catch: the trajectory stays bounded, so a sampled position
    often agrees to six places, but the scheme is first-order and not reversible.
    """
    broken = dict(md.reference_tree())
    broken[md.VERLET] = broken[md.VERLET].replace(
        "    _, forces = evaluate(system, spec)\n"
        "    for i, mass in enumerate(system.masses):\n"
        "        for axis in range(3):\n"
        "            system.velocities[i][axis] += half * forces[i][axis] / mass\n"
        "    return system\n",
        "    return system\n")
    assert broken[md.VERLET] != md.reference_tree()[md.VERLET]
    run, rendered = md.make_runner(), canonical(broken)
    failed = [t.id for t in md.build_tasks()
              if md.reward(t, run(rendered, t)) == 0.0]
    assert any("reversing" in t for t in failed), failed
    assert any("conserved" in t for t in failed), failed


def test_stackvm_is_deeper_than_minilang_which_is_why_it_exists():
    """The second domain is not "harder code" -- it is somewhere for the
    recursion to go. `src/vm/ops` is a node whose parent is itself a child."""
    def node_depth(files):
        # the nodes are where the CONTEXT.md records are; the skill directory is
        # deep in both domains and is not a node.
        return max(p.count("/") for p in files if p.endswith(CONTEXT_FILE))
    assert node_depth(stackvm.initial_files()) == 3        # src/vm/ops/CONTEXT.md
    assert node_depth(domain.initial_files()) == 2         # src/frontend/CONTEXT.md
    assert "src/vm/ops/CONTEXT.md" in stackvm.initial_files()
    routes = LocalWorld(version=1, path="src/vm").routing(stackvm.initial_files())
    assert routes == ["src/vm/ops"]


def test_the_frozen_suite_agrees_with_its_own_oracle():
    tasks = domain.build_tasks()
    run = domain.make_runner()
    rendered = canonical(domain.reference_tree())
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
