"""EvoX Genesis port: the mechanism, one component at a time, offline.

Every test here is about a piece of the paper's model, not about the domain --
the domain is a compact stand-in and its numbers prove nothing about upstream.
What the numbers *can* prove is that the mechanism is the one described: the path
coordinate bounds authority, delegation does not advance the version, the parent's
verdict is the parent's, and two agents editing one file both survive.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import posixpath
import time

import pytest

from agentdescent.aggregator import AggregatorConfig, MergeOutcome, diffs_contradict
from agentdescent.defaults import DefaultAcceptance, DefaultConflict
from agentdescent.evolution import EvolvingArtifact, RoundInfo, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.filetree import canonical, match_any
from agentdescent.policies import MergeContext, Policies, ProposalContext

from examples.genesis import _domain as domain
from examples.genesis import _jqx as jqx
from examples.genesis import _md as md
from examples.genesis import _stackvm as stackvm
from examples.genesis import genesis_recursive_worlds as genesis
from examples.genesis._delegation import (Brief, Delegation, Edit,
                                          RecursiveDelegation, render_edits)
from examples.genesis._claude_code import (CLAUDE_CODE_BRIEF,
                                           ClaudeCodeExecutor)
from examples.genesis._architect import (ARCHITECT_PROMPT, ArchitectPhase,
                                         missing_sections,
                                         _parse as parse_architect_reply)
from examples.genesis._extract import ExtractPhase
from examples.genesis._judge import ParentJudge
from examples.genesis._review import (CompletionJudge, ParentCodeReview,
                                      chain_reviews)
from examples.genesis._octopus import OctopusConflict, git_available, three_way
from examples.genesis._spatial import SpatialContract, parse_situated_edits
from examples.genesis._worktree import (Rollout, WorktreeLedger,
                                        git_worktrees_available)
from examples.genesis._suite import TestSuite as SuiteOfTests
from examples.genesis._suite import cold_start, preflight
from examples.genesis._world import (CONTEXT_FILE, KNOWN_ISSUES, ROUTING_HEADING,
                                     SKILLS_DIR, TRUNCATED,
                                     LocalWorld, WorldLog, owns, parse_routing,
                                     looks_like_file, resolve_edit_path,
                                     routing_entry, shadowed_by_module, shadows_package,
                                     under_heading)


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


def test_an_oversized_context_record_is_truncated_per_file_as_upstream_does():
    """Per file, at upstream's cap, with upstream's marker -- and not globally.

    `ContextNode.build_context/2` truncates each CONTEXT.md at
    `truncation.context_max_bytes` (default 65_536) and has no overall budget at all.
    This port had one global 8_000-char cap instead, eight times tighter, and that is
    the other half of how md's agents never saw their specification: two frozen files
    that sorted earlier spent the budget. The marker is a signal to prune the record,
    so it is upstream's string.
    """
    context = LocalWorld(version=1, path="").situate({CONTEXT_FILE: "x" * 70_000})
    assert TRUNCATED in context
    assert context.count("x") == 65_536
    # the record is not the last thing in the brief: the file listing survives it
    assert context.rstrip().endswith("(none yet)")
    # and a record under the cap is untouched, however large the whole brief gets
    assert TRUNCATED not in LocalWorld(version=1, path="").situate(
        {CONTEXT_FILE: "y" * 50_000, "spec/CONTEXT.md": "z" * 50_000},
        contracts=("spec/**",))

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
    """"Try again" is queued; only a refusal is written into the world.

    The node it opened is still recorded, and that is upstream's order rather than an
    exception to this rule: the architect creates the child directory and its
    CONTEXT.md with `make_dir` (which auto-commits) and *then* spawns the subagent, so
    the node exists whether or not the work does. What is not written is any remark
    about the attempt.
    """
    log = WorldLog()
    policy = RecursiveDelegation(manager=lambda b: [Delegation("src", "do nothing")],
                                 executor=lambda b: [], log=log, max_depth=2)
    proposals = policy.propose(_proposal_ctx({CONTEXT_FILE: "# root"},
                                             Task(id="t", prompt="x")))
    edits = parse_situated_edits(proposals[0]) if proposals else []
    assert [e["path"] for e in edits] == [CONTEXT_FILE]       # the routing entry only
    assert "refused" not in (edits[0]["content"] if edits else "")
    assert log.pending_rework                                 # queued instead
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
    # The refusal at the child's own node, and the entry for the node the manager
    # opened -- written before the child was briefed, as upstream's `make_dir` is.
    assert [e["path"] for e in edits] == [CONTEXT_FILE, f"src/{CONTEXT_FILE}"]
    assert "refused: it breaks the build" in edits[1]["content"]
    assert KNOWN_ISSUES in edits[1]["content"]
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


def test_the_contract_reaches_the_agent_even_when_a_bigger_frozen_file_sorts_first():
    """Sorted order made *which* contract an agent sees depend on filenames.

    md froze `spec/**`, `tests/**` and `md.py`; `md.py` sorts first, is 6.8 kB of
    driver, and ate a budget the 4.5 kB specification then never reached. Every
    agent inferred the public surface from the one failing test it was shown, five
    of the seven keyword parameters went missing, and the run scored 0.750 against
    a specification nobody had read. The order is the domain's now.
    """
    state = {CONTEXT_FILE: "# root\n", "aaa_driver.py": "# driver\n" + "x" * 9000,
             "spec/CONTEXT.md": "# the contract\nTHE SIGNATURE IS f(a, b=1)\n"}
    context = LocalWorld(0, "").situate(state, contracts=("spec/**", "aaa_driver.py"),
                                        max_chars=4000)
    assert "THE SIGNATURE IS f(a, b=1)" in context
    assert len(context) <= 4000


def test_the_file_listing_survives_a_contract_too_big_to_fit():
    """Truncating from the end dropped the listing, which is the actionable part:
    a node that owns no file yet only learns it from that line."""
    state = {CONTEXT_FILE: "# root\n", "spec/CONTEXT.md": "# spec\n" + "y" * 9000}
    context = LocalWorld(0, "src").situate(state, contracts=("spec/**",),
                                           max_chars=2000)
    assert TRUNCATED in context                                  # the contract gave
    assert "(none yet -- this node owns no file)" in context      # the listing stayed


@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_every_node_of_every_domain_is_given_the_whole_contract(spec):
    """At the default budget, from every node, for every domain -- not just the one
    whose frozen file happens to sort early."""
    files = spec.initial_files()
    nodes = sorted(p[:-len(CONTEXT_FILE)].rstrip("/") for p in files
                   if p.endswith(CONTEXT_FILE) and not p.startswith("spec/"))
    contract = files["spec/CONTEXT.md"].strip()
    for node in nodes:
        context = LocalWorld(0, node).situate(files, contracts=spec.FROZEN)
        assert contract in context, f"{spec.MD.name if hasattr(spec, 'MD') else node}"


def test_a_dead_backend_stops_the_run_before_it_starts():
    """Thirteen minutes of a 404 endpoint reads exactly like a broken mechanism.

    `evolve()` swallows an exception from a proposal policy, correctly -- one timed
    out call should cost its episode and no more. But then every call failing is not
    an error, it is an empty run: `reward 0.000  depth=0  accepted=400`, with the
    only evidence `2400 failed` in the usage line. So the endpoint is asked one
    question first, and that failure is loud.
    """
    def dead(prompt):
        raise RuntimeError("Error code: 404 - the API does not exist")

    with pytest.raises(SystemExit) as caught:
        preflight(dead)
    assert "ANTHROPIC_BASE_URL" in str(caught.value)
    assert "404" in str(caught.value)

    asked = []
    preflight(lambda prompt: asked.append(prompt) or "OK")       # a live one is quiet
    assert len(asked) == 1


# ---------------------------------------------------------------------------
# Cold start: the decomposition is the run's job, not the human's
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_cold_start_keeps_the_goal_the_contract_and_the_suite_and_nothing_else(spec):
    """Every file a human cannot avoid supplying, and not one more.

    A formation domain ships a `CONTEXT.md` per node, each with a routing table, and
    for every node below the root that is the system's own first job done for it --
    the paper's phase 1 is architecture and design. `--cold-start` takes it away:
    what is left is the goal, the frozen contract, the suite that scores it and any
    frozen entry script.
    """
    cold = cold_start(spec.initial_files())
    assert set(cold) <= set(spec.initial_files())
    # the root record survives, with an emptied table
    assert parse_routing(cold[CONTEXT_FILE]) == []
    # no other record, and no skill
    assert [p for p in cold if p.endswith(CONTEXT_FILE)] == [CONTEXT_FILE] + \
           [p for p in cold if p.endswith(CONTEXT_FILE) and p.startswith("spec/")]
    assert not [p for p in cold if SKILLS_DIR in p]
    # and nothing frozen was touched: scoring is identical
    frozen = [p for p in spec.initial_files() if match_any(p, spec.FROZEN)]
    assert frozen and all(cold.get(p) == spec.initial_files()[p] for p in frozen)


def test_a_cold_started_root_is_not_sent_into_the_read_only_directories():
    """`routing()` falls back to the sub-directories that exist, which was invisible
    while every domain shipped a table at the root. Cold-started, the root's only
    sub-directories are `spec/` and `tests/` -- and a manager was duly told to
    delegate into the two places it is forbidden to write."""
    cold = cold_start(md.initial_files())
    assert LocalWorld(0, "", readonly=md.FROZEN).routing(cold) == []
    assert LocalWorld(0, "").routing(cold) == ["spec", "tests"]      # the old bug
    # a directory with anything writable in it is still a route
    grown = dict(cold, **{"src/__init__.py": "x\n"})
    assert LocalWorld(0, "", readonly=md.FROZEN).routing(grown) == ["src"]


def test_a_cold_started_world_writes_the_decomposition_it_was_not_given():
    """End to end, offline, from eight files: the run opens its own nodes, records
    them in its own routing tables, and still reaches a working world."""
    tasks = domain.build_tasks()
    log = WorldLog()
    cold = cold_start(domain.initial_files())
    assert [p for p in cold if p.endswith(CONTEXT_FILE)] == [CONTEXT_FILE,
                                                             "spec/" + CONTEXT_FILE]
    strategy = SpatialContract(initial_files=cold, frozen=domain.FROZEN, log=log,
                               max_files_per_diff=6)
    delegation = RecursiveDelegation(manager=domain.offline_manager,
                                     executor=domain.offline_executor, log=log,
                                     max_depth=3, contracts=domain.FROZEN)
    result = evolve(
        tasks, domain.reward, run=domain.make_runner(),
        propose=lambda *a: (_ for _ in ()).throw(AssertionError("not installed")),
        strategy=strategy, artifact_id="world", blast_radius=0.2,
        rounds=14, n_workers=4, max_concurrency=4, self_verify=False,
        held_out_frac=0.4, seed=0,
        policies=Policies(proposal=delegation,
                          acceptance=ParentJudge(log=log, enabled=True),
                          conflict=OctopusConflict()))

    assert result.error is None
    assert result.final_reward > 0.9, result.outcomes()
    assert delegation.routes_opened > 0, "no node was ever opened"
    grown = result.state if isinstance(result.state, dict) else dict(result.state)
    records = {p: grown[p] for p in grown if p.endswith(CONTEXT_FILE)}
    assert len(records) > 1, sorted(records)
    written = [p for p in records if p != CONTEXT_FILE and not p.startswith("spec/")]
    assert written, "the run never recorded a node of its own"
    assert any(parse_routing(records[p]) for p in records), "no routing table grew"
    assert log.contract_violations == 0


def test_the_md_suite_rejects_every_deliberately_wrong_implementation():
    """A suite is only as good as what it refuses, and nothing else here checked that.

    The md domain was validated only against a *correct* implementation -- the suite
    accepts the right answer, and nothing was known about what it rejects. A field
    that returns zero everywhere satisfies every invariant in it (zero sums to zero,
    zero is the gradient of a constant, nothing moves so nothing drifts), and a real
    run shipped exactly that: `d -= box * (d / box + 0.5) // 1`, where the `// 1`
    binds after the multiplication. It reported audit reward 1.000.

    So: keep the wrong implementations, and require every one of them to die. Writing
    this found two more holes in the same sitting. A `round`-based geometry -- the one
    thing the spec explicitly forbids, because Python's round is banker's rounding --
    passed the entire suite; and the centre-of-mass-corrected temperature died to
    exactly one test.

    **Two killers, not one.** The zero field originally died to a single test, and
    that test was the negative control (`a_far_too_large_timestep_does_not_conserve_
    energy`), which exists only because "the conservation tests must be able to fail,
    or they assert nothing". One test between a false 1.000 and the truth is not a
    margin.

    This is the harness keeping its own measurement honest, and it is not what
    upstream relies on: Genesis has no objective function at all -- `grep -rio
    'fitness|reward|score'` over `apps/evo_git` returns `oom_score_adjust` and nothing
    else -- and in its place is a parent that reads the diff and runs the tests
    (`agents/manager.ex`). Coverage, for the record, would not have caught any of
    this: the zero field executes every line of every test.
    """
    report = md.MD.kill_report(md.reference_tree(), stop_after=2)
    assert set(report) == set(md.BASELINES)
    survived = sorted(name for name, killers in report.items() if not killers)
    assert not survived, f"the suite accepts these wrong implementations: {survived}"
    fragile = {name: killers for name, killers in report.items() if len(killers) < 2}
    assert not fragile, f"only one test stands between these and a false pass: {fragile}"


# ---------------------------------------------------------------------------
# The parent's other half: reading the change, not only counting tests
# ---------------------------------------------------------------------------

def _review_brief(state, objective="make the forces right"):
    return Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                 objective=objective, context="", state=state,
                 task=Task(id="t", prompt="x"), output="", reward=0.0, depth=0)


def test_the_parent_reads_the_change_and_can_reject_it_on_what_it_sees():
    """`manager.ex` states the parent's validation as three things: review the
    child's results, run the tests, AND reject code-quality anti-patterns. This port
    had the middle one, which is a number -- and a number could not see a registry
    whose minimum-image expression left every force at zero.
    """
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return '{"verdict": "reject", "reason": "evaluate() returns zeros"}'

    review = ParentCodeReview(complete, contracts=md.CONTRACTS)
    verdict = review(_review_brief(md.initial_files()),
                     [Edit("src", "src/potentials/registry.py", "def evaluate(s, p):\n"
                           "    return 0.0, [[0.0] * 3 for _ in s.positions]\n")])
    assert verdict is not None
    kind, reason = verdict
    assert kind == "rejected" and "returns zeros" in reason
    assert (review.reviewed, review.rejected) == (1, 0 + 1)
    # it was shown the whole file and the contract, because four lines of arithmetic
    # cannot be judged from a summary
    assert "return 0.0, [[0.0] * 3" in seen[0]
    assert "dudr_over_r" in seen[0]          # the spec travelled with it


def test_the_parent_accepts_what_it_has_no_complaint_about():
    review = ParentCodeReview(lambda prompt: '{"verdict": "accept", "reason": ""}')
    assert review(_review_brief(md.initial_files()),
                  [Edit("src", "src/__init__.py", "def energy(p):\n    return 1.0\n")]) is None
    assert (review.reviewed, review.rejected) == (1, 0)


def test_a_reviewer_that_cannot_speak_is_not_evidence_against_the_child():
    """An unparseable reply, or a dead backend, must not reject work. The child did
    the work; the reviewer failing is the reviewer's problem."""
    garbled = ParentCodeReview(lambda prompt: "I think it looks fine?")
    assert garbled(_review_brief(md.initial_files()),
                   [Edit("src", "src/a.py", "x = 1\n")]) is None
    assert (garbled.reviewed, garbled.rejected, garbled.unparsed) == (1, 0, 1)

    def dead(prompt):
        raise RuntimeError("502")

    broken = ParentCodeReview(dead)
    assert broken(_review_brief(md.initial_files()),
                  [Edit("src", "src/a.py", "x = 1\n")]) is None
    assert broken.rejected == 0


def test_the_parent_does_not_spend_a_call_on_bookkeeping_alone():
    """A routing note is not a change to review."""
    review = ParentCodeReview(lambda prompt: pytest.fail("should not be called"))
    assert review(_review_brief(md.initial_files()),
                  [Edit("src", "src/CONTEXT.md", "# src\n", kind="context")]) is None
    assert review.reviewed == 0


def test_the_tests_run_before_the_reviewer_because_they_are_free():
    """`chain_reviews` short-circuits: a regression needs no second opinion, and the
    deterministic check costs no model call."""
    calls = []
    cheap = lambda parent, returned: (calls.append("cheap") or ("rejected", "tests"))
    dear = lambda parent, returned: pytest.fail("the model was asked anyway")
    chained = chain_reviews(cheap, dear)
    assert chained(_review_brief({}), [Edit("src", "src/a.py", "x = 1\n")]) == ("rejected",
                                                                               "tests")
    assert calls == ["cheap"]
    # and a disabled review is skipped rather than branched around by the caller
    assert chain_reviews(None, None) is None
    only = lambda parent, returned: None
    assert chain_reviews(None, only) is only


def test_a_refusal_is_written_under_upstreams_own_heading():
    """"findings worth preserving belong in CONTEXT.md... `## Known Issues` (problems
    to avoid re-discovering)" -- `agents/manager.ex`. A refusal is the cheapest such
    finding: the next agent here would otherwise be refused for the same thing."""
    body = under_heading("# src/core\n\n## Intent\nowns vectors.py\n",
                         KNOWN_ISSUES, "- refused: out of scope")
    assert body.endswith("## Known Issues\n- refused: out of scope\n")
    again = under_heading(body, KNOWN_ISSUES, "- refused: twice")
    assert again.count(KNOWN_ISSUES) == 1           # one section, two lines
    assert again.endswith("- refused: out of scope\n- refused: twice\n")


@pytest.mark.parametrize("spec", DOMAINS, ids=DOMAIN_IDS)
def test_every_node_record_carries_upstreams_four_standard_sections(spec):
    """"The standard four sections (Intent, API Surface, Constraints, Routing Table)
    are the foundation" -- `agents/manager.ex`. API Surface was missing from every
    record in this port, on a domain whose entire failure mode was a wrong public
    surface: the 0.750 run dropped five of seven keyword parameters.

    Intent and the routing table are required of every node; a leaf has no routing
    table and constraints are inherited, so those two are checked where they appear.
    """
    files = spec.initial_files()
    records = {p: body for p, body in files.items()
               if p.endswith(CONTEXT_FILE) and not p.startswith("spec/")}
    assert records
    for path, body in records.items():
        assert "## Intent" in body, path
        assert "## API Surface" in body, f"{path} says nothing about what it exposes"
    assert any(ROUTING_HEADING in body for body in records.values())
    assert any("## Constraints" in body for body in records.values())


# ---------------------------------------------------------------------------
# The scheduling model: a worktree per episode, a commit, and the worktree gone
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not git_worktrees_available(), reason="git worktree unavailable")
def test_a_parents_merge_commit_carries_every_child_commit_as_a_parent():
    """The phylogenetic graph, as git history rather than as a diagram.

    `Git.merge_octopus/2` leaves a merge commit whose parents are the base and every
    child; `phylo_graph_node.ex` then has a `find_merge_base/2` worth calling. With
    the states kept in memory there was nothing for either to see.
    """
    ledger = WorktreeLedger()
    with Rollout(ledger) as rollout:
        base = rollout.commit("M0001", {CONTEXT_FILE: "# root\n"}, "before delegating")
        left = rollout.commit("E0002", {CONTEXT_FILE: "# root\n", "src/a.py": "a\n"},
                              "child a", [base])
        right = rollout.commit("E0003", {CONTEXT_FILE: "# root\n", "src/b.py": "b\n"},
                               "child b", [base])
        merge = rollout.commit("M0001.merge",
                               {CONTEXT_FILE: "# root\n", "src/a.py": "a\n",
                                "src/b.py": "b\n"}, "merge 2 children",
                               [base, left, right])
        graph = {line.split()[0]: line.split()[1:] for line in rollout.history()}
        assert merge[:7] in graph
        assert len(graph[merge[:7]]) >= 3          # parents, then the subject words
        assert base[:7] in graph and left[:7] in graph and right[:7] in graph
    assert (ledger.created, ledger.removed) == (4, 4)
    assert ledger.merge_commits == 1 and ledger.max_parents == 3
    assert ledger.leaked == 0


@pytest.mark.skipif(not git_worktrees_available(), reason="git worktree unavailable")
def test_every_episode_commits_in_its_own_worktree_and_the_worktree_is_removed():
    """"Commit your changes, release your worktree, and wait" -- `agents/manager.ex`.

    The release is not optional upstream: an unreleased worktree is an agent that
    never yielded, and the scheduler is cooperative. So the ledger has to balance.
    """
    ledger = WorktreeLedger()
    log = WorldLog()
    seen = []
    policy = RecursiveDelegation(
        rollout_factory=lambda: Rollout(ledger),
        manager=lambda b: ([Delegation("src/frontend", "lex"),
                            Delegation("src/backend", "eval")] if b.world.path == "src"
                           else ([Delegation("src", "all of it")] if not b.world.path
                                 else [])),
        executor=lambda b: (seen.append(b.world.path) or
                            [Edit(b.world.path, f"{b.world.path}/x.py", "x = 1\n")]),
        log=log, max_depth=3)
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root\n"}, Task(id="t", prompt="x")))

    # the two leaves first, then each manager's accountability turn at its own node
    assert seen[:2] == ["src/frontend", "src/backend"] and "src" in seen[2:]
    assert ledger.created == ledger.removed > 0, ledger.summary()
    assert ledger.leaked == 0
    assert ledger.commits >= 4            # two leaves, the base, and the merges
    assert ledger.max_parents >= 3        # a parent, and the two children it kept
    # and every episode carries the sha it could be resurrected from
    assert all(e.commit for e in log.episodes), [e.agent_id for e in log.episodes
                                                 if not e.commit]


def test_a_child_sees_the_node_record_its_parent_wrote_before_briefing_it():
    """"Create the child directory with CONTEXT.md via `make_dir` (auto-commits),
    **then** spawn the subagent" -- `agents/architect.ex`, Phase 1. A child briefed
    before that write arrives at a node its own parent's routing table does not
    mention, which is exactly the thing the routing table exists to prevent.
    """
    briefs = {}
    policy = RecursiveDelegation(
        manager=lambda b: [Delegation("src", "do it")] if not b.world.path else [],
        executor=lambda b: (briefs.__setitem__(b.world.path, b.context) or []),
        log=WorldLog(), max_depth=3)
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root\n"}, Task(id="t", prompt="x")))
    assert "src" in briefs
    assert "`./src/` -> do it" in briefs["src"], briefs["src"]


# ---------------------------------------------------------------------------
# complete_task: the run ends when an agent says so, not when the budget does
# ---------------------------------------------------------------------------

def _completion_judge(complete, state):
    return CompletionJudge(complete, tasks=domain.build_tasks(),
                           run=domain.make_runner(), reward=domain.reward,
                           state_of=lambda: state, contracts=domain.CONTRACTS,
                           objective="an integer expression language")


def test_the_root_agent_is_not_asked_while_a_test_it_can_see_still_fails():
    """Upstream's own precondition: "aim for a 100% pass rate on the given test
    suites -- treat them as the definition of done". Below that the question is not
    asked, and no model call is spent asking it."""
    judge = _completion_judge(lambda prompt: pytest.fail("asked too early"),
                              domain.initial_files())
    assert judge(RoundInfo(round=1, held_out_reward=0.0, n_items=0, committed=0,
                           rejected=0)) is False
    assert judge.asked == 0


def test_the_run_ends_when_the_root_agent_says_the_objective_is_delivered():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return '{"complete": true, "reason": "every module is implemented"}'

    judge = _completion_judge(complete, domain.reference_tree())
    info = RoundInfo(round=4, held_out_reward=1.0, n_items=4, committed=3, rejected=0)
    assert judge(info) is True
    assert judge.asked == 1 and judge.verdict == "complete"
    # the decision is about the codebase, and the port's own measurement is not shown:
    # that number comes from tests no agent may read, and handing it over would turn
    # the judgment back into a threshold.
    assert "1.0" not in prompts[0] and "held-out" not in prompts[0]
    assert "src/__init__.py" in prompts[0]          # the tree is what it judges


def test_a_round_that_accepted_nothing_is_not_scored_twice():
    """Fifty subprocesses a round, to ask about a codebase that did not change, would
    cost more than the rounds being watched."""
    runs = []
    state = domain.initial_files()
    judge = CompletionJudge(lambda p: pytest.fail("never green"),
                            tasks=domain.build_tasks(),
                            run=lambda rendered, task: runs.append(task.id) or "",
                            reward=domain.reward, state_of=lambda: state,
                            contracts=domain.CONTRACTS, objective="o")
    info = RoundInfo(round=1, held_out_reward=0.0, n_items=0, committed=0, rejected=0)
    assert judge(info) is False
    assert len(runs) == 1, runs       # and it stops at the first failing test
    assert judge(info) is False
    assert len(runs) == 1            # the same version, so not scored again


def test_an_unparseable_or_negative_answer_keeps_the_run_going():
    green = domain.reference_tree()
    assert _completion_judge(lambda p: "not sure yet", green)(
        RoundInfo(round=1, held_out_reward=1.0, n_items=1, committed=1,
                  rejected=0)) is False
    judge = _completion_judge(lambda p: '{"complete": false, "reason": "stubs left"}',
                              green)
    assert judge(RoundInfo(round=1, held_out_reward=1.0, n_items=1, committed=1,
                           rejected=0)) is False
    assert (judge.verdict, judge.reason) == ("incomplete", "stubs left")

    def dead(prompt):
        raise RuntimeError("502")

    assert _completion_judge(dead, green)(
        RoundInfo(round=1, held_out_reward=1.0, n_items=1, committed=1,
                  rejected=0)) is False


def test_a_per_test_score_cannot_see_a_test_poisoning_the_next_one():
    """One process per test is what makes one-task-per-test possible, and it is blind
    to anything that leaks between tests. A real run produced the sharpest possible
    case, and this baseline is that code with nothing changed:

        def rdf(positions, box, bins, rmax):        # in src/observe/__init__.py
            from .rdf import histogram              # <- rebinds src.observe.rdf
            ...                                     #    from this function to the
                                                    #    module. It destroys itself.

    The first call in a process works; every call after it raises. Scored one test per
    process it is perfect, and the run reported audit reward 1.000 for it. Run the way
    a person runs a suite -- `mix test`, which is what `agents/manager.ex` means by
    "run ALL tests" -- five tests fail.
    """
    broken = dict(md.reference_tree(),
                  **md.SUITE_ONLY_BASELINES["self-clobbering-import"])
    failures = md.MD.suite_failures(broken, audit=True)
    assert len(failures) == 5, failures
    assert all("not callable" in f for f in failures), failures

    # every one of those five passes on its own, which is the whole point -- and the
    # only tasks that fail are the two that run the suite as one unit, which exist so
    # that the *search* can see this at all and not only the gate
    run = md.make_runner()
    rendered = canonical(broken)
    scores = {t.id: md.reward(t, run(rendered, t)) for t in md.build_tasks()}
    named = [t for t in md.build_tasks()
             if t.meta["func"] and any(t.meta["func"] in f for f in failures)]
    assert len(named) == 5 and all(scores[t.id] == 1.0 for t in named), named
    assert {i for i, v in scores.items() if v == 0.0} == {"suite::one-process",
                                                          "audit:suite::one-process"}

    # and the reference is clean under both ways of running it
    assert md.MD.suite_failures(md.reference_tree(), audit=True) == []


# ---------------------------------------------------------------------------
# Phase 1: the architect designs the tree the implementation phase grows
# ---------------------------------------------------------------------------

def _scripted_architect(plan, seen=None):
    def complete(prompt):
        path = prompt.split("situated at the repository path `")[1].split("`")[0]
        path = "" if path == "./" else path
        if seen is not None:
            seen.append(path)
        return json.dumps(plan.get(path, {"record": f"# {path or './'}\n", "children": []}))
    return complete


def test_the_architect_designs_a_tree_and_the_run_starts_from_it():
    """Upstream a formation run is two root agents in sequence: "Phase 1: Architecture
    (Architect as root agent)" then "Phase 2: Implementation", and Phase 2 is handed
    "the architecture, directory structure, CONTEXT.md routing tables ... already in
    place (created by an Architect agent)".

    This port had neither half. A domain shipped its records hand-written, which is
    Phase 1 done by a person; `--cold-start` removed them and put nothing in their
    place. Neither is what upstream does.
    """
    plan = {
        "": {"record": "# md -- root\n\n## Intent\nGrow it.\n\n## API Surface\nmd.py\n\n"
                       "## Constraints\npure python\n\n## Routing Table\n"
                       "- `./src/` -> the library\n",
             "children": [{"path": "src", "objective": "the library"}]},
        "src": {"record": "# src\n\n## Intent\nthe entry points\n\n## API Surface\n"
                          "src/__init__.py\n\n## Constraints\nlazy imports\n\n"
                          "## Routing Table\n- `./src/core/` -> geometry\n",
                "children": [{"path": "core", "objective": "geometry"}]},
    }
    seen = []
    phase = ArchitectPhase(_scripted_architect(plan, seen), contracts=md.CONTRACTS,
                           max_depth=3)
    given = cold_start(md.initial_files())
    tree = phase.design(given, "Lennard-Jones molecular dynamics")

    assert seen[:2] == ["", "src"]                       # root first, then its child
    assert "src/core" in seen                            # a bare name is relative
    assert phase.nodes[:2] == ["", "src"] and phase.depth >= 2
    # what it wrote is a CONTEXT.md tree and nothing else
    added = set(tree) - set(given)
    assert added and all(p.endswith(CONTEXT_FILE) for p in added), added
    assert parse_routing(tree["src/" + CONTEXT_FILE]) == ["src/core"]
    # and the frozen contract is untouched by the phase
    assert tree["spec/" + CONTEXT_FILE] == given["spec/" + CONTEXT_FILE]


def test_an_architect_may_not_name_a_node_outside_its_own_subtree():
    """The spatial contract applies to designing as much as to writing: a node named
    outside the subtree is refused and counted, not quietly relocated into it."""
    plan = {"src": {"record": "# src\n", "children": [{"path": "spec/stolen", "objective": "no"},
                                                      {"path": "src/ok", "objective": "yes"}]}}
    phase = ArchitectPhase(_scripted_architect(plan), max_depth=2, root_path="src")
    tree = phase.design({CONTEXT_FILE: "# root\n"}, "o")
    assert phase.refused == 1
    assert "spec/stolen/" + CONTEXT_FILE not in tree
    assert "src/ok/" + CONTEXT_FILE in tree


def test_a_child_the_architect_named_is_always_routed_to():
    """A node its parent's table does not mention is a node no manager can reach, so a
    record that names children and forgets the table gets the table it forgot."""
    record, children = parse_architect_reply(
        json.dumps({"record": "# src\n\n## Intent\nthings\n",
                    "children": [{"path": "core", "objective": "geometry"}]}), "src/")
    # `files` rides along undeclared, which the size guard reads as "no claim made".
    assert children == [{"path": "src/core", "objective": "geometry", "files": 0}]
    assert parse_routing(record) == ["src/core"]


def test_an_unusable_reply_costs_its_node_and_not_the_phase():
    phase = ArchitectPhase(lambda prompt: "no json here", max_depth=2)
    assert phase.design({CONTEXT_FILE: "# root\n"}, "o") == {CONTEXT_FILE: "# root\n"}
    assert phase.unparsed == 1 and phase.nodes == []


def test_an_executor_is_asked_to_keep_its_own_record_current():
    """"CONTEXT.md is your long-term memory: findings worth preserving belong in
    CONTEXT.md" -- `agents/manager.ex`, and the archive shows 62 later accepted updates
    affecting 19 files. Nothing in this port asked for that, so the count was zero by
    construction rather than by measurement."""
    seen = []
    executor = md.llm_executor(lambda prompt: seen.append(prompt) or "")
    executor(Brief(world=LocalWorld(version=1, path="src/core", readonly=md.FROZEN),
                   objective="o", context="", state=md.initial_files(),
                   task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    assert "the only memory that outlives you" in seen[0]
    assert "src/core/CONTEXT.md" in seen[0]
    assert "## Known Issues" in seen[0]


def test_a_record_an_agent_writes_itself_is_counted_separately():
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: [],
        executor=lambda b: [Edit(b.world.path, "src/x.py", "x = 1\n"),
                            Edit(b.world.path, f"src/{CONTEXT_FILE}",
                                 "# src\n\n## Known Issues\n- the box may be zero\n")],
        log=log, max_depth=2, root_path="src")
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root\n", f"src/{CONTEXT_FILE}": "# src\n"},
                                 Task(id="t", prompt="x")))
    assert policy.record_updates == 1


def test_a_file_shaped_path_is_not_a_node_even_before_it_exists():
    """"A node is a **directory**, never a file" -- `agents/manager.ex`, twice.

    The delegation policy refused a path that was already a file in the tree, which is
    the case a real run hit (`src/frontend/lexer.py`, and the child wrote `CONTEXT.md`
    *inside* it). A designed tree contains nothing that exists yet, so that check sees
    nothing: an architect produced `observables/observables.py` as a node and hung two
    children under it. The shape is what is left to go on.
    """
    assert looks_like_file("src/frontend/lexer.py")
    assert looks_like_file("observables/observables.py")
    assert not looks_like_file("src/frontend") and not looks_like_file("src")

    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/observe.py", "measuring")]
                           if not b.world.path else []),
        executor=lambda b: [Edit(b.world.path, f"{b.world.path or '.'}/x.py", "x = 1\n")],
        log=log, max_depth=3)
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root\n"}, Task(id="t", prompt="x")))
    assert policy.mistaken_nodes == 1

    phase = ArchitectPhase(_scripted_architect({
        "": {"record": "# root\n", "children": [{"path": "observables.py", "objective": "no"},
                                                {"path": "observe", "objective": "yes"}]}}),
        max_depth=2)
    tree = phase.design({CONTEXT_FILE: "# root\n"}, "o")
    assert phase.mistaken_nodes == 1
    assert "observe/" + CONTEXT_FILE in tree
    assert "observables.py/" + CONTEXT_FILE not in tree


def test_every_domain_briefs_its_agents_with_an_objective_not_a_catalogue_line():
    """The architect is told what to build, not what the header calls it.

    `--mode b` was first run with `DOMAIN_BLURB` as the objective -- "Lennard-Jones
    molecular dynamics with a frozen driver (6 nodes, 14 files), test suite" -- and the
    architect filled in the rest. Its root record read the specification correctly, the
    API Surface it wrote names `src/geometry.py`, and then its routing table said
    `./geometry/`: the package root the suite imports from was never delegated to. So
    every child that read the same specification built its own `src/` inside its own
    node, and the world came out `integrate/step/src/geometry/displacement/src/`, held
    out at 0.562. The layout below `src/` is the architect's to invent; where the
    package root *is* is the contract, and a brief is where a contract goes.
    """
    from examples.genesis.genesis_recursive_worlds import (DOMAIN_BLURB, DOMAINS,
                                                           objective_for)
    for domain in DOMAINS:
        objective = objective_for(domain)
        assert objective is not DOMAIN_BLURB[domain]
        assert objective != DOMAIN_BLURB[domain]
        # Where the deliverable goes. That much is the contract in every domain --
        # the run that this test exists for lost `src/` out of a routing table.
        assert "`src/`" in objective
        assert len(objective) > len(DOMAIN_BLURB[domain])
        # And what has to be reachable from there, stated however that domain states
        # it: four domains name `src/__init__.py` and its entry points, and `fly`
        # names the four functions its frozen driver imports, because there the
        # public surface is whatever the driver reaches and nothing else.
        assert "`src/__init__.py`" in objective or "`fly.py`" in objective

    # ... and the one thing it does not say is how to decompose below that root.
    for word in ("geometry", "potentials", "forces", "integrate", "observe"):
        assert f"src/{word}" not in md.OBJECTIVE


# ---------------------------------------------------------------------------
# Blind mode: the agents write the tests, and cannot read the acceptance suite
# ---------------------------------------------------------------------------

_ACCEPTANCE = {"acceptance/test_sparse.py": (
    "TOLERANCE = 0.15\n\n\n"
    "def test_the_code_is_sparse_at_every_concentration():\n"
    "    assert 0.04 < 0.05 < TOLERANCE\n")}


def _blind_suite(**kw):
    return SuiteOfTests(name="blind", given={"spec/CONTEXT.md": "# spec\n"},
                     frozen=("spec/**",), hidden=_ACCEPTANCE,
                     requirements={"sparse": "SPEC 1.3 -- the mushroom body code."},
                     **kw)


def test_a_blind_suite_never_puts_its_tests_in_the_repository():
    """Upstream the agents write the tests; the external suite is the experimenter's.

    This port had it backwards -- a human wrote every assertion and then pasted its
    **source** into the prompt, which is the strongest hint there is. An agent handed
    the assertion is not implementing a specification, it is writing to an assertion.
    """
    suite = _blind_suite()
    assert suite.blind
    assert not any(path.startswith("acceptance/") for path in suite.initial_files())
    assert not any("test_" in path for path in suite.initial_files())


def test_a_blind_prompt_carries_the_requirement_and_not_the_assertion():
    suite = _blind_suite()
    task = [t for t in suite.build_tasks() if not t.meta.get("suite")][0]
    assert "SPEC 1.3" in task.prompt
    # The name is a requirement written as a sentence, and that much is deliberate.
    assert "the code is sparse at every concentration" in task.prompt
    # The threshold, the tolerance and the inputs are not.
    assert "TOLERANCE" not in task.prompt and "0.15" not in task.prompt
    assert "assert" not in task.prompt


def test_a_sighted_suite_still_shows_its_source():
    """The default is unchanged: `md` and the others still hand over the test."""
    task = [t for t in md.build_tasks() if not t.meta.get("suite")][0]
    assert "def test_" in task.prompt and "assert" in task.prompt


def test_the_parent_runs_the_tests_the_child_wrote():
    """`mix test`, theirs. A child whose own tests fail has not finished.

    Everything in the artifact is the agents' own work by construction -- the
    acceptance suite is merged into a scratch copy only while one evaluation runs --
    so every `test_*` found here was written by an agent.
    """
    suite = _blind_suite()
    good = {"src/a/kc.py": "X = 1\n", "src/a/test_kc.py": "def test_x():\n    assert 1\n"}
    bad = {"src/a/kc.py": "X = 1\n",
           "src/a/test_kc.py": "def test_x():\n    assert 0, 'nope'\n"}
    assert suite.own_test_failures(good) == []
    assert "nope" in suite.own_test_failures(bad)[0]
    # Scoped to one node's subtree, which is what a parent reviewing one child wants.
    assert suite.own_test_failures(bad, "src/b") == []
    assert suite.own_test_failures(bad, "src/a") != []


def test_a_node_that_wrote_code_and_no_test_is_sent_back():
    """Upstream every node is accountable for its own subtree, and tests are done.

    A node with untested code is a node whose parent has nothing to run.
    """
    suite = _blind_suite()
    review = suite.own_review()

    class _Parent:
        def __init__(self, state, path):
            self.state, self.world = state, LocalWorld(version=0, path=path)

    untested = [Edit("src/a", "src/a/kc.py", "X = 1\n")]
    verdict = review(_Parent({CONTEXT_FILE: "# root\n"}, "src/a"), untested)
    assert verdict is not None and verdict[0] == "rework"
    assert "no test" in verdict[1]

    tested = untested + [Edit("src/a", "src/a/test_kc.py", "def test_x():\n    assert 1\n")]
    assert review(_Parent({CONTEXT_FILE: "# root\n"}, "src/a"), tested) is None

    failing = [Edit("src/a", "src/a/test_kc.py", "def test_x():\n    assert 0\n")]
    verdict = review(_Parent({CONTEXT_FILE: "# root\n"}, "src/a"), failing)
    assert verdict is not None and verdict[0] == "rework"
    assert "your own tests fail" in verdict[1]


def test_the_run_reports_how_many_model_calls_were_in_flight_at_once():
    """`usage.seconds / wallclock` looks like it answers this and does not.

    `seconds` spans the whole process, including the phases that run before
    `evolve()` does; the `wallclock` a stage profile reports covers only the stage. The
    md run's ratio came out at 8.2 with four workers, which is not a measurement of
    anything. This counts the thing directly.
    """
    import threading

    from examples._common import ConcurrencyGauge

    gauge = ConcurrencyGauge()
    assert gauge.peak == 0
    inside = threading.Barrier(3, timeout=10)

    def slow(prompt):
        inside.wait()
        return prompt

    wrapped = gauge.wrap(slow)
    threads = [threading.Thread(target=wrapped, args=("x",)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gauge.peak == 3
    # And it comes back down, so a later quiet phase does not inflate the peak.
    gauge.wrap(lambda p: p)("x")
    assert gauge.peak == 3


def test_a_root_session_gets_a_bigger_turn_budget_than_a_child():
    """Upstream's own two numbers: up to 2,048 model-tool turns at the root, 128 below.

    A root agent is running the whole objective and a child one node of it, so a single
    ceiling for both either starves the root or hands every leaf a session it has no
    use for. The port had one number, 24, for both.
    """
    from examples.genesis._claude_code import ClaudeCodeExecutor

    executor = ClaudeCodeExecutor()
    assert executor.ROOT_TURNS == 2048 and executor.CHILD_TURNS == 128
    assert executor._root_turns == 2048 and executor._max_turns == 128
    root = executor._command("x", executor._root_turns)
    child = executor._command("x", executor._max_turns)
    assert root[root.index("--max-turns") + 1] == "2048"
    assert child[child.index("--max-turns") + 1] == "128"
    # One number given explicitly still means one number, for a cheap run.
    pinned = ClaudeCodeExecutor(max_turns=12)
    assert pinned._root_turns == 12 and pinned._max_turns == 12


def test_a_record_is_maintained_rather_than_written_once():
    """Upstream: 26 `CONTEXT.md` creations and **62 later accepted updates**.

    This port wrote them on exactly two occasions, and a `--mode a` run sat at 0.938
    reading a map of a layout the work had already left behind -- nothing in the
    mechanism could say the record was stale. Upstream's architect does not stop at
    design: it reviews the implementation and re-spawns refinement architects where a
    node misaligns (`agents/architect.ex`).
    """
    from examples.genesis._architect import misaligned

    record = ("## Intent\nx\n\n## API Surface\n`kc.py` does things.\n\n"
              "## Routing Table\n- `./src/a/gone/` -> nothing is here\n")
    drift = misaligned(record, {"src/a/kc.py": "x", "src/a/extra.py": "y"}, "src/a")
    assert "does not exist" in drift            # a route to a directory nobody made
    assert "extra.py" in drift                  # a file the record never mentions
    assert misaligned(record, {"src/a/kc.py": "x", "src/a/gone/y.py": "z"},
                      "src/a") == ""

    # And the hook produces a record edit rather than a verdict, because a verdict
    # cannot fix a map.
    revisions = []

    def refine(path, state):
        revisions.append(path)
        return "## Intent\nrewritten\n"

    policy = RecursiveDelegation(manager=lambda b: [], executor=lambda b: [],
                                 log=WorldLog(), refine=refine)
    state = {CONTEXT_FILE: "# root\n"}
    proposals = policy.propose(_proposal_ctx(state, Task(id="t", prompt="x")))
    assert revisions == [""]
    assert policy.records_revised == 1
    # A leaf that delegated nothing still gets its record refreshed -- a leaf is where
    # the code lands, so its API Surface is the one that goes stale first.
    assert any(CONTEXT_FILE in payload and "rewritten" in payload
               for payload in proposals)


def test_the_routing_table_is_made_to_agree_with_the_children_it_opened():
    """Upstream: the routing table "is your primary delegation tool ... the map that
    makes recursive delegation work". A record whose table disagrees with the children
    the architect just opened is a broken map, and it has to be repaired in *both*
    directions.

    Dropping a refused entry was always necessary -- a table advertising a node nobody
    may write is a trap for the next manager. Adding a forgotten one turned out to
    matter more. An architect answered with five children and a `## Routing Table`
    holding its API Surface instead: `` `__init__.py` — Exports get_regions(...) ``,
    then two sentences about FlyWire. Nothing looked wrong that run, because the
    children were queued from the *reply*. Then the budget ran out, a later phase 1
    resumed from records, rebuilt its queue from the **table**, found no routes, and
    dropped five brain regions in silence.
    """
    from examples.genesis._architect import _fix_routing

    api_surface_in_the_wrong_section = (
        "## Intent\nx\n\n## Routing Table\n"
        "- `__init__.py` — Exports `get_regions(...)`\n"
        "- All regional organization grounded in FlyWire\n")
    fixed = _fix_routing(api_surface_in_the_wrong_section,
                         [{"path": "src/r/antennal_lobe", "objective": "the AL"},
                          {"path": "src/r/mushroom_body", "objective": "the MB"}])
    assert parse_routing(fixed) == ["src/r/antennal_lobe", "src/r/mushroom_body"]
    assert "get_regions" not in fixed

    # A refused child still goes, and one the table already lists is not duplicated.
    listed = ("## Routing Table\n- `./src/r/keep/` -> stays\n"
              "- `./src/r/gone/` -> refused\n")
    fixed = _fix_routing(listed, [{"path": "src/r/keep", "objective": "stays"}])
    assert parse_routing(fixed) == ["src/r/keep"]
    assert fixed.count("src/r/keep") == 1

    # And a record with no routing section at all gets one.
    fixed = _fix_routing("## Intent\nx\n", [{"path": "a/b", "objective": "o"}])
    assert parse_routing(fixed) == ["a/b"]


def test_a_child_the_architect_calls_two_files_is_not_a_directory():
    """The counterweight to "decompose MORE aggressively", enforced rather than stated.

    Upstream says both in the same breath -- decompose harder when the objective feels
    large, *and* single responsibility, shared capability at the lowest common ancestor,
    a directory holding one short function is a directory that did not want splitting.
    The first is a command; the second needs judgement, and a model given both executes
    the first. Measured on the fly domain: 600 nodes for one simulator, three quarters
    of them pure routing, at depth 8. Upstream's 123-hour C compiler run -- 750 files,
    249 000 lines -- has **26** nodes and bottomed out at depth 5.

    So a child now declares how many files it expects to hold, and one that says two or
    fewer is refused: it is two files in this node's API Surface.
    """
    reply = {"record": "# node\n",
             "children": [{"path": "big", "objective": "a lot", "files": 9},
                          {"path": "tiny", "objective": "a function", "files": 1},
                          {"path": "pair", "objective": "two things", "files": 2},
                          {"path": "silent", "objective": "undeclared"}]}
    phase = ArchitectPhase(lambda prompt: json.dumps(reply)
                           if "`./`" in prompt or "path `./`" in prompt
                           else json.dumps({"record": "# leaf\n", "children": []}),
                           max_depth=2, max_nodes=10)
    tree = phase.design({CONTEXT_FILE: "# root\n"}, "o")
    assert phase.too_small == 2
    assert "big/" + CONTEXT_FILE in tree
    assert "tiny/" + CONTEXT_FILE not in tree and "pair/" + CONTEXT_FILE not in tree
    # An undeclared count is not a refusal -- the guard reads what is there and does
    # not invent a number the architect never gave.
    assert "silent/" + CONTEXT_FILE in tree
    # And the refused ones are not left advertised in the routing table either.
    assert parse_routing(tree[CONTEXT_FILE]) == ["big", "silent"]

    prompt = ARCHITECT_PROMPT.format(path="", path_prefix="", objective="o", context="c")
    assert '"files"' in prompt
    assert "three files or fewer, return no children" in prompt


def test_a_node_may_not_be_named_after_one_of_its_own_ancestors():
    """"Decompose MORE aggressively" has a counterweight and upstream states both.

    Single responsibility, and **shared capability belongs at the lowest common
    ancestor**. A directory named after something already above it is that rule broken
    in the one way a tree can show: whatever is really in there belongs to the ancestor,
    or the ancestor's name was wrong, and either way two places claim it. Measured: an
    architect produced `.../receptor/types/catalog/types` and `.../catalog/demographics`
    beside an existing `.../receptor/population/demographics`.
    """
    from examples.genesis._architect import repeats_an_ancestor

    assert repeats_an_ancestor("a/types/catalog/types") == "types"
    assert repeats_an_ancestor("src/brain/mushroom_body/x/mushroom_body") == "mushroom_body"
    assert repeats_an_ancestor("src/brain/circuits/cell_types") == ""
    assert repeats_an_ancestor("src") == ""

    phase = ArchitectPhase(_scripted_architect({
        "src/a": {"record": "# a\n", "children": [{"path": "src/a/a", "objective": "no"},
                                                  {"path": "src/a/b", "objective": "yes"}]}}),
        max_depth=2, root_path="src/a")
    tree = phase.design({CONTEXT_FILE: "# root\n"}, "o")
    assert phase.repeated == 1
    assert "src/a/b/" + CONTEXT_FILE in tree
    assert "src/a/a/" + CONTEXT_FILE not in tree
    # and the refused one is not left advertised in the table either
    assert parse_routing(tree["src/a/" + CONTEXT_FILE]) == ["src/a/b"]


def test_phase_one_designs_a_level_at_a_time_and_siblings_cannot_see_each_other():
    """Upstream an Architect *spawns* sub-architects, which is a statement about
    independence: every node at a depth inherits the chain down to its own parent,
    designed a level ago, so nothing in a level can depend on anything else in it.
    Running them one after another was this port's choice, and it cost the fly domain
    32 minutes for 71 nodes.

    The asks read a snapshot rather than the live tree, so a sibling can never see
    another sibling's record even if it finishes first and the result does not depend
    on which call returns when.
    """
    import threading

    seen, together, lock = [], [], threading.Lock()
    live = [0]
    plan = {"": {"record": "# root\n", "children": [{"path": "a", "objective": "A"},
                                                    {"path": "b", "objective": "B"},
                                                    {"path": "c", "objective": "C"}]}}

    def complete(prompt):
        with lock:
            live[0] += 1
            together.append(live[0])
            seen.append(prompt)
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        for path, reply in plan.items():
            if f"repository path `{path or './'}`" in prompt:
                return json.dumps(reply)
        return json.dumps({"record": "# leaf\n", "children": []})

    phase = ArchitectPhase(complete, max_depth=2, max_nodes=10, workers=3)
    phase.design({}, "o")
    assert len(phase.nodes) == 4                       # root and three children
    assert max(together) >= 2, "the three siblings were designed one after another"
    # No sibling saw another's record: each child's prompt carries the root's chain
    # and nothing from `a`, `b` or `c`.
    children = [p for p in seen if "repository path `./`" not in p]
    assert len(children) == 3
    for prompt in children:
        assert sum(prompt.count(f"`{n}/") for n in "abc") <= 1


def test_a_resumed_phase_one_hands_each_child_its_own_objective():
    """The routing line's right-hand side is the objective, not decoration.

    A phase 1 continuing from records rather than from replies has nothing else to
    give a child. Handing down the *parent's* objective instead sends a leaf the whole
    project: an architect asked to design `.../cell_types/mushroom_body` while carrying
    the root objective came back with `brain, learning, environment, simulation`,
    having redesigned the library from the top at depth five.
    """
    from examples.genesis._world import parse_routes

    record = ("## Routing Table\n"
              "- `./src/brain/` -> the connectome-grounded neural model\n"
              "- `./src/arena/` -> arena physics and odour fields\n")
    assert parse_routes(record) == [("src/brain", "the connectome-grounded neural model"),
                                    ("src/arena", "arena physics and odour fields")]
    # And the paths still come back exactly as `parse_routing` reports them.
    assert [p for p, _ in parse_routes(record)] == parse_routing(record)

    asked = []

    def complete(prompt):
        asked.append(prompt.split("THE OBJECTIVE\n", 1)[1].splitlines()[0])
        return json.dumps({"record": "# leaf\n", "children": []})

    given = {CONTEXT_FILE: "# root\n",
             "src/" + CONTEXT_FILE: "# src\n" + record}
    phase = ArchitectPhase(complete, max_depth=3, max_nodes=10, root_path="src",
                           resume=True)
    phase.design(given, "build the whole product")
    assert phase.reused == 1                       # `src` kept, no call spent
    assert asked == ["the connectome-grounded neural model",
                     "arena physics and odour fields"]
    assert "build the whole product" not in asked


def test_a_node_may_not_shadow_a_sibling_module():
    """`rdf/` next to `rdf.py` are one name to Python, and the module wins.

    Everything the child then writes in that directory is unreachable from every
    import in the repository, and it cannot be repaired from either side: the parent
    that forwards `rdf.py` into `src.observe.rdf.rdf` gets `'src.observe.rdf' is not a
    package`, because the name resolves back to the file doing the forwarding. Three
    runs and 30 000 rollouts never got that repository off 0.812. So the collision is
    refused where it is created -- and where it was inherited instead, mode A says so
    in the record, because the guard cannot undo history.
    """
    assert shadowed_by_module({"src/observe/rdf.py": "x"}, "src/observe/rdf")
    assert shadowed_by_module({"a/b.pyi": "x"}, "a/b")
    assert not shadowed_by_module({"src/observe/rdf.py": "x"}, "src/observe/thermo")
    assert not shadowed_by_module({"src/observe/rdf/rdf.py": "x"}, "src/observe/rdf")

    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda b: ([Delegation("src/observe/rdf", "the histogram")]
                           if not b.world.path else []),
        executor=lambda b: [Edit(b.world.path, f"{b.world.path or '.'}/x.py", "x = 1\n")],
        log=log, max_depth=3)
    policy.propose(_proposal_ctx({CONTEXT_FILE: "# root\n", "src/observe/rdf.py": "x"},
                                 Task(id="t", prompt="x")))
    assert policy.shadowed_nodes == 1 and policy.mistaken_nodes == 0

    # The architect refuses a designed one the same way.
    phase = ArchitectPhase(_scripted_architect({
        "": {"record": "# root\n", "children": [{"path": "rdf", "objective": "no"},
                                                {"path": "thermo", "objective": "yes"}]}}),
        max_depth=2)
    tree = phase.design({CONTEXT_FILE: "# root\n", "rdf.py": "x"}, "o")
    assert phase.mistaken_nodes == 1
    assert "thermo/" + CONTEXT_FILE in tree and "rdf/" + CONTEXT_FILE not in tree

    # And the extractor, handed one that already exists, writes it down.
    reply = json.dumps({"record": "## Intent\nObserve.\n"})
    extract = ExtractPhase(lambda prompt: reply, max_depth=2)
    described = extract.extract({CONTEXT_FILE: "# root\n", "src/observe/rdf.py": "x",
                                 "src/observe/rdf/rdf.py": "y"}, "o")
    assert extract.shadowed == 1
    note = described["src/observe/" + CONTEXT_FILE]
    assert KNOWN_ISSUES in note and "One of the two has to go." in note
    assert "the name resolves back to the forwarding file" in note


def test_a_file_may_not_shadow_a_node_directory_of_the_same_name():
    """The same collision from the other side, and the side that actually bit.

    `--mode b` reached 1.000 and its root agent still refused to sign the objective
    off: "the tree contains several empty `__init__.py` files and stub modules (e.g.
    `src/potentials/kernels/lennard_jones/__init__.py`) that are dead scaffolding".
    It was right that the node was dead and wrong about why. The manager at
    `src/potentials/kernels` had delegated `lennard_jones/` to a child and then, on
    its own accountability turn, written `lennard_jones.py` beside it. One name, and
    the module wins: the child's whole node -- 41 bytes of
    `from .lennard_jones import lennard_jones` -- became unreachable from every import
    in the repository.

    `_is_node` cannot catch this one: at delegation time the file did not exist yet.
    """
    assert shadows_package({"k/lj/__init__.py": "x"}, "k/lj.py")
    assert shadows_package({"k/lj/deep/x.py": "x"}, "k/lj.pyi")
    assert not shadows_package({"k/lj/__init__.py": "x"}, "k/harmonic.py")
    assert not shadows_package({"k/lj/__init__.py": "x"}, "k/CONTEXT.md")
    # A prefix that is not a path segment is a different directory, not a collision.
    assert not shadows_package({"k/ljx/__init__.py": "x"}, "k/lj.py")

    # The leaf path drops it...
    policy = RecursiveDelegation(
        manager=lambda b: [], log=WorldLog(),
        executor=lambda b: [Edit("k", "k/lj.py", "shadowing\n"),
                            Edit("k", "k/harmonic.py", "fine\n")])
    state = {CONTEXT_FILE: "# root\n", "k/" + CONTEXT_FILE: "# k\n",
             "k/lj/__init__.py": "from .lj import lj\n"}
    policy.propose(_proposal_ctx(state, Task(id="t", prompt="x")))
    assert policy.shadowing_edits == 1

    # ...and so does the accountability turn, which is where it actually happened.
    own = RecursiveDelegation(
        manager=lambda b: ([Delegation("k/lj", "the kernel")] if b.world.path == "k"
                           else []),
        executor=lambda b: ([Edit(b.world.path, "k/lj/__init__.py", "x = 1\n")]
                            if b.world.path == "k/lj"
                            else [Edit(b.world.path, "k/lj.py", "shadowing\n")]),
        log=WorldLog(), max_depth=3)
    own._episode(LocalWorld(version=0, path="k"), "build it",
                 {CONTEXT_FILE: "# root\n", "k/" + CONTEXT_FILE: "# k\n"},
                 _proposal_ctx({CONTEXT_FILE: "# root\n"}, Task(id="t", prompt="x")),
                 depth=0, rollout=None, base=None)
    assert own.shadowing_edits == 1


def test_a_review_finding_reaches_the_manager_that_can_act_on_it():
    """A parent that returns work keeps the finding for its own turn.

    `--mode a` over a repository whose `src/observe/rdf.py` was a zero stub shadowing
    the real implementation at `src/observe/rdf/rdf.py`. The parent review diagnosed it
    exactly -- "the actual file src/observe/rdf/rdf.py defines a function named rdf,
    not histogram, so this import will fail at runtime" -- and rejected the child over
    it. But `rdf.py` belongs to the *parent*; no child may write it. The finding went
    into the child's `## Known Issues`, the node sat in `open rework` for 10 003
    rollouts across two samplers, and the repository never moved off 0.812 -- while
    rewriting that one parent-owned file takes it to 65/65.

    Upstream review and accountability are one phase, run by the agent "ACCOUNTABLE for
    all code in its node path". So the finding travels to the turn that can act on it.
    """
    seen = []

    def manager(brief):
        return [Delegation("src/observe/rdf", "implement the histogram")] \
            if brief.world.path == "src/observe" else []

    def executor(brief):
        seen.append((brief.world.path, brief.objective))
        if brief.world.path == "src/observe/rdf":
            return [Edit("src/observe/rdf", "src/observe/rdf/rdf.py", "def rdf(): ...\n")]
        return []

    policy = RecursiveDelegation(
        manager=manager, executor=executor, log=WorldLog(), max_depth=3,
        review=lambda brief, edits: ("rejected", "defines rdf, not histogram"))
    state = {CONTEXT_FILE: "# root\n", "src/observe/" + CONTEXT_FILE: "# observe\n"}
    policy._episode(LocalWorld(version=0, path="src/observe"), "build it", state,
                    _proposal_ctx(state, Task(id="t", prompt="x")), depth=0,
                    rollout=None, base=None)

    own = [objective for path, objective in seen if path == "src/observe"]
    assert own, "the manager never took its own accountability turn"
    # The finding, the node it came from, and whose job it now is.
    assert "defines rdf, not histogram" in own[-1]
    assert "`src/observe/rdf`" in own[-1]
    assert "accountable for every file at `src/observe`" in own[-1]
    assert policy.accountability_findings == 1

    # No finding, no noise: an episode whose review returned nothing reads as before.
    quiet = RecursiveDelegation(manager=lambda b: [], executor=executor, log=WorldLog())
    seen.clear()
    quiet._episode(LocalWorld(version=0, path="src/observe"), "build it", state,
                   _proposal_ctx(state, Task(id="t", prompt="x")), depth=0,
                   rollout=None, base=None)
    assert "Your review returned work" not in seen[-1][1]
    assert quiet.accountability_findings == 0


def test_phase_one_designs_the_codebase_and_not_the_repository_around_it():
    """The root record is generated, and phase 1 starts at the package root.

    Situated at the repository root and briefed on a library at `src/`, the architect
    resolved the contradiction by deciding it *was* `src`: a record titled `# src`, an
    API Surface naming `__init__.py`, and children hung at the repository root where
    nothing imports them -- three samples out of three, and a prompt line saying its own
    path was its own directory did not move it. Upstream starts its architect on the
    codebase it creates; the specification, the suite and the entry point are the
    harness around that codebase and predate every agent. So the root is generated from
    what the domain already declares, and every node below it is still invented.
    """
    from examples.genesis._architect import harness_record
    from examples.genesis.genesis_recursive_worlds import DOMAINS, objective_for
    from examples.genesis.genesis_recursive_worlds import package_root

    for domain, spec in DOMAINS.items():
        root = package_root(domain)
        assert root == posixpath.dirname(spec.ENTRY) and "/" not in root
        record = harness_record(root, spec.FROZEN, objective_for(domain))
        # It is a record like any other: the four standard sections, and one route.
        assert missing_sections(record) == []
        assert parse_routing(record) == [root]
        # And it says what the architect kept getting wrong, in the section that binds.
        for frozen in spec.FROZEN:
            assert f"`{frozen}`" in record
        assert "Nothing here is implementation" in record


def test_the_architect_is_told_its_api_surface_stops_at_its_own_directory():
    """The record-level half of the same lesson, stated in the prompt.

    A node's API Surface names its own files; anything deeper is a child's, and the way
    to reach a child is the routing table. An architect that describes files it never
    routes to has designed something nobody is accountable for.
    """
    prompt = ARCHITECT_PROMPT.format(path="src", path_prefix="src/",
                                     objective="o", context="c")
    assert "API Surface may only name files inside your own directory" in prompt
    assert "src/<file>" in prompt
    assert "never reaches" in prompt


# ---------------------------------------------------------------------------
# An episode as a Claude Code session: permissions, scope, and what comes back
# ---------------------------------------------------------------------------

def _fake_claude(tmp_path, script):
    """A stand-in `claude` binary that edits the worktree it is run in."""
    binary = tmp_path / "claude"
    binary.write_text("#!/usr/bin/env python3\n" + script)
    binary.chmod(0o755)
    return str(binary)


FAKE = '''
import json, os, sys
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
prompt = sys.argv[sys.argv.index("-p") + 1]
open("scope.txt", "w").write(prompt)
os.makedirs("src/core", exist_ok=True)
open("src/core/vectors.py", "w").write("def minimum_image(a, b, box):\\n    return b\\n")
open("src/elsewhere.py", "w").write("# outside the node\\n")
open("tests/test_geometry.py", "w").write("# tried to edit the suite\\n")
print(json.dumps({"is_error": False, "num_turns": 7}))
'''


def test_an_episode_can_be_a_claude_code_session_and_its_edits_are_still_the_ports(tmp_path):
    """The contract does not move when the episode becomes a session.

    An edit outside the node is a request, not an error (`agents/executor.ex`); a write
    to a frozen path is dropped and counted; and what the session did is read from the
    worktree rather than parsed out of a reply.
    """
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, binary=_fake_claude(tmp_path, FAKE))
    state = dict(md.initial_files())
    edits = executor(Brief(world=LocalWorld(version=1, path="src/core",
                                            readonly=md.FROZEN),
                           objective="make displacement work", context="", state=state,
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=2))
    by_path = {e.path: e for e in edits}
    assert "src/core/vectors.py" in by_path            # its own node: work
    assert by_path["src/core/vectors.py"].owner == "src/core"
    assert "src/elsewhere.py" in by_path               # outside: still returned...
    assert not owns("src/core", "src/elsewhere.py")    # ...and the port makes it a request
    assert executor.sessions == 1 and executor.turns == 7
    assert executor.edits == 1 and executor.requests >= 1

    # the session ran in a throwaway copy, so the real state is untouched
    assert state == md.initial_files()


def test_the_session_is_fenced_before_the_contract_ever_sees_it(tmp_path):
    """Three fences, and this is the first two: the frozen globs are denied by name in
    the session's own settings, and the network tools are off."""
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, binary="claude")
    command = executor._command("do the thing", executor.CHILD_TURNS)
    assert "--disallowedTools" in command
    assert "WebFetch,WebSearch,Task" in command
    assert "--permission-mode" in command and "acceptEdits" in command
    assert "--max-turns" in command

    workspace = str(tmp_path)
    executor._write_settings(workspace)
    settings = json.loads((tmp_path / ".claude" / "settings.local.json").read_text())
    denied = settings["permissions"]["deny"]
    assert "Write(tests/**)" in denied and "Edit(spec/**)" in denied
    assert any(d.startswith("Write(md.py") for d in denied)


def test_the_brief_tells_the_session_what_it_may_not_touch(tmp_path):
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, binary=_fake_claude(tmp_path, FAKE))
    executor(Brief(world=LocalWorld(version=1, path="src/core", readonly=md.FROZEN),
                   objective="o", context="", state=dict(md.initial_files()),
                   task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=2))
    # the fake binary wrote the prompt it was given into the worktree, and the worktree
    # is gone -- so read it back from the edits instead
    # (the prompt names the node and the frozen set)
    prompt = CLAUDE_CODE_BRIEF.format(path="src/core", objective="o",
                                      frozen=", ".join(md.FROZEN), failure="")
    assert "write ONLY files under it" in prompt
    assert "tests/**" in prompt and "read-only" in prompt
    assert "judged by; editing it is not a way to pass it" in prompt


def test_a_session_that_dies_costs_its_episode_and_nothing_else(tmp_path):
    dying = _fake_claude(tmp_path, 'import sys\nif "--version" in sys.argv:\n'
                                   '    print("0.0.0"); raise SystemExit(0)\n'
                                   'raise SystemExit(3)\n')
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, binary=dying)
    edits = executor(Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                           objective="o", context="", state=dict(md.initial_files()),
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    assert edits == [] and executor.failed == 1


#: A session that does some work and then never finishes. The wall, not the turn
#: budget, is what ends an episode in practice: of 52 executor sessions in one fly
#: run, 27 ran into the wall and the busiest reached 97 of its 128 turns.
SLOW = '''
import os, sys, time
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
os.makedirs("src/core", exist_ok=True)
open("src/core/vectors.py", "w").write("def minimum_image(a, b, box):\\n    return b\\n")
time.sleep(120)
'''


def test_a_session_that_runs_out_of_wall_keeps_what_it_wrote(tmp_path):
    """The wall is an interruption, not a verdict.

    A session killed at the wall wrote what it wrote, and the port reads the worktree
    rather than the exit status, so the work comes back. What the wall costs is the
    report -- there is no JSON after a SIGKILL -- which is why the turn counter read
    309 for a run whose transcripts held 2 607 assistant turns, and why the timeout
    is counted apart from every other way a session can fail.
    """
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, timeout=3.0,
                                  binary=_fake_claude(tmp_path, SLOW))
    edits = executor(Brief(world=LocalWorld(version=1, path="src/core",
                                            readonly=md.FROZEN),
                           objective="o", context="", state=dict(md.initial_files()),
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=2))
    assert [e.path for e in edits] == ["src/core/vectors.py"]
    assert executor.failed == 1 and executor.timeouts == 1
    assert executor.turns == 0          # no report to read: the count is not the truth


#: A session that starts a server the way the brief tells it to run the suite, and
#: then hangs. `subprocess.run(timeout=)` signals only the CLI, so one run left a
#: `python3 _cli.py serve` listening with its working directory already deleted.
LEAKY = '''
import os, subprocess, sys, time
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
subprocess.Popen([sys.executable, "-c",
                  "import os, sys, time\\n"
                  "open(sys.argv[1], 'w').write(str(os.getpid()))\\n"
                  "time.sleep(120)", "MARKER"])
while not os.path.exists("MARKER") or not open("MARKER").read().strip():
    time.sleep(0.05)
time.sleep(120)
'''


def test_a_session_does_not_outlive_the_episode_that_started_it(tmp_path):
    """The group is killed, not the process.

    An executor has a shell because it has to run the suite it is judged by, and a
    suite run starts servers. The session leads its own process group and the group
    goes when the session does -- otherwise the thing the session started is still
    holding a port when the next 43 episodes come round.
    """
    marker = tmp_path / "grandchild.pid"
    binary = _fake_claude(tmp_path, LEAKY.replace("MARKER", str(marker)))
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, timeout=5.0, binary=binary)
    executor(Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                   objective="o", context="", state=dict(md.initial_files()),
                   task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    assert executor.timeouts == 1
    pid = int(marker.read_text().strip())
    for _ in range(100):                       # the signal is delivered, not instant
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        raise AssertionError(f"the session's grandchild {pid} outlived the episode")


#: Reports back whatever reasoning cap reached the CLI, which is how the run's own
#: number is checked rather than the one the class would have defaulted to.
ECHO_THINKING = '''
import json, os, sys
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
open("thinking.txt", "w").write(os.environ.get("MAX_THINKING_TOKENS", "(unset)"))
print(json.dumps({"is_error": False, "num_turns": 2}))
'''


def test_the_executor_gets_the_same_session_settings_as_every_other_role(tmp_path):
    """The role that does the work is not the role that runs on defaults.

    The architect, the manager, the reviewer and the extractor were all built from the
    run's session settings; the executor was built from three of its own arguments and
    inherited neither the wall nor the reasoning cap. It ran a whole fly domain at a
    900 s default nobody had chosen while `--timeout 600` and `--thinking-tokens 2048`
    applied to every other role -- so the numbers have to reach it too.
    """
    executor = ClaudeCodeExecutor(frozen=md.FROZEN, timeout=17.0, thinking_tokens=2048,
                                  binary=_fake_claude(tmp_path, ECHO_THINKING))
    edits = executor(Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                           objective="o", context="", state=dict(md.initial_files()),
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    assert executor._timeout == 17.0
    assert {e.path: e.content for e in edits}["thinking.txt"] == "2048"


def test_the_session_wall_is_not_the_model_call_timeout():
    """Two different numbers, and sharing one was how the executor lost both.

    `--timeout` is the timeout on one model call -- `examples/_common` says so in its
    own help, and this port defaults it to 120 s. A session is a loop of many calls,
    so a 120 s "timeout" would kill every one of them before it read a file.
    """
    args = genesis.build_parser().parse_args([])
    assert args.timeout == 120.0                       # one model call
    assert args.session_timeout == ClaudeCodeExecutor.TIMEOUT
    assert args.session_timeout > args.timeout


def test_the_session_decisions_are_made_before_the_roles_that_read_them():
    """`--mode a` builds the extractor, and it is the first role built.

    Both the session settings and the "is there a CLI at all" fallback were written
    below it, so a `--mode a --agent-sessions` run raised `NameError: use_sessions`
    where it looked like it would fall back to a completion. Order is the fix, and
    order is what this checks.
    """
    tree = ast.parse(inspect.getsource(genesis.main))
    defined = {}
    mode_a = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_session_kwargs":
            defined["_session_kwargs"] = node.lineno
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) \
                and node.id == "use_sessions":
            defined.setdefault("use_sessions", node.lineno)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) \
                and node.left.attr == "mode" \
                and any(getattr(c, "value", None) == "a" for c in node.comparators):
            mode_a.append(node.lineno)
    assert set(defined) == {"_session_kwargs", "use_sessions"}
    extractor = max(mode_a)                     # the branch that builds the extractor
    assert defined["_session_kwargs"] < extractor
    assert defined["use_sessions"] < extractor


#: An architect that starts implementing. The record is the job; the module beside
#: it is the failure mode, and it costs nothing here because the workspace is a copy.
BUSY_ARCHITECT = '''
import json, os, sys
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
os.makedirs("src", exist_ok=True)
open("src/CONTEXT.md", "w").write("# src\\n\\n## Routing Table\\n- `./core/` (2 files) -> cores\\n")
open("src/eager.py", "w").write("# an architect that could not help itself\\n")
print(json.dumps({"is_error": False, "num_turns": 4}))
'''


def test_a_design_session_that_writes_code_is_counted_not_believed(tmp_path):
    """`strays=0` has to mean nothing strayed.

    The counter was reported on every phase-1 line and incremented nowhere, by a
    helper that walked the workspace with a module its file never imported -- so it
    said zero whatever the architect did. The session owns the workspace and deletes
    it, so the count is taken there, before it is gone.
    """
    from examples.genesis._architect_session import ArchitectSession

    designer = ArchitectSession(binary=_fake_claude(tmp_path, BUSY_ARCHITECT))
    record, children = designer({}, "src", "design it")
    assert record and [c["path"] for c in children] == ["src/core"]
    assert designer.strays == 1                       # src/eager.py, and not the record
    assert "strays=1" in designer.summary()


#: Reports back the session's own situation: what it was invoked with, and where the
#: CLI was told to keep its state.
ECHO_ENV = '''
import json, os, sys
if "--version" in sys.argv:
    print("0.0.0 (fake)"); raise SystemExit(0)
open("argv.txt", "w").write("\\n".join(sys.argv))
open("home.txt", "w").write(os.environ.get("CLAUDE_CONFIG_DIR", "(unset)"))
open("ident.txt", "w").write(",".join(k for k in HOST if os.environ.get(k)))
print(json.dumps({"is_error": False, "num_turns": 1}))
'''


def test_a_session_does_not_inherit_the_host_s_identity_or_its_state(tmp_path,
                                                                    monkeypatch):
    """A session launched from a session is not that session.

    Inherited, the identity variables make every episode in a run *be* the host: one
    fly run's 52 episodes each wrote a transcript named with the host's session id, and
    their TodoWrite state -- keyed by that id -- landed in the host's own task list.
    `CLAUDE_CONFIG_DIR` then moves transcripts, todos and synced skills out of
    `~/.claude`, which one run had left 685 project directories in.
    """
    from examples.genesis._session import (HOST_SESSION_VARS, session_env,
                                           session_home)

    for name in HOST_SESSION_VARS:
        monkeypatch.setenv(name, "the-host")
    env = session_env()
    assert not [k for k in HOST_SESSION_VARS if k in env]
    assert env["CLAUDE_CONFIG_DIR"] == session_home()
    assert not session_home().startswith(os.path.expanduser("~/.claude"))

    script = ECHO_ENV.replace("HOST", repr(list(HOST_SESSION_VARS)))
    executor = ClaudeCodeExecutor(frozen=md.FROZEN,
                                  binary=_fake_claude(tmp_path, script))
    edits = executor(Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                           objective="o", context="", state=dict(md.initial_files()),
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    wrote = {e.path: e.content for e in edits}
    assert wrote["ident.txt"] == ""                    # none of them reached the session
    assert wrote["home.txt"] == session_home()


def test_a_session_with_its_own_key_is_not_given_the_host_s_whole_situation(tmp_path,
                                                                           monkeypatch):
    """31 850 input tokens plain against 1 317 bare, same endpoint, same prompt.

    A CLI launched from inside a Claude Code session inherits that session's MCP
    servers, skills, agent list, the user's email and a system prompt about reviewing
    pull requests -- a 24x prefix on every turn of every episode, none of it about the
    objective. `--bare` drops it, and reads credentials strictly from
    `ANTHROPIC_API_KEY`, which is why it is used only when the run brought one: a
    session billed to the local CLI's own sign-in makes no API call at all with it.
    """
    from examples.genesis._session import isolation_flags

    assert isolation_flags({}) == []
    assert isolation_flags({"ANTHROPIC_AUTH_TOKEN": "t"}) == []    # OAuth-shaped: no
    assert isolation_flags({"ANTHROPIC_API_KEY": "k"}) == ["--bare",
                                                           "--strict-mcp-config"]

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    script = ECHO_ENV.replace("HOST", "[]")
    executor = ClaudeCodeExecutor(frozen=md.FROZEN,
                                  binary=_fake_claude(tmp_path, script))
    edits = executor(Brief(world=LocalWorld(version=1, path="src", readonly=md.FROZEN),
                           objective="o", context="", state=dict(md.initial_files()),
                           task=md.build_tasks()[0], output="FAIL", reward=0.0, depth=1))
    argv = {e.path: e.content for e in edits}["argv.txt"].splitlines()
    assert "--bare" in argv and "--strict-mcp-config" in argv
    # the fences are not what bare drops: the deny list is still on the command line
    assert "--disallowedTools" in argv


def test_bare_mode_is_asked_only_for_the_tools_it_carries():
    """`--bare` does not expose `Write`, and no flag brings it back.

    Measured against the real CLI: a bare session given `--tools Read,Write,Glob,Grep`
    answers "I only have a file Read tool available". An executor in one run spent a
    turn discovering it -- `No such tool available: Write. Write is disabled for this
    session` -- and then wrote every file through `cat > f << EOF` instead, while the
    same code without `--bare` had made 1,910 successful `Write` calls. `Edit` is there
    and creates a file that does not exist, which is how all seven phase-1 records in
    that run were written by sessions with no shell at all.
    """
    from examples.genesis._session import READ_WRITE_TOOLS, available_tools

    bare = ["--bare", "--strict-mcp-config"]
    assert available_tools(READ_WRITE_TOOLS, []) == list(READ_WRITE_TOOLS)
    kept = available_tools(READ_WRITE_TOOLS, bare)
    assert "Write" not in kept and "Edit" in kept
    # a role whose only job is to write one file keeps a way to write it
    assert "Edit" in available_tools(("Read", "Glob", "Grep", "Write"), bare)

    executor = ClaudeCodeExecutor(frozen=md.FROZEN, binary="claude")
    command = executor._command("do it", 8, {"ANTHROPIC_API_KEY": "k"})
    allowed = command[command.index("--allowedTools") + 1].split(",")
    assert "--bare" in command and "Write" not in allowed and "Edit" in allowed
    plain = executor._command("do it", 8, {})
    assert "--bare" not in plain
    assert "Write" in plain[plain.index("--allowedTools") + 1].split(",")


def test_the_brief_keeps_the_session_inside_the_checkout():
    """The blind property is a rule, not a wall, and the rule has to be stated.

    Four of twelve episodes in one run ran `find /` and read a previous run's output
    from `/tmp`; one opened the very `_cli.py` that answered the acceptance failure it
    had been handed to reproduce. A shell can read whatever the process can, so the
    worktree bounds what survives and not what is seen.
    """
    brief = CLAUDE_CODE_BRIEF.format(path="src", objective="o", frozen="REQUIREMENTS.md",
                                     failure="")
    assert "Do not read outside it" in brief
    assert "find /" in brief
    assert "not this repository's state" in brief


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


# -- phase 1's architect as a session -------------------------------------------

def test_a_routing_heading_is_read_at_any_level_or_numbering():
    """The routing table is the map delegation runs on, so a heading written the way
    an agent writes markdown -- deeper, numbered, lowercased -- must still open it.

    Measured: a session architect wrote a complete five-child routing table under a
    heading the exact-prefix match rejected, and the tree recorded the node as a leaf.
    """
    from examples.genesis._world import parse_route_sizes, parse_routes

    for heading in ("## Routing Table", "### Routing Table", "## Routing table",
                    "## Routing", "#### Routing", "## 4. Routing Table",
                    "## Routing Table (children)"):
        body = f"{heading}\n\n- `./src/brain/` (3 files) -> the circuits\n"
        assert parse_routes(body) == [("src/brain", "the circuits")], heading
        assert parse_route_sizes(body) == {"src/brain": 3}, heading
    # and a heading that only mentions routing is still not the section
    assert parse_routes("## Non-routing notes\n\n- `./src/brain/` -> x\n") == []


def test_a_routing_line_may_declare_the_child_s_size():
    """A session architect writes `CONTEXT.md` and returns no structured reply, so the
    file-count that refuses a child too small to be a directory has to live in the
    routing line. A record without counts still routes; it just declares nothing.
    """
    from examples.genesis._world import parse_route_sizes, parse_routes

    sized = ("## Routing Table\n\n"
             "- `./src/brain/` (12 files) -> circuits\n"
             "- `./src/web/` -> the page\n")
    assert parse_routes(sized) == [("src/brain", "circuits"), ("src/web", "the page")]
    assert parse_route_sizes(sized) == {"src/brain": 12}
    assert parse_route_sizes("## Routing Table\n\n- `./src/brain/` -> x\n") == {}


def test_the_session_architect_returns_what_the_completion_one_does():
    """`ArchitectSession` is a drop-in for the completion: same return shape, so the
    phase's guards -- the file-count threshold, the shadowing refusals -- apply to
    both. A record the session did not write is `(None, [])`, which the phase counts
    as an unusable reply, because a node nobody designed is one event either way.
    """
    from examples.genesis._architect_session import _children

    record = ("## Routing Table\n\n"
              "- `./src/brain/` (12 files) -> circuits\n"
              "- `web/` (4 files) -> the page\n")
    kids = _children(record, "src/")
    assert [k["path"] for k in kids] == ["src/brain", "src/web"]
    assert [k["files"] for k in kids] == [12, 4]
    assert all(k["objective"] for k in kids)


def test_the_design_rules_are_shared_by_both_delivery_paths():
    """One set of rules, two ways to deliver the record. Duplicating them is how the
    completion path and the session path would silently grow different trees.
    """
    from examples.genesis._architect import ARCHITECT_PROMPT, ARCHITECT_RULES

    assert ARCHITECT_PROMPT.startswith(ARCHITECT_RULES)
    assert "ONE JSON object" in ARCHITECT_PROMPT
    assert "ONE JSON object" not in ARCHITECT_RULES


def test_a_design_session_is_given_no_shell():
    """The implementation executor gets Bash because it has to run the suite it is
    judged by. An architect designs and does not implement, and a shell is how a
    design session becomes an implementation session by accident.
    """
    from examples.genesis._architect_session import ArchitectSession

    command = ArchitectSession(model="m").session._command("hi")
    allowed = command[command.index("--allowedTools") + 1]
    assert "Bash" not in allowed
    assert "Bash" in command[command.index("--disallowedTools") + 1]


# -- the manager and the extractor as sessions ----------------------------------

def test_a_manager_plan_is_read_line_by_line():
    """A plan of five children whose fourth line is malformed delegates four, not
    none. That is the whole reason the deliverable is lines and not JSON: the failure
    these sessions exist to remove is "the structured reply did not parse".
    """
    from examples.genesis._roles import _plan

    assert _plan("# heading\n"
                 "- `src/brain` -> build the circuits\n"
                 "src/world -> arena and odour\n"
                 "a line with no arrow at all\n"
                 "* src/web: the page\n") == [
        ("src/brain", "build the circuits"),
        ("src/world", "arena and odour"),
        ("src/web", "the page")]
    assert _plan("") == []          # an empty plan means "handle it here"


def test_a_review_verdict_that_does_not_speak_is_not_a_rejection():
    """The child did the work; a reviewer that cannot say ACCEPT or REJECT is not
    evidence against it. Same rule the completion reviewer already held.
    """
    from examples.genesis._roles import _verdict

    assert _verdict("ACCEPT\nlooks fine") == ("accept", "looks fine")
    assert _verdict("**REJECT**\nimport does not resolve") == (
        "reject", "import does not resolve")
    assert _verdict("I think it is probably ok") == (None, "")
    assert _verdict("") == (None, "")


def test_a_reviewing_session_is_shown_the_child_s_real_files():
    """`ReviewSession` runs over the candidate -- the parent's state with the child's
    edits applied -- so Read and Grep reach the work itself. The completion reviewer
    saw a diff rendered and truncated at 12 000 characters.
    """
    import examples.genesis._roles as roles

    seen = {}

    class _Fake:
        sessions = failed = turns = 0

        def run(self, state, prompt, *, read):
            seen["state"] = dict(state)
            seen["prompt"] = prompt
            return {read[0]: "ACCEPT\n"}

        def summary(self):
            return ""

    review = roles.ReviewSession.__new__(roles.ReviewSession)
    review._contracts = ()
    review.session = _Fake()
    review.reviewed = review.rejected = review.unparsed = 0

    class _World:
        path = "src"

        def situate(self, state, contracts=()):
            return "(context)"

    class _Parent:
        world = _World()
        state = {"src/CONTEXT.md": "# src\n"}
        objective = "build it"

    class _Edit:
        kind = "work"

        def __init__(self, path, content):
            self.path, self.content = path, content

    assert review(_Parent(), [_Edit("src/brain/kc.py", "X = 1\n")]) is None
    assert seen["state"]["src/brain/kc.py"] == "X = 1\n"
    assert "src/brain/kc.py" in seen["prompt"]


def test_an_extractor_session_replaces_files_in_the_prompt():
    """Upstream's ContextExtractor reads code with a tool. Without one this port had
    to put the node's own files in the prompt -- capped at eight files of six thousand
    characters, and the run before that cap invented an API surface off file names.
    """
    import inspect

    from examples.genesis._extract import ExtractPhase
    from examples.genesis._roles import EXTRACT_BRIEF

    assert "session" in inspect.signature(ExtractPhase.__init__).parameters
    # the record is a file it writes, not a reply it returns
    assert "{record}" in EXTRACT_BRIEF
    # and it is told it may not change the code
    assert "may not change the code" in EXTRACT_BRIEF


def test_a_read_only_role_gets_no_shell_and_keeps_only_what_it_was_asked_for(tmp_path):
    """The fence that holds is the deny list and the readback, not the allow list.

    `--allowedTools` is an auto-approve list, not a whitelist: measured over one run's
    role sessions, the manager used `Edit` 41 times and the reviewer 6, and `Edit` was
    never in either one's allowed tools. `--disallowedTools` *is* a fence -- no role
    session in that run ran a shell, and none reached the network tools -- and the
    containment for everything else is :meth:`AgentSession.run`, which returns the paths
    the caller named and nothing else. `agent/tools.ex` gives `:read` agents the read
    half plus `context_write`; that is what this comes to here.
    """
    from examples.genesis._roles import ExtractSession, ManagerSession

    for role in (ExtractSession(model="m"),
                 ManagerSession(lambda p, o: (p, o), model="m")):
        command = role.session._command("hi")
        denied = command[command.index("--disallowedTools") + 1]
        assert "Bash" in denied and "WebFetch" in denied and "Task" in denied

    # ...and a session that writes where it was not asked is not believed.
    from examples.genesis._session import AgentSession

    session = AgentSession(binary=_fake_claude(tmp_path, BUSY_ARCHITECT))
    got = session.run({}, "design it", read=["src/CONTEXT.md"])
    assert set(got) == {"src/CONTEXT.md"}          # src/eager.py was written, not read
    assert session.changed == ["src/eager.py"]     # seen and counted, never returned
