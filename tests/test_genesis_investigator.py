"""Upstream's ``Investigator``, narrow: a read-only investigation that persists.

The narrow scope is deliberate. Upstream's investigator also fans out to
``subagent_investigator`` at child nodes, and that part is *not* ported here -- the
port's session work gives a manager read tools of its own, which is what fan-out is
for. What is left, and what nothing else in this port does, is the other half: a
read-only delegation whose findings are **persisted into the version**, so a later
agent inherits them instead of re-discovering them.

The gap it fills is that ``_MANAGER_PROMPT`` carries the ``CONTEXT.md`` chain and a
listing of files, never their contents -- this port's agents have no read tool. An
investigator is the one role allowed to read a node's code and leave a note about it.
"""

from __future__ import annotations

import json

from agentdescent.filetree import canonical
from agentdescent.evolution import Task
from agentdescent.policies import ProposalContext

from examples.genesis._delegation import (Brief, Delegation, Edit,
                                          RecursiveDelegation)
from examples.genesis._investigator import (FINDINGS_HEADING, Investigator)
from examples.genesis._suite import llm_manager
from examples.genesis._world import CONTEXT_FILE, LocalWorld, WorldLog

_STATE = {
    CONTEXT_FILE: "# root\n",
    "src/" + CONTEXT_FILE: "# src\n",
    "src/core.py": "from src.util import g\n\ndef f():\n    return g()\n",
    "src/util.py": "def g():\n    return 42\n",
    "src/deep/" + CONTEXT_FILE: "# deep\n",
    "src/deep/leaf.py": "def h():\n    return 0\n",
}


def _brief(state, path, objective="what does this node do?"):
    world = LocalWorld(version=1, path=path)
    return Brief(world=world, objective=objective,
                 context=world.situate(state), state=dict(state),
                 task=Task(id="t", prompt="x"), output="", reward=0.0, depth=1)


def _ctx(state, reward=0.0):
    return ProposalContext(rendered=canonical(state),
                           task=Task(id="t", prompt="x"), output="",
                           reward=reward, base_version=1)


def _reply(findings="- `src/core.py` imports `g` from `src/util.py`.\n"):
    return json.dumps({"findings": findings, "answer": "two modules."})


# ---------------------------------------------------------------------------
# what it writes, and where
# ---------------------------------------------------------------------------

def test_findings_are_recorded_under_the_nodes_own_record():
    investigator = Investigator(lambda p: _reply())
    (edit,) = investigator(_brief(_STATE, "src"))
    assert edit.owner == "src"
    assert edit.path == f"src/{CONTEXT_FILE}"
    assert FINDINGS_HEADING in edit.content
    assert "- `src/core.py` imports `g` from `src/util.py`." in edit.content
    assert edit.content.startswith("# src\n"), "the record is extended, not replaced"
    assert (investigator.calls, investigator.recorded) == (1, 1)


def test_an_existing_record_is_extended_and_a_missing_one_is_created():
    existing = Investigator(lambda p: _reply())(_brief(_STATE, "src"))[0]
    assert existing.kind == "context"                 # src/CONTEXT.md was there
    fresh_state = {CONTEXT_FILE: "# root\n", "new/core.py": "x = 1\n"}
    fresh = Investigator(lambda p: _reply())(_brief(fresh_state, "new"))[0]
    assert fresh.kind == "record"                     # the write that creates it


def test_the_investigator_is_shown_the_nodes_files_not_its_childrens():
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return _reply()

    Investigator(complete)(_brief(_STATE, "src"))
    assert "return 42" in seen[0], "the node's own code must reach the reader"
    assert "return 0" not in seen[0], "a child subtree is not this node's to read"
    assert "src/deep/" in seen[0], "but the reader is told the tree continues"


def test_the_question_travels_into_the_prompt():
    seen = []
    Investigator(lambda p: seen.append(p) or _reply())(
        _brief(_STATE, "src", objective="where is g defined?"))
    assert "where is g defined?" in seen[0]


# ---------------------------------------------------------------------------
# the cases that must not be conflated
# ---------------------------------------------------------------------------

def test_nothing_worth_recording_costs_nothing():
    investigator = Investigator(lambda p: json.dumps({"findings": "", "answer": "none"}))
    assert investigator(_brief(_STATE, "src")) == []
    assert (investigator.calls, investigator.empty, investigator.unparsed) == (1, 1, 0)


def test_a_node_with_no_files_is_not_investigated():
    investigator = Investigator(lambda p: _reply())
    assert investigator(_brief({CONTEXT_FILE: "# root\n"}, "empty")) == []
    assert (investigator.calls, investigator.empty) == (0, 1)


def test_a_broken_reply_and_a_dead_call_are_counted_as_unparsed():
    for complete in (lambda p: "no json at all",
                     lambda p: json.dumps({"answer": "no findings key"}),
                     lambda p: (_ for _ in ()).throw(RuntimeError("down"))):
        investigator = Investigator(complete)
        assert investigator(_brief(_STATE, "src")) == []
        assert investigator.unparsed == 1


def test_a_frozen_record_is_never_written():
    investigator = Investigator(lambda p: _reply(), frozen=("spec/**",))
    state = {"spec/" + CONTEXT_FILE: "# spec\n", "spec/x.md": "x\n"}
    assert investigator(_brief(state, "spec")) == []
    assert investigator.refused == 1


# ---------------------------------------------------------------------------
# the seam: a read-only delegation runs the investigator, not the recursion
# ---------------------------------------------------------------------------

def _policy(investigator, log=None, **kw):
    return RecursiveDelegation(
        manager=lambda brief: ([Delegation("src", "what is here?", readonly=True)]
                               if brief.world.path == "" else []),
        executor=lambda brief: [Edit(brief.world.path, "src/should_not_exist.py", "x\n")],
        log=log or WorldLog(), max_depth=3, investigator=investigator, **kw)


def test_a_read_only_delegation_runs_the_investigator_and_folds_its_record():
    log = WorldLog()
    policy = _policy(Investigator(lambda p: _reply()), log=log)
    policy.propose(_ctx(_STATE, reward=1.0))
    assert policy.investigations == 1 and policy.finding_edits == 1
    roles = [e.role for e in log.episodes]
    assert "investigator" in roles
    assert "executor" not in roles, "a read-only delegation must not run the executor"


def test_the_finding_record_rides_in_the_proposal():
    policy = _policy(Investigator(lambda p: _reply()))
    (proposal,) = policy.propose(_ctx(_STATE, reward=1.0))
    assert f"src/{CONTEXT_FILE}" in proposal
    assert FINDINGS_HEADING in proposal


def test_a_read_only_delegation_without_an_investigator_is_ordinary_work():
    """A flag with nothing behind it must not silently drop the delegation."""
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda brief: ([Delegation("src", "do it", readonly=True)]
                               if brief.world.path == "" else []),
        executor=lambda brief: [Edit(brief.world.path, "src/core.py", "x = 1\n")],
        log=log, max_depth=3)                      # no investigator installed
    policy.propose(_ctx(_STATE, reward=1.0))
    assert "executor" in [e.role for e in log.episodes]
    assert policy.investigations == 0


def test_a_write_that_is_not_the_nodes_record_is_refused():
    """Source edits are not an investigation, and an ancestor's record is not ours."""
    def wrong(brief):
        return [Edit("src", "src/core.py", "hacked\n")]

    policy = _policy(wrong)
    policy.propose(_ctx(_STATE, reward=1.0))
    assert policy.investigations == 1
    assert policy.finding_edits == 0
    assert policy.investigation_refused == 1


# ---------------------------------------------------------------------------
# the manager's protocol
# ---------------------------------------------------------------------------

def test_llm_manager_reads_the_investigate_flag():
    def complete(prompt):
        return json.dumps({"delegations": [
            {"path": "src", "objective": "build it"},
            {"path": "src/deep", "objective": "what is in there?", "investigate": True}]})

    delegations = llm_manager(complete)(_brief(_STATE, ""))
    assert [(d.path, d.readonly) for d in delegations] == [
        ("src", False), ("src/deep", True)]


def test_llm_manager_still_accepts_a_reply_without_the_flag():
    def complete(prompt):
        return json.dumps({"delegations": [{"path": "src", "objective": "build it"}]})

    (delegation,) = llm_manager(complete)(_brief(_STATE, ""))
    assert delegation.readonly is False
