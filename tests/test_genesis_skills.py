"""Upstream's ``SkillExtractor``, and the read half that makes it matter.

The port could already *inherit* a skill: ``LocalWorld.skills()`` collects
``.agents/skills/`` along the node chain, which is ``hierarchical_skill_names/2`` and
what the paper lists among what an accepted version carries (3.1). What it could not
do is write one, and -- because its agents have no read tool -- what it could not do
either is hand an inherited skill's *body* to anyone. A name line telling an agent to
"read one before using it" is an instruction it cannot follow.

So there are two halves under test here and they are one feature: the extractor
(:mod:`examples.genesis._skills`) and ``LocalWorld.skill_bodies``, without which the
extractor's output is inert.
"""

from __future__ import annotations

import json

from agentdescent.evolution import Task
from agentdescent.filetree import canonical
from agentdescent.policies import ProposalContext

from examples.genesis._delegation import (Brief, Delegation, Edit,
                                          RecursiveDelegation)
from examples.genesis._skills import (SkillExtractor, is_skill_path, skill_name,
                                      skill_path)
from examples.genesis._world import CONTEXT_FILE, SKILLS_DIR, LocalWorld, WorldLog


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_STATE = {
    CONTEXT_FILE: "# root\n",
    "src/" + CONTEXT_FILE: "# src\n",
    "src/core.py": "def f():\n    return 1\n",
    "src/util.py": "def g():\n    return 2\n",
}


def _brief(state, path="", objective="grow the repository"):
    world = LocalWorld(version=1, path=path, skill_bodies=True)
    return Brief(world=world, objective=objective,
                 context=world.situate(state), state=dict(state),
                 task=Task(id="t", prompt="x"), output="", reward=0.0, depth=0)


def _ctx(state, reward=0.0):
    return ProposalContext(rendered=canonical(state),
                           task=Task(id="t", prompt="x"), output="",
                           reward=reward, base_version=1)


def _work(*paths):
    return [Edit(owner=p.rsplit("/", 1)[0] if "/" in p else "", path=p, content="x = 1\n")
            for p in paths]


def _one_skill(node="src", name="core-first", body="# core first\n", **kw):
    item = {"node": node, "name": name, "description": "import core first",
            "body": body}
    item.update(kw)
    return json.dumps({"skills": [item], "reason": "a gotcha"})


# ---------------------------------------------------------------------------
# paths and names
# ---------------------------------------------------------------------------

def test_skill_paths_are_node_scoped_under_the_agents_directory():
    assert skill_path("", "deploy") == f"{SKILLS_DIR}/deploy.md"
    assert skill_path("src", "deploy") == f"src/{SKILLS_DIR}/deploy.md"
    assert skill_path("./src/", "deploy") == f"src/{SKILLS_DIR}/deploy.md"
    assert is_skill_path("src/.agents/skills/x.md")
    assert not is_skill_path("src/x.py")
    assert not is_skill_path("src/.agents/readme.md")
    assert skill_name("src/.agents/skills/x.md") == "x"
    assert skill_name("src/x.py") is None


# ---------------------------------------------------------------------------
# the extractor: what it writes
# ---------------------------------------------------------------------------

def test_extractor_writes_a_skill_with_upstreams_frontmatter():
    """`skill_add`'s format -- name/description/parameters -- not the model's YAML."""
    extractor = SkillExtractor(lambda p: _one_skill(parameters=[
        {"name": "mod", "type": "str", "description": "the module", "required": True}]))
    edits = extractor(_brief(_STATE), _work("src/core.py"))
    assert len(edits) == 1
    edit = edits[0]
    assert (edit.kind, edit.owner, edit.path) == (
        "skill", "", f"src/{SKILLS_DIR}/core-first.md")
    assert edit.content.startswith("---\nname: core-first\n")
    assert 'description: "import core first"' in edit.content
    assert "parameters:\n  - name: mod\n    type: str\n" in edit.content
    assert "    required: true\n" in edit.content
    assert edit.content.rstrip().endswith("# core first")
    assert (extractor.calls, extractor.extracted, extractor.unparsed) == (1, 1, 0)


def test_a_skill_can_be_placed_at_the_root():
    extractor = SkillExtractor(lambda p: _one_skill(node=""))
    (edit,) = extractor(_brief(_STATE), _work("src/core.py"))
    assert edit.path == f"{SKILLS_DIR}/core-first.md"


def test_the_root_brief_ignores_which_node_the_work_came_from():
    """The extractor is a root-level agent: it distils work from anywhere and picks
    the node, which is `skill_enable`'s "where it is most relevant"."""
    extractor = SkillExtractor(lambda p: _one_skill(node="src"))
    (edit,) = extractor(_brief(_STATE), _work("src/core.py", "src/util.py"))
    assert edit.path.startswith("src/")


# ---------------------------------------------------------------------------
# the extractor: what it refuses, and why each refusal is counted separately
# ---------------------------------------------------------------------------

def test_a_duplicate_of_an_existing_skill_is_not_written():
    """Upstream: "use `skill_list` ... Avoid duplicating existing skills." """
    state = dict(_STATE)
    state[f"src/{SKILLS_DIR}/core-first.md"] = "---\nname: core-first\n---\nold\n"
    extractor = SkillExtractor(lambda p: _one_skill())
    assert extractor(_brief(state), _work("src/core.py")) == []
    assert (extractor.deduped, extractor.extracted) == (1, 0)


def test_a_name_used_elsewhere_is_a_duplicate_even_at_another_node():
    state = dict(_STATE)
    state[f"{SKILLS_DIR}/core-first.md"] = "---\nname: core-first\n---\nold\n"
    extractor = SkillExtractor(lambda p: _one_skill(node="src"))
    assert extractor(_brief(state), _work("src/core.py")) == []
    assert extractor.deduped == 1


def test_the_written_skill_shows_in_the_extractors_next_prompt():
    """The whole point of `skill_list` before `skill_add`: the second call sees the
    first call's skill and can refuse to duplicate it."""
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return _one_skill()

    state = dict(_STATE)
    extractor = SkillExtractor(complete)
    first = extractor(_brief(state), _work("src/core.py"))
    for edit in first:
        state[edit.path] = edit.content
    second = extractor(_brief(state), _work("src/util.py"))
    assert second == []
    assert f"src/{SKILLS_DIR}/core-first.md" in seen[1]


def test_malformed_names_nodes_and_bodies_are_refused():
    for kw in ({"name": "Not Kebab"}, {"name": "core_first"}, {"name": ""},
               {"body": "   "}, {"node": "src/nope"}, {"node": "src/core.py"}):
        extractor = SkillExtractor(lambda p, kw=kw: _one_skill(**kw))
        assert extractor(_brief(_STATE), _work("src/core.py")) == [], kw
        assert extractor.refused == 1, kw


def test_a_skill_may_not_be_written_into_the_frozen_contract():
    extractor = SkillExtractor(lambda p: _one_skill(node="spec"),
                               frozen=("spec/**",))
    state = dict(_STATE, **{"spec/" + CONTEXT_FILE: "# spec\n"})
    assert extractor(_brief(state), _work("src/core.py")) == []
    assert extractor.refused == 1


def test_at_most_max_skills_are_written_per_contribution():
    reply = json.dumps({"skills": [
        {"node": "src", "name": f"s{i}", "description": "d", "body": "# body\n"}
        for i in range(5)]})
    extractor = SkillExtractor(lambda p: reply, max_skills=2)
    assert len(extractor(_brief(_STATE), _work("src/core.py"))) == 2
    assert extractor.extracted == 2          # the rest are capped, not refused


# ---------------------------------------------------------------------------
# the extractor: the three "nothing happened" cases that must not look alike
# ---------------------------------------------------------------------------

def test_finding_nothing_is_a_valid_answer_and_not_a_parse_failure():
    """Upstream: "perfectly valid to find no skills worth extracting"."""
    extractor = SkillExtractor(lambda p: json.dumps({"skills": [], "reason": "nothing"}))
    assert extractor(_brief(_STATE), _work("src/core.py")) == []
    assert (extractor.calls, extractor.extracted, extractor.unparsed) == (1, 0, 0)


def test_an_unparseable_reply_is_counted():
    for reply in ("no json here", "{not json}", "[]", '{"skills": "nope"}'):
        extractor = SkillExtractor(lambda p, reply=reply: reply)
        assert extractor(_brief(_STATE), _work("src/core.py")) == []
        assert extractor.unparsed == 1, reply


def test_a_dead_call_costs_a_skill_and_not_the_contribution():
    def boom(prompt):
        raise RuntimeError("endpoint down")

    extractor = SkillExtractor(boom)
    assert extractor(_brief(_STATE), _work("src/core.py")) == []
    assert extractor.unparsed == 1


def test_bookkeeping_alone_makes_no_call():
    extractor = SkillExtractor(lambda p: _one_skill())
    bookkeeping = [Edit("", CONTEXT_FILE, "# root\n", kind="record")]
    assert extractor(_brief(_STATE), bookkeeping) == []
    assert extractor.calls == 0


# ---------------------------------------------------------------------------
# the read half: without it the write half is inert
# ---------------------------------------------------------------------------

def test_skill_bodies_reach_the_brief_only_when_asked_for():
    skill = f"src/{SKILLS_DIR}/core-first.md"
    state = dict(_STATE, **{skill: "---\nname: core-first\n---\nimport core first\n"})
    named = LocalWorld(version=1, path="src").situate(state)
    assert skill in named and "import core first" not in named
    carried = LocalWorld(version=1, path="src", skill_bodies=True).situate(state)
    assert skill in carried and "import core first" in carried


def test_skill_bodies_are_inherited_down_the_chain():
    skill = f"src/{SKILLS_DIR}/core-first.md"
    state = dict(_STATE, **{skill: "---\nname: core-first\n---\nimport core first\n"})
    deep = LocalWorld(version=1, path="src/core", skill_bodies=True).situate(state)
    assert "import core first" in deep


# ---------------------------------------------------------------------------
# the seam: a root episode distils what it just accepted
# ---------------------------------------------------------------------------

def test_a_rework_rollout_does_not_redistil_the_same_contribution():
    """A rework is the same contribution retried, not a new one to learn from."""
    state = {CONTEXT_FILE: "# root\n", "src/" + CONTEXT_FILE: "# src\n"}
    log = WorldLog()
    policy = RecursiveDelegation(
        manager=lambda brief: ([Delegation("src", "build it")]
                               if brief.world.path == "" else []),
        executor=lambda brief: [Edit("src", "src/core.py", "def f():\n    return 1\n")],
        log=log, max_depth=2,
        skills=SkillExtractor(lambda p: _one_skill(node="src")))
    log.request_rework("src", "not yet")
    policy.propose(_ctx(state, reward=1.0))
    assert policy.skills_written == 0


def test_a_skill_rides_in_the_proposal_the_engine_commits():
    state = {CONTEXT_FILE: "# root\n"}
    policy = RecursiveDelegation(
        manager=lambda brief: [],                 # root does the work itself
        executor=lambda brief: [Edit("", "src/core.py", "def f():\n    return 1\n")],
        log=WorldLog(), max_depth=2,
        skills=SkillExtractor(lambda p: _one_skill(node="src")))
    (proposal,) = policy.propose(_ctx(state, reward=1.0))
    assert f"src/{SKILLS_DIR}/core-first.md" in proposal
    assert "def f():" in proposal


def test_a_skill_is_not_written_when_the_episode_produced_only_bookkeeping():
    state = {CONTEXT_FILE: "# root\n"}
    policy = RecursiveDelegation(
        manager=lambda brief: [],
        executor=lambda brief: [Edit("", "src/core.py", "def f():\n    return 1\n",
                                     kind="record")],
        log=WorldLog(), max_depth=2,
        skills=SkillExtractor(lambda p: _one_skill()))
    (proposal,) = policy.propose(_ctx(state, reward=1.0))
    assert SKILLS_DIR not in proposal
    assert policy.skills_written == 0


def test_a_managed_root_distils_too():
    """The contribution is what the episode returned, children included."""
    state = {CONTEXT_FILE: "# root\n", "src/" + CONTEXT_FILE: "# src\n"}
    policy = RecursiveDelegation(
        manager=lambda brief: ([Delegation("src", "build it")]
                               if brief.world.path == "" else []),
        executor=lambda brief: [Edit("src", "src/core.py", "def f():\n    return 1\n")],
        log=WorldLog(), max_depth=2,
        skills=SkillExtractor(lambda p: _one_skill(node="src")))
    (proposal,) = policy.propose(_ctx(state, reward=1.0))
    assert f"src/{SKILLS_DIR}/core-first.md" in proposal
    assert policy.skills_written == 1


# ---------------------------------------------------------------------------
# the trust region: skills do not compete with the work they distil
# ---------------------------------------------------------------------------

def test_skills_are_bounded_separately_from_the_work():
    policy = RecursiveDelegation(manager=lambda b: [], executor=lambda b: [],
                                 log=WorldLog(), max_edits=1, max_skill_edits=2)
    work = _work("a.py", "b.py")                       # one survives, one is trimmed
    skills = [Edit("", f"{SKILLS_DIR}/s{i}.md", "---\n---\n", kind="skill")
              for i in range(3)]
    bound = policy._bound(work + skills)
    assert len([e for e in bound if e.kind == "skill"]) == 2
    assert len([e for e in bound if e.kind != "skill"]) == 1
    assert policy.truncated == 1


def test_no_skills_means_the_bounds_are_unchanged():
    """`skills=None` must be the old behaviour exactly, counter and all."""
    policy = RecursiveDelegation(manager=lambda b: [], executor=lambda b: [],
                                 log=WorldLog(), max_edits=1)
    assert policy.max_skill_edits == 2            # present, but nothing produces one
    bound = policy._bound(_work("a.py", "b.py", "c.py"))
    assert len(bound) == 1 and policy.truncated == 2
