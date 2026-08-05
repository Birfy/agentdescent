"""Offline tests for the ACE (Agentic Context Engineering) example.

Exercise the ACE playbook representation, grow-and-refine de-dup, BIO entity
decoding, task shaping, and scoring -- no network or LLM calls.
"""

from agentdescent.evolution import Task
from examples.ace.ace_context_evolution import (
    ACEPlaybook,
    _entities,
    build_tasks,
    estimate_calls,
    jaccard,
    make_reward,
)


def test_playbook_appends_and_dedupes_exact():
    pb = ACEPlaybook()
    state = {}
    d = pb.to_diff(state, "revenue :: tag deferred revenue as a contract liability",
                   "w0", 1, "a")
    assert d is not None
    state.update(d.ops)
    # exact duplicate -> no diff (incremental delta, content-addressed)
    assert pb.to_diff(state, "revenue :: tag deferred revenue as a contract liability",
                      "w0", 1, "a") is None


def test_playbook_grow_and_refine_rejects_near_duplicate():
    pb = ACEPlaybook(dedup_threshold=0.8)
    state = {}
    d = pb.to_diff(state, "debt :: senior notes are long-term debt", "w0", 1, "a")
    state.update(d.ops)
    # near-duplicate in the same section is pruned (embedding de-dup proxy)
    assert pb.to_diff(state, "debt :: senior notes are long term debt now",
                      "w1", 1, "a") is None
    # the same lesson in a *different* section is allowed (localised bullets)
    assert pb.to_diff(state, "revenue :: senior notes are long-term debt",
                      "w1", 1, "a") is not None


def test_playbook_never_rewrites_existing_bullets():
    pb = ACEPlaybook()
    state = {}
    for i in range(3):
        d = pb.to_diff(state, f"shares :: rule number {i}", "w", 1, "a")
        assert d is not None
        # each admitted bullet is a NEW key -- existing ones are untouched.
        assert set(d.ops).isdisjoint(set(state))
        state.update(d.ops)
    assert len(state) == 3


def test_playbook_render_groups_by_section():
    pb = ACEPlaybook()
    state = {}
    for prop in ("revenue :: A", "debt :: B", "revenue :: C"):
        d = pb.to_diff(state, prop, "w", 1, "a")
        state.update(d.ops)
    rendered = pb.render(state)
    assert "## debt" in rendered and "## revenue" in rendered
    assert rendered.index("## debt") < rendered.index("## revenue")  # sorted


def test_jaccard_bounds():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b c", "x y z") == 0.0
    assert 0.0 < jaccard("a b c d", "a b c e") < 1.0


def test_bio_entity_decoding():
    names = ["O", "B-Revenues", "I-Revenues", "B-Goodwill"]
    tokens = ["Total", "net", "revenues", "were", "and", "goodwill"]
    tags = [0, 1, 2, 0, 0, 3]  # a 2-token Revenues span + a 1-token Goodwill span
    spans = _entities(tokens, tags, names)
    assert spans == [(1, 3, "Revenues"), (5, 6, "Goodwill")]


def test_build_tasks_single_entity_and_topk():
    names = ["O", "B-Revenues", "I-Revenues", "B-Goodwill", "B-Debt"]
    rows = []
    # 3 Revenues rows, 2 Goodwill rows, 1 Debt row, plus a two-entity row (dropped)
    for _ in range(3):
        rows.append({"tokens": ["rev", "100"], "ner_tags": [0, 1]})
    for _ in range(2):
        rows.append({"tokens": ["gw", "50"], "ner_tags": [0, 3]})
    rows.append({"tokens": ["debt", "9"], "ner_tags": [0, 4]})
    rows.append({"tokens": ["100", "50"], "ner_tags": [1, 3]})  # two entities -> skipped
    tasks = build_tasks(rows, names, limit=99, top_k=2)
    # top-2 concepts = Revenues, Goodwill; Debt filtered out, two-entity row dropped
    targets = {t.meta["target"] for t in tasks}
    assert targets == {"Revenues", "Goodwill"}
    assert all("<<" in t.prompt and ">>" in t.prompt for t in tasks)
    assert all(set(t.meta["candidates"]) == {"Goodwill", "Revenues"} for t in tasks)


def test_reward_exact_and_substring():
    reward = make_reward()
    task = Task(id="t", prompt="q", meta={"target": "Goodwill"})
    assert reward(task, "Goodwill") == 1.0
    assert reward(task, "The tag is Goodwill.") == 1.0            # standalone token
    assert reward(task, "goodwill") == 1.0                        # case-insensitive
    assert reward(task, "DebtInstrumentFaceAmount") == 0.0


def test_estimate_calls_scales():
    assert estimate_calls(6, 2, 12) > estimate_calls(3, 2, 12)
    assert estimate_calls(6, 4, 12) > estimate_calls(6, 2, 12)
