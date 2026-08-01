"""Offline tests for the EvoSkill example.

Exercise the faithful numeric scorer (unit-aware tolerance + multi-tolerance
weighting), the bounded top-K frontier, doc retrieval, and the loop. No network.
"""

from examples.evoskill_skill_discovery import (
    Frontier,
    PASS_THRESHOLD,
    extract_numbers,
    fuzzy_match_answer,
    retrieve_context,
    run_evoskill,
    score_multi_tolerance,
)


def test_extract_numbers_scales_units():
    assert 2602.0 in extract_numbers("2,602")
    scaled = extract_numbers("2,602 million")
    assert 2602e6 in scaled and 2602.0 in scaled     # scaled AND bare kept


def test_fuzzy_match_tolerance():
    assert fuzzy_match_answer("2602", "2,602", 0.0)
    assert fuzzy_match_answer("2602", "2650", 0.05)          # within 5%
    assert not fuzzy_match_answer("2602", "3000", 0.05)      # outside 5%
    assert fuzzy_match_answer("507", "about 507 million dollars", 0.0)


def test_fuzzy_match_multi_number_requires_all():
    assert fuzzy_match_answer("100 and 200", "we found 200 and 100", 0.0)
    assert not fuzzy_match_answer("100 and 200", "only 100", 0.0)


def test_multi_tolerance_weighting():
    assert abs(score_multi_tolerance("2,602", "2,602") - 1.0) < 1e-9   # exact -> 1
    assert score_multi_tolerance("", "2602") == 0.0                    # empty -> 0
    partial = score_multi_tolerance("2650", "2602")   # matches loose tols only
    assert 0.0 < partial < 1.0


def test_pass_threshold_is_0_8():
    assert PASS_THRESHOLD == 0.8


def test_frontier_bounded_topk():
    f = Frontier(max_size=2)
    assert f.update({"a": "1"}, 0.5)          # room
    assert f.update({"b": "2"}, 0.7)          # room
    assert not f.update({"c": "3"}, 0.4)      # full, worse than worst(0.5) -> reject
    assert f.update({"d": "4"}, 0.9)          # replaces the worst member
    assert abs(f.select_parent()[1] - 0.9) < 1e-9   # parent = best


def test_retrieve_context_picks_relevant_lines():
    doc = "\n".join(["defense spending 1940 was 2602",
                     "an unrelated line about weather", "veterans 507"])
    ctx = retrieve_context(doc, "total national defense expenditures 1940", n_lines=1)
    assert "defense" in ctx


def _skill_helps_stub():
    """Base agent fails (no skill) but succeeds once any skill is in context."""
    class Stub:
        def __call__(self, prompt):
            if "Skill Proposer" in prompt:
                return "create defense-lookup\nLook up defense totals."
            if "Skill Generator" in prompt:
                return "Defense lookup\n- Find the national defense row, report millions."
            return "Answer: 2602" if "skill:" in prompt else "Answer: 0"
    return Stub()


def _skill_useless_stub():
    """Base agent always fails; no skill ever helps (drives rollback / no-gain)."""
    class Stub:
        def __call__(self, prompt):
            if "Skill Proposer" in prompt:
                return "create noop\nTry harder."
            if "Skill Generator" in prompt:
                return "No-op\n- does not help."
            return "Answer: 0"
    return Stub()


def _tasks(n_train=4, n_val=3):
    train = [{"question": "defense 1940?", "answer": "2602",
              "source_files": "", "difficulty": "hard"} for _ in range(n_train)]
    val = [{"question": "defense?", "answer": "2602",
            "source_files": "", "difficulty": "hard"} for _ in range(n_val)]
    return train, val


def test_loop_discovers_a_helpful_skill():
    train, val = _tasks()
    res = run_evoskill(_skill_helps_stub(), {}, train, val, iterations=3, seed=0)
    assert res.seed_score == 0.0
    assert res.best_score > res.seed_score
    assert len(res.skills) >= 1


def test_sync_gate_no_false_improvement():
    """Sync frontier: a skill that never helps must not raise the best score."""
    train, val = _tasks()
    res = run_evoskill(_skill_useless_stub(), {}, train, val, iterations=3, seed=0)
    assert res.seed_score == 0.0
    assert res.best_score == res.seed_score          # gate reports no false gain


def test_async_sgd_keeps_helpful_skill():
    """Async SGD path (SgdSkillAggregator): a helpful skill is validated and kept."""
    train, val = _tasks()
    res = run_evoskill(_skill_helps_stub(), {}, train, val, iterations=6, seed=0,
                       asynchronous=True, async_ratio=2, max_seconds=20,
                       eval_concurrency=1, batch_size=2, val_every=1)
    assert res.seed_score == 0.0
    assert res.best_score > res.seed_score
    assert len(res.skills) >= 1


def test_async_sgd_rolls_back_useless_skill():
    """Async SGD path: skills with no held-out gain are rolled back to checkpoint."""
    train, val = _tasks()
    res = run_evoskill(_skill_useless_stub(), {}, train, val, iterations=6, seed=0,
                       asynchronous=True, async_ratio=2, max_seconds=20,
                       eval_concurrency=1, batch_size=2, val_every=1)
    assert res.seed_score == 0.0
    assert res.best_score == 0.0                      # never improved
    assert res.skills == {}                           # rolled back to empty checkpoint


def test_eval_at_end_scores_final_library():
    """eval_at_end: apply skills during the run, score the final library once."""
    train, val = _tasks()
    res = run_evoskill(_skill_helps_stub(), {}, train, val, iterations=6, seed=0,
                       asynchronous=True, async_ratio=2, max_seconds=20,
                       eval_concurrency=1, batch_size=2, eval_at_end=True)
    assert res.seed_score == 0.0
    assert res.best_score > 0.0                       # final eval sees the applied skills
    assert len(res.skills) >= 1


# ---------------------------------------------------------------------------
# The library is a real directory now (SkillLibraryTree)
# ---------------------------------------------------------------------------


def test_the_library_is_keyed_by_path_and_installs_as_a_directory():
    """The migration's point: the artifact is a directory, not a name->text dict."""
    import tempfile

    from agentdescent.filetree import load_tree, materialize

    from examples.evoskill_skill_discovery import skills_of

    train, val = _tasks()
    res = run_evoskill(_skill_helps_stub(), {}, train, val, iterations=4, seed=0,
                       eval_concurrency=1, batch_size=2)
    assert res.skills, "nothing was learned, so there is nothing to install"
    # every state key is a real relative path, one directory per skill
    assert all(p.startswith("skills/") and p.endswith("/SKILL.md") for p in res.tree)
    assert skills_of(res.tree) == res.skills

    dest = tempfile.mkdtemp()
    materialize(res.tree, dest)
    assert load_tree(dest) == res.tree            # round-trips as a directory


def test_the_retriever_prompt_is_unchanged_by_the_move_to_paths():
    """A faithful port's prompt is part of what it reproduces.

    `render()` is now the artifact's lossless serialisation rather than the prompt
    text, so `run` reassembles the prompt -- and it has to come out identical, or
    the port measures something else than it did before."""
    from agentdescent.filetree import parse_tree

    from examples.evoskill_skill_discovery import (
        SkillLibraryTree, render_skills, skill_path, skills_of)

    skills = {"defense-lookup": "Find the national defense row.\n- report millions",
              "table-headers": "Locate the nearest header row first."}
    tree = {skill_path(n): b for n, b in skills.items()}
    rendered = SkillLibraryTree().render(tree)
    assert render_skills(skills_of(parse_tree(rendered))) == render_skills(skills)


def test_one_skill_per_proposal_and_the_name_protocol_survives():
    from examples.evoskill_skill_discovery import SkillLibraryTree

    s = SkillLibraryTree()
    d = s.to_diff({}, "defense lookup :: find the row", "w0", 1, "skill_library")
    assert d.ops == {"skills/defense-lookup/SKILL.md": "find the row"}
    assert s.to_diff(d.ops, "defense lookup :: find the row", "w0", 1, "a") is None
    assert s.to_diff({}, "no separator here", "w0", 1, "a") is None
