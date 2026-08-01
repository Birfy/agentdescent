"""`examples/skill_dir_evolution.py` -- the directory-evolution example.

It is offline and takes seconds, so unlike the algorithm ports there is no reason
to test only its helpers: run the loop. What it has to demonstrate is the one
claim the example exists to make -- that the *directory* is what changed the
score, not the model.
"""

import os

from examples.skill_dir_evolution import (
    SKILL_NAME,
    make_rows,
    make_skill_dir,
    offline_agent,
    offline_reflector,
    to_tasks,
)

from agentdescent import evolve_skill_dir, load_tree


def _run(**kwargs):
    path = make_skill_dir()
    result = evolve_skill_dir(
        path, to_tasks(make_rows(12)), agent=offline_agent(),
        reflect_with=offline_reflector(), score="contains",
        rounds=3, n_workers=2, max_concurrency=1, **kwargs)
    return path, result


def test_the_example_fixes_the_wrong_column():
    path, result = _run()
    assert result.error is None, result.error
    assert result.final_reward == 1.0
    assert "COLUMN: amount" in result.state["references/rules.md"]
    assert result.outcomes().get("committed", 0) >= 1


def test_the_skill_is_what_moved_the_score_not_the_agent():
    # the same agent, the same tasks, with the skill's answer already correct vs
    # still wrong: if the directory were not being read these would agree.
    path = make_skill_dir()
    tasks = to_tasks(make_rows(12))
    before = evolve_skill_dir(
        path, tasks, agent=offline_agent(), reflect_with=lambda p: "",
        score="contains", rounds=1, n_workers=1, max_concurrency=1)
    assert before.final_reward == 0.0          # wrong column, nothing proposed

    _, after = _run()
    assert after.final_reward == 1.0


def test_the_run_never_writes_to_the_source_directory():
    path, result = _run()
    assert load_tree(path)["references/rules.md"].strip() == "COLUMN: id"


def test_write_to_installs_the_evolved_skill():
    path, result = _run()
    plan = result.write_to(path)
    assert plan["backup"]
    assert "COLUMN: amount" in load_tree(path)["references/rules.md"]
    assert os.path.basename(path) == SKILL_NAME
