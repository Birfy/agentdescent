"""The M0 runner, offline.

`--dry-run` resolves the skill, loads the tasks and builds the objective without
calling anything. That is the part worth checking before spending tokens, so it
is the part this test drives -- with real directories on disk, not mocks of the
resolution.
"""

from __future__ import annotations

import json
import os

import pytest

from agentdescent.runners import LAYOUTS, layout_prefix
from examples.dsh.evolve_dsh_skill import build_objective, load_tasks, main


def make_skill(root, name, body="do the thing\n"):
    path = os.path.join(str(root), ".dsh", "skills", name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def write_tasks(path, rows):
    with open(str(path), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write((row if isinstance(row, str) else json.dumps(row)) + "\n")
    return str(path)


@pytest.fixture()
def project(tmp_path):
    """A project root the resolver will find, holding one dsh skill."""
    os.makedirs(tmp_path / "proj" / ".git")
    make_skill(tmp_path / "proj", "sql")
    return tmp_path / "proj"


# ---------------------------------------------------------------------------
# Loading tasks
# ---------------------------------------------------------------------------


def test_jsonl_with_gold_becomes_tasks_carrying_it(tmp_path):
    path = write_tasks(tmp_path / "t.jsonl",
                       [{"prompt": "q1", "gold": "a1"}, {"prompt": "q2", "gold": "a2"}])
    tasks = load_tasks(path)
    assert [t.prompt for t in tasks] == ["q1", "q2"]
    assert [t.meta["gold"] for t in tasks] == ["a1", "a2"]


def test_a_bare_line_is_taken_as_a_prompt(tmp_path):
    """A plain list of questions is a legitimate task file for the B rung."""
    path = write_tasks(tmp_path / "t.txt", ["just a question", "another one"])
    tasks = load_tasks(path)
    assert [t.prompt for t in tasks] == ["just a question", "another one"]
    assert all("gold" not in t.meta for t in tasks)


def test_blank_lines_are_skipped_and_ids_stay_dense(tmp_path):
    path = write_tasks(tmp_path / "t.jsonl", [{"prompt": "a"}, "", {"prompt": "b"}])
    assert [t.id for t in load_tasks(path)] == ["t0", "t1"]


def test_a_json_line_without_a_prompt_names_its_line_number(tmp_path):
    path = write_tasks(tmp_path / "t.jsonl", [{"prompt": "ok"}, {"gold": "orphan"}])
    with pytest.raises(ValueError, match=r":2\b"):
        load_tasks(path)


def test_an_empty_file_is_an_error_not_an_empty_run(tmp_path):
    with pytest.raises(ValueError, match="no tasks"):
        load_tasks(write_tasks(tmp_path / "t.jsonl", []))


# ---------------------------------------------------------------------------
# The objective ladder
# ---------------------------------------------------------------------------


class Args:
    check = None
    check_timeout = 30.0
    score = "contains"


def test_gold_scoring_is_used_when_every_task_has_one(tmp_path):
    tasks = load_tasks(write_tasks(tmp_path / "t.jsonl", [{"prompt": "q", "gold": "42"}]))
    reward, described = build_objective(Args(), tasks)
    assert "contains" in described
    assert reward(tasks[0], "the answer is 42") == 1.0


def test_a_check_command_wins_over_score_when_both_are_available(tmp_path):
    """Rung B beats rung A: the user named an objective, so use it."""
    args = Args()
    args.check = "true"
    tasks = load_tasks(write_tasks(tmp_path / "t.jsonl", [{"prompt": "q", "gold": "42"}]))
    reward, described = build_objective(args, tasks)
    assert "exit 0" in described
    assert reward(tasks[0], "anything at all") == 1.0


def test_missing_gold_refuses_rather_than_scoring_everything_zero(tmp_path):
    """Guessing here produces a confidently wrong artifact and nothing catches it."""
    tasks = load_tasks(write_tasks(tmp_path / "t.jsonl",
                                   [{"prompt": "q", "gold": "1"}, {"prompt": "r"}]))
    with pytest.raises(SystemExit, match="--check"):
        build_objective(Args(), tasks)


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------


def test_dry_run_resolves_the_real_skill_and_calls_nothing(project, tmp_path, capsys):
    tasks = write_tasks(tmp_path / "t.jsonl", [{"prompt": "q", "gold": "a"}])
    code = main(["--skill", "sql", "--tasks", tasks, "--cwd", str(project), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "project-dsh" in out and "rank 100" in out
    assert str(project) in out
    assert "nothing was written" in out


def test_dry_run_needs_no_api_key_and_no_sdk(project, tmp_path, monkeypatch, capsys):
    """`dsh()` imports its SDK lazily, so building the agent must not need it."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    tasks = write_tasks(tmp_path / "t.jsonl", ["a bare question"])
    code = main(["--skill", "sql", "--tasks", tasks, "--cwd", str(project),
                 "--check", "true", "--dry-run"])
    assert code == 0
    assert "command exit 0" in capsys.readouterr().out


def test_an_unknown_skill_reports_instead_of_evolving_something_else(project, tmp_path, capsys):
    tasks = write_tasks(tmp_path / "t.jsonl", [{"prompt": "q", "gold": "a"}])
    code = main(["--skill", "absent", "--tasks", tasks, "--cwd", str(project),
                 "--dry-run"])
    assert code == 1
    assert "--list" in capsys.readouterr().err


def test_a_flat_skill_is_refused_because_evolving_it_would_take_its_whole_root(
        tmp_path, capsys):
    """A flat skill lives *directly* in a root shared with every other skill.

    Treating it as a directory would load its siblings as part of the artifact
    and write them all back.
    """
    project = tmp_path / "proj"
    os.makedirs(project / ".git")
    root = project / ".dsh" / "skills"
    os.makedirs(root)
    for name in ("flat", "neighbour"):
        with open(root / f"{name}.md", "w", encoding="utf-8") as fh:
            fh.write("body\n")
    tasks = write_tasks(tmp_path / "t.jsonl", [{"prompt": "q", "gold": "a"}])
    code = main(["--skill", "flat", "--tasks", tasks, "--cwd", str(project),
                 "--dry-run"])
    assert code == 1
    assert "SKILL.md" in capsys.readouterr().err


def test_list_shows_the_winning_skill_and_its_root(project, capsys):
    assert main(["--list", "--cwd", str(project)]) == 0
    out = capsys.readouterr().out
    assert "sql" in out and "project-dsh" in out


def test_list_with_no_skills_shows_the_roots_it_searched(tmp_path, capsys):
    """The useful answer to "nothing found" is *where* it looked."""
    os.makedirs(tmp_path / "bare" / ".git")
    assert main(["--list", "--cwd", str(tmp_path / "bare")]) == 1
    out = capsys.readouterr().out
    assert "project-dsh" in out and "absent" in out


# ---------------------------------------------------------------------------
# The layout -- the silent one
# ---------------------------------------------------------------------------


def test_the_dsh_layout_matches_the_harness_rank_100_root():
    """Stage a candidate where the harness does not look and nothing reports it.

    The agent just never sees the skill, every candidate scores alike, and the
    run reads as a method that does not work. A rollout workspace holds no
    `.git`, so the harness resolves its project root to the workspace and scans
    `<workspace>/.dsh/skills/<name>` -- which is what this layout must produce.
    """
    assert layout_prefix("dsh_skill", "sql") == ".dsh/skills/sql"
    assert layout_prefix("dsh_skill_library", "sql") == ".dsh/skills"
    assert LAYOUTS["dsh_skill"] != LAYOUTS["claude_skill"]
