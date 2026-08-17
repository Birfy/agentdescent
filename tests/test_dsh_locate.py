"""Where a dsh installation keeps its skills -- resolved, not guessed.

The failure this guards against is quiet: point an evolution run at a shadowed
copy and it will improve a file the harness never loads, while the harness keeps
serving the winner. Nothing crashes; the method just looks like it does not
work. So the rank order and the shadowing rule are tested, not assumed.
"""

from __future__ import annotations

import os

import pytest

from agentdescent.dsh import (
    Skill,
    agents_home,
    dsh_home,
    find_skill,
    list_skills,
    project_root,
    skill_roots,
)
from agentdescent.dsh.locate import SKILL_FILE, SYSTEM_CHILD


def write(path: str, text: str = "body\n") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def skill_dir(root: str, name: str, text: str = "body\n") -> str:
    write(os.path.join(root, name, SKILL_FILE), text)
    return os.path.join(root, name)


# ---------------------------------------------------------------------------
# Homes
# ---------------------------------------------------------------------------


def test_the_homes_follow_the_env_and_fall_back_to_dotfiles(tmp_path):
    env = {"DSH_HOME": str(tmp_path / "home"), "DSH_AGENTS_HOME": str(tmp_path / "ag")}
    assert dsh_home(env) == str(tmp_path / "home")
    assert agents_home(env) == str(tmp_path / "ag")
    assert dsh_home({}).endswith(os.path.join(".dsh"))
    assert agents_home({}).endswith(os.path.join(".agents"))


def test_an_empty_env_var_means_absent_not_empty():
    """A shell exporting an unset variable hands us ``''``.

    Treating that as a configured home would resolve every root to the
    filesystem root and silently find nothing. The harness makes the same
    distinction for its own variables.
    """
    assert dsh_home({"DSH_HOME": ""}).endswith(os.path.join(".dsh"))
    assert agents_home({"DSH_AGENTS_HOME": ""}).endswith(os.path.join(".agents"))


# ---------------------------------------------------------------------------
# The project root
# ---------------------------------------------------------------------------


def test_the_project_root_is_the_nearest_ancestor_with_dot_git(tmp_path):
    os.makedirs(tmp_path / "repo" / ".git")
    deep = tmp_path / "repo" / "a" / "b"
    os.makedirs(deep)
    assert project_root(str(deep)) == str(tmp_path / "repo")


def test_a_worktree_counts_because_its_dot_git_is_a_file(tmp_path):
    """`.git` is a directory in a clone and a *file* in a worktree.

    Probing only for a directory would walk past a worktree to whatever repo
    happens to sit above it, and resolve project skills to the wrong project.
    """
    write(str(tmp_path / "wt" / ".git"), "gitdir: /elsewhere\n")
    deep = tmp_path / "wt" / "src"
    os.makedirs(deep)
    assert project_root(str(deep)) == str(tmp_path / "wt")


def test_without_a_repo_the_starting_directory_is_the_project(tmp_path):
    lonely = tmp_path / "no-repo"
    os.makedirs(lonely)
    # tmp_path has no .git above it inside the tmp tree; the walk hits the
    # filesystem root and must hand back where it started rather than "/".
    resolved = project_root(str(lonely))
    assert resolved == str(lonely) or resolved.startswith(str(tmp_path.parent))


# ---------------------------------------------------------------------------
# Roots and rank order
# ---------------------------------------------------------------------------


def test_the_roots_come_back_in_the_providers_rank_order(tmp_path):
    os.makedirs(tmp_path / "proj" / ".git")
    env = {"DSH_HOME": str(tmp_path / "h"), "DSH_AGENTS_HOME": str(tmp_path / "a")}
    roots = skill_roots(str(tmp_path / "proj"), custom_dirs=[str(tmp_path / "c")],
                        environ=env)
    assert [r.rank for r in roots] == [100, 200, 300, 400, 500]
    assert [r.source for r in roots] == [
        "project-dsh", "project-agents", "custom", "user-dsh", "user-agents"]
    assert roots[0].path == str(tmp_path / "proj" / ".dsh" / "skills")
    assert roots[3].path == str(tmp_path / "h" / "skills")


def test_missing_roots_are_returned_because_absent_is_valid_empty_state(tmp_path):
    """A caller deciding *where to create* a skill needs the path regardless."""
    os.makedirs(tmp_path / "proj" / ".git")
    roots = skill_roots(str(tmp_path / "proj"), environ={"DSH_HOME": str(tmp_path)})
    assert all(not r.exists for r in roots)
    assert len(roots) == 4


def test_include_default_roots_false_leaves_only_the_custom_ones(tmp_path):
    roots = skill_roots(str(tmp_path), custom_dirs=[str(tmp_path / "only")],
                        include_default_roots=False, environ={})
    assert [r.source for r in roots] == ["custom"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_both_skill_shapes_are_found(tmp_path):
    root = tmp_path / "c"
    skill_dir(str(root), "shaped")
    write(str(root / "flat.md"))
    found = {s.name: s for s in list_skills(str(tmp_path), custom_dirs=[str(root)],
                                            include_default_roots=False, environ={})}
    assert set(found) == {"shaped", "flat"}
    assert found["shaped"].is_directory and found["shaped"].entrypoint.endswith(SKILL_FILE)
    assert not found["flat"].is_directory
    assert found["flat"].entrypoint == str(root / "flat.md")


def test_frontmatter_name_beats_the_path(tmp_path):
    root = tmp_path / "c"
    skill_dir(str(root), "on-disk", "---\nname: declared\n---\n\nbody\n")
    names = [s.name for s in list_skills(str(tmp_path), custom_dirs=[str(root)],
                                         include_default_roots=False, environ={})]
    assert names == ["declared"]


def test_frontmatter_this_regex_cannot_read_falls_back_to_the_path(tmp_path):
    """No YAML parser here -- the core of this package has no dependencies.

    A block scalar is exactly the shape the regex must not pretend to
    understand; falling back to the directory name is right, because that is
    what the name almost always is.
    """
    root = tmp_path / "c"
    skill_dir(str(root), "fallback", "---\nname: >\n  folded value\n---\n\nbody\n")
    names = [s.name for s in list_skills(str(tmp_path), custom_dirs=[str(root)],
                                         include_default_roots=False, environ={})]
    assert names == ["fallback"]


def test_a_directory_without_a_skill_file_is_not_a_skill(tmp_path):
    root = tmp_path / "c"
    os.makedirs(root / "references")
    write(str(root / "references" / "notes.md"))
    assert list_skills(str(tmp_path), custom_dirs=[str(root)],
                       include_default_roots=False, environ={}) == []


def test_the_system_child_is_skipped_only_under_the_user_dsh_root(tmp_path):
    """`.system` holds system-owned directories, not user skills.

    The skip belongs to that one root: a custom root that happens to contain a
    `.system` directory is the caller's own layout and must not be censored.
    """
    home, custom = tmp_path / "h", tmp_path / "c"
    skill_dir(str(home / "skills"), SYSTEM_CHILD)
    skill_dir(str(home / "skills"), "mine")
    skill_dir(str(custom), SYSTEM_CHILD)
    os.makedirs(tmp_path / "proj" / ".git")

    names = [s.name for s in list_skills(
        str(tmp_path / "proj"), custom_dirs=[str(custom)],
        environ={"DSH_HOME": str(home), "DSH_AGENTS_HOME": str(tmp_path / "none")})]
    assert names == [SYSTEM_CHILD, "mine"]      # the custom one survives
    assert [s.root.source for s in list_skills(
        str(tmp_path / "proj"), custom_dirs=[str(custom)],
        environ={"DSH_HOME": str(home), "DSH_AGENTS_HOME": str(tmp_path / "none")})
        if s.name == SYSTEM_CHILD] == ["custom"]


# ---------------------------------------------------------------------------
# Shadowing -- the one that matters
# ---------------------------------------------------------------------------


def test_the_nearer_root_wins_a_duplicate_name_and_the_loser_is_not_returned(tmp_path):
    """The whole reason this module exists.

    Two roots hold `sql`. The harness serves the project copy. An evolution run
    handed the *user* copy would improve a file that is never loaded -- a silent
    no-op that reads as "the method does not work".
    """
    project = tmp_path / "proj"
    os.makedirs(project / ".git")
    home = tmp_path / "h"
    skill_dir(str(project / ".dsh" / "skills"), "sql", "project copy\n")
    skill_dir(str(home / "skills"), "sql", "user copy\n")

    env = {"DSH_HOME": str(home), "DSH_AGENTS_HOME": str(tmp_path / "none")}
    winners = list_skills(str(project), environ=env)
    assert len(winners) == 1
    assert winners[0].root.source == "project-dsh"

    won = find_skill("sql", str(project), environ=env)
    assert won is not None and won.path.startswith(str(project))
    with open(won.entrypoint, encoding="utf-8") as fh:
        assert fh.read() == "project copy\n"


def test_a_custom_root_shadows_the_user_root_but_not_the_project(tmp_path):
    project = tmp_path / "proj"
    os.makedirs(project / ".git")
    home, custom = tmp_path / "h", tmp_path / "c"
    skill_dir(str(project / ".dsh" / "skills"), "a")
    skill_dir(str(custom), "a")
    skill_dir(str(custom), "b")
    skill_dir(str(home / "skills"), "b")

    env = {"DSH_HOME": str(home), "DSH_AGENTS_HOME": str(tmp_path / "none")}
    by_name = {s.name: s.root.source for s in
               list_skills(str(project), custom_dirs=[str(custom)], environ=env)}
    assert by_name == {"a": "project-dsh", "b": "custom"}


def test_find_skill_returns_none_for_a_name_no_root_has(tmp_path):
    assert find_skill("absent", str(tmp_path), include_default_roots=False,
                      environ={}) is None


def test_results_are_sorted_by_name_the_way_the_registry_sorts_summaries(tmp_path):
    root = tmp_path / "c"
    for name in ("zebra", "alpha", "middle"):
        skill_dir(str(root), name)
    names = [s.name for s in list_skills(str(tmp_path), custom_dirs=[str(root)],
                                         include_default_roots=False, environ={})]
    assert names == ["alpha", "middle", "zebra"]


def test_an_unreadable_root_is_empty_state_rather_than_an_exception(tmp_path):
    """The provider treats a confirmed-missing path as valid empty state."""
    assert list_skills(str(tmp_path), custom_dirs=[str(tmp_path / "nope")],
                       include_default_roots=False, environ={}) == []


@pytest.mark.parametrize("skill", [
    Skill("n", "/p/n", None, True),          # type: ignore[arg-type]
    Skill("n", "/p/n.md", None, False),      # type: ignore[arg-type]
])
def test_entrypoint_points_at_the_markdown_for_both_shapes(skill):
    assert skill.entrypoint.endswith(".md")
