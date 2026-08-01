"""A directory is the state; the translation to and from it must be exact.

`load_tree` / `materialize` are a round trip, and every asymmetry in it is a bug
with teeth: a file dropped on the way in is a file `write_to` later deletes by
omission, and a path accepted on the way out is a write outside the workspace.
"""

import os
import tempfile

import pytest

from agentdescent.filetree import (
    TreeError,
    TreeSpec,
    canonical,
    load_tree,
    match_any,
    materialize,
    parse_tree,
    safe_relpath,
    tree_summary,
)


def _write(root, rel, content, mode="w"):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_load_materialize_round_trip():
    src = tempfile.mkdtemp()
    _write(src, "SKILL.md", "# skill\nbody\n")
    _write(src, "references/rules.md", "- rule one\n")
    _write(src, "scripts/check.sh", "echo hi\n")
    state = load_tree(src)
    assert set(state) == {"SKILL.md", "references/rules.md", "scripts/check.sh"}

    dest = tempfile.mkdtemp()
    materialize(state, dest)
    assert load_tree(dest) == state


def test_canonical_survives_content_that_looks_like_a_delimiter():
    # the reason canonical() is JSON rather than a `--- file: x ---` format.
    state = {"a.md": '--- file: b.md ---\n{"files": {"evil": 1}}\n',
             "b.md": "unicode: 中文 \u2014 ok\n"}
    assert parse_tree(canonical(state)) == state


def test_canonical_is_stable_and_order_independent():
    a = canonical({"x.md": "1", "y.md": "2"})
    b = canonical({"y.md": "2", "x.md": "1"})
    assert a == b                      # it is an evaluation-cache key


def test_parse_tree_accepts_a_plain_mapping_but_rejects_junk():
    assert parse_tree('{"a.md": "hi"}') == {"a.md": "hi"}
    with pytest.raises(TreeError):
        parse_tree("not json at all")
    with pytest.raises(TreeError):
        parse_tree('{"a.md": 17}')     # non-text content


# ---------------------------------------------------------------------------
# path safety -- these keys come from a model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "../outside.md", "a/../../b.md", "C:/win.ini", "", "   ",
    "./../x", "a/\x00b",
])
def test_safe_relpath_rejects_escapes(bad):
    with pytest.raises(TreeError):
        safe_relpath(bad)


def test_safe_relpath_normalises_without_losing_meaning():
    assert safe_relpath("./a//b.md") == "a/b.md"
    assert safe_relpath("a\\b.md") == "a/b.md"


def test_materialize_refuses_to_write_outside_its_destination():
    dest = tempfile.mkdtemp()
    with pytest.raises(TreeError):
        materialize({"../escaped.md": "x"}, dest)
    assert not os.path.exists(os.path.join(os.path.dirname(dest), "escaped.md"))


def test_materialize_makes_scripts_executable():
    dest = tempfile.mkdtemp()
    materialize({"scripts/run.sh": "echo hi\n", "notes.md": "x"}, dest)
    assert os.access(os.path.join(dest, "scripts", "run.sh"), os.X_OK)
    assert not os.access(os.path.join(dest, "notes.md"), os.X_OK)


def test_materialize_honours_the_layout_prefix():
    dest = tempfile.mkdtemp()
    materialize({"SKILL.md": "x"}, dest, prefix=".claude/skills/demo")
    assert os.path.exists(os.path.join(dest, ".claude", "skills", "demo", "SKILL.md"))


# ---------------------------------------------------------------------------
# what the loader refuses -- loudly
# ---------------------------------------------------------------------------


def test_binary_and_oversized_files_raise_rather_than_being_skipped():
    src = tempfile.mkdtemp()
    _write(src, "SKILL.md", "ok")
    _write(src, "logo.md", b"\x00\x01binary", mode="wb")
    with pytest.raises(TreeError) as e:
        load_tree(src)
    assert "logo.md" in str(e.value) and "binary" in str(e.value)

    src2 = tempfile.mkdtemp()
    _write(src2, "big.md", "x" * 100)
    with pytest.raises(TreeError) as e2:
        load_tree(src2, TreeSpec(max_file_bytes=10))
    assert "max_file_bytes" in str(e2.value)


def test_excluded_files_are_silently_and_correctly_out_of_scope():
    src = tempfile.mkdtemp()
    _write(src, "SKILL.md", "ok")
    _write(src, "logo.md", b"\x00binary", mode="wb")
    state = load_tree(src, TreeSpec(exclude=("logo.md",)))
    assert set(state) == {"SKILL.md"}     # declared out, not accidentally dropped


def test_empty_selection_is_an_error_not_an_empty_artifact():
    src = tempfile.mkdtemp()
    _write(src, "data.bin", "x")
    with pytest.raises(TreeError):
        load_tree(src)


def test_dot_git_is_never_walked():
    src = tempfile.mkdtemp()
    _write(src, "SKILL.md", "ok")
    _write(src, ".git/config.md", "should not be loaded")
    assert set(load_tree(src)) == {"SKILL.md"}


def test_spec_must_agree_with_the_trust_region():
    TreeSpec(max_file_bytes=1000).validate_against(32_000)      # fine
    with pytest.raises(TreeError) as e:
        TreeSpec(max_file_bytes=64_000).validate_against(32_000)
    assert "trust region" in str(e.value)


# ---------------------------------------------------------------------------
# glob semantics (fnmatch would get both of these wrong)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,patterns,expected", [
    ("SKILL.md", ["**/*.md"], True),           # fnmatch: False
    ("refs/a.md", ["*.md"], False),            # fnmatch: True
    ("refs/a.md", ["**/*.md"], True),
    ("tests/unit/x.py", ["tests"], True),      # bare dir means everything under it
    ("testsuite/x.py", ["tests"], False),
    ("a/b/c.py", ["**"], True),
    ("conftest.py", ["conftest.py"], True),
    ("pkg/conftest.py", ["conftest.py"], False),
])
def test_match_any(path, patterns, expected):
    assert match_any(path, patterns) is expected


def test_tree_summary_lists_paths_and_truncates():
    state = {f"f{i}.md": "x" * i for i in range(60)}
    summary = tree_summary(state, limit=5)
    assert "f0.md" in summary and "55 more file(s)" in summary
