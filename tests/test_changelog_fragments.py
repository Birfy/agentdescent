"""Changelog fragments: ``changelog.d/*.md`` folded into CHANGELOG.md.

PRs add one fragment file each instead of editing CHANGELOG.md, so they stop
conflicting on the same lines. These tests pin the contract of
``tools/changelog_fragments.py`` on temporary trees, and check the real
``changelog.d`` of this checkout so a malformed fragment fails ``pytest -q``.
"""

import os
from pathlib import Path

import pytest

from tools import changelog_fragments as cf

REPO = Path(__file__).resolve().parent.parent

CHANGELOG = """\
# Changelog

Intro.

## [Unreleased]

### Added

- Existing added entry.

### Fixed

- Existing fixed entry
  that wraps.

## [1.0.0] \u2014 2026-01-01

### Fixed

- Old fix.

[Unreleased]: https://github.com/o/r/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/o/r/releases/tag/v1.0.0
"""

EMPTY_UNRELEASED = """\
# Changelog

## [Unreleased]

## [1.0.0] \u2014 2026-01-01

### Fixed

- Old fix.

[Unreleased]: https://github.com/o/r/compare/v0.9.0...HEAD
[1.0.0]: https://github.com/o/r/compare/v0.9.0...v1.0.0
"""


def make_repo(tmp_path, changelog=CHANGELOG, **fragments):
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    (directory / "README.md").write_text("# Not a fragment\n", encoding="utf-8")
    for name, body in fragments.items():
        (directory / (name.replace("_", "-") + ".md")).write_text(body, encoding="utf-8")
    return tmp_path


def fragment_files(root):
    return sorted(p.name for p in (root / "changelog.d").iterdir())


def release(root, *extra):
    return cf.main(["--root", str(root), "release", *extra])


def test_real_fragments_are_valid():
    fragments, errors = cf.collect(REPO)
    assert errors == []
    assert all(f.category in cf.CATEGORIES for f in fragments)


def test_check_accepts_valid_fragments_and_ignores_readme(tmp_path, capsys):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n", b_2="Added\nTwo.\n")
    (root / "changelog.d" / ".gitkeep").write_text("")
    assert cf.main(["--root", str(root), "check"]) == 0
    assert "2 changelog fragment(s) OK" in capsys.readouterr().out
    fragments, errors = cf.collect(root)
    assert [f.name for f in fragments] == ["a-1.md", "b-2.md"]
    assert errors == []


def test_readme_is_never_folded(tmp_path):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root) == 0
    assert "Not a fragment" not in (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert fragment_files(root) == ["README.md"]


@pytest.mark.parametrize(
    "body, message",
    [
        ("Bugfix\nText.\n", "first line must be one of"),
        ("fixed\nText.\n", "first line must be one of"),
        ("\nFixed\nText.\n", "first line must be one of"),
        ("Fixed\n", "entry text is empty"),
        ("Fixed\n\n   \n", "entry text is empty"),
        ("Fixed\n- \n", "entry text is empty"),
        ("", "first line must be one of"),
    ],
)
def test_check_rejects_bad_fragments(tmp_path, capsys, body, message):
    root = make_repo(tmp_path, bad=body)
    assert cf.main(["--root", str(root), "check"]) == 1
    err = capsys.readouterr().err
    assert "changelog.d/bad.md" in err and message in err


def test_check_flags_non_markdown_files(tmp_path, capsys):
    root = make_repo(tmp_path)
    (root / "changelog.d" / "notes.txt").write_text("Fixed\nx\n")
    assert cf.main(["--root", str(root), "check"]) == 1
    assert "notes.txt: fragments must be .md files" in capsys.readouterr().err


def test_check_without_directory_is_ok(tmp_path):
    assert cf.main(["--root", str(tmp_path), "check"]) == 0


def test_release_appends_below_existing_bullets_and_deletes_fragments(tmp_path):
    root = make_repo(
        tmp_path,
        n1_fix="Fixed\nNew fix (#1).\n",
        n2_add="Added\nNew feature\nacross two lines (#2).\n",
    )
    assert release(root) == 0
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text == (
        CHANGELOG.replace(
            "- Existing added entry.\n",
            "- Existing added entry.\n\n- New feature\n  across two lines (#2).\n",
        ).replace(
            "  that wraps.\n",
            "  that wraps.\n\n- New fix (#1).\n",
        )
    )
    assert fragment_files(root) == ["README.md"]


def test_release_creates_missing_headings_in_keep_a_changelog_order(tmp_path):
    root = make_repo(
        tmp_path,
        s_sec="Security\nSec.\n",
        c_chg="Changed\nChg.\n",
        r_rem="Removed\nRem.\n",
    )
    assert release(root) == 0
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = text.split("## [1.0.0]")[0]
    headings = [l for l in unreleased.splitlines() if l.startswith("### ")]
    assert headings == [
        "### Added", "### Changed", "### Removed", "### Fixed", "### Security",
    ]
    assert "### Changed\n\n- Chg.\n\n### Removed\n\n- Rem.\n\n### Fixed" in text
    assert "  that wraps.\n\n### Security\n\n- Sec.\n\n## [1.0.0]" in text


def test_release_into_empty_unreleased(tmp_path):
    root = make_repo(tmp_path, EMPTY_UNRELEASED, a_1="Fixed\nOne.\n", b_2="Added\nTwo.\n")
    assert release(root) == 0
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Two.\n\n### Fixed\n\n- One.\n\n"
        "## [1.0.0]"
    )


def test_release_is_deterministic_regardless_of_creation_order(tmp_path):
    names = {"b_2": "Fixed\nSecond.\n", "a_1": "Fixed\nFirst.\n", "c_3": "Fixed\nThird.\n"}
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first = make_repo(tmp_path / "one", **names)
    second = make_repo(tmp_path / "two", **dict(reversed(list(names.items()))))
    assert release(first) == 0 and release(second) == 0
    one = (first / "CHANGELOG.md").read_text(encoding="utf-8")
    assert one == (second / "CHANGELOG.md").read_text(encoding="utf-8")
    assert one.index("- First.") < one.index("- Second.") < one.index("- Third.")


def test_release_is_idempotent_without_fragments(tmp_path, capsys):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root) == 0
    after_first = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release(root) == 0
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == after_first
    assert "no fragments to fold" in capsys.readouterr().out


def test_release_normalises_bullet_markers_and_indentation(tmp_path):
    root = make_repo(
        tmp_path,
        a_1="Fixed\n- Marker written by hand\n  continued line.\n",
        b_2="Fixed\nPara one\n\n    Para two, indented.\n",
    )
    assert release(root) == 0
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "- Marker written by hand\n  continued line.\n" in text
    assert "- Para one\n\n  Para two, indented.\n" in text
    assert "- - " not in text


def test_release_with_invalid_fragment_changes_nothing(tmp_path, capsys):
    root = make_repo(tmp_path, good="Fixed\nOk.\n", bad="Nope\nText.\n")
    assert release(root) == 1
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert fragment_files(root) == ["README.md", "bad.md", "good.md"]
    assert "no changes made" in capsys.readouterr().err


def test_release_without_unreleased_section_changes_nothing(tmp_path):
    root = make_repo(tmp_path, "# Changelog\n\n## [1.0.0] \u2014 2026-01-01\n", a_1="Fixed\nOne.\n")
    assert release(root) == 1
    assert fragment_files(root) == ["README.md", "a-1.md"]


def test_release_cuts_a_dated_section(tmp_path, capsys):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root, "--version", "1.1.0", "--date", "2026-02-03") == 0
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [1.1.0] \u2014 2026-02-03\n\n### Added\n" in text
    assert "- One.\n\n## [1.0.0] \u2014 2026-01-01" in text
    assert (
        "[Unreleased]: https://github.com/o/r/compare/v1.1.0...HEAD\n"
        "[1.1.0]: https://github.com/o/r/compare/v1.0.0...v1.1.0\n"
        "[1.0.0]: https://github.com/o/r/releases/tag/v1.0.0\n"
    ) in text
    assert text.count("## [Unreleased]") == 1
    assert "cut [1.1.0]" in capsys.readouterr().out


def test_release_version_without_fragments_cuts_existing_bullets_once(tmp_path):
    root = make_repo(tmp_path)
    assert release(root, "--version", "1.1.0", "--date", "2026-02-03") == 0
    cut = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [1.1.0] \u2014 2026-02-03\n" in cut
    # Re-running now finds an empty [Unreleased]: nothing to release, no change.
    assert release(root, "--version", "1.1.1", "--date", "2026-02-04") == 0
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == cut


def test_release_version_leaves_empty_unreleased_alone(tmp_path):
    root = make_repo(tmp_path, EMPTY_UNRELEASED)
    assert release(root, "--version", "1.1.0", "--date", "2026-02-03") == 0
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == EMPTY_UNRELEASED


def test_release_refuses_an_existing_version(tmp_path):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root, "--version", "1.0.0", "--date", "2026-02-03") == 1
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert fragment_files(root) == ["README.md", "a-1.md"]


@pytest.mark.parametrize(
    "args",
    [
        ["--version", "1.1.0"],
        ["--date", "2026-02-03"],
        ["--version", "v1.1", "--date", "2026-02-03"],
        ["--version", "1.1.0", "--date", "2026-13-40"],
        ["--version", "1.1.0", "--date", "20260203"],
    ],
)
def test_release_rejects_bad_version_or_date(tmp_path, args):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    with pytest.raises(SystemExit) as exc:
        release(root, *args)
    assert exc.value.code == 2
    assert fragment_files(root) == ["README.md", "a-1.md"]


def test_fenced_headings_are_not_mistaken_for_categories(tmp_path):
    text = CHANGELOG.replace(
        "- Existing added entry.\n",
        "- Existing added entry.\n\n```md\n### Security\n```\n",
    )
    root = make_repo(tmp_path, text, s_sec="Security\nSec.\n")
    assert release(root) == 0
    out = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert out.count("### Security") == 2
    assert "  that wraps.\n\n### Security\n\n- Sec.\n\n## [1.0.0]" in out


def test_history_below_unreleased_is_untouched(tmp_path):
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root, "--version", "1.1.0", "--date", "2026-02-03") == 0
    out = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    tail = CHANGELOG[CHANGELOG.index("## [1.0.0]"): CHANGELOG.index("[Unreleased]:")]
    assert tail in out


# -- atomic write / fault injection (#297) ------------------------------------
#
# ``run_release`` used to open CHANGELOG.md with mode "w", which truncates the
# file before the replacement bytes are on disk. A crash (or an injected
# OSError) after that truncate would leave an empty or partial history. The
# fix stages the full replacement in a sibling temp file, fsyncs+closes it,
# then ``os.replace``s; fragments are removed only after replace succeeds.


def test_release_writes_via_atomic_replace(tmp_path, monkeypatch):
    """Happy path still folds and deletes, and the durable write goes through
    ``os.replace`` rather than truncating CHANGELOG.md in place."""
    import os as _os

    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    original = (root / "CHANGELOG.md").read_bytes()
    replaces = []
    real_replace = _os.replace

    def tracking_replace(src, dst):
        replaces.append((str(src), str(dst)))
        # The destination must still hold the original bytes at replace time:
        # a truncate-then-write path would already have emptied it.
        assert Path(dst).read_bytes() == original
        return real_replace(src, dst)

    monkeypatch.setattr(cf.os, "replace", tracking_replace)

    assert release(root) == 0
    assert replaces, "release must call os.replace for the CHANGELOG write"
    assert any(Path(dst).name == "CHANGELOG.md" for _src, dst in replaces)
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "- One." in text
    assert fragment_files(root) == ["README.md"]


def test_release_survives_a_write_failure(tmp_path, monkeypatch):
    """If staging the replacement raises mid-write, CHANGELOG.md and every
    fragment must be left untouched and the failure surfaced."""
    import io

    root = make_repo(tmp_path, a_1="Fixed\nOne.\n", b_2="Added\nTwo.\n")
    before_changelog = (root / "CHANGELOG.md").read_bytes()
    before_fragments = fragment_files(root)

    class BoomFile:
        def write(self, data):
            raise OSError("injected write failure")

        def flush(self):
            pass

        def fileno(self):
            return -1

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def boom_fdopen(fd, *a, **k):
        # Close the real fd so we do not leak the mkstemp handle; then return
        # a stub that fails on write.
        try:
            cf.os.close(fd)
        except OSError:
            pass
        return BoomFile()

    monkeypatch.setattr(cf.os, "fdopen", boom_fdopen)

    err = io.StringIO()
    rc = cf.run_release(root, out=io.StringIO(), err=err)
    assert rc == 1
    assert "failed to write CHANGELOG.md" in err.getvalue()
    assert (root / "CHANGELOG.md").read_bytes() == before_changelog
    assert fragment_files(root) == before_fragments
    leftovers = [p.name for p in root.iterdir() if p.name.startswith(".CHANGELOG")]
    assert leftovers == []


def test_release_survives_a_flush_failure(tmp_path, monkeypatch):
    """Failure while flushing/closing the staged file also preserves history."""
    import io

    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    before_changelog = (root / "CHANGELOG.md").read_bytes()
    before_fragments = fragment_files(root)

    real_fdopen = cf.os.fdopen

    def flaky_fdopen(fd, *a, **k):
        handle = real_fdopen(fd, *a, **k)
        real_flush = handle.flush

        def boom_flush():
            real_flush()
            raise OSError("injected flush failure")

        handle.flush = boom_flush
        return handle

    monkeypatch.setattr(cf.os, "fdopen", flaky_fdopen)

    err = io.StringIO()
    rc = cf.run_release(root, out=io.StringIO(), err=err)
    assert rc == 1
    assert "failed to write CHANGELOG.md" in err.getvalue()
    assert (root / "CHANGELOG.md").read_bytes() == before_changelog
    assert fragment_files(root) == before_fragments


def test_release_survives_a_replace_failure(tmp_path, monkeypatch):
    """If the final os.replace fails, the original CHANGELOG.md stays and
    fragments remain available for a retry."""
    import io

    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    before_changelog = (root / "CHANGELOG.md").read_bytes()
    before_fragments = fragment_files(root)

    def boom_replace(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(cf.os, "replace", boom_replace)

    err = io.StringIO()
    rc = cf.run_release(root, out=io.StringIO(), err=err)
    assert rc == 1
    assert "failed to write CHANGELOG.md" in err.getvalue()
    assert (root / "CHANGELOG.md").read_bytes() == before_changelog
    assert fragment_files(root) == before_fragments


def test_release_success_still_folds_and_is_idempotent(tmp_path, capsys):
    """Successful atomic path: folded text, fragments gone, re-run is a no-op."""
    root = make_repo(tmp_path, a_1="Fixed\nOne.\n")
    assert release(root) == 0
    after = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "- One." in after
    assert fragment_files(root) == ["README.md"]
    assert release(root) == 0
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == after
    assert "no fragments to fold" in capsys.readouterr().out
