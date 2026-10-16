# Changelog fragments

Do not edit `CHANGELOG.md` in a pull request. Every PR used to add a bullet at
the same spot under `[Unreleased]`, so each merge made all the other open PRs
conflict. Instead, add **one new file** here; new files never conflict. The
maintainer folds the fragments into `CHANGELOG.md` when cutting a release.

## Writing a fragment

Create `changelog.d/<number>-<slug>.md`, where `<number>` is the issue (or, if
there is none, the PR) number and `<slug>` is a few lowercase words, for example
`changelog.d/286-sigmoid-novelty-overflow.md`. The number plus slug keeps names
unique across PRs.

- **First line:** the category, exactly one of `Added`, `Changed`,
  `Deprecated`, `Removed`, `Fixed`, `Security`.
- **Remaining lines:** the entry text, in the same style as the existing
  `CHANGELOG.md` bullets (Markdown, wrapped around 80 columns, mention the issue
  as `(#286)`). Do not write the leading `- `; it is added for you, and
  continuation lines are indented automatically.

Example (`changelog.d/286-sigmoid-novelty-overflow.md`):

```markdown
Fixed
`sigmoid_novelty_weights` no longer raises `OverflowError` on large negative
scores, and a non-finite score no longer poisons the weight total (#286).
```

One fragment is one entry. A PR with several user-visible changes adds several
fragments. Files other than `*.md` (and this `README.md`) are rejected, and
dotfiles such as `.gitkeep` are ignored.

## Checking

`pytest -q` validates every fragment (`tests/test_changelog_fragments.py`). You
can also run the check on its own:

```bash
python -m tools.changelog_fragments check
```

## Releasing (maintainers)

```bash
# Fold the fragments into CHANGELOG.md under [Unreleased] and delete them.
python -m tools.changelog_fragments release

# Same, and cut the release: [Unreleased] becomes "## [X.Y.Z] — YYYY-MM-DD",
# a fresh empty [Unreleased] is added, and the compare links are updated.
python -m tools.changelog_fragments release --version X.Y.Z --date YYYY-MM-DD
```

Entries are grouped by category in Keep a Changelog order and appended below
any bullets already under that heading, in file-name order. With no fragments
(and no `--version`) the command does nothing, so it is safe to re-run. The
script does not touch `agentdescent/__init__.py`; the version still lives only
there.
