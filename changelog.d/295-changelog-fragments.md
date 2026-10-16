Added
Changelog fragments (#295): a PR now adds one file `changelog.d/<number>-<slug>.md`
instead of editing `CHANGELOG.md`, so open PRs stop conflicting on the same
lines. `python -m tools.changelog_fragments check` validates them (and runs in
`pytest -q`); `release [--version X.Y.Z --date YYYY-MM-DD]` folds them into
`[Unreleased]` by category and optionally cuts the release section. See
`changelog.d/README.md`.
