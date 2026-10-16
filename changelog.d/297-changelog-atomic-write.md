Fixed
`tools.changelog_fragments` release no longer truncates `CHANGELOG.md` before
the replacement is safely on disk (#297). The folded text is staged in a
sibling temporary file, flushed and closed, then atomically replaced with
`os.replace`; fragments are removed only after that replace succeeds. A
write, flush, close or replace failure leaves the original history and every
fragment untouched.
