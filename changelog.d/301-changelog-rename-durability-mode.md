Fixed
`tools.changelog_fragments` release now syncs `CHANGELOG.md`'s parent
directory after the atomic replace and before deleting any fragment, and
keeps the existing file's POSIX permission bits instead of publishing the
private `0600` staging file (#301). If the directory sync fails, the error
says `CHANGELOG.md` may already have been replaced and every fragment is
kept. Non-POSIX platforms skip the directory sync and mode copy explicitly.
