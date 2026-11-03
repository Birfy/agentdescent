Fixed
`init PATH` and `algorithms run NAME PATH` now refuse a PATH that does not
exist before guessing kind: exit 2 with `target '…' does not exist (cwd …)` and
no spec written, instead of falling through `guess_kind` to `skill_dir` and a
kind-mismatch or "is not a directory" error. With `--kind text`, a path-looking
missing target still writes the spec but warns that the path string will be used
as the prompt; `plan` surfaces the same warning. Directory kinds' `plan` says
`does not exist` when the target is absent (#706).
