Fixed
A worker CLI now runs in the rollout workspace even when it resolves its project
directory from `$PWD` (#371). `worker_env()` set no `PWD`, so the caller's was
passed through and OpenCode (Bun) worked in the directory `agentdescent` was
launched from: it never saw the staged candidate, could read the dataset's gold
labels and the run store, and with the default `build` agent could edit the
user's real files. `worker_env(workspace)` now sets `PWD` to the workspace for
both `isolate` values (`extra` can still override it), and `code_runner`'s
trimmed child environment sets it too.
