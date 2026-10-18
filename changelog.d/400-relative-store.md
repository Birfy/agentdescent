Fixed
`--store <relative path>` now works for `resume` and every other detached launch
(#400). The launcher runs with the run directory as its cwd and was handed the
still-relative `log.txt` path, so `agentdescent --store ./store resume <id>` died
with `could not start the run: ... FileNotFoundError: './store/<id>/log.txt'`
while an absolute store worked. `RunDir` now always holds an absolute path.
