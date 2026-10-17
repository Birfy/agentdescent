Fixed
Candidate code in a container sandbox now sees `PWD=/work`, matching its actual
working directory (#377). `docker`/`podman exec -w /work` moves the process but
leaves `$PWD` as whatever the image or the `run` set, so a CLI that finds its
project from `$PWD` could look past the staged candidate. `exec_prefix()` now sets
`PWD` to the container workdir for Docker and Podman, never the host workspace
path; the engine CLI still gets the host environment and the
candidate environment allowlist is unchanged.
