# Sandboxes — the environment a rollout runs in

> **Plugs into [`evolve`](evolution.md) via** `code_runner(..., sandbox_pool=...)`,
> or `policies=Policies(sandbox_provider=..., sandbox_spec=...)`.

Once a candidate is *code* rather than text — [evolving a
directory](directory-evolution.md), [DGM](algo-dgm.md), a
[`code_runner`](directory-evolution.md#3-runners-giving-a-real-agent-the-candidate)
of any kind — running it needs somewhere to run. Two questions follow, and they
are independent:

* **lifetime** — who owns a workspace, when does it come back, what happens if
  its owner dies. The pool answers this for every provider;
* **isolation** — what the candidate can reach while it runs. The *provider*
  answers this, and the answers differ by a lot.

## Lifetime: leases, not deletion by age

Every rollout leases a workspace from a pool and gives it back. That is a change
of ownership rather than of behaviour: the default is still one fresh directory
per call, deleted afterwards.

```python
from agentdescent.sandbox import SandboxPool, WorkspaceProvider

pool = SandboxPool(WorkspaceProvider(), max_sandboxes=8)
run = code_runner([...], sandbox_pool=pool)     # share it, and the cap is shared
```

**One ceiling, not two.** `max_concurrency` worker threads and
`eval_concurrency` scoring threads both stage workspaces. Sharing one pool is
what makes the limit a limit; with a pool each, the real ceiling is their sum.
`acquire` blocks when the pool is full — failing would push retry logic into
every caller, and growing would make the ceiling a suggestion.

**Reuse is opt-in.** `SandboxSpec(reuse=True)` hands back a warm workspace,
reset to empty first. The default is a fresh one because `code_runner`'s frozen
overlay assumes a clean directory: a workspace still holding the previous
candidate's files produces a score that looks entirely ordinary and is wrong.

**Reclaiming what an owner abandoned.** Each workspace holds a
`.agentdescent-lease.json` naming its owner, and the owner renews it while
working. `WorkspaceProvider.reap()` deletes the ones nobody is renewing.

That replaces "delete anything older than six hours", which was wrong in both
directions. A directory's mtime does not change while a process works *inside*
it, so a long rollout looked abandoned; and on a shared `$TMPDIR` — a parameter
sweep, several containers on one volume — one run's live workspace looked
collectable to every other run on the machine. Renewal is a statement by the
owner; its absence is the thing worth acting on.

A live owner gets one grace period: between one and two TTLs, an expired lease
whose process is still running is left alone. Past that it goes regardless, so a
recycled pid cannot keep a dead run's directory alive forever.

!!! note "Where the numbers are"
    `sandbox_wait_s`, `sandbox_setup_s` and the created/reused/failure counts
    reach `EvolutionResult` when you pass the pool a `Meter`. `evolve()` does not
    wire one automatically: its default [executor](execution.md) is handed the
    actors directly and never opens a pool of its own, so on a default run these
    fields stay zero rather than being quietly approximate.

## One ceiling across processes

`SandboxPool` bounds what a **process** creates. Two runs on one machine each
respect their own limit and together exceed the machine's — the same mistake
`max_concurrency` and `eval_concurrency` made before they shared a gate, one
level up.

```python
from agentdescent.sandbox_shared import SharedSandboxPool

pool = SharedSandboxPool(root="/var/tmp/agentdescent-pool",
                         capacity=8,      # the machine's ceiling
                         holders=2)       # how many runs expect to share it
```

No server. Every sandbox already carries a lease file naming its owner and when
it was last renewed — the file `reap` reads to decide what is abandoned. Counting
those answers a different question with the same data: how many sandboxes are
alive on this machine right now.

**Quota, not just capacity.** A ceiling alone lets whoever asks first take
everything, so a long run starves a short one. Each holder is guaranteed
`capacity // holders` (at least one) and may exceed it only while others are
under theirs. Without that, sharing a ceiling is worse than not sharing: the runs
interfere and none of them can tell.

**A dead holder's slot comes back.** Its lease stops being renewed and stops
being counted — the same rule that reclaims its directory.

!!! warning "Advisory, and cooperative"
    Admission reads the lease directory and then acquires; nothing holds the
    directory still in between. Two processes admitting at the same instant can
    both see room and both take it, so the ceiling is a strong tendency rather
    than a hard cap. That is the right trade for a bound whose job is to stop a
    machine thrashing — but do not use it where an exact limit is a correctness
    property.

!!! note "One machine, verified; several, not"
    Nothing here assumes a single machine, and nothing here has been run on two.
    The lease directory would have to be somewhere both can see, and shared
    filesystems have their own opinions about atomicity. That is the work a
    cross-machine claim needs, and it has not been done.

## Isolation strength — three levels

Lifetime and isolation are different questions. The pool answers the first for
every provider; which provider you choose answers the second.

| provider | filesystem | network | privileges | limits |
|---|---|---|---|---|
| `WorkspaceProvider` (default) | **the whole host** | **the host's** | your user's | timeout only (POSIX: rlimits) |
| `ContainerProvider` | the workspace only | off unless asked | no capabilities, not root | memory / CPU, every platform |
| remote / microVM | — | — | — | not implemented |

**What the default does not stop.** A trimmed environment redirects a *lookup*,
not a *path*. `HOME` points inside the workspace, so `~/.aws/credentials` misses
— and `open("/Users/you/.aws/credentials")` does not. The default provider is
appropriate for code you would run yourself; it is not a boundary.

```python
from agentdescent.sandbox import SandboxPool
from agentdescent.sandbox_container import ContainerProvider

pool = SandboxPool(ContainerProvider("python:3.11-slim"), max_sandboxes=8)
run = code_runner(["python", "main.py"], test_cmd=["pytest", "-q"],
                  sandbox_pool=pool)
```

**Docker or podman, whichever is there.** The engine is detected unless you name
one (`engine="podman"`); the flags used are the ones both accept, and the test
suite runs the same isolation assertions against each engine that is installed.
No Python dependency — the core still installs with none.

Staging is unchanged: the tree is materialised on the host and bind-mounted at
`/work`, so `FileTree`, the frozen overlay and fixtures all behave exactly as
before. Only execution moves.

!!! warning "macOS and Windows: the workspace must be in a shared path"
    The engine runs inside a VM there, and it shares only part of the host. The
    system temporary directory — where a workspace goes by default — is usually
    outside it, and the container then starts with an empty `/work`, so the first
    symptom is the candidate failing to open its own files.

    The provider checks for this at acquire time and says so. Stage under your
    home directory (`workspace_root=`), or add the path to the VM
    (`colima start --mount <path>:w`, or Docker Desktop's File Sharing).

Defaults are closed and opened one field at a time:

| `SandboxSpec` field | effect |
|---|---|
| `network` | `None`/`"none"` → `--network none`. Only `"inherit"` connects it |
| `memory_mb` / `cpu` | `--memory` / `--cpus`; unset means no flag, not a default ceiling |
| `env_allowlist` | variable **names**; values are read on the host and passed one at a time. Nothing is inherited in bulk |
| `image` | overrides the provider's image per rollout |

Always on: read-only root with a writable `/work` and `/tmp`, `--cap-drop ALL`,
`no-new-privileges`, a pids limit, and the container runs as your uid so the
files it writes are yours to delete.

!!! warning "Stronger, not absolute"
    Containers share the host kernel and escapes exist. This is a large step up
    from a trimmed environment and it is not a licence to run deliberately
    hostile code. For that you want a VM boundary, which this does not provide.

## The environment is part of a measurement

A score is a statement about a candidate *under a toolchain*. Two rollouts run
under different images are not two samples of one quantity, and averaging them
produces a number that answers no question anyone asked.

So `SandboxSpec.fingerprint()` becomes part of the [evaluation cache
key](verifier.md#the-evaluation-cache) — one environment's measurement can never
answer another's question — and a run whose acceptance decision compared
measurements from different environments raises `env_mismatch`, which
[`bench/`](efficiency.md#the-configuration-matrix-bench) marks rather than
averaging in.

The fingerprint is empty while there is only one environment, which is why a
default run's cache keys are unchanged.

## Related

* [Evolving a directory](directory-evolution.md) — what puts code in a sandbox
* [Where rollouts run](execution.md) — the process a sandbox is acquired from
* [Efficiency](efficiency.md) — what sandbox setup and waiting actually cost
