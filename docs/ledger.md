# The ledger — the versioned artifact store

*Module:* [`agentdescent.ledger`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/ledger.py)
· *API:* [`Ledger`, `Snapshot`, `CASConflict`, …](api.md#the-ledger)

The ledger is the parameter server: the one place an artifact's current value
lives, and the only thing that decides whether a proposed change becomes the new
value. It is backed by a **real git repository**, and every merge is a
**compare-and-swap** against the version the proposer read.

```
artifacts/<id>.json     one JSON blob per artifact (its serialized state)
versions.json           artifact id -> integer version
branch dev              where the loop commits
branch stable           what production reads
```


## More than one writer

CAS plus a version vector was already the right primitive; what it lacked was a
critical section that spans **processes**. `threading.Lock` makes one process's
workers take turns and says nothing to a second process, so two of them could
interleave a read-modify-write of `versions.json` — and lose a commit by a route
CAS cannot see, because both writers read the same head and both were right when
they read it.

In practice the failure arrived even earlier, as git: `index.lock`, or a
checkout refusing to overwrite a file the other process had just written.

Every path that touches the repository now holds an advisory file lock —
`fcntl` where there is one, an atomically-created directory where there is not —
kept in `.git/` rather than the working tree, because a lock file in the tree is
committed by `add -A` and then blocks the next checkout.

`versions.json` is written to a sibling and renamed. `open(..., "w")` truncates
first and fills after, so a reader in another process can see an empty file where
the version vector should be.

**CAS still does its job.** The lock serialises writers; it does not make them
agree. A writer holding a version that has moved is told to rebase exactly as
before — which is what the retry loop in any multi-writer caller is for:

The aggregator does this for you now — `AggregatorConfig(cas_attempts=3)`, with
a jittered backoff, because two writers backing off by the same amount collide
again on the same schedule. Setting it to 1 restores the old behaviour: settle
the evidence back into the pool and wait a round.

**Rebasing means re-applying the diff to the new head**, not re-sending the
candidate. The candidate was computed against a head that no longer exists;
committing it would discard whatever won the race — the lost update CAS exists to
prevent, arriving through the retry meant to preserve it.

Writing it by hand, outside the aggregator, looks like this:

```python
for _ in range(attempts):
    head = ledger.head_version(Ledger.DEV)
    candidate = ledger.snapshot(Ledger.DEV).get(aid).apply(diff)   # re-apply
    try:
        ledger.commit(candidate, base_version=head)
        break
    except CASConflict:
        continue        # somebody committed first; rebase onto their head
```

With one writer none of this fires: every commit goes through a single merger, so
a conflict is unreachable. `result.cas_conflicts` is how much contention a
multi-writer configuration is actually producing.

!!! note "Keep the ledger outside the sandbox"
    A ledger inside a sandbox is destroyed with it. Point `repo_path` at a
    location the sandbox does not own.

## Why git, and why that is not overkill

Three properties are needed and git already has all of them: an append-only
history, atomic commits, and content-addressed storage that deltas similar blobs.
Two more come for free and turn out to matter more:

* **The history is the audit trail.** `result.ledger_log` is `git log`. When a
  run does something surprising, the sequence of merge decisions is right there,
  with the diff that caused each one.
* **A run resumes.** Pass the same `repo_path=` to `evolve()` and it picks up
  the artifact where it stopped, because the state is on disk, not in a process.

```python
result = evolve(tasks, reward, agent=agent, repo_path="./my-run")
# ...later, same call, same path: continues from the committed head
```

Omit `repo_path` and the run gets a scratch repo that is removed when the call
returns. A caller-supplied path is **never** deleted.

!!! note "Git runs with an isolated config"
    A personal `~/.gitconfig` — `commit.gpgsign`, `core.hooksPath`, a template
    dir — must not be able to fail the ledger's own bookkeeping commits, so every
    invocation passes an isolated environment. Your global git setup cannot break
    a run, and a run cannot touch your global setup.

## Compare-and-swap is the whole concurrency story

```python
snap = ledger.snapshot(Ledger.DEV)      # read: artifacts + their versions
artifact = snap.get("my_skill")
base_vv = {"my_skill": snap.version.get("my_skill", 0)}

new_version = ledger.commit(candidate, base_vv, branch=Ledger.DEV,
                            message="merge w2:a91f -> my_skill")
```

`commit` succeeds only if the artifact is still at `base_vv`. If another merge
landed first it raises `CASConflict`, and the aggregator settles those evidence
cards back into the pool for another look — they lost a *race*, not a
*comparison*, which is a distinction the reference aggregator is careful about.

That is the only lock in the system. Workers never block on each other; they
propose against whatever version they read, and disagreement about which version
that was is handled by CAS plus the [staleness policy](staleness.md).

`commit_atomic` extends the same guarantee across several artifacts at once, for
a change that only makes sense applied together.

## Two branches: `dev` and `stable`

| branch | what it is | who reads it |
|---|---|---|
| `dev` | every accepted merge, immediately | the workers |
| `stable` | a merge that has *survived* `promote_after_k` rounds | production |

This is the EMA of weight averaging, expressed as a branch. A change that looks
good on one round's held-out sample and then regresses never reaches `stable`;
the loop keeps moving on `dev` regardless, so the confirmation costs no
throughput.

`finalize()` publishes the current `dev` head to `stable` at the end of a clean
run — without it, a run that stops the moment it hits `target_reward` would leave
the artifact it was *for* one confirmation short of the branch anyone reads.

## Serialization is yours

The ledger does not know what an artifact is. You give it two functions:

```python
ledger = Ledger(repo_path,
                serialize=lambda a: {"state": a.state, "blast_radius": a.blast_radius},
                deserialize=lambda aid, version, payload: MyArtifact(aid, ...))
```

`evolve()` installs a pair for `EvolvingArtifact`. Supply your own only when you
implement [`Evolvable`](data-model.md) yourself.

!!! warning "Every commit stores the whole artifact"
    `artifacts/<id>.json` holds the complete state, not a patch — git does the
    delta-compression underneath. Fine for a playbook or a prompt; worth knowing
    for a [file tree](directory-evolution.md), which is why `TreeSpec` caps
    `max_total_bytes`.

## Failure is reported, not raised

```python
from agentdescent import LedgerFailure     # (GitError, OSError, JSONDecodeError)

try:
    ...
except LedgerFailure as e:
    ...
```

A held `index.lock`, a full `$TMPDIR`, a corrupt JSON blob: none of these are
allowed to escape `evolve()` as an exception. They end the run, and the partial
result comes back with `result.error` set — because an artifact evolved over
nine rounds is worth more than a clean traceback about the tenth.

`ContractRejected` is the one refusal that is a *decision* rather than a
failure: a diff declared a dependency on a superseded contract major, so it can
never be safe to apply.

## Reading a run afterwards

```python
print(result.ledger_log)         # the merge history, newest first
```

```
a91f0c2 merge w2:8fd10e3c:4 -> my_skill
6bd44e1 evoskill: select best frontier member
b912f19 genesis
```

Or open the repo with any git tool you already know — it is an ordinary
repository.
