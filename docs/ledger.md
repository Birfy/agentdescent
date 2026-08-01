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
