# The data model — what a gradient is here

*Module:* [`agentdescent.evolvable`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolvable.py)
· *API:* [`Evolvable`, `Diff`, `EvidenceCard`, `Contract`, `VersionVector`](api.md#the-data-model)

Four types carry everything that flows through the system. They are small on
purpose: the whole framework is "propose diffs, merge diffs", and these say what
a diff *is*.

| Deep learning | AgentDescent |
|---|---|
| parameter tensor θ | a library of `Evolvable` artifacts |
| gradient *g* | a `Diff` + its `EvidenceCard` |
| the step | the aggregator's merge decision |
| — | `VersionVector` on the diff: what it was derived against |

The last row has no counterpart, and it is the one that matters. A gradient is
computed against the current parameters and applied immediately. A diff is
proposed against a *version*, and by the time it is merged the world may have
moved — so a diff has to carry what it was looking at.

## `Evolvable` — the unit of evolution

Anything satisfying this protocol can be evolved. There is no registry and no
base class to inherit:

```python
class Evolvable(Protocol):
    id: str
    version: int
    contract: Contract
    blast_radius: float

    def diff(self, other) -> Diff: ...
    def apply(self, diff: Diff) -> "Evolvable": ...
    def evidence_eval(self, evidence: EvidenceCard) -> float: ...
```

`evidence_eval` scores the artifact against the trajectories an evidence card
carries. It was `cheap_eval` until that collided with
[`ThreeLayerVerifier.cheap_eval`](verifier.md), which takes an *artifact* and
means something else — and both are called from the same forty lines of the
aggregator. The old name survives as an alias.

A fourth method, `full_eval(task_set)`, used to be required here and was called
by nothing: ground truth reaches the aggregator through the verifier's `eval_fn`,
which the *domain* supplies, so an artifact never had to know how to score itself.
Requiring a method the engine does not use is a tax on everyone implementing the
protocol, so it is gone.

`apply` returns a **new** artifact rather than mutating — the aggregator scores
candidates side by side, and in-place mutation would make that impossible.

`blast_radius` is an estimate of how much of the task surface this artifact
touches, and it is what [governance](governance.md) sorts on. It is *estimated,
not annotated*: a skill triggered by every task is pulled into the slow layer
automatically, while a harness patch touching one task cluster can ride the fast
layer.

Most users never implement this: [`EvolvingArtifact`](evolution.md) is the
implementation `evolve()` uses, pairing a flat `{key: value}` state with a
[strategy](strategies.md). Implement your own only when your artifact is not a
flat dict — see [`domains/router.py`](orchestrator.md#why-a-synthetic-domain-exists-at-all) for a
worked example.

## `Diff` — the gradient

```python
@dataclass
class Diff:
    diff_id: str
    target: str                  # artifact id
    ops: Dict[str, Any]          # the payload; key -> new value
    contract_breaking: bool = False
    author: str = "unknown"      # which worker produced it
```

`ops` is the **op-space the optimizer reasons over**, and everything the
aggregator does is defined on its keys:

| relation | test | consequence |
|---|---|---|
| overlap | `diffs_conflict` — shared keys | *not* a conflict on its own |
| contradiction | `diffs_contradict` — shared key, different value | resolved on held-out score |
| complement | disjoint keys | **fused** into one candidate |

Overlap alone is deliberately not treated as a conflict: two workers proposing
the *same* value for a key are duplicates, and collapsing them is the point of
content-addressing.

!!! tip "Choosing your key space is a design decision, not a formality"
    The keys decide what can be merged concurrently. `AppendRules` hashes the
    proposal text, so every distinct lesson is its own key and almost everything
    fuses. `SingleSlot` has one key, so every proposal contradicts every other
    and the best one wins. [`FileTree`](directory-evolution.md) uses file paths,
    which is why two workers editing different files merge for free. Same
    machinery, three very different concurrency profiles.

A `None` value **deletes** the key, on both sides: `apply` pops it, and `diff`
emits one for every key the target no longer has. `dict.update` could express add
and replace but not remove — invisible for a rules playbook, disqualifying for a
file tree where a key is a path, and it left `a.apply(a.diff(b))` quietly
different from `b`.

`Diff.size()` is `len(ops)`, which is what the aggregator's trust region caps
(`trust_region_ops`, default 6), alongside `trust_region_chars` (32 000) per
value. Both exist to stop a runaway reflector; both are reported as `oversized`
in [`result.outcomes()`](evolution.md#why-did-nothing-commit) rather than
dropped in silence.

## `EvidenceCard` — the gradient metadata

```python
@dataclass
class EvidenceCard:
    diff: Diff
    base_version: VersionVector     # what it was proposed against
    touched: List[str]
    before_after_delta: float = 0.0 # the proposer's own local measurement
    trajectory_refs: List[str] = () # the failures that justify it
    cost_tokens: int = 0
    cost_wallclock: float = 0.0
```

The card outlives the diff it justifies. When a diff is discarded for staleness
the card settles back into the evidence pool and can be reused — the rollout that
produced it was expensive, and the *observation* is still true even when the
patch no longer applies.

`before_after_delta` is folded into the acceptance test as extra evidence for the
candidate: it is the closest thing here to a gradient magnitude. It is populated
only when `self_verify=True`, which costs a second rollout per proposal —
[worth it for a cheap model, rarely worth it for a coding agent](directory-evolution.md#cost-the-first-order-design-constraint).

## `VersionVector` and staleness

`VersionVector` is just `{artifact_id: version}`. Diffs record only the artifacts
they read, so most are sparse.

```python
vv_staleness(head, base)   # eta = max over touched artifacts of (head - base)
vv_dominates(a, b)         # is a at least as new as b everywhere they overlap?
```

`eta = 0` means the diff was proposed against the current head. Larger means the
world moved underneath it, and what happens next is the
[staleness policy](staleness.md)'s decision — discard, rebase, or accept.

## `Contract` — the interface that must not silently change

```python
Contract(input_schema="task", output_schema="text", side_effects=(), major=1)
```

Artifacts depend on each other's *interfaces*, not their contents. A change that
breaks one is a semver-major event, so the [ledger](ledger.md) records the
contract an artifact was **registered** with and refuses any later commit whose
major disagrees — a breaking change has to be re-registered deliberately rather
than merged like an ordinary diff.

In a single-artifact `evolve()` run this never fires; it exists for the
multi-artifact library the design targets. Until recently it never fired at all:
`Contract`, `is_compatible_with` and `ContractRejected` all existed and nothing
called any of them, while the docstrings described the enforcement as if it were
there.

## `stable_hash`

```python
from agentdescent import stable_hash
```

Python randomises `hash()` of strings per process unless `PYTHONHASHSEED` is
pinned, so anything reproducible — seeding an RNG, assigning a tensor-parallel
section — must hash through here instead. If you write a custom
[strategy](strategies.md) or [parallel strategy](parallelism.md) that hashes
keys, use it, or your `seed=` argument silently means nothing.

## Errors

`ContractError` is the base for "the caller broke a documented contract" —
distinct from a backend failure (a rate limit, a dead endpoint), which the engine
absorbs and reports so partial results survive. A contract violation makes the
run meaningless, so the engines let it propagate. Its subclasses are
[`ProposalContractError` and `RewardContractError`](evolution.md).
