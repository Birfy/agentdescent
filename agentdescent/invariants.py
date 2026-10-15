"""Declarative, machine-checkable governance invariants (design doc, section 6).

The governance layer already has two mechanisms: an L0 *id list* for the
artifacts the loop may only read (``governance.FROZEN_IDS``) and an
L1/L2 *layer* derived from blast radius. What it does not have is a way to
say "this artifact may only be changed in these ways" as a *contract* -- a
predicate the merge pipeline checks on every diff, before the expensive
measurement, so a policy-violating proposal is rejected without buying a
held-out sweep.

This module is that missing piece. An :class:`Invariant` is a small
declarative rule attached to an artifact's :class:`Contract`; the aggregator
checks them per merge and reports a ``violates-invariant`` outcome instead of
committing (or even measuring) a diff that breaks one. Enforcement is
**unconditional and vacuous by default**: an artifact whose contract declares
no invariants is untouched, and one that declares any is checked -- there is
no opt-in switch to forget, because a governance rule that installs but never
runs is exactly the failure mode ``Policies.require_supported`` exists to
prevent.

The built-in kinds cover the failure modes that matter for self-modifying
agents, and a custom kind is a resolved predicate (a ``workspec.Ref``) that
gets the base artifact, the diff and the candidate and returns violation
strings. Everything is duck-typed on the ``Evolvable`` protocol, so custom
domains can declare invariants without implementing anything new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolvable import Contract, Diff

__all__ = [
    "BUILTIN_INVARIANTS",
    "Invariant",
    "check_dependency_graph",
    "check_invariants",
    "unknown_kind",
]


@dataclass(frozen=True)
class Invariant:
    """One declarative rule an artifact's diffs must satisfy.

    ``kind`` names a built-in rule (see :data:`BUILTIN_INVARIANTS`) or a
    custom rule when ``ref`` is given. ``params`` are the rule's arguments,
    positional and in the order each built-in documents. ``ref`` is a
    ``workspec.Ref`` ("module:attribute" + config) resolving to a predicate
    ``(base, diff, candidate) -> Iterable[str]``; when present, it is the rule
    and ``kind`` is ignored (use ``"custom"`` for readability).
    """

    kind: str
    params: Tuple[Any, ...] = ()
    ref: Optional[Any] = None


# -- built-in rules -----------------------------------------------------------
#
# Each checker has the same shape: ``(base, diff, candidate, params) ->
# List[str]`` where each string is one violation (empty = the rule holds).
# ``base`` and ``candidate`` are Evolvables; ``diff`` is the change that turns
# one into the other. Checkers are pure: they read and never write.


def _check_mutable_keys(base: Any, diff: Diff, candidate: Any,
                        params: Sequence[Any]) -> List[str]:
    """``mutable_keys=([...])`` -- the diff may only touch the declared keys.

    The one-key analog of ``FileTree``'s ``writable``: an artifact that
    declares which keys it may change makes every other edit a policy
    violation rather than an accidental mutation. ``params`` **is** the key
    set: ``mutable_keys=("a", "b")`` permits exactly those keys, and an empty
    ``params`` permits nothing (the artifact is read-only to diffs)."""
    allowed = set(params)
    outside = sorted(set(diff.ops) - allowed)
    if outside:
        return [
            f"diff touches key(s) {outside} outside the declared mutable set "
            f"{sorted(allowed)}"
        ]
    return []


def _check_diff_size_bounded(base: Any, diff: Diff, candidate: Any,
                             params: Sequence[Any]) -> List[str]:
    """``diff_size_bounded=(N,)`` -- at most N op keys per commit.

    A per-commit size cap expressed as an invariant: the diff that changes the
    artifact is bounded, so a runaway reflector cannot rewrite the whole state
    in one step however the trust region is configured."""
    limit = int(params[0]) if params else 0
    if len(diff.ops) > limit:
        return [f"diff has {len(diff.ops)} op(s); invariant caps it at {limit}"]
    return []


def _check_diff_char_bounded(base: Any, diff: Diff, candidate: Any,
                             params: Sequence[Any]) -> List[str]:
    """``diff_char_bounded=(N,)`` -- total value size per commit ≤ N chars.

    The character analog of :func:`_check_diff_size_bounded`, for the case
    where one op (a whole file, a long instruction) is the risk rather than
    the number of ops."""
    limit = int(params[0]) if params else 0
    chars = sum(len(str(v)) for v in diff.ops.values())
    if chars > limit:
        return [
            f"diff carries {chars} chars of value; invariant caps it at {limit}"
        ]
    return []


def _check_blast_radius_nondecreasing(base: Any, diff: Diff, candidate: Any,
                                      params: Sequence[Any]) -> List[str]:
    """``blast_radius_nondecreasing`` -- the candidate may not shrink its radius.

    ``governance.classify`` sorts an artifact into L2 (fast, async) or L1
    (slow, oracle-gated) by ``blast_radius <= FAST_MAX``. An artifact that can
    shrink its own radius could quietly re-classify itself *below* the
    threshold its governance depends on -- the exact direction of change a
    self-modifying system must not be able to make. Vacuous for the reference
    domain (blast radius is fixed at construction) and load-bearing for any
    domain that evolves it."""
    if candidate.blast_radius < base.blast_radius:
        return [
            f"blast_radius shrank {base.blast_radius} -> "
            f"{candidate.blast_radius}; an artifact may not relax its own "
            "governance layer"
        ]
    return []


def _check_depends_no_self(base: Any, diff: Diff, candidate: Any,
                           params: Sequence[Any]) -> List[str]:
    """``depends_no_self`` -- no contract may declare a dependency on itself.

    Duck-typed on ``depends_on``, so it is a no-op on today's single-artifact
    contracts and becomes load-bearing the moment multi-artifact dependencies
    land: a self-referential dependency is a cycle of length one, and it is
    caught here at merge time as well as at registration."""
    aid = getattr(base, "id", None)
    for contract in (getattr(base, "contract", None),
                     getattr(candidate, "contract", None)):
        for dep in getattr(contract, "depends_on", ()) or ():
            if dep == aid:
                return [f"contract depends_on itself ({aid!r})"]
    return []


#: Every built-in kind, name -> checker. ``check_invariants`` reports an
#: unknown kind as a violation rather than silently skipping it: a declared
#: rule that never runs is the failure a caller cannot detect.
BUILTIN_INVARIANTS: Dict[str, Callable[..., List[str]]] = {
    "mutable_keys": _check_mutable_keys,
    "diff_size_bounded": _check_diff_size_bounded,
    "diff_char_bounded": _check_diff_char_bounded,
    "blast_radius_nondecreasing": _check_blast_radius_nondecreasing,
    "depends_no_self": _check_depends_no_self,
}

#: The violation string for a kind that has no checker. Kept as a name so a
#: test can match on it and so ``check_invariants`` reads as a predicate.
unknown_kind = "unknown invariant kind"


def _violations(value: Any, inv: Invariant) -> List[str]:
    """Normalise a custom predicate's return to a list of violation strings.

    A predicate may return ``None`` (no violations), a string, or an iterable
    of strings; anything else is a caller contract violation reported as such
    rather than raising inside the merge path."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        out = list(value)
    except TypeError:
        return [f"invariant predicate {inv.kind!r} returned "
                f"{type(value).__name__}, not a string or iterable of strings"]
    return [str(v) for v in out]


def check_invariants(base: Any, diff: Diff, candidate: Any) -> List[str]:
    """Return every violation of ``base.contract``'s invariants by ``diff``.

    ``base`` and ``candidate`` are the before/after Evolvables and ``diff``
    the change between them. Returns an empty list when the contract declares
    no invariants -- which is the default and the untouched path. Call this
    **before** measuring a candidate, so a policy-violating diff is rejected
    without buying the expensive held-out sweep.

    Unknown kinds and broken custom predicates are reported as violations too:
    a governance rule that cannot run is a governance failure, not a reason to
    skip the gate."""
    contract = getattr(base, "contract", None)
    invariants = getattr(contract, "invariants", ()) or ()
    if not invariants:
        return []
    out: List[str] = []
    for inv in invariants:
        if inv.ref is not None:
            resolve = getattr(inv.ref, "resolve", None)
            if not callable(resolve):
                out.append(f"{unknown_kind}: {inv.kind!r} ref has no resolve()")
                continue
            try:
                fn = resolve()
            except Exception as e:  # noqa: BLE001 - a broken ref is a violation
                out.append(f"invariant predicate {inv.kind!r} failed to resolve: {e}")
                continue
            if not callable(fn):
                out.append(
                    f"invariant predicate {inv.kind!r} resolved to "
                    f"{type(fn).__name__}, which is not callable")
                continue
            try:
                out.extend(_violations(fn(base, diff, candidate), inv))
            except Exception as e:  # noqa: BLE001 - the predicate is user code
                out.append(f"invariant predicate {inv.kind!r} raised: {e}")
            continue
        checker = BUILTIN_INVARIANTS.get(inv.kind)
        if checker is None:
            out.append(f"{unknown_kind}: {inv.kind!r}")
            continue
        out.extend(checker(base, diff, candidate, inv.params))
    return out


def check_dependency_graph(contracts: Mapping[str, Contract]) -> List[str]:
    """Registration-time graph checks over every artifact's ``depends_on``.

    Self-dependencies and cycles are structural facts about the contract set,
    not about any single diff, so they are checked once against the full graph
    rather than per merge. Duck-typed on ``depends_on``: on today's
    single-artifact contracts (no ``depends_on`` attribute, or empty) this
    returns ``[]``, and it becomes load-bearing when multi-artifact
    dependencies land.

    ``contracts`` maps artifact id -> its Contract (as the ledger records on
    :meth:`~agentdescent.ledger.Ledger.register`).
    """
    deps: Dict[str, Tuple[str, ...]] = {
        aid: tuple(getattr(contract, "depends_on", ()) or ())
        for aid, contract in contracts.items()
    }
    out: List[str] = []
    for aid, depends in deps.items():
        if aid in depends:
            out.append(f"{aid!r} depends_on itself")
    # cycle detection: walk each edge, following dependencies depth-first
    # with a per-path seen set (a node revisited on the *same* path is a cycle).
    def cycle_from(start: str, path: Tuple[str, ...]) -> Optional[str]:
        if start in path:
            return " -> ".join((*path, start))
        for dep in deps.get(start, ()):
            found = cycle_from(dep, (*path, start))
            if found is not None:
                return found
        return None

    for aid in deps:
        cycle = cycle_from(aid, ())
        if cycle is not None and f"dependency cycle: {cycle}" not in out:
            out.append(f"dependency cycle: {cycle}")
    return out


# -- contract (de)serialisation ------------------------------------------------
#
# A contract declares the invariants the merge gate enforces, so it has to
# survive the ledger round-trip: ``ledger.snapshot`` materialises artifacts
# from ``artifacts/<id>.json``, and an artifact whose invariants evaporate on
# the way out of storage is a governance rule that silently stops running. The
# payload is JSON-scalar (a ``Ref`` is already JSON-scalar by construction), so
# an old payload without a ``contract`` key reads back as ``None`` and the
# caller's default contract applies -- the untouched path.


def _invariant_to_dict(inv: Invariant) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": inv.kind, "params": list(inv.params)}
    if inv.ref is not None:
        out["ref"] = {
            "target": inv.ref.target,
            "config": dict(getattr(inv.ref, "config", {})),
            "call": bool(getattr(inv.ref, "call", True)),
        }
    return out


def _invariant_from_dict(d: Mapping[str, Any]) -> Invariant:
    ref = None
    if d.get("ref"):
        from .workspec import Ref          # lazy: avoids an import cycle
        ref = Ref(target=d["ref"]["target"], config=dict(d["ref"].get("config", {})),
                  call=bool(d["ref"].get("call", True)))
    return Invariant(kind=d["kind"], params=tuple(d.get("params", ())), ref=ref)


def contract_to_dict(contract: Contract) -> Dict[str, Any]:
    """The JSON-scalar form of a contract, for the ledger's artifact payload."""
    return {
        "input_schema": contract.input_schema,
        "output_schema": contract.output_schema,
        "side_effects": list(contract.side_effects),
        "major": contract.major,
        "invariants": [_invariant_to_dict(i)
                       for i in getattr(contract, "invariants", ())],
    }


def contract_from_dict(data: Mapping[str, Any]) -> Contract:
    """Reconstruct a contract from :func:`contract_to_dict`'s form.

    Unknown keys are tolerated (an old payload, or one written by a future
    version) -- a contract that refuses to load would take the ledger down
    with it, which is the opposite of what derived data is for."""
    return Contract(
        input_schema=data.get("input_schema", "any"),
        output_schema=data.get("output_schema", "any"),
        side_effects=tuple(data.get("side_effects", ())),
        major=data.get("major", 1),
        invariants=tuple(_invariant_from_dict(i)
                         for i in data.get("invariants", ())),
    )
