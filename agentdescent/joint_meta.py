"""Joint multi-slot evolution: search the whole decision plane, not one rule at a time.

Everything in the meta stack so far evolves **one** slot: ``meta_run`` /
``continuous_meta_run`` fix a ``slot`` and search its value, ``meta_population_run``
runs several rules of one slot as islands, and ``portfolio_report`` only *moves
budget* between slots after the fact. Nothing ever searches the slots
**together** -- and that is the gap #388 names (P7): two decision rules that are
individually neutral can be jointly decisive, and a change to one slot can only
be judged against the others it runs beside.

This module is that rung. A **joint candidate** is a bundle ``{slot: value}``
over a chosen subset of :data:`~agentdescent.meta.SLOTS`; the outer ``evolve()``
holds the whole bundle as its artifact, each rollout installs **all** the slots
on ``Policies`` and runs the inner search, and the reward is the inner
held-out curve. The per-slot gates, the key-space diff merge, and the archive
are reused unchanged -- a bundle proposal names one or more slots as
``=== <slot> ===`` blocks, so the aggregator unions edits to different slots
(a coherent multi-slot change) and resolves two edits to the same slot on
held-out score exactly as it does for any strategy.

The second half of #388 is **isolating untrusted rules**: a joint change is
only adopted if it does not *regress* any slot on held-out problems. The
:func:`joint_meta_validate` attribution scores each slot's **leave-one-out
contribution** (the joint score minus the score with only that slot reverted);
a slot whose contribution is negative is *untrusted*, and with ``isolate=True``
it is rolled back while the rest of the bundle is adopted -- so one over-eager
rule cannot smuggle a regression into the archive.

The archive stays the same shape: an adopted bundle writes one ordinary
``meta-rule`` record per slot (with its own ``problem_descriptor``), so
``_archived_rules``, ``exam_archive`` and ``continuous_meta_run`` keep seeding
and examining it with no special case.
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from .evolvable import Diff
from .evolution import EvolutionResult, Task, evolve
from .meta import (
    MetaOutcome, MetaReward, MetaRunReport, Problem, SlotSpec, _archived_rules,
    archive_rule, auc, problem_descriptor, slot_reflector,
)

__all__ = [
    "PoliciesBundle",
    "joint_problem",
    "JointMetaReport",
    "joint_meta_validate",
    "joint_meta_run",
    "joint_archive_rule",
]


# ---------------------------------------------------------------------------
# The artifact: one bundle of slot values
# ---------------------------------------------------------------------------

_BLOCK = re.compile(r"^===\s*([A-Za-z_][\w\- ]*?)\s*===\s*$", re.MULTILINE)


def _parse_blocks(text: str,
                  allowed: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Split ``=== slot ===`` blocks into ``{slot: body}``.

    Only headers naming an ``allowed`` slot (typically ``self.specs``) split the
    text, so a slot's own value that happens to contain a ``=== ... ===`` line
    is *content*, not a delimiter. The last block for a slot wins, and text
    before the first accepted header is ignored (the reflector's prose)."""
    ok = set(allowed) if allowed is not None else None
    matches = [m for m in _BLOCK.finditer(text or "")
               if ok is None or m.group(1).strip() in ok]
    out: Dict[str, str] = {}
    for index, match in enumerate(matches):
        slot = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        out[slot] = body
    return out


@dataclass
class PoliciesBundle:
    """The joint artifact: a bundle of per-slot :class:`SlotSpec` values.

    It is both a :class:`~agentdescent.evolvable.Strategy` (``keys`` / ``initial``
    / ``render`` / ``to_diff`` for the outer ``evolve()``) and a spec-like object
    (``compile`` / ``describe`` for validation and the reflector). The state is
    ``{slot: rendered value}``; the key space is the slot names, so the
    aggregator's existing union-on-distinct-keys and resolve-on-same-key rules
    apply with no new machinery.

    A proposal names one or more slots, each as ``=== <slot> ===`` followed
    by that slot's complete new value; several blocks form a coherent joint
    change evaluated as one candidate. A bare proposal (no header) is accepted only when
    exactly one slot's gate compiles it -- otherwise it is ambiguous and
    refused, because guessing which slot a raw rule belongs to is how a joint
    search would silently corrupt the wrong one.

    ``seed_values`` lets a caller start the bundle from the archive's best
    per-slot rules instead of each spec's own initial value.
    """

    specs: Mapping[str, SlotSpec]
    seed_values: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("PoliciesBundle needs at least one slot spec")

    @property
    def slots(self) -> List[str]:
        return list(self.specs)

    def keys(self) -> Sequence[str]:
        """The key space is the slot names -- what the aggregator merges on."""
        return list(self.specs)

    def _slot_initial(self, slot: str) -> str:
        if self.seed_values is not None and slot in self.seed_values:
            return self.seed_values[slot]
        spec = self.specs[slot]
        return spec.render(spec.initial())

    def initial(self) -> Dict[str, str]:
        return {slot: self._slot_initial(slot) for slot in self.specs}

    def render(self, state: Mapping[str, str]) -> str:
        blocks = []
        for slot in sorted(state):
            blocks.append(f"=== {slot} ===\n{state[slot]}")
        return "\n".join(blocks)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        text = (proposal or "").strip()
        blocks = _parse_blocks(text, self.specs)
        if blocks:
            # Every valid block in the proposal becomes one key of the diff, so a
            # reflector may propose a coherent MULTI-slot change; the aggregator
            # evaluates it as one candidate and joint_meta_validate attributes it
            # per slot (and can isolate the harmful ones).
            ops: Dict[str, str] = {}
            for slot, value in blocks.items():
                value = value.strip()
                if slot not in self.specs or not value or state.get(slot) == value:
                    continue
                if not self._compiles(slot, value):
                    continue
                ops[slot] = value
            if not ops:
                return None
            key = "+".join(sorted(ops))
            return Diff(diff_id=f"{author}:{key}:{base_version}", target=target,
                        ops=ops, author=author)
        # No header: accept only if exactly one slot's gate compiles the text.
        if not text:
            return None
        accepted = [slot for slot in self.specs if self._compiles(slot, text)]
        if len(accepted) != 1:
            return None
        slot = accepted[0]
        if state.get(slot) == text:
            return None
        return Diff(diff_id=f"{author}:{slot}:{base_version}", target=target,
                    ops={slot: text}, author=author)

    def _compiles(self, slot: str, value: str) -> bool:
        try:
            self.specs[slot].compile(value)
        except Exception:  # noqa: BLE001 - a gate that raises refuses the value
            return False
        return True

    def compile(self, rendered: str) -> Dict[str, Any]:
        """``{slot: compiled object}``. A slot missing from ``rendered`` falls
        back to its initial value, so a partial render is still runnable."""
        blocks = _parse_blocks(rendered or "", self.specs)
        out: Dict[str, Any] = {}
        for slot, spec in self.specs.items():
            text = blocks.get(slot) or self._slot_initial(slot)
            out[slot] = spec.compile(text)
        return out

    def describe(self) -> str:
        parts = [
            "The value is a **bundle** of decision slots. Each slot is one block:",
            "```",
            "=== <slot> ===",
            "<that slot's complete new value>",
            "```",
            "Emit one block per slot you change; change several together when "
            "they interact (the joint change is evaluated as one candidate). "
            "The slots are:",
        ]
        for slot in self.specs:
            parts.append(f"- `{slot}`: {self.specs[slot].describe()}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# The inner problem
# ---------------------------------------------------------------------------


def joint_problem(
    tasks: Sequence[Task],
    reward: Callable[[Task, str], float],
    *,
    base: Any = None,
    **evolve_kwargs: Any,
) -> Problem:
    """An inner ``evolve()`` whose candidate is the **whole bundle**.

    ``(bundle, seed) -> MetaOutcome`` where ``bundle`` is
    :meth:`PoliciesBundle.compile`'s ``{slot: object}``. Every slot in the
    bundle is installed on ``base`` (a ``Policies``) at once, so the inner run
    sees the joint decision plane -- not just one rule. ``base`` defaults to a
    bare ``Policies()``.
    """
    from .policies import Policies

    base = base if base is not None else Policies()
    kwargs = dict(evolve_kwargs)
    kwargs["verbose"] = False

    def problem(bundle: Any, seed: int) -> MetaOutcome:
        policies = base
        for slot, value in bundle.items():
            policies = policies.merged_with(**{slot: value})
        result = evolve(list(tasks), reward, seed=seed, policies=policies, **kwargs)
        return MetaOutcome.from_result(result)

    return problem


# ---------------------------------------------------------------------------
# Validation: attribute a joint change to its slots, isolate the untrusted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointMetaReport:
    """What a :func:`joint_meta_run` decided, per slot and jointly.

    ``slot_gains`` is each slot's **leave-one-out contribution** -- the joint
    score minus the score with only that slot reverted to its seed. A slot is
    ``untrusted`` when that contribution is **negative** (reverting it would
    improve the held-out score); ``isolated`` are the untrusted slots rolled
    back before adoption (empty when the whole bundle was rejected). ``value``
    is the adopted bundle (the corrected one when isolation fired)."""

    seed_value: str
    value: str
    verdict: str
    joint_gain: float
    slot_gains: Dict[str, float]
    slot_values: Dict[str, str]
    untrusted: Tuple[str, ...]
    isolated: Tuple[str, ...]
    reason: str


def _named(problems: Union[Sequence[Problem], Mapping[str, Problem]]) -> Dict[str, Problem]:
    if isinstance(problems, Mapping):
        return dict(problems)
    return {f"p{i}": p for i, p in enumerate(problems)}


def _evaluate(bundle: Mapping[str, Any], named: Mapping[str, Problem],
              seeds: Sequence[int], score: MetaReward) -> Dict[str, float]:
    """Per-problem mean reward of a compiled bundle over ``seeds``."""
    out: Dict[str, float] = {}
    for name, problem in named.items():
        out[name] = statistics.fmean(
            float(score(problem(dict(bundle), s))) for s in seeds)
    return out


def _mean_gain(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    keys = list(before)
    return statistics.fmean(after[k] - before[k] for k in keys) if keys else 0.0


def joint_meta_validate(
    bundle_spec: PoliciesBundle,
    before_value: str,
    after_value: str,
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    seeds: Sequence[int] = (0,),
    meta_reward: Optional[MetaReward] = None,
    min_gain: float = 0.0,
    isolate: bool = True,
) -> Dict[str, Any]:
    """Attribute a joint change to its slots, and isolate the ones that hurt.

    * **joint** -- the after bundle's mean gain over the before bundle on
      ``val_problems`` at ``seeds``.
    * **per slot (leave-one-out)** -- each slot's **contribution** is the joint
      score minus the score with *only that slot* reverted to its seed. This is
      what credits a slot that is load-bearing inside a synergy (reverting it
      collapses the joint gain) and still exposes one that drags the bundle
      down (reverting it *raises* the score). It is not the "change it alone
      from the seed" marginal, which reads 0 for every slot of a coupled
      landscape -- exactly the change P7 exists to find.
    * **untrusted** -- a slot whose contribution is **negative** (reverting it
      would improve the held-out score). ``min_gain`` is the bar for the joint
      and corrected gain, not for a slot's contribution, so a coupled change is
      not rejected for having no lone effect.
    * **isolation** -- with ``isolate=True`` an untrusted slot is rolled back to
      its seed value and the corrected bundle re-validated; it is adopted only
      if the corrected gain still clears ``min_gain``. With ``isolate=False``
      any untrusted slot rejects the whole bundle (fail-closed).

    Returns a plain dict (JSON-friendly for a CLI/report): ``joint_gain``,
    ``slot_gains`` (leave-one-out contributions), ``untrusted``, ``isolated``,
    ``verdict`` (``adopt`` / ``reject``), ``value`` (the bundle to adopt),
    ``reason``.
    """
    named = _named(val_problems)
    if not named:
        raise ValueError("joint_meta_validate() got no val_problems")
    score = meta_reward or auc
    before = bundle_spec.compile(before_value)
    after = bundle_spec.compile(after_value)
    before_scores = _evaluate(before, named, seeds, score)
    after_scores = _evaluate(after, named, seeds, score)
    joint_gain = _mean_gain(before_scores, after_scores)

    slot_gains: Dict[str, float] = {}
    for slot in bundle_spec.specs:
        # leave-one-out: revert just this slot from the adopted bundle and see
        # how much the joint score drops -- its contribution to the synergy.
        loo = dict(after)
        loo[slot] = before[slot]
        loo_scores = _evaluate(loo, named, seeds, score)
        slot_gains[slot] = _mean_gain(loo_scores, after_scores)
    untrusted = tuple(slot for slot, gain in slot_gains.items() if gain < 0.0)

    def result(verdict: str, value: str, isolated: Sequence[str], gain: float,
               reason: str) -> Dict[str, Any]:
        return {
            "joint_gain": gain,
            "slot_gains": dict(slot_gains),
            "untrusted": untrusted,
            "isolated": tuple(isolated),
            "verdict": verdict,
            "value": value,
            "reason": reason,
        }

    if not untrusted:
        verdict = "adopt" if joint_gain >= min_gain else "reject"
        reason = (f"joint gain {joint_gain:+.3f} over {len(named)} problem(s), "
                  f"no slot regressed")
        if verdict == "reject":
            reason = (f"joint gain {joint_gain:+.3f} did not clear "
                      f"min_gain {min_gain:+.3f}")
        return result(verdict, after_value, (), joint_gain, reason)

    if isolate:
        corrected = _parse_blocks(after_value, bundle_spec.specs)
        before_blocks = _parse_blocks(before_value, bundle_spec.specs)
        for slot in untrusted:
            if slot in before_blocks:
                corrected[slot] = before_blocks[slot]
            else:
                corrected.pop(slot, None)
        corrected_value = bundle_spec.render(corrected)
        corrected_scores = _evaluate(bundle_spec.compile(corrected_value),
                                     named, seeds, score)
        corrected_gain = _mean_gain(before_scores, corrected_scores)
        if corrected_gain >= min_gain:
            return result(
                "adopt", corrected_value, untrusted, corrected_gain,
                f"adopted after isolating untrusted slot(s) {list(untrusted)}; "
                f"corrected joint gain {corrected_gain:+.3f}")

    return result(
        "reject", after_value, (), joint_gain,
        f"untrusted slot(s) {list(untrusted)} regressed on held-out problems")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _seed_bundle(
    store: Any, slots: Sequence[str], specs: Mapping[str, SlotSpec],
    *,
    config: Optional[Dict[str, Any]], artifact_id: str,
) -> Dict[str, str]:
    """Seed each slot from the archive's best rule for it, else its own initial."""
    seeded: Dict[str, str] = {}
    for slot in slots:
        spec = specs[slot]
        rules = _archived_rules(store, slot=slot, spec=spec, config=config,
                                artifact_id=artifact_id)
        if rules:
            seeded[slot] = rules[0].rendered or spec.render(spec.initial())
        else:
            seeded[slot] = spec.render(spec.initial())
    return seeded


def joint_meta_run(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slots: Sequence[str],
    specs: Mapping[str, SlotSpec],
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (4, 5, 6, 7),
    min_gain: float = 0.0,
    isolate: bool = True,
    store: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    run_id: Optional[str] = None,
    **evolve_kwargs: Any,
) -> Tuple[JointMetaReport, EvolutionResult]:
    """Evolve several decision slots **together**, validate them jointly, decide.

    ``problems`` are bundle-aware :class:`Problem` s -- ``(bundle, seed) ->
    MetaOutcome``, exactly what :func:`joint_problem` builds -- so each rollout
    runs the inner search with **all** the slots installed. The outer artifact
    is the :class:`PoliciesBundle`; a proposal may change several slots
    together, and the aggregator unions distinct-slot edits and resolves
    same-slot contradictions on held-out score.

    After ``evolve()`` the bundle is validated by :func:`joint_meta_validate`:
    each slot's **leave-one-out contribution** is attributed, and an untrusted
    slot (a negative contribution) is isolated (rolled back) when
    ``isolate=True``. On ``adopt`` the bundle is
    written to ``store`` (one ``meta-rule`` record per slot) when a store is
    given, seeded from the archive's per-slot best when ``store`` is given.

    ``val_problems``/``val_seeds`` must be isolated from the training set, the
    same contract ``meta_run`` enforces. Returns ``(report, result)``.
    """
    if propose is None and model is None:
        raise ValueError("joint_meta_run() needs propose= or model=")
    if not slots:
        raise ValueError("joint_meta_run() got no slots")
    missing = [s for s in slots if s not in specs]
    if missing:
        raise ValueError(f"no spec for slot(s) {missing}")
    named = _named(problems)
    val_named = _named(val_problems)
    if not named:
        raise ValueError("joint_meta_run() got no problems")
    if not val_named:
        raise ValueError("joint_meta_run() needs val_problems to judge transfer")
    overlap = set(named) & set(val_named)
    if overlap:
        raise ValueError(f"problems and val_problems share names {sorted(overlap)}")
    if set(seeds) & set(val_seeds):
        raise ValueError("seeds and val_seeds must be disjoint")

    specs = {slot: specs[slot] for slot in slots}
    seed_values = (_seed_bundle(store, slots, specs, config=config,
                                artifact_id=artifact_id) if store is not None else None)
    bundle_spec = PoliciesBundle(specs, seed_values=seed_values)
    initial = bundle_spec.initial()
    seed_value = bundle_spec.render(initial)

    reflector = propose if propose is not None else slot_reflector(model, bundle_spec)
    score = meta_reward or auc
    # Seed-major interleaving, so every problem lands on both sides of the
    # positional held-out cut (the meta_evolve lesson).
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for seed in seeds for name in named]

    def run(rendered: str, task: Task) -> str:
        bundle = bundle_spec.compile(rendered)
        outcome = named[task.meta["problem"]](bundle, int(task.meta["seed"]))
        return outcome.to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(score(MetaOutcome.from_json(output)))
        except (TypeError, ValueError, KeyError):
            return 0.0

    result = evolve(
        tasks, reward, run=run, propose=reflector, strategy=bundle_spec,
        initial_state=initial, self_verify=False, solved_threshold=1.1,
        **evolve_kwargs)
    value = result.rendered or seed_value

    validation = joint_meta_validate(
        bundle_spec, seed_value, value, val_named, seeds=val_seeds,
        meta_reward=meta_reward, min_gain=min_gain, isolate=isolate)

    adopted_value = validation["value"]
    adopted = validation["verdict"] == "adopt"
    report = JointMetaReport(
        seed_value=seed_value, value=adopted_value,
        verdict=validation["verdict"], joint_gain=float(validation["joint_gain"]),
        slot_gains={k: float(v) for k, v in validation["slot_gains"].items()},
        slot_values=_parse_blocks(adopted_value, bundle_spec.specs),
        untrusted=tuple(validation["untrusted"]),
        isolated=tuple(validation["isolated"]),
        reason=validation["reason"])

    if adopted and store is not None:
        joint_archive_rule(
            store, report=report, result=result, slots=slots, specs=specs,
            config=config, artifact_id=artifact_id,
            run_id=run_id or f"joint-{int(time.time())}")
    try:
        result.meta = {
            "verdict": report.verdict, "joint_gain": report.joint_gain,
            "slot_gains": report.slot_gains, "untrusted": list(report.untrusted),
            "isolated": list(report.isolated), "reason": report.reason,
        }
    except AttributeError:
        pass
    return report, result


def joint_archive_rule(
    store: Any,
    *,
    report: JointMetaReport,
    result: EvolutionResult,
    slots: Sequence[str],
    specs: Mapping[str, SlotSpec],
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    run_id: str = "joint",
) -> List[Any]:
    """Write an adopted bundle to the archive as one ``meta-rule`` per slot.

    Only slots whose value actually **changed** and were **not isolated** are
    written -- a rolled-back slot would otherwise inject its *seed* text with a
    negative contribution, and an unchanged slot an "adopt" with gain 0. Each
    adopted slot's ``final_reward`` is its leave-one-out contribution (its part
    of the bundle's held-out gain), so ``_archived_rules`` ranks a load-bearing
    slot on the synergy it enabled rather than on a lone marginal of 0. The
    record's ``problem`` descriptor carries the bundle id and sibling slots
    (top-level keys ``problem_from_archive`` ignores), so the coupling is not
    lost; ``seed_value`` is the bundle's real per-slot seed.
    """
    blocks = _parse_blocks(report.value, list(slots))
    seed_blocks = _parse_blocks(report.seed_value, list(slots))
    adopted = set(slots) - set(report.isolated)
    siblings = [s for s in slots if s not in report.isolated]
    records = []
    for slot in slots:
        if slot not in adopted:
            continue
        spec = specs[slot]
        text = blocks.get(slot)
        seed_text = seed_blocks.get(slot) or spec.render(spec.initial())
        if text is None or text == seed_text:
            continue                       # unchanged: nothing was adopted here
        gain = float(report.slot_gains.get(slot, 0.0))
        descriptor = problem_descriptor(slot)
        descriptor["joint_run"] = run_id
        descriptor["joint_slots"] = list(siblings)
        state = _rendered_state(spec, text)
        result_i = EvolutionResult(
            state=state, rendered=text, final_reward=gain,
            history=[], ledger_log=[])
        meta_report = MetaRunReport(
            slot=slot, seed_value=seed_text, value=text,
            validation={}, mean_gain=gain, transfer=None, verdict="adopt",
            reason=f"joint adoption with contribution {gain:+.3f}")
        records.append(archive_rule(
            store, meta_report, result_i, run_id=f"{run_id}:{slot}", slot=slot,
            spec=spec, config=config, problem=descriptor,
            artifact_id=artifact_id))
    return records


def _rendered_state(spec: SlotSpec, text: str) -> Dict[str, str]:
    """The strategy state a rendered slot value means.

    ``SingleSlot``/``SourceSlot`` hold one value under ``spec.key``; a
    ``ParamSlot`` holds ``{name: value}`` and can parse its own text. The
    archived state must be the shape a later ``initial_state=`` seeding expects,
    so this is derived from the spec rather than assumed.
    """
    parse = getattr(spec, "_parse", None)
    if parse is not None:
        try:
            return parse(text)
        except Exception:  # noqa: BLE001 - fall back to the single-slot shape
            pass
    return {getattr(spec, "key", "value"): text}
