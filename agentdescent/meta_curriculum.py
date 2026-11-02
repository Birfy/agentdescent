"""Meta-curriculum: the meta loop generates the problems it is judged on.

``continuous_meta_run`` trains on the archive's problems and validates on the
caller's; the task distribution is the least-examined part of that design, the
same gap ``curriculum_evolve`` (#329) closed one level down. There, a
challenger generated frontier *tasks* for the solver; here it generates frontier
*problems* for the meta loop -- the decision rule's own solver-challenger:

* a generated problem is a runnable inner :class:`~agentdescent.meta.Problem`
  the challenger draws from a *problem-generation spec*, validated by a
  mandatory gate before it ever reaches the sampler;
* learnability is two-phase and measured online, exactly as in ``selfplay``:
  a generated problem the rule already solves (first score at the ceiling) is
  dropped at once, one it never learns is dropped as stuck, one it learns is
  counted -- so the frontier keeps turning toward the rule's current edge;
* the challenger **never touches the final evaluation**: the validation set is
  fixed from ``val_problems`` and the held-out split is computed from the seed
  tasks alone, so what the run reports is always measured on problems the
  challenger never generated;
* the archive still drives continuity: the search starts from the best archived
  rule for the slot/spec/config (in a fresh ledger repo), and on ``adopt`` the
  rule is written back with its problem descriptor, so the next session starts
  where this one was validated with a problem distribution the meta loop itself
  has been pushing outward.

The sampler is ``selfplay.FrontierSampler`` applied one level up: the frontier
holds ``(problem, name)`` pairs instead of tasks with gold answers, and
``record(task_id, score)`` receives the *meta reward* of the outcome, which is
the learnability signal here.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .evolution import EvolutionResult, Task, evolve
from .meta import (
    MetaOutcome, MetaReward, MetaRunReport, Problem, SlotSpec,
    _mean_gain, _require_disjoint, _transfer_from, archive_rule, auc,
    meta_validate, problem_descriptor,
)
from .meta_audit import _digest, audited_meta_validate
from .selfplay import FrontierSampler
from .verified_seed import verified_rules

__all__ = [
    "MetaChallengerConfig",
    "MetaProblemChallenger",
    "curriculum_meta_run",
]


@dataclass(frozen=True)
class MetaChallengerConfig:
    """The problem-challenger's dials: how much of the problem budget goes to
    the frontier, what counts as learned or stuck, and how long a generator may
    keep trying. The same numbers ``curriculum_evolve`` uses, one level up."""

    #: How many valid generated problems each challenger step installs.
    generated_per_round: int = 4
    #: Fraction of a worker's picks that come from the frontier (the rest are
    #: seed problems, so the meta loop never forgets the fixed curriculum).
    frontier_share: float = 0.5
    #: A problem whose first meta-reward is already at or above ``1 - before_gap``
    #: is not on the frontier; it is dropped on its first attempt.
    before_gap: float = 0.05
    #: A problem whose ``last - first`` meta-reward never exceeds this is stuck
    #: and dropped. The challenger is told the stuck count, so it learns not to
    #: generate problems its own meta loop cannot learn.
    stuck_gap: float = 0.0
    #: How many attempts a problem may have before it is judged.
    max_attempts: int = 8
    #: The challenger commits to a new problem-generation spec only when the
    #: measured learnability report beats this.
    r0_floor: float = 0.05
    #: Valid generation attempts per challenger step before giving up.
    max_generate_attempts: int = 20
    #: The frontier pool is capped; the oldest learned/stuck problems leave first.
    max_frontier_pool: int = 64


@dataclass
class MetaProblemChallenger:
    """The problem-generation artifact for the meta loop.

    ``generate(descriptor, seed) -> config`` draws a candidate problem
    configuration; ``validate(config) -> Problem`` is the **mandatory validity
    gate** -- it must raise on anything that does not build a runnable inner
    :class:`~agentdescent.meta.Problem`, and the returned problem is what the
    sampler installs. ``propose(descriptor, report) -> descriptor`` turns the
    measured frontier report (``r0``, ``learned``, ``stuck``) into the next
    generation spec. All three are plain callables, so an offline fixture can be
    deterministic and a live run can use a model.
    """

    generate: Callable[[str, int], Dict[str, Any]]
    validate: Callable[[Dict[str, Any]], Problem]
    propose: Callable[[str, Dict[str, float]], str]
    #: The current problem-generation spec. Starts here; every challenger step
    #: may replace it.
    spec: str = ""
    config: MetaChallengerConfig = field(default_factory=MetaChallengerConfig)


class _MetaCurriculumDriver:
    """The challenger step between meta rounds: measure, propose, regenerate."""

    def __init__(self, challenger: MetaProblemChallenger,
                 sampler: FrontierSampler, problems: Dict[str, Problem],
                 *, solved_threshold: float, rng: random.Random) -> None:
        self.challenger = challenger
        self.sampler = sampler
        self.problems = problems
        self._solved_threshold = solved_threshold
        self._rng = rng
        self._rounds_closed = 0
        self._reports: List[Dict[str, float]] = []
        self._next_index = 0
        #: names of the generated problems in ``problems`` (the seed problems
        #: are never pruned)
        self._generated: List[str] = []

    def refresh(self) -> None:
        """Generate, validate and install the next frontier batch of problems."""
        cfg = self.challenger.config
        batch: List[Task] = []
        for _ in range(cfg.max_generate_attempts):
            if len(batch) >= cfg.generated_per_round:
                break
            try:
                payload = self.challenger.generate(
                    self.challenger.spec, self._rng.getrandbits(32))
                problem = self.challenger.validate(payload)
            except (ValueError, TypeError, KeyError):
                self.sampler.count_invalid()
                continue
            name = f"frontier-{self._next_index}"
            self._next_index += 1
            self.problems[name] = problem
            self._generated.append(name)
            task = Task(id=name, prompt=f"frontier problem {name}",
                        meta={"problem": name, "seed": self._rng.getrandbits(32),
                              "generated": True, "spec": self.challenger.spec})
            batch.append(task)
        if batch:
            self.sampler.install(batch)
        self._prune()

    def _prune(self) -> None:
        """Forget generated problems the sampler no longer holds.

        The sampler drops learned/stuck problems and caps its pool at
        ``max_frontier_pool``; without this the name -> Problem map kept every
        problem ever generated, growing by ``generated_per_round`` per round
        for the life of the run. Called between rounds only, so no rollout is
        in flight on a pruned name.
        """
        live = {t.id for t in self.sampler._frontier}
        keep = []
        for name in self._generated:
            if name in live:
                keep.append(name)
            else:
                self.problems.pop(name, None)
        self._generated = keep

    def after_round(self, info: Any) -> None:
        """Close the meta round: judge the frontier, feed the challenger."""
        self._rounds_closed += 1
        report = self.sampler.sweep(self._solved_threshold)
        self._reports.append(report)
        try:
            self.challenger.spec = self.challenger.propose(
                self.challenger.spec, report)
        except Exception:  # noqa: BLE001 - a broken proposer must not fail the run
            self.sampler._stats["invalid"] += 1
        self.refresh()

    def final_stats(self) -> Dict[str, Any]:
        return {
            **self.sampler.stats(),
            "r0_mean": (sum(r["r0"] for r in self._reports) / len(self._reports)
                        if self._reports else 0.0),
            "rounds_closed": self._rounds_closed,
        }


def curriculum_meta_run(
    store: ConsolidationStore,
    slot: str,
    spec: SlotSpec,
    challenger: MetaProblemChallenger,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    *,
    problems: Mapping[str, Problem],
    val_problems: Mapping[str, Problem],
    config: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0, 1),
    val_seeds: Sequence[int] = (10, 11),
    min_gain: float = 0.0,
    transfer_floor: float = 0.0,
    solved_threshold: float = 0.999,
    run_id: Optional[str] = None,
    repo_dir: Optional[str] = None,
    min_exam: Optional[str] = None,
    meta_audit: Optional[Any] = None,
    bias_threshold: float = 0.1,
    oracle: Optional[MetaReward] = None,
    audit_store: Optional[Any] = None,
    audit_rate: float = 0.1,
    audit_seed: int = 0,
    **evolve_kwargs: Any,
) -> Tuple[Dict[str, Any], EvolutionResult]:
    """Evolve a decision slot against a problem distribution the meta loop generates.

    The solver is ``meta_evolve``'s: the slot value compiles through ``spec``,
    a rollout is one inner problem under that value scored by ``meta_reward``.
    The challenger is the task distribution's own solver-challenger: between
    outer rounds it generates, validates and installs frontier *problems* into
    the solver's sampler, and the sampler mixes them with the seed problems
    (``problems``) so the meta loop's problem distribution evolves as the rule
    does. The search starts from the best archived rule for the slot/spec/config
    (in a fresh ledger repo under ``repo_dir``), the validation set is fixed at
    ``val_problems`` and never mixed in, and on ``adopt`` the rule is written
    back with its problem descriptor.

    Parameters
    ----------
    store:
        The :class:`~agentdescent.consolidation.ConsolidationStore` for
        continuity: seeds the search from its best rule and receives the adopted
        rule back.
    slot, spec, propose, model:
        As :func:`~agentdescent.meta.meta_run`.
    challenger:
        The problem-generation artifact (:class:`MetaProblemChallenger`).
    problems, val_problems:
        The seed problems (the fixed curriculum the frontier is mixed into) and
        the validation problems (never touched by the challenger).
    config:
        This session's configuration, written back with an adopted rule.
    meta_reward, seeds, val_seeds, min_gain, transfer_floor:
        As :func:`~agentdescent.meta.meta_run`.
    solved_threshold:
        What a generated problem must reach to count as learned by the rule.
    run_id, repo_dir:
        The archived ``run_id`` and the base directory for the session's fresh
        ledger repo (a shared repo would resume and ignore the seed). A
        ``meta-cur-<slot>-<unix time>-<random>`` id is generated when omitted.
    min_exam, meta_audit, bias_threshold:
        The seeding gate applied to archived rules via
        :func:`~agentdescent.verified_seed.verified_rules`. ``min_exam=None``
        (the default) reproduces the old ``_archived_rules`` behaviour exactly;
        ``"flagged"`` / ``"passed"`` require that exam status or better, and a
        supplied ``meta_audit`` store additionally excludes rules whose gain
        bias exceeds ``bias_threshold``.
    oracle, audit_store, audit_rate, audit_seed:
        When both ``oracle`` and ``audit_store`` are supplied the validation
        table is :func:`~agentdescent.meta_audit.audited_meta_validate`: a
        probability sample of ``(problem, seed)`` pairs is scored by ``oracle``
        and persisted, and the verdict is written back with
        ``audit_store.record_verdict``. The verdict itself still rides the
        verifier, never the oracle. Supplying only one of the two raises.
    **evolve_kwargs:
        The outer :func:`evolve` call's settings (``rounds``, ``n_workers``,
        ``held_out_frac`` ...), except ``task_sampler`` / ``on_round`` /
        ``run`` / ``reward`` / ``propose`` / ``strategy`` / ``repo_path`` /
        ``initial_state``.

    Returns ``(report, result)`` where ``report`` is a dict carrying the
    decision fields (``slot``, ``value``, ``seed_value``, ``validation``,
    ``mean_gain``, ``transfer``, ``verdict``, ``reason``, ``run_id``) plus the
    frontier statistics (``generated_total``, ``learned``, ``stuck``,
    ``r0_mean``, ``frontier_n``); ``result`` is the solver's
    :class:`~agentdescent.evolution.EvolutionResult` with ``result.meta`` set to
    the same report.
    """
    for taken in ("strategy", "run", "reward", "agent", "initial_state",
                  "task_sampler", "on_round", "repo_path"):
        if taken in evolve_kwargs:
            raise TypeError(f"curriculum_meta_run() sets {taken}= itself")
    if not problems or not val_problems:
        raise ValueError("curriculum_meta_run() needs problems and val_problems")
    if propose is None and model is None:
        raise ValueError("curriculum_meta_run() needs propose= or model=")
    if (oracle is None) != (audit_store is None):
        raise ValueError(
            "curriculum_meta_run() needs both oracle= and audit_store= to "
            "audit the meta gain, or neither")
    _require_disjoint(problems, val_problems, seeds, val_seeds)
    score = meta_reward or auc
    if repo_dir is not None:
        os.makedirs(repo_dir, exist_ok=True)
    rid = run_id or f"meta-cur-{slot}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    rules = list(verified_rules(
        store, slot=slot, spec=spec, config=config, artifact_id="meta-rule",
        min_exam=min_exam, meta_audit=meta_audit,
        bias_threshold=bias_threshold).rules)
    prior = rules[0] if rules else None
    prior_state = dict(prior.state) if prior is not None else None
    prior_value = (spec.render(prior_state) if prior_state
                   else spec.render(spec.initial()))

    named = dict(problems)
    seed_tasks = _seed_tasks(named, seeds)
    sampler = FrontierSampler(challenger, seed=seeds[0] if seeds else 0)
    driver = _MetaCurriculumDriver(
        challenger, sampler, named, solved_threshold=solved_threshold,
        rng=random.Random((seeds[0] if seeds else 0) + 1))

    def run_fn(rendered: str, task: Task) -> str:
        try:
            value = spec.compile(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": f"compile: {error}"}).to_json()
        problem = named[task.meta["problem"]]
        outcome = problem(value, int(task.meta["seed"]))
        return outcome.to_json()

    def reward_fn(task: Task, output: str) -> float:
        try:
            return float(score(MetaOutcome.from_json(output)))
        except (TypeError, ValueError, KeyError):
            return 0.0

    def on_round(info: Any) -> None:
        driver.after_round(info)

    driver.refresh()  # seed the frontier before the first round
    result = evolve(
        list(seed_tasks), reward_fn, run=run_fn, propose=propose,
        strategy=spec, task_sampler=sampler, on_round=on_round,
        initial_state=prior_state if prior_state is not None
        else spec.initial(),
        repo_path=tempfile.mkdtemp(prefix="agentdescent-meta-cur-", dir=repo_dir),
        **evolve_kwargs)
    value = result.rendered or prior_value

    train_validation = meta_validate(
        spec, prior_value, value, problems, seeds=seeds, meta_reward=score)
    if audit_store is not None:
        validation = audited_meta_validate(
            spec, prior_value, value, val_problems, seeds=val_seeds,
            meta_reward=score, oracle=oracle, store=audit_store, slot=slot,
            run_id=rid, sample_rate=audit_rate, seed=audit_seed)
    else:
        validation = meta_validate(
            spec, prior_value, value, val_problems, seeds=val_seeds,
            meta_reward=score)
    mean_gain = _mean_gain(validation)
    train_gain = _mean_gain(train_validation)
    transfer = _transfer_from(train_gain, mean_gain)
    if mean_gain <= min_gain:
        verdict, reason = "reject", (
            f"evolved value gained {mean_gain:+.3f} on unseen problems, "
            f"not above min_gain {min_gain:+.3f}")
    elif transfer is not None and transfer < transfer_floor:
        verdict, reason = "reject", (
            f"transfer ratio {transfer:.3f} is below transfer_floor "
            f"{transfer_floor:.3f} (train gain {train_gain:+.3f}, "
            f"val gain {mean_gain:+.3f})")
    else:
        verdict, reason = "adopt", (
            f"mean gain {mean_gain:+.3f} on unseen problems, "
            f"transfer {transfer if transfer is None else round(transfer, 3)}")

    audit_fields: Dict[str, Any] = {}
    if audit_store is not None:
        audited_n = sum(1 for _ in audit_store.iter_records(run_id=rid))
        oracle_gain = audit_store.rectified_gain(run_id=rid)
        gain_bias = audit_store.gain_bias(run_id=rid)
        audit_store.record_verdict(
            run_id=rid, slot=slot, verdict=verdict, reason=reason,
            mean_gain=mean_gain, transfer=transfer, audited_n=audited_n,
            oracle_gain=oracle_gain, value_digest=_digest(value))
        audit_fields = {
            "audited_n": audited_n,
            "oracle_gain": oracle_gain,
            "gain_bias": gain_bias,
            "audit_store": audit_store.path,
        }

    frontier = driver.final_stats()
    if verdict == "adopt":
        report = MetaRunReport(
            slot=slot, seed_value=prior_value, value=value,
            validation=validation, mean_gain=mean_gain, transfer=transfer,
            verdict=verdict, reason=reason)
        archive_rule(store, report, result, run_id=rid, slot=slot, spec=spec,
                     config=config, problem=problem_descriptor(slot),
                     artifact_id="meta-rule")

    report = {
        "slot": slot, "seed_value": prior_value, "value": value,
        "validation": validation, "mean_gain": mean_gain, "transfer": transfer,
        "verdict": verdict, "reason": reason,
        "run_id": rid if verdict == "adopt" else None,
        "seeded_from": prior.run_id if prior is not None else None,
        "frontier": frontier,
        **audit_fields,
    }
    try:
        result.meta = report
    except AttributeError:
        pass
    return report, result


def _seed_tasks(problems: Mapping[str, Problem], seeds: Sequence[int]) -> List[Task]:
    """One outer task per ``(problem, seed)`` pair, seed-major so every problem
    lands on both sides of ``evolve``'s positional held-out split."""
    return [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                 meta={"problem": name, "seed": seed})
            for seed in seeds for name in problems]
