"""Portfolio evolution: the framework evolves the rule that governs its spend.

``portfolio_report`` is a fixed, honest reallocation policy -- grow the layers
that pay, shrink the ones that do not, hold the ones with too little evidence.
The framework's own thesis is recursion: the rule that decides how the framework
spends should itself be improved by the framework. This module makes the policy
**evolvable**.

The reallocation rule is four numbers -- ``floor``, ``grow``, ``shrink``,
``min_samples`` -- compiled into a :class:`ReallocationPolicy` by the same
:class:`~agentdescent.meta.ParamSlot` machinery the meta stack uses. Crucially,
``ReallocationPolicy.apply`` **delegates to the shipped ``portfolio_report``**,
so the evolved policy is the *same* rule with different numbers: the baseline is
``ReallocationPolicy()`` (the defaults) and there is no second implementation to
drift from the one the framework actually runs.

A *problem* is one **layer window** -- a set of :class:`LayerBudget` rows, the
unit ``portfolio_report`` already consumes. Its reward is the projected total
return the policy's reallocation would buy on that window (each layer's measured
return scaled by its budget ratio). ``portfolio_evolve`` runs the ordinary
``evolve()`` over those windows -- tasks per ``(window, seed)``, the meta reward
the projected return, ``propose`` rewriting the four numbers -- so the framework
searches its own spending rule on its own measured history, validates the
evolved rule on windows the search never trained on, and reports the gain over
the fixed baseline.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .evolution import EvolutionResult, Task, evolve
from .meta import (
    MetaOutcome, MetaReward, _check_meta_reward, auc, meta_validate,
)
from .portfolio import LayerBudget, layers_from_archive, portfolio_report

__all__ = [
    "ReallocationPolicy",
    "reallocation_spec",
    "portfolio_problem",
    "windows_from_archive",
    "portfolio_evolve",
]


@dataclass(frozen=True)
class ReallocationPolicy:
    """The evolvable reallocation rule: four numbers, applied to a layer window.

    ``apply`` is ``portfolio_report`` with these numbers, not a re-implementation
    -- the evolved rule and the shipped baseline cannot drift apart. The default
    values are exactly ``portfolio_report``'s own defaults, so
    ``ReallocationPolicy()`` *is* the fixed policy the framework ships.
    """

    floor: float = 0.05
    grow: float = 0.5
    shrink: float = 0.5
    min_samples: int = 3

    def apply(self, rows: Sequence[LayerBudget]) -> Dict[str, float]:
        """``layer name -> new budget`` over a window, by ``portfolio_report``."""
        return portfolio_report(
            rows, floor=self.floor, min_samples=int(self.min_samples),
            grow=self.grow, shrink=self.shrink).reallocated


def reallocation_spec() -> Any:
    """The evolvable spec: the four numbers as a :class:`~agentdescent.meta.ParamSlot`.

    The factory is :class:`ReallocationPolicy`, so ``spec.compile(rendered)`` is
    the callable rule the problems run; the default rendering is the shipped
    fixed policy, which is also the baseline ``portfolio_evolve`` measures
    against.
    """
    from .meta import ParamSlot
    return ParamSlot(
        factory=lambda floor, grow, shrink, min_samples: ReallocationPolicy(
            floor=float(floor), grow=float(grow), shrink=float(shrink),
            min_samples=int(min_samples)),
        params={"floor": 0.05, "grow": 0.5, "shrink": 0.5, "min_samples": 3.0},
        bounds={"floor": (0.0, 1.0), "grow": (0.0, 5.0), "shrink": (0.0, 1.0),
                "min_samples": (1.0, 10.0)},
        title="# Portfolio reallocation parameters")


def _projected_return(policy: ReallocationPolicy,
                      rows: Sequence[LayerBudget]) -> float:
    """The mean return the policy's reallocation would buy on a window.

    Each layer's measured return is scaled by its budget ratio under the policy;
    the mean over the window is the number. It is deliberately **not clipped**:
    ``LayerBudget.return_`` is any finite number and ``grow`` may exceed 1, so
    clipping let every aggressive policy hit the same ceiling and look equal.
    The default ``auc`` reward still normalises to ``[0, 1]``; a custom
    ``meta_reward`` sees the raw projection. The identity policy (no change)
    scores the plain mean return, so a policy that grows the paying layers and
    shrinks the rest scores above it.
    """
    if not rows:
        return 0.0
    new = policy.apply(rows)
    total = 0.0
    for row in rows:
        old = max(float(row.budget), 1e-9)
        ratio = float(new.get(row.name, row.budget)) / old
        total += float(row.return_) * ratio
    return total / len(rows)


def portfolio_problem(rows: Sequence[LayerBudget]) -> Callable:
    """A :class:`~agentdescent.meta.Problem` over one layer window.

    ``(policy, seed) -> MetaOutcome`` where the outcome's final is the projected
    return of applying ``policy`` to the window's rows. The caller's
    ``meta_reward`` scores that outcome (in the search and in
    :func:`~agentdescent.meta.meta_validate`), so the problem itself does not
    take one.
    """

    def problem(policy: Any, seed: int) -> MetaOutcome:
        projected = _projected_return(policy, rows)
        return MetaOutcome(curve=[projected, projected], final=projected,
                           rollouts=len(rows))

    return problem


def _window_from_records(records: Sequence[Any]) -> List[LayerBudget]:
    """One window: aggregate a chunk of rule records into layer rows (per slot).

    The aggregates match :func:`~agentdescent.portfolio.layers_from_archive`:
    return = mean reward, spent = mean rounds per rule, budget = total rounds,
    samples = rule count -- so the window is the portfolio's own view.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        slot = (rec.problem or {}).get("slot")
        if not slot:
            # same skip as layers_from_archive: a record with no slot has no
            # layer, and a "meta:?" row would misalign the two views.
            continue
        row = grouped.setdefault(slot, {"rewards": [], "n": 0, "rounds": 0.0})
        row["n"] += 1
        row["rewards"].append(float(rec.final_reward))
        row["rounds"] += float(rec.rounds)
    out: List[LayerBudget] = []
    for slot, row in grouped.items():
        n = int(row["n"])
        out.append(LayerBudget(
            name=f"meta:{slot}", budget=row["rounds"],
            spent=row["rounds"] / n, return_=statistics.fmean(row["rewards"]),
            samples=n, note=f"{n} rule(s)"))
    return out


def windows_from_archive(
    store: ConsolidationStore,
    *,
    windows: int = 3,
    val_frac: float = 0.4,
    artifact_id: str = "meta-rule",
) -> Tuple[List[List[LayerBudget]], List[List[LayerBudget]]]:
    """Train/validation **layer windows** from the archive's rule records.

    The records (with a problem descriptor) are ordered by ``created_at`` and cut
    into ``windows`` contiguous chunks; each chunk becomes one window of layer
    rows (:func:`_window_from_records`). The windows are then split by position
    into train and validation -- the same positional isolation ``evolve()``
    applies -- so the evolved rule is validated on windows it never trained on.

    An archive with fewer than two windows cannot be split and returns two empty
    lists; :func:`portfolio_evolve` then asks the caller for explicit windows.
    """
    records = [rec for rec in store.iter_archive()
               if rec.artifact_id == artifact_id and rec.problem]
    if len(records) < 2:
        return [], []
    records.sort(key=lambda rec: rec.created_at)
    n = max(1, min(int(windows), len(records)))
    # contiguous chunks; the remainder joins the last chunk
    size, extra = divmod(len(records), n)
    chunks: List[List[Any]] = []
    start = 0
    for i in range(n):
        take = size + (1 if i < extra else 0)
        chunks.append(records[start:start + take])
        start += take
    layer_windows = [_window_from_records(chunk) for chunk in chunks if chunk]
    layer_windows = [window for window in layer_windows if window]
    cut = max(1, int(len(layer_windows) * val_frac))
    if len(layer_windows) < 2:
        return [], []
    return layer_windows[:-cut], layer_windows[-cut:]


def _evolve_policy(
    spec: Any,
    problems: Mapping[str, Callable],
    *,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]],
    model: Optional[Any],
    seeds: Sequence[int],
    meta_reward: Optional[MetaReward] = None,
    initial_state: Optional[Dict[str, str]] = None,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """The ordinary engine over the reallocation parameters.

    The same shape ``meta_evolve`` builds for a slot, without the ``Policies``
    slot constraint: the artifact is the rendered parameter block, ``run``
    installs it as a :class:`ReallocationPolicy`, and the reward is the projected
    return scored by ``meta_reward`` (``auc`` by default). Closing over that
    score -- as ``meta_parts`` does -- is what makes the knob drive the search
    instead of only the post-hoc scorecard.
    """
    from .meta import slot_reflector

    score = meta_reward or auc
    _check_meta_reward(score)
    named = dict(problems)
    # seed-major interleaving, so every window lands on both sides of the
    # positional held-out cut (the meta_evolve lesson)
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for seed in seeds for name in named]
    reflector = propose if propose is not None else slot_reflector(model, spec)
    spec.compile(spec.render(spec.initial()))       # the seed must pass its gate

    def run(rendered: str, task: Task) -> str:
        try:
            policy = spec.compile(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": f"compile: {error}"}).to_json()
        outcome = named[task.meta["problem"]](policy, int(task.meta["seed"]))
        return outcome.to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(score(MetaOutcome.from_json(output)))
        except (TypeError, ValueError, KeyError):
            return 0.0

    return evolve(
        tasks, reward, run=run, propose=reflector, strategy=spec,
        initial_state=initial_state or spec.initial(), self_verify=False,
        solved_threshold=1.1, **evolve_kwargs)


def _mean_gain(report: Mapping[str, Mapping[str, float]]) -> float:
    rows = list(report.values())
    return statistics.fmean(row["gain"] for row in rows) if rows else 0.0


def portfolio_evolve(
    store: Optional[ConsolidationStore] = None,
    *,
    windows: Optional[Sequence[Sequence[LayerBudget]]] = None,
    val_windows: Optional[Sequence[Sequence[LayerBudget]]] = None,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0, 1),
    val_seeds: Sequence[int] = (2, 3),
    baseline: Optional[str] = None,
    **evolve_kwargs: Any,
) -> Tuple[Dict[str, Any], EvolutionResult]:
    """Evolve the reallocation rule on the archive's (or the caller's) windows.

    ``windows``/``val_windows`` are the train and validation layer windows; when
    omitted they come from :func:`windows_from_archive(store)`. The four
    reallocation numbers are the artifact, ``propose`` (or ``model``) rewrites
    them, and the meta reward is the projected return.

    The baseline is the shipped fixed policy (``ReallocationPolicy()``): the
    report carries the evolved policy's projected return on both splits, its
    **gain** over the baseline on the validation windows, and an
    ``adopt``/``reject`` verdict (adopt only when that gain is strictly
    positive, so a refused no-op is not called an adoption). ``meta_reward``
    scores the projected return in the search *and* in the validation.
    ``baseline`` is the incumbent parameter block to search from and validate
    against; it defaults to the shipped fixed policy, so a caller that never
    passes it is unchanged. Returns ``(report, result)`` where
    ``result.rendered`` is the evolved parameter block.
    """
    if propose is None and model is None:
        raise ValueError("portfolio_evolve() needs propose= or model=")
    if windows is None or val_windows is None:
        if store is None:
            raise ValueError(
                "portfolio_evolve() needs a store= (or explicit windows= and "
                "val_windows=)")
        auto_train, auto_val = windows_from_archive(store)
        windows = windows if windows is not None else auto_train
        val_windows = val_windows if val_windows is not None else auto_val
    if not windows or not val_windows:
        raise ValueError(
            "portfolio_evolve() needs both train and validation windows; run "
            "continuous_meta_run (or a meta session) first, or pass windows=")

    spec = reallocation_spec()
    train = {f"w{i}": portfolio_problem(list(w))
             for i, w in enumerate(windows)}
    val = {f"v{i}": portfolio_problem(list(w))
           for i, w in enumerate(val_windows)}

    seed_value = (baseline if baseline is not None
                  else spec.render(spec.initial()))
    seed_state = spec.initial()
    if baseline is not None:
        try:
            seed_state.update(spec._parse(baseline))
        except Exception:  # noqa: BLE001 - an unparseable baseline fails in validate
            pass
    result = _evolve_policy(spec, train, propose=propose, model=model,
                            seeds=seeds, meta_reward=meta_reward,
                            initial_state=seed_state, **evolve_kwargs)
    value = result.rendered or seed_value
    train_validation = meta_validate(spec, seed_value, value, train,
                                     seeds=seeds, meta_reward=meta_reward)
    val_validation = meta_validate(spec, seed_value, value, val,
                                   seeds=val_seeds, meta_reward=meta_reward)
    train_gain = _mean_gain(train_validation)
    val_gain = _mean_gain(val_validation)
    verdict = "adopt" if val_gain > 0.0 else "reject"
    reason = (f"evolved reallocation projects {val_gain:+.3f} over the fixed "
              f"baseline on {len(val_validation)} unseen window(s)")
    if verdict == "reject":
        reason += "; adoption needs a strictly positive gain"
    report = {
        "seed_value": seed_value, "value": value,
        "train": train_validation, "val": val_validation,
        "train_gain": train_gain, "val_gain": val_gain,
        "verdict": verdict, "reason": reason,
    }
    try:
        result.meta = report
    except AttributeError:
        pass
    return report, result
