"""The algorithms shipped with the package. Importing this registers them."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from . import Algorithm, AlgorithmError, BuildContext, Overrides, Param, register
from .rrsi.rrsi_policies import RRsiConfig, build_rrsi_bundle

#: Kinds whose strategy is a directory tree or a code/plugin tree.
_TREE_KINDS = ("skill_dir", "agent_dir", "agent_code", "plugin")

_RRSI_HELP = {
    "beta0": "large-gain cost rule: tolerance",
    "beta1": "large-gain cost rule: slope",
    "w_s": "within-band score weight",
    "w_c": "within-band cost weight",
    "w_n": "within-band novelty weight",
    "noise_band": "historical-best floor tolerance (score >= S* - noise_band); the "
                  "paper estimates it by re-evaluating the unchanged base",
    "b_min": "smallest edit budget (independent edits per candidate)",
    "b_max": "largest edit budget, at the start of the anneal",
    "T": "rounds the edit budget anneals over",
    "stall_window": "rounds without movement of S* that count as a stall",
    "prune_window": "rounds of component yield a component must show gain over to stay",
    "max_history_chars": "how much per-edit history fits in the reflection prompt",
}


def _rrsi_params():
    out = []
    for f in dataclasses.fields(RRsiConfig):
        out.append(Param(f.name, "int" if isinstance(f.default, int)
                         and not isinstance(f.default, bool) else "float", f.default,
                         _RRSI_HELP.get(f.name, ""),
                         minimum=1 if f.name in ("b_min", "b_max") else 0,
                         # selection rejects a noise delta outside [0, 1]
                         maximum=1 if f.name == "noise_band" else None))
    out.append(Param("protected", "list[str]", [],
                     "strings that must not leak into the artifact: task ids, gold "
                     "answers, names from the evolve set (never the final holdout). "
                     "A candidate containing one is rejected before it is measured"))
    out.append(Param("base_score", "float", 0.0,
                     "the base artifact's evolve-set score, to seed the historical-best "
                     "floor S* (0 keeps the floor at the first measurement)"))
    return tuple(out)


_CONFIG_FIELDS = {f.name for f in dataclasses.fields(RRsiConfig)}


def _build_rrsi(ctx: BuildContext, p: Dict[str, Any]) -> Overrides:
    if ctx.reflect is None:
        raise AlgorithmError("algorithm: rrsi proposes with the reflection model, and "
                             "the spec resolved none")
    try:
        config = RRsiConfig(**{k: v for k, v in p.items() if k in _CONFIG_FIELDS})
    except (TypeError, ValueError) as e:
        raise AlgorithmError(f"algorithm.params: {e}") from None
    if config.b_min > config.b_max:
        raise AlgorithmError(
            f"algorithm.params: b_min ({config.b_min}) must not exceed b_max ({config.b_max})")
    bundle, _state = build_rrsi_bundle(
        ctx.reflect, protected=tuple(p["protected"]), config=config,
        base_score=p["base_score"])
    notes = []
    if not p["protected"]:
        notes.append(
            "algorithm rrsi: `protected` is empty, so the leakage screen has nothing to "
            "check and will never reject a candidate. Put the evolve set's distinctive "
            "strings (gold answers, task ids) there -- never the final holdout -- or the "
            "committed artifact can be a list of answers.")
    return Overrides(policies={"screen": bundle.screen, "proposal": bundle.proposal,
                               "acceptance": bundle.acceptance}, notes=notes)


register(Algorithm(
    name="rrsi",
    summary="Regularized recursive self-improvement: an annealed edit budget, a "
            "leakage screen before measurement, and a historical-best acceptance floor.",
    kinds=("text",),
    fidelity="mechanism_microport",
    params=_rrsi_params(),
    build=_build_rrsi,
    source="Google RRSI (regularized harness selection); docs/algo-rrsi.md",
    extra_calls="none (the reflection model proposes as it would anyway; the screen "
                "is deterministic and runs before any held-out sweep)",
    doc="algo-rrsi.md"))


def _build_archive(ctx: BuildContext, p: Dict[str, Any]) -> Overrides:
    from ..selection import Archive

    try:
        archive = Archive(sampling=p["sampling"], temperature=p["temperature"],
                          seed=p["seed"])
    except ValueError as e:
        raise AlgorithmError(f"algorithm.params: {e}") from None
    return Overrides(
        policies={"selection": archive},
        notes=["algorithm archive_selection: only the parent-selection rule changes "
               "(it makes the engine keep a candidate population); proposals, the "
               "merge and the acceptance gate are the engine's."])


register(Algorithm(
    name="archive_selection",
    summary="Parent sampling from an archive of every accepted candidate, tempered by "
            "novelty (DGM's rule by default) instead of always extending the head.",
    kinds=("text",) + _TREE_KINDS,
    fidelity="preset",
    params=(
        Param("sampling", "str", "sigmoid_novelty",
              "sigmoid_novelty: Darwin Gödel Machine's rule, sigmoid(10*(score-0.5)) / "
              "(1+times chosen); novelty: softmax over score divided by (1+times chosen); "
              "performance: softmax over score; best: always the top scorer (SICA); "
              "uniform: the ablation",
              choices=("sigmoid_novelty", "novelty", "performance", "uniform", "best")),
        Param("temperature", "float", 1.0,
              "softmax temperature for performance / novelty; must be > 0",
              minimum=1e-6),
        Param("seed", "int", 0, "seed of the parent draw"),
    ),
    build=_build_archive,
    source="Darwin Gödel Machine (choose_selfimproves), ADAS archive, SICA best-agent",
    extra_calls="none"))


# ACE, GEPA and SkillOpt register themselves; importing them makes the names.
from . import ace  # noqa: E402,F401
from . import gepa  # noqa: E402,F401
from . import skillopt  # noqa: E402,F401
from . import dgm  # noqa: E402,F401
