"""The ``agentdescent meta-campaign`` surface: continuous meta-learning as a loop.

`meta_campaign.meta_campaign_run` is the bounded loop over
`continuous_meta_run`: each session evolves a decision-slot value, validates it
on unseen problems and archives the adopted rule, while `RSIController` decides
when the campaign should stop. This module makes that callable from a JSON spec,
the CLI and the MCP server.

A spec points at the inner domain by ``module:attribute``:

    {
      "archive": "~/.agentdescent/archive",
      "slot": "task_sampler",
      "spec": "mypkg.meta:slot_spec",
      "tasks_for": "mypkg.meta:tasks_for",
      "run": "mypkg.meta:run",
      "reward": "mypkg.meta:reward",
      "inner_propose": "mypkg.meta:inner_propose",
      "inner_strategy": "mypkg.meta:inner_strategy",
      "propose": "mypkg.meta:propose",
      "problems": "mypkg.meta:problems",
      "val_problems": "mypkg.meta:val_problems",
      "campaign": {"max_sessions": 3, "patience": 2},
      "inner": {"rounds": 5, "n_workers": 1},
      "evolve": {"rounds": 1, "n_workers": 1}
    }

`agentdescent meta-campaign init --demo --out meta.json` writes the offline
example from `agentdescent.meta_demo`. `plan` validates refs and configs without
calling builders or creating the archive; `run` returns the campaign report.
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .meta import SlotSpec
from .meta_campaign import MetaCampaignConfig, meta_campaign_run
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, load_json,
    parse_csv, require_callable, require_instance, require_mapping,
    require_text, resolve_builder, resolve_ref,
)

__all__ = [
    "meta_campaign_init",
    "meta_campaign_plan",
    "meta_campaign_run_surface",
]


def _label(value: Any) -> str:
    return (os.path.abspath(os.path.expanduser(value))
            if isinstance(value, str) else "<inline>")


def _config(value: Any) -> MetaCampaignConfig:
    if value is None:
        return MetaCampaignConfig()
    data = require_mapping(value, "campaign")
    allowed = {field.name for field in fields(MetaCampaignConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SurfaceError(
            "unknown campaign key(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(sorted(allowed)))
    try:
        return MetaCampaignConfig(**data)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid campaign config: {error}") from None


def _int_value(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise SurfaceError(f"{name} must be an int")
    return value


def _float_value(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SurfaceError(f"{name} must be a number")
    return float(value)


def _int_list(value: Any, name: str) -> Tuple[int, ...]:
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in parse_csv(value, name=name))
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"{name} must be integers: {error}") from None


def _callbacks(spec: Mapping[str, Any], *,
               allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    callbacks = {}
    for name in ("tasks_for", "run", "reward", "inner_propose", "propose"):
        value = resolve_ref(spec.get(name), name=name, allow=allow)
        callbacks[name] = require_callable(value, name)
    for name in ("meta_reward", "cost_fn"):
        value = spec.get(name)
        if value is None:
            callbacks[name] = None
        else:
            callbacks[name] = require_callable(
                resolve_ref(value, name=name, allow=allow), name)
    return callbacks


def _validate(spec: Mapping[str, Any], *,
              allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    archive = require_text(spec.get("archive"), "archive")
    slot = require_text(spec.get("slot"), "slot")
    config = _config(spec.get("campaign"))
    spec_value = resolve_ref(spec.get("spec"), name="spec", allow=allow)
    if not callable(spec_value) and not isinstance(spec_value, SlotSpec):
        raise SurfaceError(
            f"spec must be a builder or SlotSpec, got {type(spec_value).__name__}")
    strategy = resolve_ref(
        spec.get("inner_strategy"), name="inner_strategy", allow=allow)
    if strategy is None:
        raise SurfaceError("inner_strategy must be a builder or object")
    problems = spec.get("problems")
    val_problems = spec.get("val_problems")
    for name, value in (("problems", problems), ("val_problems", val_problems)):
        if value is None:
            continue
        resolved = resolve_ref(value, name=name, allow=allow)
        if not callable(resolved) and not isinstance(resolved, Mapping):
            raise SurfaceError(
                f"{name} must be a builder or mapping, got "
                f"{type(resolved).__name__}")
    callbacks = _callbacks(spec, allow=allow)
    inner = require_mapping(spec.get("inner") or {}, "inner")
    evolve = require_mapping(spec.get("evolve") or {}, "evolve")
    return {
        "archive": os.path.abspath(os.path.expanduser(archive)),
        "archive_exists": os.path.isdir(os.path.expanduser(archive)),
        "slot": slot,
        "config": config,
        "inner": inner,
        "evolve": evolve,
        "seeds": _int_list(spec.get("seeds"), "seeds") or (0,),
        "val_seeds": _int_list(spec.get("val_seeds"), "val_seeds") or (1, 2),
        "callbacks": callbacks,
        "spec_ref": spec.get("spec"),
        "strategy_ref": spec.get("inner_strategy"),
        "problems_ref": problems,
        "val_problems_ref": val_problems,
        "meta_reward_ref": spec.get("meta_reward"),
        "cost_fn_ref": spec.get("cost_fn"),
    }


def _build(spec: Mapping[str, Any], *,
           allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    checked = _validate(spec, allow=allow)
    checked["spec_object"] = require_instance(
        resolve_builder(spec.get("spec"), name="spec", allow=allow),
        SlotSpec, "spec")
    checked["inner_strategy"] = resolve_builder(
        spec.get("inner_strategy"), name="inner_strategy", allow=allow)
    if checked["inner_strategy"] is None:
        raise SurfaceError("inner_strategy builder returned None")
    problems = spec.get("problems")
    if problems is None:
        checked["problems"] = None
    else:
        built = resolve_builder(problems, name="problems", allow=allow)
        checked["problems"] = require_mapping(built, "problems")
    val_problems = spec.get("val_problems")
    if val_problems is None:
        checked["val_problems"] = None
    else:
        built = resolve_builder(val_problems, name="val_problems", allow=allow)
        checked["val_problems"] = require_mapping(built, "val_problems")
    checked["artifact_id"] = str(spec.get("artifact_id") or "meta-rule")
    checked["train_runs"] = _int_value(
        spec.get("train_runs"), "train_runs", 2)
    checked["val_runs"] = _int_value(spec.get("val_runs"), "val_runs", 1)
    checked["min_gain"] = _float_value(
        spec.get("min_gain"), "min_gain", 0.0)
    checked["transfer_floor"] = _float_value(
        spec.get("transfer_floor"), "transfer_floor", 0.0)
    checked["run_id"] = spec.get("run_id")
    repo_dir = spec.get("repo_dir")
    checked["repo_dir"] = (os.path.abspath(os.path.expanduser(repo_dir))
                           if repo_dir is not None else None)
    return checked


def load_spec(value: Any) -> Dict[str, Any]:
    payload = load_json(value, name="meta-campaign spec")
    spec = require_mapping(payload, "meta-campaign spec")
    for key in ("archive", "slot", "spec", "tasks_for", "run", "reward",
                "inner_propose", "inner_strategy", "propose"):
        if key not in spec:
            raise SurfaceError(f"meta-campaign spec needs {key!r}")
    return spec


def meta_campaign_plan(path: Any, *, allow: Optional[Sequence[str]] = None
                       ) -> Dict[str, Any]:
    spec = load_spec(path)
    checked = _validate(spec, allow=allow)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "archive": checked["archive"],
        "archive_exists": checked["archive_exists"],
        "slot": checked["slot"],
        "campaign": asdict(checked["config"]),
        "refs": {
            "slot_spec": checked["spec_ref"],
            "inner_strategy": checked["strategy_ref"],
            "problems": checked["problems_ref"],
            "val_problems": checked["val_problems_ref"],
            "meta_reward": checked["meta_reward_ref"],
            "cost_fn": checked["cost_fn_ref"],
        },
        "seeds": list(checked["seeds"]),
        "val_seeds": list(checked["val_seeds"]),
        "next": ["agentdescent meta-campaign run --spec <spec.json>"],
    }


def meta_campaign_run_surface(path: Any, *, out: Optional[str] = None,
                              allow: Optional[Sequence[str]] = None
                              ) -> Dict[str, Any]:
    spec = load_spec(path)
    built = _build(spec, allow=allow)
    store = ConsolidationStore(built["archive"])
    callbacks = built["callbacks"]
    report = meta_campaign_run(
        store, built["slot"], built["spec_object"],
        config=built["config"],
        run_id=built["run_id"],
        repo_dir=built["repo_dir"],
        cost_fn=callbacks["cost_fn"],
        tasks_for=callbacks["tasks_for"], run=callbacks["run"],
        reward=callbacks["reward"],
        inner_propose=callbacks["inner_propose"],
        inner_strategy=built["inner_strategy"],
        propose=callbacks["propose"],
        meta_reward=callbacks["meta_reward"],
        problems=built["problems"],
        val_problems=built["val_problems"],
        seeds=built["seeds"], val_seeds=built["val_seeds"],
        artifact_id=built["artifact_id"],
        train_runs=built["train_runs"], val_runs=built["val_runs"],
        inner=built["inner"], min_gain=built["min_gain"],
        transfer_floor=built["transfer_floor"],
        **built["evolve"])
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "archive": built["archive"],
        "report": report.to_dict(),
        "next": (
            ["inspect sessions[*].verdict and archived_run_id"]
            if report.complete else
            ["inspect control.decisions and detail for the failed session"]),
    }
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def meta_campaign_init(out: Any, *, demo: bool = False) -> Dict[str, Any]:
    target = os.path.abspath(os.path.expanduser(require_text(out, "out")))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(target))[0]
    if demo:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "archive": os.path.join(directory, f"{stem}-archive"),
            "repo_dir": os.path.join(directory, f"{stem}-repos"),
            "slot": "task_sampler",
            "spec": "agentdescent.meta_demo:slot_spec",
            "tasks_for": "agentdescent.meta_demo:tasks_for",
            "run": "agentdescent.meta_demo:run",
            "reward": "agentdescent.meta_demo:reward",
            "inner_propose": "agentdescent.meta_demo:inner_propose",
            "inner_strategy": "agentdescent.meta_demo:inner_strategy",
            "propose": "agentdescent.meta_demo:propose",
            "problems": "agentdescent.meta_demo:problems",
            "val_problems": "agentdescent.meta_demo:val_problems",
            "campaign": {"max_sessions": 2, "patience": 5},
            "inner": {"rounds": 5, "n_workers": 1, "target_reward": 1.0},
            "evolve": {"rounds": 1, "n_workers": 1},
            "seeds": [0, 1, 2, 3],
            "val_seeds": [10, 11],
        }
    else:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "archive": os.path.join(directory, f"{stem}-archive"),
            "repo_dir": os.path.join(directory, f"{stem}-repos"),
            "slot": "task_sampler",
            "spec": "your_package.meta:slot_spec",
            "tasks_for": "your_package.meta:tasks_for",
            "run": "your_package.meta:run",
            "reward": "your_package.meta:reward",
            "inner_propose": "your_package.meta:inner_propose",
            "inner_strategy": "your_package.meta:inner_strategy",
            "propose": "your_package.meta:propose",
            "problems": "your_package.meta:problems",
            "val_problems": "your_package.meta:val_problems",
            "campaign": {"max_sessions": 2, "patience": 5},
            "inner": {"rounds": 5, "n_workers": 1},
            "evolve": {"rounds": 1, "n_workers": 1},
        }
    atomic_write_json(target, spec)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": target,
        "demo": demo,
        "next": [
            f"agentdescent meta-campaign plan --spec {target}",
            f"agentdescent meta-campaign run  --spec {target} --json",
        ],
    }
