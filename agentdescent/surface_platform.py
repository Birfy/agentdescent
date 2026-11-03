"""The ``agentdescent platform`` surface: a durable, multi-cycle RSI campaign.

`platform_campaign.PlatformCampaign` is the library seam over the loop kernel,
compute allocator, promotion decision, capability foundry and continual
experience. This module makes that seam usable from a JSON spec, the CLI and the
MCP server, without hiding the Python callbacks it needs.

A spec names the run database and the two configs, then resolves the domain
callbacks by ``module:attribute``::

    {
      "runs_db": "platform-runs.sqlite3",
      "platform": {"tenant": "t1", "workspace": "ws", "max_generations": 3},
      "campaign": {"max_cycles": 5, "target": 0.9, "patience": 2},
      "compute": "mypkg.wiring:compute",
      "step": "mypkg.steps:step",
      "evidence": "mypkg.steps:evidence",
      "capability": "mypkg.steps:capability"
    }

``compute`` may be a ref or an inline model/option description, and may be
omitted (a single ``direct`` strategy over a zero-cost local model). Run
``agentdescent platform init --demo --out platform.json`` for a runnable offline
example, then ``platform plan`` and ``platform run``.

The surface owns no orchestration logic: it validates the spec, builds the
already-tested objects, and prints ``PlatformCampaignReport.to_dict()``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .budgeted_loop import BudgetedStep, BudgetedStepConfig
from .capability_foundry import CapabilityFoundry
from .compute_policy import (
    ComputeAction, ComputeController, ComputeOption,
)
from .durable_runs import SqliteRunStore
from .experience import ExperienceEngine, ExperienceRecord
from .loop_kernel import LoopContext, LoopObservation
from .model_router import ModelRouter, ModelSpec
from .platform import PlatformConfig, RSIPlatform
from .platform_campaign import CampaignConfig, PlatformCampaign
from .search_fabric import SearchFabric
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, load_json,
    load_jsonl, require_callable, require_instance, require_mapping,
    require_text, resolve_builder, resolve_ref,
)

__all__ = [
    "platform_init",
    "platform_plan",
    "platform_run",
    "demo_step",
]


def _text(value: Any, name: str) -> str:
    return require_text(value, name)


def _default_runs_db() -> str:
    home = os.environ.get("AGENTDESCENT_HOME") or "~/.agentdescent"
    return os.path.join(os.path.abspath(os.path.expanduser(home)),
                        "platform.sqlite3")


def _config(cls: Any, payload: Any, name: str) -> Any:
    if payload is None:
        return cls()
    data = require_mapping(payload, name)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SurfaceError(
            f"unknown {name} key(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}")
    try:
        return cls(**data)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid {name}: {error}") from None


def _compute(value: Any, *, allow: Optional[Sequence[str]] = None) -> ComputeController:
    if value is None:
        models = [ModelSpec("local", cost=0.0)]
        options = [ComputeOption("direct", ComputeAction.DIRECT, calls=1)]
        return ComputeController(
            options, ModelRouter(models, cost_weight=0.0, exploration=0.0),
            cost_weight=0.0, exploration=0.0)
    if isinstance(value, str):
        return require_instance(
            resolve_builder(value, name="compute", allow=allow),
            ComputeController, "compute")
    try:
        data = require_mapping(value, "compute")
        models = [ModelSpec(**require_mapping(row, "compute.models[]"))
                  for row in data.get("models", [])]
        options = [ComputeOption(**require_mapping(row, "compute.options[]"))
                   for row in data.get("options", [])]
        if not models or not options:
            raise SurfaceError(
                "compute needs at least one model and one option, or a ref")
        router = ModelRouter(
            models, cost_weight=float(data.get("cost_weight", 0.0)),
            exploration=float(data.get("exploration", 0.0)),
            seed=int(data.get("seed", 0)))
        return ComputeController(
            options, router,
            cost_weight=float(data.get("cost_weight", 0.0)),
            exploration=float(data.get("exploration", 0.0)),
            max_calls=data.get("max_calls"), seed=int(data.get("seed", 0)))
    except SurfaceError:
        raise
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"invalid inline compute: {error}") from None


def _rows(value: Any, name: str) -> Tuple[ExperienceRecord, ...]:
    if value is None:
        return ()
    try:
        return tuple(ExperienceRecord.from_dict(row)
                     for row in load_jsonl(value, name=name))
    except (TypeError, ValueError, KeyError) as error:
        raise SurfaceError(f"{name} is not ExperienceRecord JSONL: {error}") from None


def _optional_ref(value: Any, name: str, *, builder: bool = False,
                  allow: Optional[Sequence[str]] = None) -> Any:
    if value is None:
        return None
    return (resolve_builder(value, name=name, allow=allow) if builder
            else resolve_ref(value, name=name, allow=allow))


_BUDGETED_KEYS = ("fabric", "state", "config", "risk", "evidence_of")


def _budgeted_data(value: Any) -> Dict[str, Any]:
    data = require_mapping(value, "step.budgeted")
    unknown = sorted(set(data) - set(_BUDGETED_KEYS))
    if unknown:
        raise SurfaceError(
            "unknown step.budgeted key(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(_BUDGETED_KEYS))
    return data


def _budgeted_step(value: Any, *, allow: Optional[Sequence[str]] = None
                   ) -> BudgetedStep:
    data = _budgeted_data(value)
    fabric = require_instance(
        resolve_builder(data.get("fabric"), name="step.budgeted.fabric",
                        allow=allow), SearchFabric, "step.budgeted.fabric")
    state = require_callable(
        resolve_ref(data.get("state"), name="step.budgeted.state", allow=allow),
        "step.budgeted.state")
    config = _config(
        BudgetedStepConfig, data.get("config"), "step.budgeted.config")
    risk = _optional_ref(data.get("risk"), "step.budgeted.risk", allow=allow)
    evidence_of = _optional_ref(
        data.get("evidence_of"), "step.budgeted.evidence_of", allow=allow)
    for label, fn in (("step.budgeted.risk", risk),
                      ("step.budgeted.evidence_of", evidence_of)):
        if fn is not None:
            require_callable(fn, label)
    budgeted = BudgetedStep(
        fabric, state, config=config, risk=risk, evidence_of=evidence_of)

    def run_step(context: LoopContext, decision: Any = None) -> LoopObservation:
        # The platform passes ``(context, compute_decision)``; BudgetedStep owns
        # its own budget decision and only needs the context.
        return budgeted(context)

    return run_step


def _step(spec: Mapping[str, Any], *,
          allow: Optional[Sequence[str]] = None) -> Any:
    value = spec.get("step")
    if isinstance(value, Mapping):
        if set(value) != {"budgeted"}:
            raise SurfaceError(
                "a step object must contain exactly the 'budgeted' block")
        return _budgeted_step(value["budgeted"], allow=allow)
    return require_callable(
        resolve_ref(value, name="step", allow=allow), "step")


def _validate_step(spec: Mapping[str, Any], *,
                   allow: Optional[Sequence[str]] = None) -> None:
    value = spec.get("step")
    if not isinstance(value, Mapping):
        require_callable(
            resolve_ref(value, name="step", allow=allow), "step")
        return
    if set(value) != {"budgeted"}:
        raise SurfaceError(
            "a step object must contain exactly the 'budgeted' block")
    data = _budgeted_data(value["budgeted"])
    fabric = resolve_ref(
        data.get("fabric"), name="step.budgeted.fabric", allow=allow)
    if not callable(fabric) and not isinstance(fabric, SearchFabric):
        raise SurfaceError(
            "step.budgeted.fabric must be a builder or SearchFabric")
    require_callable(
        resolve_ref(data.get("state"), name="step.budgeted.state", allow=allow),
        "step.budgeted.state")
    _config(BudgetedStepConfig, data.get("config"), "step.budgeted.config")
    for name in ("risk", "evidence_of"):
        value = _optional_ref(
            data.get(name), f"step.budgeted.{name}", allow=allow)
        if value is not None:
            require_callable(value, f"step.budgeted.{name}")


def _build(spec: Mapping[str, Any], *, allow: Optional[Sequence[str]] = None
           ) -> Tuple[RSIPlatform, PlatformCampaign, Dict[str, Any]]:
    runs_db = spec.get("runs_db") or spec.get("store") or _default_runs_db()
    runs_db = _text(runs_db, "runs_db")
    if spec.get("platform") is None:
        raise SurfaceError("platform spec needs a platform config")
    platform_config = _config(PlatformConfig, spec.get("platform"), "platform")
    campaign_config = _config(CampaignConfig, spec.get("campaign"), "campaign")
    compute = _compute(spec.get("compute"), allow=allow)
    step = _step(spec, allow=allow)
    evidence = _optional_ref(spec.get("evidence"), "evidence", allow=allow)
    capability = _optional_ref(spec.get("capability"), "capability", allow=allow)
    if evidence is not None:
        require_callable(evidence, "evidence")
    if capability is not None:
        require_callable(capability, "capability")
    foundry = _optional_ref(spec.get("foundry"), "foundry", builder=True,
                            allow=allow)
    if foundry is not None:
        require_instance(foundry, CapabilityFoundry, "foundry")
    if capability is not None and foundry is None:
        raise SurfaceError("capability needs a foundry ref")
    experience = _optional_ref(spec.get("experience"), "experience",
                               builder=True, allow=allow)
    if experience is not None:
        require_instance(experience, ExperienceEngine, "experience")
    decision_log = _optional_ref(spec.get("decision_log"), "decision_log",
                                 builder=True, allow=allow)
    telemetry = _optional_ref(spec.get("telemetry"), "telemetry",
                              builder=True, allow=allow)
    gate = _optional_ref(spec.get("gate"), "gate", allow=allow)
    risk_gate = _optional_ref(spec.get("risk_gate"), "risk_gate", allow=allow)
    for label, value in (("gate", gate), ("risk_gate", risk_gate)):
        if value is not None:
            require_callable(value, label)
    platform = RSIPlatform(
        SqliteRunStore(os.path.abspath(os.path.expanduser(runs_db))),
        platform_config, compute=compute, foundry=foundry,
        experience=experience, decision_log=decision_log,
        telemetry=telemetry, gate=gate, risk_gate=risk_gate)
    campaign = PlatformCampaign(platform, campaign_config)
    callbacks = {
        "step": step, "evidence": evidence, "capability": capability,
        "validation": _rows(spec.get("validation"), "validation"),
        "anchors": _rows(spec.get("anchors"), "anchors"),
        "worker_id": str(spec.get("worker_id") or "platform-worker"),
    }
    return platform, campaign, callbacks


def load_spec(value: Any) -> Dict[str, Any]:
    payload = load_json(value, name="platform spec")
    spec = require_mapping(payload, "platform spec")
    if not spec.get("step"):
        raise SurfaceError("platform spec needs a step ref")
    return spec


def _label(value: Any) -> str:
    return (os.path.abspath(os.path.expanduser(value))
            if isinstance(value, str) else "<inline>")


def _validate(spec: Mapping[str, Any], *, allow: Optional[Sequence[str]] = None
              ) -> Dict[str, Any]:
    runs_db = spec.get("runs_db") or spec.get("store") or _default_runs_db()
    runs_db = _text(runs_db, "runs_db")
    if spec.get("platform") is None:
        raise SurfaceError("platform spec needs a platform config")
    platform_config = _config(PlatformConfig, spec.get("platform"), "platform")
    campaign_config = _config(CampaignConfig, spec.get("campaign"), "campaign")
    compute = spec.get("compute")
    if compute is None or isinstance(compute, Mapping):
        _compute(compute, allow=allow)
    else:
        resolve_ref(compute, name="compute", allow=allow)
    _validate_step(spec, allow=allow)
    for name in ("evidence", "capability", "gate", "risk_gate"):
        value = _optional_ref(spec.get(name), name, allow=allow)
        if value is not None:
            require_callable(value, name)
    for name in ("foundry", "experience", "decision_log", "telemetry"):
        _optional_ref(spec.get(name), name, builder=True, allow=allow)
    validation = _rows(spec.get("validation"), "validation")
    anchors = _rows(spec.get("anchors"), "anchors")
    return {
        "runs_db": os.path.abspath(os.path.expanduser(runs_db)),
        "platform": asdict(platform_config),
        "campaign": asdict(campaign_config),
        "validation_rows": len(validation),
        "anchor_rows": len(anchors),
    }


def platform_plan(path: Any, *, allow: Optional[Sequence[str]] = None
                  ) -> Dict[str, Any]:
    """Validate the spec and resolve every ref without running or creating a DB."""
    spec = load_spec(path)
    checked = _validate(spec, allow=allow)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "runs_db": checked["runs_db"],
        "platform": checked["platform"],
        "campaign": checked["campaign"],
        "refs": {
            "step": spec.get("step"),
            "evidence": spec.get("evidence"),
            "capability": spec.get("capability"),
            "foundry": spec.get("foundry"),
            "experience": spec.get("experience"),
        },
        "validation_rows": checked["validation_rows"],
        "anchor_rows": checked["anchor_rows"],
        "next": ["agentdescent platform run --spec <spec.json>"],
    }


def platform_run(path: Any, *, out: Optional[str] = None,
                 allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run a full :class:`PlatformCampaign` and return its JSON report."""
    spec = load_spec(path)
    platform, campaign, callbacks = _build(spec, allow=allow)
    report = campaign.run(
        worker_id=callbacks["worker_id"], step=callbacks["step"],
        evidence=callbacks["evidence"], capability=callbacks["capability"],
        validation=callbacks["validation"], anchors=callbacks["anchors"])
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": _label(path),
        "runs_db": platform.store.path,
        "report": report.to_dict(),
        "next": [
            "inspect cycles[*].loop.generations for the accepted artifact",
            "agentdescent ops publish --runs-db <runs_db> ...",
        ],
    }
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def platform_init(out: Any, *, demo: bool = False) -> Dict[str, Any]:
    """Write a starter spec; ``--demo`` writes the offline runnable example."""
    target = os.path.abspath(os.path.expanduser(_text(out, "out")))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    if demo:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "runs_db": os.path.join(directory, "platform-runs.sqlite3"),
            "platform": {
                "tenant": "demo", "workspace": "default",
                "max_generations": 3, "target": 0.75,
            },
            "campaign": {
                "max_cycles": 2, "target": 0.75, "patience": 1,
            },
            "compute": {
                "models": [{"name": "local", "cost": 0.0}],
                "options": [{"name": "direct", "action": "direct", "calls": 1}],
                "cost_weight": 0.0, "exploration": 0.0,
            },
            "step": "agentdescent.surface_platform:demo_step",
        }
    else:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "runs_db": os.path.join(directory, "platform-runs.sqlite3"),
            "platform": {"tenant": "t1", "workspace": "default",
                         "max_generations": 3},
            "campaign": {"max_cycles": 3, "patience": 1},
            "compute": "your_package.wiring:compute",
            "step": "your_package.steps:step",
            "evidence": "your_package.steps:evidence",
            "capability": "your_package.steps:capability",
            "foundry": "your_package.wiring:foundry",
            "experience": "your_package.wiring:experience",
        }
    atomic_write_json(target, spec)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": target,
        "demo": demo,
        "next": [
            f"agentdescent platform plan --spec {target}",
            f"agentdescent platform run  --spec {target} --json",
        ],
    }


def demo_step(context: LoopContext, decision: Any) -> LoopObservation:
    """A deterministic offline step for ``platform init --demo``."""
    reward = min(1.0, 0.6 + 0.15 * int(context.generation))
    return LoopObservation(
        reward, cost=float(getattr(decision, "expected_cost", 0.0)),
        artifact_id=f"demo-{context.generation}")
