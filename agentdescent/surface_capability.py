"""The ``agentdescent capability`` surface: failures to an admitted capability.

`capability_loop.CapabilityLoop` already composes the production-trace
pipeline, tool synthesis/admission and the quality-diversity archive. This
module makes that composition runnable from a JSON spec, the CLI and the MCP
server.

A spec names the traces, the goal and the three already-built objects:

    {
      "traces": "traces.jsonl",
      "goal": "parse a CSV line",
      "descriptor": [0, 0],
      "pipeline": "mypkg.wiring:build_pipeline",
      "scout": "mypkg.wiring:build_scout",
      "archive": "mypkg.wiring:build_archive",
      "make_request": "mypkg.wiring:make_request",
      "kind": "tool",
      "tags": ["csv"]
    }

``agentdescent capability init --demo --out capability.json`` writes a runnable
offline example (and its traces). The surface owns the hand-off only; the
pipeline, synthesis, admission and archive stay in their tested modules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .artifact_graph import ArtifactGraph
from .capability_archive import CapabilityArchive
from .capability_foundry import CapabilityFoundry, CapabilityRequest
from .capability_loop import CapabilityLoop
from .capability_scout import CapabilityScout
from .data_flywheel import DataFlywheel, FlywheelConfig, TraceSample
from .proofs import ProofInvariant, ProofRunner
from .quality_diversity import QualityDiversityArchive
from .safe_selfmod import ModificationSpec, SafeSelfModifier
from .surface_common import (
    SURFACE_SCHEMA_VERSION, SurfaceError, atomic_write_json, load_json,
    load_jsonl, parse_csv, require_instance, require_mapping, require_text,
    resolve_builder, resolve_ref,
)
from .task_foundry import FoundryConfig, TaskCandidate, TaskFoundry
from .task_pipeline import TaskPipeline
from .tool_synthesis import ToolSpec, ToolSynthesizer

__all__ = [
    "capability_init",
    "capability_plan",
    "capability_run",
    "demo_archive",
    "demo_make_request",
    "demo_pipeline",
    "demo_scout",
]


def load_traces(value: Any) -> Tuple[TraceSample, ...]:
    rows = []
    for index, row in enumerate(load_jsonl(value, name="traces")):
        trace_id = row.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise SurfaceError(f"traces[{index}].trace_id is required")
        task = row.get("task", row.get("prompt"))
        if task is None:
            raise SurfaceError(f"traces[{index}] needs task or prompt")
        rows.append(TraceSample(
            trace_id=trace_id, task=task, output=row.get("output"),
            reward=row.get("reward", 0.0), cost=row.get("cost", 0.0),
            tags=row.get("tags") or (), error=row.get("error"),
            metadata=row.get("metadata") or {}))
    return tuple(rows)


def load_spec(value: Any) -> Dict[str, Any]:
    payload = load_json(value, name="capability spec")
    spec = require_mapping(payload, "capability spec")
    for key in ("traces", "goal", "descriptor", "pipeline", "scout",
                "archive", "make_request"):
        if key not in spec:
            raise SurfaceError(f"capability spec needs {key!r}")
    return spec


def _options(spec: Mapping[str, Any]) -> Dict[str, Any]:
    goal = require_text(spec.get("goal"), "goal")
    descriptor = spec.get("descriptor")
    if not isinstance(descriptor, Sequence) or isinstance(descriptor, (str, bytes)):
        raise SurfaceError("descriptor must be a JSON array of ints")
    return {
        "goal": goal,
        "make_request": spec.get("make_request"),
        "descriptor": tuple(int(value) for value in descriptor),
        "kind": str(spec.get("kind") or "tool"),
        "tags": tuple(str(tag) for tag in parse_csv(
            spec.get("tags") or (), name="tags")),
        "parents": tuple(str(parent) for parent in parse_csv(
            spec.get("parents") or (), name="parents")),
        "metadata": require_mapping(spec.get("metadata") or {}, "metadata"),
    }


def _validate(spec: Mapping[str, Any], *,
              allow: Optional[Sequence[str]] = None) -> Tuple[Tuple[TraceSample, ...], Dict[str, Any]]:
    options = _options(spec)
    for name in ("pipeline", "scout", "archive"):
        value = resolve_ref(spec.get(name), name=name, allow=allow)
        if callable(value):
            continue
        expected = {"pipeline": TaskPipeline, "scout": CapabilityScout,
                    "archive": CapabilityArchive}[name]
        if not isinstance(value, expected):
            raise SurfaceError(
                f"{name} must be a builder or {expected.__name__}, "
                f"got {type(value).__name__}")
    request = resolve_ref(
        spec.get("make_request"), name="make_request", allow=allow)
    if not callable(request):
        raise SurfaceError("make_request must be callable")
    return load_traces(spec.get("traces")), options


def _build(spec: Mapping[str, Any], *,
           allow: Optional[Sequence[str]] = None
           ) -> Tuple[CapabilityLoop, Tuple[TraceSample, ...], Dict[str, Any]]:
    options = _options(spec)
    pipeline = require_instance(
        resolve_builder(spec.get("pipeline"), name="pipeline", allow=allow),
        TaskPipeline, "pipeline")
    scout = require_instance(
        resolve_builder(spec.get("scout"), name="scout", allow=allow),
        CapabilityScout, "scout")
    archive = require_instance(
        resolve_builder(spec.get("archive"), name="archive", allow=allow),
        CapabilityArchive, "archive")
    make_request = resolve_ref(
        spec.get("make_request"), name="make_request", allow=allow)
    if not callable(make_request):
        raise SurfaceError("make_request must be callable")
    traces = load_traces(spec.get("traces"))
    loop = CapabilityLoop(pipeline, scout, archive)
    options["make_request"] = make_request
    return loop, traces, options


def capability_plan(path: Any, *, allow: Optional[Sequence[str]] = None
                    ) -> Dict[str, Any]:
    spec = load_spec(path)
    traces, options = _validate(spec, allow=allow)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": (os.path.abspath(os.path.expanduser(path))
                 if isinstance(path, str) else "<inline>"),
        "traces": len(traces),
        "goal": options["goal"],
        "descriptor": list(options["descriptor"]),
        "refs": {
            "pipeline": spec.get("pipeline"),
            "scout": spec.get("scout"),
            "archive": spec.get("archive"),
            "make_request": spec.get("make_request"),
        },
        "next": ["agentdescent capability run --spec <spec.json>"],
    }


def capability_run(path: Any, *, out: Optional[str] = None,
                   allow: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    spec = load_spec(path)
    loop, traces, options = _build(spec, allow=allow)
    report = loop.run(traces, **options)
    payload: Dict[str, Any] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": (os.path.abspath(os.path.expanduser(path))
                 if isinstance(path, str) else "<inline>"),
        "report": report.to_dict(),
        "next": (
            ["capability admitted; inspect archive and tool synthesis"]
            if report.admitted else
            ["inspect report.verdict and report.pipeline.foundry"]),
    }
    if out is not None:
        payload["out"] = atomic_write_json(out, payload)
    return payload


def capability_init(out: Any, *, demo: bool = False) -> Dict[str, Any]:
    target = os.path.abspath(os.path.expanduser(require_text(out, "out")))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    traces_path = os.path.join(directory, "capability-traces.jsonl")
    if demo:
        with open(traces_path, "w", encoding="utf-8") as handle:
            for index in range(4):
                handle.write(json_line({
                    "trace_id": f"demo-trace-{index}",
                    "task": f"Add the two numbers in line {index}: {index} + {index + 1}",
                    "output": "unknown",
                    "reward": 0.0,
                }))
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "traces": traces_path,
            "goal": "add two integers from a short line",
            "descriptor": [0, 0],
            "pipeline": "agentdescent.surface_capability:demo_pipeline",
            "scout": "agentdescent.surface_capability:demo_scout",
            "archive": "agentdescent.surface_capability:demo_archive",
            "make_request": "agentdescent.surface_capability:demo_make_request",
            "kind": "tool",
            "tags": ["demo"],
        }
    else:
        spec = {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "traces": traces_path,
            "goal": "what the missing capability should do",
            "descriptor": [0, 0],
            "pipeline": "your_package.wiring:build_pipeline",
            "scout": "your_package.wiring:build_scout",
            "archive": "your_package.wiring:build_archive",
            "make_request": "your_package.wiring:make_request",
            "kind": "tool",
            "tags": [],
        }
    atomic_write_json(target, spec)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "ok": True,
        "spec": target,
        "traces": traces_path,
        "demo": demo,
        "next": [
            f"agentdescent capability plan --spec {target}",
            f"agentdescent capability run  --spec {target} --json",
        ],
    }


def json_line(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"


def demo_pipeline() -> TaskPipeline:
    flywheel = DataFlywheel(FlywheelConfig())
    foundry = TaskFoundry(
        FoundryConfig(min_validation=0.0, min_tasks=1, max_candidates=20,
                      per_seed=1, holdout_fraction=0.5),
        generator=lambda seed, attempt: TaskCandidate(
            task_id=f"{seed.task_id}-{attempt}",
            prompt=f"{seed.prompt} (variant {attempt})",
            difficulty=0.5),
        validator=lambda candidate: 1.0)
    return TaskPipeline(flywheel, foundry)


def demo_scout() -> CapabilityScout:
    synthesizer = ToolSynthesizer(
        generator=lambda request, generation, prior: ToolSpec(
            name=f"demo-tool-{generation}", description="demo tool",
            source="def demo_tool(value):\n    return value + 1\n"),
        validator=lambda spec: 1.0)
    gate = SafeSelfModifier(set(), max_risk=0.5, canary_risk=0.2)
    prover = ProofRunner([
        ProofInvariant("has-source", lambda artifact: bool(
            getattr(artifact, "source", ""))),
    ])
    foundry = CapabilityFoundry(
        gate, graph=ArtifactGraph(), prover=prover, require_review=False)
    return CapabilityScout(synthesizer, foundry)


def demo_archive() -> CapabilityArchive:
    return CapabilityArchive(
        QualityDiversityArchive([2, 2], seed=0), ArtifactGraph())


def demo_make_request(candidate: Any) -> CapabilityRequest:
    spec = candidate.spec
    return CapabilityRequest(
        gap_id="demo-gap", tool=spec, validation=candidate.validation,
        modification=ModificationSpec(
            spec.name, spec.source, "tool", requested_capabilities=()),
        before=0.0, after=1.0)
