"""Production failures must become an archived capability in one call."""

import json

import pytest

from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_archive import CapabilityArchive
from agentdescent.capability_foundry import CapabilityFoundry, CapabilityRequest
from agentdescent.capability_loop import CapabilityLoop
from agentdescent.capability_scout import CapabilityScout
from agentdescent.data_flywheel import DataFlywheel, TraceSample
from agentdescent.deliberation import DeliberationReport
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.quality_diversity import QualityDiversityArchive
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.task_foundry import FoundryConfig, TaskCandidate, TaskFoundry
from agentdescent.task_pipeline import TaskPipeline
from agentdescent.tool_synthesis import (
    ToolSpec, ToolSynthesisRequest, ToolSynthesizer, ToolValidation,
)

_SOURCE = "def parse_csv(text):\n    return text.split(',')\n"
_UNSAFE_SOURCE = "import os\n\ndef run():\n    return os.getcwd()\n"


def _traces(*, reward=0.1, count=5):
    return [
        TraceSample(
            f"t{index}", f"failed task {index}",
            output=f"bad output {index}", reward=reward)
        for index in range(count)
    ]


def _pipeline():
    def generator(seed, _attempt):
        return TaskCandidate(
            seed.task_id, seed.prompt, difficulty=seed.difficulty)

    foundry = TaskFoundry(
        FoundryConfig(per_seed=1, holdout_fraction=0.5),
        generator=generator, validator=lambda _candidate: 1.0)
    return TaskPipeline(DataFlywheel(), foundry)


def _synthesizer(*, produce=True, source=_SOURCE, name="csv-tool"):
    def generator(request: ToolSynthesisRequest, _generation, _prior):
        if not produce:
            return None
        return ToolSpec(name, "parse a CSV line", source)

    return ToolSynthesizer(generator, lambda _spec: ToolValidation(True, 0.9))


def _capability_foundry():
    return CapabilityFoundry(
        SafeSelfModifier(set()), graph=ArtifactGraph(),
        prover=ProofRunner([
            ProofInvariant("has-source", lambda artifact: bool(artifact.source)),
        ]))


def _make_request(candidate):
    tool = candidate.spec
    return CapabilityRequest(
        gap_id="csv", tool=tool, validation=candidate.validation,
        review=DeliberationReport("approve", 0.8, 0.9, (), "consensus"),
        modification=ModificationSpec(tool.name, tool.source, "tool"),
        before=0.5, after=0.7, canary=0.65)


def _archive(*, bins=(2, 2)):
    return CapabilityArchive(QualityDiversityArchive(bins=bins), ArtifactGraph())


def _loop(*, archive=None, synthesizer=None, foundry=None):
    return CapabilityLoop(
        _pipeline(),
        CapabilityScout(
            synthesizer or _synthesizer(),
            foundry or _capability_foundry()),
        archive or _archive())


def _run(loop, traces=None, **overrides):
    kwargs = {
        "goal": "parse a CSV line",
        "make_request": _make_request,
        "descriptor": (0, 0),
        "tags": ("csv",),
    }
    kwargs.update(overrides)
    return loop.run(traces if traces is not None else _traces(), **kwargs)


def test_failure_traces_become_an_admitted_archived_capability():
    archive = _archive()
    report = _run(_loop(archive=archive))

    assert report.pipeline.verdict == "ready"
    assert report.scout is not None
    assert report.scout.outcome is not None
    assert report.scout.outcome.status == "promoted"
    assert report.archive is not None
    assert report.archive.added
    assert report.admitted
    assert report.verdict == "admitted"
    assert "csv-tool" in archive.graph
    assert archive.retrieve("parse_csv", k=1)[0].artifact.artifact_id == "csv-tool"
    elite = archive.archive.get((0, 0))
    assert elite is not None
    assert elite.score == pytest.approx(0.9)
    assert elite.metadata["gap_id"] == "csv"
    assert json.loads(json.dumps(report.to_dict()))["verdict"] == "admitted"


def test_pipeline_without_failures_stops_before_the_scout():
    report = _run(_loop(), _traces(reward=0.9))

    assert report.verdict == "pipeline-empty"
    assert report.verdict.startswith("pipeline-")
    assert report.scout is None
    assert report.archive is None
    assert not report.admitted


def test_scout_without_a_valid_candidate_does_not_touch_the_archive():
    archive = _archive()
    report = _run(
        _loop(archive=archive, synthesizer=_synthesizer(produce=False)))

    assert report.verdict == "capability-no-candidate"
    assert report.verdict.startswith("capability-")
    assert report.scout is not None
    assert report.scout.verdict == "no-candidate"
    assert report.archive is None
    assert len(archive.graph) == 0


def test_capability_foundry_rejection_does_not_reach_the_archive():
    archive = _archive()
    report = _run(
        _loop(archive=archive, synthesizer=_synthesizer(source=_UNSAFE_SOURCE)))

    assert report.verdict == "capability-rejected"
    assert report.scout is not None
    assert report.scout.outcome is not None
    assert report.scout.outcome.status == "rejected"
    assert report.archive is None
    assert len(archive.graph) == 0


def test_descriptor_with_the_wrong_dimension_raises():
    archive = _archive(bins=(2, 2))

    with pytest.raises(ValueError, match="dimension"):
        _run(_loop(archive=archive), descriptor=(0,))

    assert len(archive.graph) == 0


def test_duplicate_capability_is_refused_and_the_archive_is_unchanged():
    archive = _archive()
    first = _run(_loop(archive=archive))
    assert first.admitted
    assert len(archive.graph) == 1

    second = _run(_loop(archive=archive))

    assert second.verdict == "duplicate"
    assert second.archive is not None
    assert not second.archive.added
    assert second.archive.duplicate_of == "csv-tool"
    assert not second.admitted
    assert len(archive.graph) == 1


def test_capability_that_cannot_replace_the_elite_is_not_admitted():
    archive = _archive()
    archive.add("existing", "train an image classifier", score=0.95,
                descriptor=(0, 0))

    report = _run(_loop(archive=archive))

    assert report.verdict == "not-elite"
    assert report.archive is not None
    assert not report.archive.added
    assert report.archive.duplicate_of is None
    assert not report.admitted
    assert len(archive.graph) == 1


def test_invalid_types_and_inputs_are_refused():
    loop = _loop()

    with pytest.raises(TypeError):
        CapabilityLoop(object(), loop.scout, loop.archive)
    with pytest.raises(TypeError):
        CapabilityLoop(loop.pipeline, object(), loop.archive)
    with pytest.raises(TypeError):
        CapabilityLoop(loop.pipeline, loop.scout, object())

    with pytest.raises(ValueError):
        _run(loop, goal="")
    with pytest.raises(TypeError):
        _run(loop, make_request=None)
    with pytest.raises(ValueError):
        _run(loop, descriptor=())
    with pytest.raises(TypeError):
        _run(loop, descriptor="00")
    with pytest.raises(TypeError):
        _run(loop, descriptor=(0.5, 0))
    with pytest.raises(TypeError):
        _run(loop, tags="csv")
    with pytest.raises(TypeError):
        _run(loop, metadata=[])
    with pytest.raises(TypeError):
        _run(loop, [{"trace_id": "x"}])
