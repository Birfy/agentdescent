"""Failure tasks must become a reviewed tool through the existing seams."""

import pytest

from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_foundry import CapabilityFoundry, CapabilityRequest
from agentdescent.capability_scout import CapabilityScout
from agentdescent.deliberation import DeliberationReport
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.task_foundry import TaskCandidate
from agentdescent.tool_synthesis import (
    ToolSpec, ToolSynthesisRequest, ToolSynthesizer, ToolValidation,
)

_SOURCE = "def parse_csv(text):\n    return text.split(',')\n"


def _tasks():
    return [TaskCandidate(f"t{i}", f"fix csv line {i}", difficulty=0.6)
            for i in range(3)]


def _synthesizer(*, produce=True, seen=None):
    def generator(request: ToolSynthesisRequest, generation, prior):
        if seen is not None:
            seen.append(request)
        if not produce:
            return None
        return ToolSpec("csv-tool", "parse a CSV line", _SOURCE)

    return ToolSynthesizer(generator, lambda spec: ToolValidation(True, 0.9))


def _foundry():
    return CapabilityFoundry(
        SafeSelfModifier(set()), graph=ArtifactGraph(),
        prover=ProofRunner([
            ProofInvariant("has-source", lambda art: bool(art.source)),
        ]))


def _request(candidate):
    tool = candidate.spec
    return CapabilityRequest(
        gap_id="csv", tool=tool, validation=candidate.validation,
        review=DeliberationReport("approve", 0.8, 0.9, (), "consensus"),
        modification=ModificationSpec(tool.name, tool.source, "tool"),
        before=0.5, after=0.7, canary=0.65)


def test_failure_tasks_become_an_admitted_tool():
    scout = CapabilityScout(_synthesizer(), _foundry())
    report = scout.run(_tasks(), goal="parse CSV", make_request=_request)
    assert report.verdict == "promoted"
    assert report.outcome is not None
    assert report.outcome.artifact_id == "csv-tool"
    assert report.synthesis.selected is not None
    data = report.to_dict()
    assert data["outcome"]["status"] == "promoted"
    assert data["synthesis"]["selected"] == "csv-tool"


def test_examples_and_candidate_budget_are_independent():
    seen = []
    scout = CapabilityScout(_synthesizer(seen=seen), _foundry(),
                            max_examples=2, max_candidates=3)
    report = scout.run(_tasks(), goal="parse CSV", make_request=_request)
    assert report.tasks == 3
    assert len(seen[0].examples) == 2
    assert seen[0].max_candidates == 3
    assert seen[0].constraints["task_ids"] == ["t0", "t1", "t2"]


def test_non_promoted_admission_is_not_reported_as_success():
    scout = CapabilityScout(_synthesizer(), _foundry())
    rejected = scout.run(
        _tasks(), goal="parse CSV",
        make_request=lambda candidate: CapabilityRequest(
            gap_id="csv", tool=candidate.spec, validation=candidate.validation,
            review=DeliberationReport("reject", 0.8, 0.9, (), "consensus"),
            modification=ModificationSpec(
                candidate.spec.name, candidate.spec.source, "tool"),
            before=0.5, after=0.7, canary=0.65))
    assert rejected.verdict == "rejected"
    assert rejected.outcome is not None and rejected.outcome.artifact_id == ""

    deferred = scout.run(
        _tasks(), goal="parse CSV",
        make_request=lambda candidate: CapabilityRequest(
            gap_id="csv", tool=candidate.spec, validation=candidate.validation,
            review=DeliberationReport("approve", 0.8, 0.9, (), "consensus"),
            modification=ModificationSpec(
                candidate.spec.name, candidate.spec.source, "tool"),
            before=0.5, after=0.4, canary=0.65))
    assert deferred.verdict == "rolled_back"


def test_no_candidate_or_no_request_is_explicit():
    no_candidate = CapabilityScout(_synthesizer(produce=False), _foundry())
    report = no_candidate.run(_tasks(), goal="parse CSV",
                              make_request=_request)
    assert report.verdict == "no-candidate"
    assert report.outcome is None

    scout = CapabilityScout(_synthesizer(), _foundry())
    report = scout.run(_tasks(), goal="parse CSV",
                       make_request=lambda _candidate: None)
    assert report.verdict == "no-request"


def test_invalid_inputs_are_refused():
    scout = CapabilityScout(_synthesizer(), _foundry())
    with pytest.raises(ValueError):
        scout.run([], goal="x", make_request=_request)
    with pytest.raises(ValueError):
        scout.run(_tasks(), goal="", make_request=_request)
    with pytest.raises(TypeError):
        scout.run([{"task_id": "x"}], goal="x", make_request=_request)
    with pytest.raises(TypeError):
        CapabilityScout(object(), _foundry())
    with pytest.raises(ValueError):
        CapabilityScout(_synthesizer(), _foundry(), max_examples=0)
    with pytest.raises(ValueError):
        CapabilityScout(_synthesizer(), _foundry(), max_candidates=True)
    with pytest.raises(TypeError):
        scout.run(_tasks(), goal="x", make_request=lambda _c: object())
