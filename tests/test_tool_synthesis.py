"""Tool synthesis must keep the best validated candidate and the audit trail."""

import pytest

from agentdescent.tool_synthesis import (
    EnvironmentSpec,
    ToolSynthesisRequest,
    ToolSynthesizer,
    ToolSpec,
    ToolValidation,
)


def _spec(i, source="def f():\n    return 1\n"):
    return ToolSpec(f"tool-{i}", f"tool {i}", source, capabilities=("files:read",))


def test_selects_the_highest_scoring_valid_candidate():
    scores = {0: 0.2, 1: 0.9, 2: 0.5}
    synth = ToolSynthesizer(
        lambda request, i, prior: _spec(i),
        lambda spec: ToolValidation(spec.name == "tool-1", scores[int(spec.name[-1])]))
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=3))
    assert len(report.candidates) == 3
    assert report.selected.spec.name == "tool-1"
    assert len(report.accepted) == 1


def test_generator_none_and_errors_are_recorded_not_fatal():
    def generator(_request, i, _prior):
        if i == 0:
            return None
        if i == 1:
            raise RuntimeError("model")
        return _spec(i)

    synth = ToolSynthesizer(generator, lambda _spec: True)
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=3))
    assert report.selected.spec.name == "tool-2"
    assert any("RuntimeError" in c.validation.reason for c in report.candidates)


def test_selector_can_override_the_default():
    synth = ToolSynthesizer(
        lambda _r, i, _prior: _spec(i),
        lambda _spec: 1.0,
        selector=lambda candidates: candidates[2])
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=4))
    assert report.selected.spec.name == "tool-2"


def test_generator_sees_earlier_rejections():
    seen = []

    def generator(_request, i, prior):
        seen.append(tuple(prior))
        return _spec(i)

    def validate(spec):
        if spec.name == "tool-0":
            return ToolValidation(False, 0.0, "missing capability")
        return True

    synth = ToolSynthesizer(generator, validate)
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=2))
    assert seen[0] == ()
    assert seen[1][0].validation.reason == "missing capability"
    assert report.selected.spec.name == "tool-1"


def test_invalid_selector_results_fall_back_to_the_best_valid_candidate():
    synth = ToolSynthesizer(
        lambda _r, i, _prior: _spec(i),
        lambda spec: ToolValidation(spec.name == "tool-1", 1.0),
        selector=lambda candidates: candidates[0])
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=2))
    assert report.selected.spec.name == "tool-1"
    assert report.selected.validation.valid

    synth = ToolSynthesizer(
        lambda _r, i, _prior: _spec(i),
        lambda _spec: True,
        selector=lambda _candidates: "not a candidate")
    report = synth.synthesize(ToolSynthesisRequest("goal", max_candidates=2))
    assert report.selected.spec.name == "tool-0"


def test_environment_spec_and_invalid_inputs():
    env = EnvironmentSpec("env", (_spec(1), _spec(2)))
    assert len(env.tools) == 2
    with pytest.raises(ValueError):
        EnvironmentSpec("", (_spec(1),))
    with pytest.raises(ValueError):
        EnvironmentSpec("env", ())
    with pytest.raises(TypeError):
        EnvironmentSpec("env", ("tool",))
    with pytest.raises(ValueError):
        ToolSynthesisRequest("")
    with pytest.raises(ValueError):
        ToolSynthesisRequest("goal", max_candidates=0)
    with pytest.raises(ValueError):
        ToolSpec("", "d", "source")
    with pytest.raises(ValueError):
        ToolSpec("n", "d", "source", capabilities="files")
    with pytest.raises(ValueError):
        ToolValidation(True, score=float("nan"))
