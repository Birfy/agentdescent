import hashlib
import json
from types import MappingProxyType

import pytest

from examples.rrsi.round import (
    HarnessCandidate,
    RoundRecord,
    ScreeningResult,
    run_round,
)
from examples.rrsi.selection import Measurement, SelectionConfig


CFG = SelectionConfig(beta0=0.1, beta1=1.0, w_s=0.0, w_c=1.0, w_n=0.5)
INCUMBENT = Measurement(score=0.70, cost=100.0)
BASE = {"README.md": "incumbent"}


def candidate(
    name, contents, *, base_version="v1", base_sha256=None, components=("skill",)
):
    if base_sha256 is None:
        base_sha256 = hashlib.sha256(
            json.dumps(
                BASE, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
    return HarnessCandidate(
        name, base_version, base_sha256, {"README.md": contents}, components
    )


def invoke(propose, screen, evaluate, **overrides):
    args = dict(
        round_index=3,
        base_version="v1",
        base_files=BASE,
        incumbent=INCUMBENT,
        historical_best=0.72,
        delta=0.05,
        config=CFG,
        incumbent_counts={"skill": 1},
        propose=propose,
        screen=screen,
        evaluate=evaluate,
    )
    args.update(overrides)
    return run_round(**args)


def test_screen_veto_happens_before_expensive_evaluation():
    evaluated = []

    def propose(base):
        assert base.version == "v1"
        return [
            candidate(
                "leak", "contains suspicious grader path", base_sha256=base.sha256
            ),
            candidate("clean", "allowed", base_sha256=base.sha256),
        ]

    def screen(item):
        if item.variant == "leak":
            return ScreeningResult(False, "test veto")
        return ScreeningResult(True, "static checks passed")

    def evaluate(item):
        evaluated.append(item.variant)
        return Measurement(0.84, 95.0)

    result = invoke(propose, screen, evaluate)

    assert evaluated == ["clean"]
    assert [c.outcome for c in result.record.candidates] == [
        "screen_rejected",
        "evaluated",
    ]
    assert result.record.candidates[0].measurement is None
    assert result.record.candidates[0].reason == "test veto"
    assert result.record.candidates[1].reason == "static checks passed"


def test_proposal_and_survivors_share_one_immutable_base_snapshot():
    seen_base_ids = []
    calls = []

    def propose(base):
        seen_base_ids.append(id(base.files))
        assert base.version == "v1"
        assert isinstance(base.files, MappingProxyType)
        with pytest.raises(TypeError):
            base.files["README.md"] = "mutated"
        calls.append(dict(base.files))
        return [
            candidate("a", base.files["README.md"] + " A", base_sha256=base.sha256),
            candidate("b", base.files["README.md"] + " B", base_sha256=base.sha256),
        ]

    def screen(item):
        assert item.base_version == "v1"
        assert isinstance(item.files, MappingProxyType)
        return ScreeningResult(True)

    result = invoke(propose, screen, lambda _item: Measurement(0.80, 100.0))
    assert len(set(seen_base_ids)) == 1
    assert calls == [BASE]
    assert result.record.base_version == "v1"
    assert all(c.base_version == "v1" for c in result.record.candidates)
    assert result.record.base_sha256
    assert all(
        c.source_base_sha256 == result.record.base_sha256
        for c in result.record.candidates
    )


@pytest.mark.parametrize("mutation_stage", ["screen", "evaluate"])
def test_all_candidate_artifacts_are_snapshotted_before_callbacks(mutation_stage):
    files_b = {"x": "original B"}
    components_b = ["skill"]

    def propose(base):
        return [
            HarnessCandidate("a", base.version, base.sha256, {"x": "A"}, ("skill",)),
            HarnessCandidate("b", base.version, base.sha256, files_b, components_b),
        ]

    def mutate_b():
        files_b["x"] = "mutated by callback A"
        components_b.append("memory")

    def screen(item):
        if item.variant == "a" and mutation_stage == "screen":
            mutate_b()
        if item.variant == "b":
            assert dict(item.files) == {"x": "original B"}
            assert item.components == ("skill",)
        return ScreeningResult(True)

    def evaluate(item):
        if item.variant == "a" and mutation_stage == "evaluate":
            mutate_b()
        if item.variant == "b":
            assert dict(item.files) == {"x": "original B"}
            assert item.components == ("skill",)
        return Measurement(0.8, 100.0)

    result = invoke(propose, screen, evaluate)

    candidate_b = next(c for c in result.record.candidates if c.variant == "b")
    expected_sha = hashlib.sha256(
        json.dumps({"x": "original B"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert candidate_b.outcome == "evaluated"
    assert candidate_b.components == ("skill",)
    assert candidate_b.artifact_sha256 == expected_sha


def test_screen_and_evaluator_errors_have_explicit_outcomes_and_no_fake_scores():
    def propose(_base):
        return [
            candidate("reject", "r"),
            candidate("screen-error", "s"),
            candidate("eval-error", "e"),
            candidate("bad-metric", "m"),
        ]

    def screen(item):
        if item.variant == "reject":
            return ScreeningResult(False, "critic rejected")
        if item.variant == "screen-error":
            raise RuntimeError("screen unavailable")
        return ScreeningResult(True)

    def evaluate(item):
        if item.variant == "eval-error":
            raise TimeoutError("measurement timeout")
        return Measurement(float("nan"), 10.0)

    result = invoke(propose, screen, evaluate)
    assert [c.outcome for c in result.record.candidates] == [
        "screen_rejected",
        "screen_error",
        "evaluation_error",
        "invalid_measurement",
    ]
    assert [c.reason for c in result.record.candidates] == [
        "critic rejected",
        "RuntimeError: screen unavailable",
        "TimeoutError: measurement timeout",
        "evaluator returned an invalid measurement",
    ]
    assert all(
        c.measurement is None and not c.admissible for c in result.record.candidates
    )
    assert result.winner is None


def test_round_selection_keeps_score_cost_and_selected_artifact_consistent():
    measured = {
        "lower": Measurement(0.84, 110.0),
        "higher": Measurement(0.92, 120.0),
    }
    result = invoke(
        lambda _base: [candidate("lower", "L"), candidate("higher", "H")],
        lambda _item: ScreeningResult(True),
        lambda item: measured[item.variant],
    )

    assert result.winner is not None and result.winner.variant == "higher"
    selected = next(c for c in result.record.candidates if c.variant == "higher")
    assert selected.admissible
    assert selected.measurement == measured["higher"]
    assert selected.delta_score == pytest.approx(0.22)
    assert selected.delta_cost == pytest.approx(0.20)
    assert result.record.selected_variant == "higher"
    assert result.record.next_best_score == pytest.approx(0.92)


def test_guard_callback_error_is_recorded_and_does_not_abort_other_candidates():
    def guard(_incumbent, measurement):
        if measurement.score < 0.9:
            return [object()]  # malformed callback output is handled like an error
        return []

    result = invoke(
        lambda _base: [candidate("guard-error", "A"), candidate("good", "B")],
        lambda _item: ScreeningResult(True),
        lambda item: Measurement(
            0.84 if item.variant == "guard-error" else 0.92, 100.0
        ),
        guard_fn=guard,
    )

    first, second = result.record.candidates
    assert first.outcome == "guard_error"
    assert "guard_fn results must be strings" in first.reason
    assert first.measurement == Measurement(0.84, 100.0)
    assert not first.admissible
    assert second.outcome == "evaluated" and second.admissible
    assert result.winner is not None and result.winner.variant == "good"


def test_guard_error_does_not_relabel_invalid_proposal_with_colliding_name():
    def propose(base):
        return [
            object(),
            HarnessCandidate(
                "<invalid-0>",
                base.version,
                base.sha256,
                {"README.md": "valid artifact"},
                ("skill",),
            ),
        ]

    def guard(_incumbent, _measurement):
        raise RuntimeError("guard failure")

    result = invoke(
        propose,
        lambda _item: ScreeningResult(True),
        lambda _item: Measurement(0.9, 90.0),
        guard_fn=guard,
    )

    malformed, guarded = result.record.candidates
    assert malformed.variant == guarded.variant == "<invalid-0>"
    assert malformed.outcome == "proposal_error"
    assert "non-HarnessCandidate" in malformed.reason
    assert guarded.outcome == "guard_error"
    assert guarded.reason == "RuntimeError: guard failure"


def test_round_record_roundtrips_through_json():
    result = invoke(
        lambda _base: [candidate("ok", "changed")],
        lambda _item: ScreeningResult(True),
        lambda _item: Measurement(0.84, 90.0),
    )
    restored = RoundRecord.from_json(result.record.to_json())
    assert restored == result.record


def test_proposal_failure_is_recorded_and_preserves_historical_best():
    def fail(_base):
        raise RuntimeError("no proposal")

    result = invoke(fail, lambda _item: ScreeningResult(True), lambda _item: INCUMBENT)
    assert result.winner is None
    assert result.record.proposal_error == "RuntimeError: no proposal"
    assert result.record.candidates == ()
    assert result.record.next_best_score == 0.72


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"base_version": "v0"}, "different base version"),
        ({"base_sha256": "0" * 64}, "different base digest"),
    ],
)
def test_candidate_from_different_base_is_not_screened_or_evaluated(kwargs, expected):
    called = []
    result = invoke(
        lambda _base: [candidate("wrong-base", "x", **kwargs)],
        lambda _item: called.append("screen") or ScreeningResult(True),
        lambda _item: called.append("eval") or Measurement(1.0, 1.0),
    )
    assert called == []
    assert result.record.candidates[0].outcome == "proposal_error"
    assert expected in result.record.candidates[0].reason
    assert result.record.candidates[0].measurement is None


def test_malformed_proposal_fields_still_produce_serializable_error_record():
    def propose(base):
        return [
            HarnessCandidate(
                "bad", object(), base.sha256, {"README.md": "x"}, (object(),)
            )
        ]

    result = invoke(
        propose,
        lambda _item: ScreeningResult(True),
        lambda _item: Measurement(1.0, 1.0),
    )

    item = result.record.candidates[0]
    assert item.outcome == "proposal_error"
    assert "base_version must be a string" in item.reason
    assert item.base_version == ""
    assert item.components == ()
    assert RoundRecord.from_json(result.record.to_json()) == result.record
