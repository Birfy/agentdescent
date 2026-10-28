"""A higher target score must not buy a broken invariance."""

import pytest

from agentdescent.metamorphic import (
    MetamorphicRelation,
    MetamorphicSuite,
    invariance,
    relation,
)


def _identity_run(task):
    return task


def test_invariance_passes_when_the_candidate_ignores_the_transform():
    suite = MetamorphicSuite([invariance("case", lambda task: task.lower())])
    report = suite.evaluate(lambda task: task.lower(), ["ABC", "DeF"])
    assert report.verdict == "pass"
    assert report.checks == 2
    assert report.passed == 2
    assert report.weighted_rate == 1.0
    assert suite.accept(lambda task: task.lower(), ["ABC"])


def test_invariance_fails_and_names_the_relation():
    def run(task):
        return task

    suite = MetamorphicSuite([invariance("case", lambda task: task.lower())])
    report = suite.evaluate(run, ["ABC"])
    assert report.verdict == "fail"
    assert report.failed == 1
    assert report.violations[0]["relation"] == "case"
    assert report.per_relation["case"]["rate"] == 0.0
    assert not suite.accept(run, ["ABC"])


def test_a_transform_that_should_change_the_output_uses_a_relation_check():
    suite = MetamorphicSuite([
        relation("double", lambda task: task + task,
                 lambda original, transformed, task: len(transformed) == 2 * len(original)),
    ])
    assert suite.evaluate(_identity_run, ["ab"]).verdict == "pass"


def test_relations_are_scored_separately_and_weighted():
    suite = MetamorphicSuite([
        invariance("always", lambda task: task, weight=3.0),
        invariance("broken", lambda task: task + "!"),
    ])
    report = suite.evaluate(_identity_run, ["a", "b"])
    assert report.checks == 4
    assert report.passed == 2
    assert report.weighted_rate == pytest.approx(3.0 / 4.0)
    assert report.per_relation["always"]["rate"] == 1.0
    assert report.per_relation["broken"]["rate"] == 0.0


def test_a_failing_run_is_recorded_not_crashing_the_suite():
    def run(task):
        if task == "boom":
            raise RuntimeError("no model")
        return task

    suite = MetamorphicSuite([invariance("id", lambda task: task)])
    report = suite.evaluate(run, ["ok", "boom"])
    assert report.verdict == "fail"
    assert report.checks == 2
    assert any("RuntimeError" in str(row.get("error", "")) for row in report.violations)


def test_violation_cap_keeps_reports_bounded_but_counts_every_check():
    suite = MetamorphicSuite([invariance("broken", lambda task: task + "!")])
    report = suite.evaluate(_identity_run, list(range(10)), max_violations=3)
    assert report.failed == 10
    assert len(report.violations) == 3


def test_accept_allows_a_bounded_failure_rate():
    suite = MetamorphicSuite([
        invariance("always", lambda task: task, weight=3.0),
        invariance("broken", lambda task: task + "!"),
    ])
    assert suite.accept(_identity_run, ["a"], min_rate=0.7)
    assert not suite.accept(_identity_run, ["a"], min_rate=0.8)


def test_per_relation_counts_are_integers():
    suite = MetamorphicSuite([invariance("id", lambda task: task)])
    report = suite.evaluate(_identity_run, ["a", "b"])
    assert report.per_relation["id"]["passed"] == 2
    assert isinstance(report.per_relation["id"]["passed"], int)
    assert isinstance(report.per_relation["id"]["total"], int)


def test_empty_task_set_is_insufficient_not_a_silent_pass():
    suite = MetamorphicSuite([invariance("id", lambda task: task)])
    report = suite.evaluate(_identity_run, [])
    assert report.verdict == "insufficient"
    assert report.checks == 0
    assert not suite.accept(_identity_run, [])


def test_duplicate_or_empty_relations_and_bad_min_rate_are_refused():
    with pytest.raises(ValueError):
        MetamorphicSuite([])
    with pytest.raises(ValueError):
        MetamorphicSuite([invariance("x", lambda t: t), invariance("x", lambda t: t)])
    with pytest.raises(ValueError):
        MetamorphicRelation("x", lambda t: t, lambda a, b, c: True, weight=0)
    with pytest.raises(ValueError):
        MetamorphicRelation(None, lambda t: t, lambda a, b, c: True)
    with pytest.raises(ValueError):
        MetamorphicRelation("   ", lambda t: t, lambda a, b, c: True)
    suite = MetamorphicSuite([invariance("id", lambda t: t)])
    with pytest.raises(ValueError):
        suite.accept(_identity_run, ["a"], min_rate=1.5)
