"""The end-to-end self-evolution bench has to run the whole chain.

Its job is to prove the rungs compose: seed -> population -> exam -> portfolio ->
harness -> chain audit, each producing a measurement on a deterministic domain.
The fixtures pin the two things that make it trustworthy: every rung runs and
reports `ok`, and a rung that raises is recorded as a failed stage rather than
crashing the bench.
"""

import json

from bench.self_evolution import BenchReport, Stage, _record, main


def test_the_bench_runs_every_rung(capsys):
    code = main(["--rounds", "1", "--islands", "1", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["all_ok"] is True
    assert set(out["stages"]) == {
        "seed", "population", "exam", "portfolio", "harness", "chain"}
    for name, stage in out["stages"].items():
        assert stage["ok"] is True, name
    # the numbers that make "the chain works" concrete
    assert out["stages"]["seed"]["detail"]["archive_count"] >= 1
    assert out["stages"]["exam"]["detail"]["examined"] >= 1
    assert out["stages"]["portfolio"]["detail"]["layers"]
    assert out["stages"]["harness"]["detail"]["selected"] in ("quick", "deep")
    assert out["stages"]["chain"]["detail"]["rings"]


def test_a_failing_stage_is_recorded_not_raised():
    report = BenchReport()

    def boom():
        raise RuntimeError("rung broke")

    _record(report, "boom", boom)
    assert not report.all_ok
    assert report.stages[0].ok is False
    assert "RuntimeError: rung broke" in report.stages[0].note
