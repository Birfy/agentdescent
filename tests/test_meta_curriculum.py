"""Meta-curriculum has to be right where the problem distribution self-generates.

The meta loop evolves a decision rule; the challenger generates the *problems*
it is judged on -- so the one gate that must never move is the final
evaluation. The fixtures pin the surfaces that make this honest: the challenger
installs generated problems into the training sampler and never touches the
validation set, the validity gate counts rather than crashes, learnability is
measured and reported (learned / stuck / r0), the archive still seeds the search
and receives adopted rules, and the same isolation guards hold.
"""

import re

import pytest

from agentdescent import (
    ConsolidationStore,
    MetaAuditStore,
    MetaChallengerConfig,
    MetaOutcome,
    MetaProblemChallenger,
    ParamSlot,
    SingleSlot,
    Task,
    curriculum_meta_run,
)
from agentdescent.sampling import DifficultyWeighted


# -- the deterministic landscape -------------------------------------------------


def _c_of(rendered):
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    return float(m.group(1)) if m else 0.2


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _nudge_up(rendered, task, output, reward):
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def _helpful(value, seed):
    # monotone in c, so every nudge strictly improves the held-out gate and the
    # rule is free to climb toward the challenger's frontier
    c = getattr(value, "c", 0.2)
    r = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[r, min(1.0, r + 0.05)], final=r)


def _helpful_val(value, seed):
    return _helpful(value, seed)


def _frontier_problem(d):
    def problem(value, seed):
        c = getattr(value, "c", 0.2)
        if c >= d:
            return MetaOutcome(curve=[1.0, 1.0], final=1.0)   # learned
        return MetaOutcome(curve=[0.1, 0.2], final=0.2)        # on the frontier
    return problem


def _challenger(start_d=0.4, step=0.1, *, bad=False):
    spec = [f"d:{start_d:.2f}"]

    def generate(descriptor, seed):
        m = re.search(r"d:([0-9.]+)", descriptor)
        return {"d": float(m.group(1)) if m else start_d}

    def validate(config):
        if bad:
            raise ValueError("degenerate problem")
        return _frontier_problem(config["d"])

    def propose(descriptor, report):
        m = re.search(r"d:([0-9.]+)", descriptor)
        d = float(m.group(1)) if m else start_d
        if report.get("learned", 0) > 0:
            spec[0] = f"d:{min(1.0, d + step):.2f}"
        return spec[0]

    return MetaProblemChallenger(
        generate=generate, validate=validate, propose=propose, spec=spec[0],
        config=MetaChallengerConfig(generated_per_round=2, frontier_share=0.4,
                                    max_attempts=4, max_generate_attempts=20))


def _common(store, tmp_path):
    return dict(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up, config={"dataset": "meta-cur", "scorer": "exact"},
        seeds=(0, 1, 2, 3), val_seeds=(10, 11), rounds=3, n_workers=1,
        repo_dir=str(tmp_path / "repos"))


def _seed_rule(store, tmp_path, c, run_id="seed"):
    from agentdescent import EvolutionResult, problem_descriptor
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=0.7, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot",
        config={"dataset": "meta-cur", "scorer": "exact"},
        problem=problem_descriptor("task_sampler"),
        artifact_id="meta-rule")


# -- the product surface ---------------------------------------------------------


def test_challenger_generates_problems_but_never_touches_validation(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report, result = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val})
    # generated problems entered the training sampler
    assert report["frontier"]["generated_total"] >= 1
    assert report["frontier"]["invalid_total"] == 0
    # the validation table is exactly the caller's problems -- no frontier row
    assert set(report["validation"].keys()) == {"va"}
    # the decision rides the run result
    assert result.meta["verdict"] == report["verdict"]


def test_the_rule_climbs_toward_the_frontier_and_learns(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _seed_rule(store, tmp_path, 0.3)
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(start_d=0.5),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val})
    # the search started from the archived rule and moved outward
    assert report["seeded_from"] == "seed"
    assert _c_of(report["value"]) >= 0.3
    # learnability was measured and reported
    assert "learned" in report["frontier"]
    assert "stuck" in report["frontier"]
    assert "r0_mean" in report["frontier"]


def test_invalid_generation_is_counted_not_crashed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(bad=True, step=0.3),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val})
    # the validity gate refused some payloads, and the run went on
    assert report["frontier"]["invalid_total"] >= 1
    assert report["frontier"]["generated_total"] >= 0
    assert report["verdict"] in ("adopt", "reject")


def test_verdict_and_archival(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val}, run_id="cur-1")
    assert report["verdict"] in ("adopt", "reject")
    if report["verdict"] == "adopt":
        assert report["run_id"] == "cur-1"
        rec = store.find("cur-1")
        assert rec is not None and rec.problem["slot"] == "task_sampler"


def test_requires_problems_and_validation(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        curriculum_meta_run(store, slot="task_sampler", spec=_sampler_slot(),
                            challenger=_challenger(), propose=_nudge_up,
                            problems={}, val_problems={"va": _helpful_val})
    with pytest.raises(ValueError):
        curriculum_meta_run(store, slot="task_sampler", spec=_sampler_slot(),
                            challenger=_challenger(), propose=_nudge_up,
                            problems={"ta": _helpful}, val_problems={})


def test_overlapping_train_and_validation_is_refused(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        curriculum_meta_run(
            store, slot="task_sampler", spec=_sampler_slot(),
            challenger=_challenger(), propose=_nudge_up,
            problems={"a": _helpful}, val_problems={"a": _helpful_val},
            seeds=(0, 1), val_seeds=(1, 2))


# -- verified seeding ------------------------------------------------------------


def test_min_exam_passed_excludes_a_flagged_rule_and_falls_back_to_seed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _seed_rule(store, tmp_path, 0.4, run_id="flagged-rule")
    store.mark_exam("flagged-rule", exam="flagged", flags=1, note="failed once")
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val}, min_exam="passed")
    assert report["seeded_from"] is None
    assert _c_of(report["seed_value"]) == 0.2


def test_min_exam_passed_seeds_from_a_passed_rule(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _seed_rule(store, tmp_path, 0.4, run_id="passed-rule")
    store.mark_exam("passed-rule", exam="passed", flags=0, note="ok")
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val}, min_exam="passed")
    assert report["seeded_from"] == "passed-rule"
    assert _c_of(report["seed_value"]) == 0.4


# -- audited validation ---------------------------------------------------------


def test_audited_validation_records_pairs_but_the_verifier_decides(tmp_path):
    plain_store = ConsolidationStore(str(tmp_path / "archive-plain"))
    plain, _ = curriculum_meta_run(
        **_common(plain_store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val})

    store = ConsolidationStore(str(tmp_path / "archive-audited"))
    audit = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val},
        oracle=lambda outcome: outcome.final, audit_store=audit,
        audit_rate=1.0)

    assert report["audited_n"] >= 1
    assert "gain_bias" in report and "oracle_gain" in report
    assert report["audit_store"] == str(tmp_path / "audit.jsonl")
    assert len(list(audit.iter_records())) >= 1
    assert len(audit.verdicts()) == 1
    assert audit.verdicts()[0]["verdict"] == report["verdict"]
    # the oracle only lands in the store: the table/verdict are the verifier's
    assert report["validation"] == plain["validation"]
    assert report["mean_gain"] == plain["mean_gain"]
    assert report["verdict"] == plain["verdict"]


@pytest.mark.parametrize("audit_kwargs", [
    {"oracle": lambda outcome: outcome.final},
    {"audit_store": MetaAuditStore()},
])
def test_audit_needs_both_oracle_and_store(tmp_path, audit_kwargs):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError, match="both oracle"):
        curriculum_meta_run(
            **_common(store, tmp_path), challenger=_challenger(),
            problems={"ta": _helpful, "tb": _helpful},
            val_problems={"va": _helpful_val}, **audit_kwargs)


# -- the two unmerged defects ---------------------------------------------------


def test_generated_problems_are_bounded_by_the_frontier_pool():
    # the name -> Problem map must not keep every problem ever generated
    from agentdescent.meta_curriculum import _MetaCurriculumDriver
    from agentdescent.selfplay import FrontierSampler
    import random as _random
    challenger = _challenger()
    challenger.config = MetaChallengerConfig(
        generated_per_round=4, max_frontier_pool=6, max_generate_attempts=20)
    sampler = FrontierSampler(challenger, seed=0)
    problems = {"ta": _helpful, "tb": _helpful}
    driver = _MetaCurriculumDriver(challenger, sampler, problems,
                                   solved_threshold=0.999, rng=_random.Random(1))
    driver.refresh()
    for _ in range(25):
        driver.after_round(None)
    assert sampler.stats()["generated_total"] >= 100
    assert {"ta", "tb"} <= set(problems)
    assert len(problems) <= 2 + 6
    live = {t.id for t in sampler._frontier}
    assert set(problems) - {"ta", "tb"} == live


def test_default_run_ids_do_not_collide_within_one_second(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    ids = []
    for i in range(2):
        common = _common(store, tmp_path)
        common["repo_dir"] = str(tmp_path / f"repos{i}")
        report, _ = curriculum_meta_run(
            **common, challenger=_challenger(),
            problems={"ta": _helpful, "tb": _helpful},
            val_problems={"va": _helpful_val})
        assert report["verdict"] == "adopt", report["reason"]
        ids.append(report["run_id"])
    assert ids[0] != ids[1]


def test_default_call_still_seeds_from_an_unexamined_rule(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _seed_rule(store, tmp_path, 0.3, run_id="legacy")
    report, _ = curriculum_meta_run(
        **_common(store, tmp_path), challenger=_challenger(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val})
    assert report["seeded_from"] == "legacy"
    assert "audited_n" not in report
    assert "gain_bias" not in report
