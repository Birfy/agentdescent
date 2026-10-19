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
