"""Self-rewarding meta-evolution has to be right where the judge learns truth.

The judge is a small linear regressor trained only on oracle labels; the run
alternates — evolve with the current judge, then fit a better one on newly
oracle-labelled outcomes — and reports the judge's agreement with the oracle on
outcomes it never trained on. The fixtures pin the surfaces that make this
honest: features are curve-shape (never the answer), the judge learns to agree
with the oracle, the run refits and reports measured agreement, and an empty
proposer is refused.
"""

import pytest

from agentdescent import (
    LearnableMetaJudge,
    MetaOutcome,
    ParamSlot,
    outcome_features,
    self_rewarding_meta_run,
)
from agentdescent.sampling import DifficultyWeighted


def _helping(value, seed):
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0, rollouts=2)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3, rollouts=2)


def _helping_val(value, seed):
    return _helping(value, seed)


def _slotted(value, seed):
    # a landscape where quality tracks the curve's *shape*, not any single value
    c = getattr(value, "c", 0.2)
    if c >= 0.4:
        return MetaOutcome(curve=[0.6, 0.9, 1.0], final=0.9, rollouts=3)
    if c >= 0.3:
        return MetaOutcome(curve=[0.3, 0.6], final=0.6, rollouts=2)
    return MetaOutcome(curve=[0.1, 0.2], final=0.2, rollouts=1)


def _slotted_val(value, seed):
    return _slotted(value, seed)


def _slotted_landscape(value, seed):
    # quality tracks the c band, with per-seed variation so a single candidate
    # produces feature variety (real problems vary per (problem, seed))
    c = getattr(value, "c", 0.2)
    band = 0.9 if c >= 0.4 else (0.6 if c >= 0.3 else 0.2)
    jitter = (seed % 4) * 0.02
    v = min(1.0, band + jitter)
    width = 0.02 + (seed % 4) * 0.015        # curve width varies per seed
    return MetaOutcome(curve=[max(0.0, v - width), v], final=v, rollouts=1)


def _slotted_landscape_val(value, seed):
    return _slotted_landscape(value, seed)


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _nudge_up():
    import re
    def propose(rendered, task, output, reward):
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose


def test_outcome_features_are_curve_shape():
    feats = outcome_features(_helping(_sampler_slot().compile(
        "# Policy parameters\nc: 0.4"), 0))
    assert len(feats) == 3
    assert 0 <= feats[0] <= 1.0          # mean best-so-far
    assert feats[1] == 1.0               # final
    assert feats[2] == 1.0               # intercept term


def test_the_judge_learns_to_agree_with_the_oracle():
    from agentdescent import final_reward
    judge = LearnableMetaJudge()
    slot = _sampler_slot()
    pairs = [(p, float(final_reward(p))) for p in
             [_slotted(slot.compile("# Policy parameters\nc: 0.4"), 0),
              _slotted(slot.compile("# Policy parameters\nc: 0.2"), 0),
              _slotted(slot.compile("# Policy parameters\nc: 0.3"), 0),
              _slotted(slot.compile("# Policy parameters\nc: 0.5"), 1)]]
    judge.fit(pairs)
    assert judge.n_labels == 4
    # agreement on held-out outcomes it never trained on
    held = [(p, float(final_reward(p))) for p in
            [_slotted(slot.compile("# Policy parameters\nc: 0.6"), 0),
             _slotted(slot.compile("# Policy parameters\nc: 0.2"), 5),
             _slotted(slot.compile("# Policy parameters\nc: 0.7"), 2)]]
    assert judge.agreement(held) >= 0.5


def test_a_few_labels_are_no_claim():
    judge = LearnableMetaJudge()
    assert judge.agreement([(_helping(0, 0), 1.0)]) == 0.5


def test_the_fallback_is_the_safe_auc_and_does_not_saturate():
    from agentdescent import final_reward
    judge = LearnableMetaJudge()
    slot = _sampler_slot()
    # 2 pairs is below the fit floor: the judge falls back to the safe auc
    pairs = [(p, float(final_reward(p))) for p in
             [_slotted(slot.compile("# Policy parameters\nc: 0.2"), 0),
              _slotted(slot.compile("# Policy parameters\nc: 0.4"), 1)]]
    judge.fit(pairs)
    assert not judge.fitted
    outcomes = [_slotted(slot.compile("# Policy parameters\nc: 0.2"), 0),
                _slotted(slot.compile("# Policy parameters\nc: 0.4"), 1),
                _slotted(slot.compile("# Policy parameters\nc: 0.5"), 0)]
    scores = [judge(o) for o in outcomes]
    # the old fixed-weight fallback returned 1.0 for every outcome; the safe
    # fallback still separates different qualities and stays in [0, 1]
    assert len({round(s, 4) for s in scores}) >= 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_labels_come_from_several_candidates():
    from agentdescent import final_reward
    from agentdescent.self_rewarding import _sample_oracle_pairs
    slot = _sampler_slot()
    pairs = _sample_oracle_pairs({"a": _slotted}, slot,
                                 ["# Policy parameters\nc: 0.2",
                                  "# Policy parameters\nc: 0.4"],
                                 (0, 1), final_reward, 4)
    assert len(pairs) == 4
    finals = {p.final for p, _ in pairs}
    # labels span both candidate values, not one rule's per-seed noise
    assert 0.2 in finals and 0.9 in finals


def test_self_rewarding_meta_run_refits_and_reports():
    from agentdescent import final_reward
    train = {"a": _slotted_landscape, "b": _slotted_landscape}
    val = {"v": _slotted_landscape_val}
    report, result = self_rewarding_meta_run(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        oracle=final_reward, propose=_nudge_up(),
        seeds=(0, 1, 2, 3), val_seeds=(4, 5), judge_sample=6,
        self_reward_rounds=2, rounds=1, n_workers=1, repo_path=None)
    assert report.judge_labels >= 6
    assert report.judge_fitted                  # labels spanned several candidates
    assert 0.0 <= report.judge_agreement <= 1.0
    assert report.value
    assert result.meta["judge_labels"] == report.judge_labels
    assert result.meta["judge_fitted"] == report.judge_fitted
    assert "oracle_gain" in result.meta


def test_self_rewarding_meta_run_needs_a_proposer():
    from agentdescent import final_reward
    with pytest.raises(ValueError):
        self_rewarding_meta_run({"a": _slotted}, {"v": _slotted_val},
                                slot="task_sampler", spec=_sampler_slot(),
                                oracle=final_reward)
