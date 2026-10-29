"""Traffic must become a clean dataset, not a pile of duplicates and secrets."""

import pytest

from agentdescent.data_flywheel import (
    DataFlywheel,
    FlywheelConfig,
    FlywheelReport,
    TraceSample,
    redact,
)


def _sample(i, reward=0.2, task=None, output=None, error=None, metadata=None):
    return TraceSample(f"t{i}", task or f"task {i}", output or f"out {i}",
                       reward=reward, cost=1, error=error,
                       metadata=metadata or {"keep": i})


def test_reward_band_errors_and_sampling_are_filtered():
    wheel = DataFlywheel(FlywheelConfig(min_reward=0.0, max_reward=0.5,
                                        sample_rate=1.0))
    report = wheel.mine([
        _sample(1, 0.2),
        _sample(2, 0.8),
        _sample(3, 0.2, error="tool"),
    ])
    assert report.accepted_count == 1
    assert report.rejection_counts()["reward-band"] == 1
    assert report.rejection_counts()["error"] == 1


def test_exact_duplicates_keep_one_and_near_duplicates_use_the_cap():
    exact = DataFlywheel(FlywheelConfig(max_per_cluster=2))
    report = exact.mine([_sample(i, task="same", output="same") for i in range(5)])
    assert report.accepted_count == 1
    assert report.rejection_counts()["duplicate"] == 4

    near = DataFlywheel(FlywheelConfig(max_per_cluster=3, near_duplicate_threshold=0.8),
                        similarity=lambda *_: 0.9)
    report = near.mine([_sample(i) for i in range(6)])
    assert report.accepted_count == 3
    assert report.rejection_counts()["cluster-cap"] == 3


def test_exact_copy_of_a_near_duplicate_member_is_still_a_duplicate():
    wheel = DataFlywheel(
        FlywheelConfig(max_per_cluster=3, near_duplicate_threshold=0.8),
        similarity=lambda *_: 0.9)
    report = wheel.mine([
        _sample(1, task="a", output="a"),
        _sample(2, task="b", output="b"),
        _sample(3, task="b", output="b"),
    ])
    assert [sample.trace_id for sample in report.accepted] == ["t1", "t2"]
    assert report.rejection_counts()["duplicate"] == 1


def test_a_similarity_chain_cannot_bypass_the_cluster_cap():
    def chain(left, right):
        return 0.9 if abs(int(left.task[1:]) - int(right.task[1:])) == 1 else 0.0

    wheel = DataFlywheel(FlywheelConfig(max_per_cluster=2, near_duplicate_threshold=0.8),
                         similarity=chain)
    report = wheel.mine([_sample(i, task=f"s{i}", output=f"o{i}") for i in range(6)])
    assert [sample.trace_id for sample in report.accepted] == ["t0", "t1", "t3", "t4"]
    assert report.rejection_counts()["cluster-cap"] == 2


def test_near_duplicate_hook_and_similarity_errors():
    wheel = DataFlywheel(FlywheelConfig(near_duplicate_threshold=0.8, max_per_cluster=1),
                         similarity=lambda left, right: 0.9)
    report = wheel.mine([_sample(1), _sample(2)])
    assert report.accepted_count == 1
    assert (report.rejection_counts().get("near-duplicate", 0)
            + report.rejection_counts().get("cluster-cap", 0)) == 1
    bad = DataFlywheel(FlywheelConfig(), similarity=lambda *_: float("nan"))
    with pytest.raises(ValueError):
        bad.mine([_sample(1), _sample(2)])


def test_redaction_is_recursive_and_key_based():
    payload = {
        "task": "normal question",
        "api_key": "secret",
        "nested": {"email": "a@b.c", "note": "keep", "phone": "123"},
    }
    masked = redact(payload, keys=("phone",))
    assert masked["api_key"] == "[REDACTED]"
    assert masked["nested"]["email"] == "[REDACTED]"
    assert masked["nested"]["phone"] == "[REDACTED]"
    assert masked["nested"]["note"] == "keep"


def test_dataset_candidate_has_hash_and_redacted_rows():
    wheel = DataFlywheel(FlywheelConfig(redact_keys=("secret",)))
    report = wheel.mine([_sample(1, metadata={"secret_value": "x", "keep": 1})])
    dataset = wheel.to_dataset(report, dataset_id="prod", version=1)
    assert len(dataset.rows) == 1
    assert dataset.rows[0]["metadata"]["secret_value"] == "[REDACTED]"
    assert dataset.to_dict()["content_hash"] == dataset.content_hash
    with pytest.raises(ValueError):
        wheel.to_dataset(FlywheelReport((), (), 0), dataset_id="empty", version=1)


def test_sampling_is_deterministic():
    config = FlywheelConfig(sample_rate=0.5, seed=7)
    one = DataFlywheel(config).mine([_sample(i) for i in range(50)])
    two = DataFlywheel(config).mine([_sample(i) for i in range(50)])
    assert [s.trace_id for s in one.accepted] == [s.trace_id for s in two.accepted]


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        TraceSample("", "task")
    with pytest.raises(ValueError):
        TraceSample("t", "task", reward=float("nan"))
    with pytest.raises(ValueError):
        TraceSample("t", "task", cost=-1)
    with pytest.raises(ValueError):
        FlywheelConfig(min_reward=1, max_reward=0)
    with pytest.raises(ValueError):
        FlywheelConfig(sample_rate=2)
    with pytest.raises(ValueError):
        FlywheelConfig(max_samples=0)
    wheel = DataFlywheel()
    with pytest.raises(TypeError):
        wheel.mine([{"trace_id": "x"}])
