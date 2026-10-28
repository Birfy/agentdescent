"""The QD archive has to keep one elite per cell and report it honestly."""

import random

import pytest

from agentdescent.quality_diversity import (
    Elite,
    QualityDiversityArchive,
    descriptor_distance,
    illumination_report,
)


def test_one_elite_per_cell_and_strict_improvement():
    archive = QualityDiversityArchive(bins=(2, 2))
    assert archive.add("a", 0.4, (0, 0)) is True
    assert archive.add("b", 0.3, (0, 0)) is False
    assert archive.add("c", 0.5, (0, 0)) is True
    assert archive.get((0, 0)).artifact == "c"
    assert archive.filled == 1


def test_coverage_and_qd_score_are_the_sums_not_the_best():
    archive = QualityDiversityArchive(bins=(2, 2))
    archive.add("a", 0.4, (0, 0))
    archive.add("b", 0.8, (0, 1))
    archive.add("c", 0.1, (1, 0))
    assert archive.filled == 3
    assert archive.total == 4
    assert archive.coverage == 0.75
    assert archive.qd_score == pytest.approx(1.3)
    assert archive.best.artifact == "b"


def test_empty_archive_reports_empty_not_a_fake_score():
    archive = QualityDiversityArchive(bins=(3,))
    report = illumination_report(archive)
    assert report.verdict == "empty"
    assert report.best_score is None
    assert report.qd_score == 0.0
    assert archive.best is None
    assert archive.select() is None


@pytest.mark.parametrize("bins", [[], [0], [2, -1], [True]])
def test_invalid_bins_are_refused(bins):
    with pytest.raises((ValueError, TypeError)):
        QualityDiversityArchive(bins=bins)


@pytest.mark.parametrize("descriptor", [(0,), (0, 0, 0), (2, 0), (0, -1), ("a", 0)])
def test_invalid_descriptors_are_refused(descriptor):
    archive = QualityDiversityArchive(bins=(2, 2))
    with pytest.raises((ValueError, TypeError)):
        archive.add("a", 0.1, descriptor)


def test_ties_keep_the_first_arrival_not_an_opaque_comparison():
    archive = QualityDiversityArchive(bins=(1,))
    assert archive.add("first", 0.5, (0,)) is True
    assert archive.add("second", 0.5, (0,)) is False
    assert archive.get((0,)).artifact == "first"


def test_selection_is_reproducible_with_an_explicit_rng():
    archive = QualityDiversityArchive(bins=(2, 2))
    for i, cell in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        archive.add(i, 0.1 * (i + 1), cell)
    picks_a = [archive.select(random.Random(7)).artifact for _ in range(4)]
    picks_b = [archive.select(random.Random(7)).artifact for _ in range(4)]
    assert picks_a == picks_b


def test_novelty_mode_moves_away_from_a_repeated_cell():
    archive = QualityDiversityArchive(bins=(1, 2))
    archive.add("left", 0.5, (0, 0))
    archive.add("right", 0.5, (0, 1))
    # Fill the visit counter of one cell; with two cells the other must be
    # favoured until the counts equalise.
    rng = random.Random(0)
    for _ in range(20):
        archive.select(rng, mode="novelty")
    left, right = archive.visits((0, 0)), archive.visits((0, 1))
    assert left > 0 and right > 0
    assert max(left, right) <= 2 * min(left, right)


def test_select_modes_validate_and_best_does_not_consume_a_visit():
    archive = QualityDiversityArchive(bins=(1,))
    archive.add("only", 0.9, (0,))
    assert archive.select(mode="best").artifact == "only"
    assert archive.visits((0,)) == 0
    with pytest.raises(ValueError):
        archive.select(mode="random")


def test_remove_and_clear_leave_no_ghost_visits():
    archive = QualityDiversityArchive(bins=(2,))
    archive.add("a", 0.5, (0,))
    archive.select(random.Random(0))
    assert archive.remove((0,)) is True
    assert archive.remove((0,)) is False
    assert archive.visits((0,)) == 0
    archive.add("b", 0.2, (1,))
    archive.clear()
    assert archive.filled == 0


def test_serialisation_omits_opaque_artifacts_but_keeps_layout():
    archive = QualityDiversityArchive(bins=(2,))
    archive.add(object(), 0.7, (1,), parent_id="p", generation=3,
                metadata={"slot": "selection"})
    data = archive.to_dict()
    assert data["bins"] == [2]
    assert data["cells"][0]["descriptor"] == [1]
    assert data["cells"][0]["score"] == 0.7
    assert data["cells"][0]["parent_id"] == "p"
    assert data["cells"][0]["generation"] == 3
    assert "artifact" not in data["cells"][0]


def test_illumination_verdicts_track_coverage():
    archive = QualityDiversityArchive(bins=(6,))
    assert illumination_report(archive).verdict == "empty"
    archive.add("a", 0.1, (0,))
    assert illumination_report(archive).verdict == "sparse"
    archive.add("b", 0.2, (1,))
    archive.add("c", 0.3, (2,))
    assert illumination_report(archive).verdict == "partial"
    archive.add("d", 0.4, (3,))
    archive.add("e", 0.5, (4,))
    archive.add("f", 0.6, (5,))
    report = illumination_report(archive)
    assert report.verdict == "illuminated"
    assert report.coverage == 1.0
    assert "QD-score=" in report.detail


def test_elite_rejects_non_finite_scores_and_negative_generation():
    with pytest.raises(ValueError):
        Elite(artifact=None, score=float("nan"), descriptor=(0,))
    with pytest.raises(ValueError):
        Elite(artifact=None, score=0.1, descriptor=(0,), generation=-1)


def test_descriptor_distance_is_manhattan_and_checks_length():
    assert descriptor_distance((0, 2), (3, 5)) == 6
    with pytest.raises(ValueError):
        descriptor_distance((0,), (0, 1))
