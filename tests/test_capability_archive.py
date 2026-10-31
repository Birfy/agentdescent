"""The QD archive, graph and semantic dedup must describe one population."""

import pytest

from agentdescent.artifact_graph import Artifact, ArtifactGraph
from agentdescent.capability_archive import CapabilityArchive
from agentdescent.quality_diversity import QualityDiversityArchive


def _archive():
    return CapabilityArchive(
        QualityDiversityArchive(bins=(2, 2)), ArtifactGraph(),
        dedup_threshold=0.9)


def test_add_dedup_elite_and_report_stay_in_sync():
    archive = _archive()
    first = archive.add("csv-tool", "parse csv with split", score=0.5,
                        descriptor=(0, 0), kind="tool", tags=("csv",))
    assert first.added and first.elite
    duplicate = archive.add("csv-copy", "parse csv with split ", score=0.9,
                            descriptor=(0, 1), kind="tool")
    assert not duplicate.added
    assert duplicate.duplicate_of == "csv-tool"
    assert len(archive.graph) == 1

    second = archive.add("classifier", "train a classifier on images",
                         score=0.7, descriptor=(1, 1), kind="skill")
    assert second.added
    lower = archive.add("classifier-copy", "train a classifier on images now",
                        score=0.2, descriptor=(1, 1), kind="skill")
    assert not lower.added
    assert len(archive.graph) == 2

    report = archive.report()
    assert report.duplicates >= 1
    assert report.illumination.filled == 2
    assert report.illumination.coverage == pytest.approx(0.5)
    assert report.graph.artifacts == 2
    assert archive.retrieve("parse csv", k=1)[0].artifact.artifact_id == "csv-tool"
    selected = archive.select(seed=0)
    assert selected is not None


def test_elite_replacement_removes_the_old_graph_and_fingerprint():
    archive = _archive()
    archive.add("old", "parse csv files with the split method", score=0.3,
                descriptor=(0, 0))
    archive.add("new", "train an image classifier using convolution",
                score=0.9, descriptor=(0, 0))
    assert len(archive.graph) == 1
    assert "old" not in archive.graph
    assert "new" in archive.graph
    # The evicted behaviour must not block a fresh copy in another cell.
    again = archive.add("old2", "parse csv files with the split method!",
                        score=0.95, descriptor=(1, 1))
    assert again.added
    assert len(archive.graph) == 2


def test_add_is_atomic_when_the_graph_rejects_the_artifact():
    archive = _archive()
    archive.add("p", "parse csv", score=0.5, descriptor=(0, 0))
    with pytest.raises(ValueError):
        archive.add("p2", "x", score=0.6, descriptor=(0, 1),
                    parents=("x", "x"))
    with pytest.raises(ValueError):
        archive.add("q", "y", score=0.7, descriptor=(1, 1),
                    parents=("q",))
    assert [elite.artifact for elite in archive.archive.elites] == ["p"]
    assert len(archive.graph) == 1


def test_graph_remove_decrements_document_frequency_once():
    graph = ArtifactGraph()
    graph.add(Artifact("keep", "retry retry backoff jitter alpha"))
    graph.add(Artifact("old", "retry retry retry gamma delta"))
    assert graph._df["retry"] == 2
    assert graph.remove("old")
    assert graph._df["retry"] == 1


def test_graph_failure_restores_the_previous_elite():
    archive = _archive()
    archive.add("a", "parse csv", score=0.5, descriptor=(0, 0))
    with pytest.raises(ValueError):
        archive.add("b", "different capability", score=0.9,
                    descriptor=(0, 0), parents=("b",))
    elite = archive.archive.get((0, 0))
    assert elite is not None and elite.artifact == "a"
    assert len(archive.graph) == 1
    assert "b" not in archive.graph


def test_prepopulated_graph_seeds_dedup():
    graph = ArtifactGraph()
    graph.add(Artifact("x", "parse csv with split"))
    archive = CapabilityArchive(
        QualityDiversityArchive(bins=(2, 2)), graph)
    duplicate = archive.add("y", "parse csv with split", score=0.5,
                            descriptor=(0, 0))
    assert not duplicate.added
    assert duplicate.duplicate_of == "x"
    assert len(graph) == 1


def test_invalid_inputs_and_duplicate_ids_are_refused():
    archive = _archive()
    archive.add("a", "parse csv", score=0.5, descriptor=(0, 0))
    with pytest.raises(ValueError):
        archive.add("a", "different", score=0.6, descriptor=(0, 1))
    with pytest.raises(ValueError):
        archive.add("", "x", score=0.5, descriptor=(0, 0))
    with pytest.raises(ValueError):
        archive.add("b", "", score=0.5, descriptor=(0, 0))
    with pytest.raises(ValueError):
        archive.add("b", "x", score=float("nan"), descriptor=(0, 0))
    with pytest.raises(ValueError):
        CapabilityArchive(QualityDiversityArchive((2, 2)), ArtifactGraph(),
                          dedup_threshold=2)
    with pytest.raises(TypeError):
        CapabilityArchive(object(), ArtifactGraph())
