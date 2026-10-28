"""Retrieval must find the useful artifact, not the whole library."""

import pytest

from agentdescent.artifact_graph import Artifact, ArtifactGraph


def _graph(**kwargs):
    graph = ArtifactGraph(**kwargs)
    graph.extend([
        Artifact("skill-pytest", "pytest fixture isolation for database tests",
                 kind="skill", score=0.8, tags=("pytest", "db")),
        Artifact("skill-pytest-2", "pytest fixtures and parametrization",
                 kind="skill", score=0.4, tags=("pytest",)),
        Artifact("prompt-react", "react hooks and component state",
                 kind="prompt", score=0.9, tags=("frontend",)),
        Artifact("cjk", "引用来源后再下结论", kind="skill", tags=("cjk",)),
    ])
    return graph


def test_search_ranks_the_specific_artifact_first():
    graph = _graph()
    hits = graph.search("pytest fixture isolation", k=3)
    assert hits[0].artifact.artifact_id == "skill-pytest"
    assert "isolation" in hits[0].matched_terms
    assert all(hit.score > 0 for hit in hits)


def test_kind_tags_exclude_and_min_score_filters():
    graph = _graph()
    assert {hit.artifact.artifact_id for hit in graph.search("pytest", kind="skill")} == {
        "skill-pytest", "skill-pytest-2"}
    assert [hit.artifact.artifact_id for hit in graph.search("pytest", tags=("db",))] == [
        "skill-pytest"]
    assert graph.search("pytest", exclude=("skill-pytest",))[0].artifact.artifact_id == "skill-pytest-2"
    assert graph.search("pytest", min_score=10.0) == ()


def test_trust_weight_breaks_a_lexical_tie():
    graph = ArtifactGraph(trust_weight=1.0)
    graph.add(Artifact("low", "same body", score=0.1))
    graph.add(Artifact("high", "same body", score=0.9))
    assert graph.search("same")[0].artifact.artifact_id == "high"


def test_compose_reports_coverage_budget_and_skips_oversized_artifacts():
    graph = _graph()
    composed = graph.compose("pytest fixture isolation", k=3, max_chars=80)
    assert composed.hits
    assert composed.text
    assert 0.0 <= composed.coverage <= 1.0
    assert "isolation" in composed.text
    assert all(len(hit.artifact.text) <= 80 for hit in composed.hits)
    assert graph.compose("pytest", k=1, max_chars=5).skipped


def test_missing_terms_are_explicit():
    composed = _graph().compose("pytest quantum", k=2)
    assert "quantum" in composed.missing_terms
    assert composed.coverage < 1.0


def test_unicode_query_works_and_mixed_scripts_stay_searchable():
    graph = _graph()
    hits = graph.search("引用来源", k=1)
    assert hits[0].artifact.artifact_id == "cjk"
    graph.add(Artifact("mixed", "使用pytest夹具隔离数据库", kind="skill"))
    assert "mixed" in {hit.artifact.artifact_id for hit in graph.search("pytest", kind="skill")}
    assert graph.search("夹具", kind="skill")[0].artifact.artifact_id == "mixed"


def test_graph_edges_roots_leaves_and_report():
    graph = ArtifactGraph()
    graph.add(Artifact("root", "base skill"))
    graph.add(Artifact("child", "derived skill", parents=("root",)))
    graph.add(Artifact("leaf", "final skill", parents=("child",)))
    assert graph.children("root") == ("child",)
    assert graph.parents("child") == ("root",)
    assert graph.neighbors("child") == ("leaf", "root")
    assert graph.roots() == ("root",)
    assert graph.leaves() == ("leaf",)
    report = graph.report()
    assert report.artifacts == 3 and report.edges == 2
    assert "2 edges" in report.detail
    assert report.kinds["artifact"] == 3


def test_search_is_deterministic_on_ties():
    graph = ArtifactGraph()
    graph.add(Artifact("b", "same words"))
    graph.add(Artifact("a", "same words"))
    assert [hit.artifact.artifact_id for hit in graph.search("same")] == ["a", "b"]


def test_score_strings_are_coerced_and_bare_tags_are_refused():
    artifact = Artifact("a", "text", score="0.5")
    assert artifact.score == 0.5
    with pytest.raises(ValueError):
        Artifact("a", "text", tags="pytest")
    with pytest.raises(ValueError):
        Artifact("a", "text", parents="parent")


def test_self_parent_and_cycles_are_refused():
    graph = ArtifactGraph()
    with pytest.raises(ValueError):
        graph.add(Artifact("a", "text", parents=("a",)))
    graph.add(Artifact("a", "text"))
    graph.add(Artifact("b", "text", parents=("a",)))
    with pytest.raises(ValueError):
        graph.add(Artifact("a", "other", parents=("b",)))


def test_compose_backfills_when_the_top_hit_exceeds_the_budget():
    graph = ArtifactGraph()
    graph.add(Artifact("long", "pytest " + "x" * 100, score=0.9))
    graph.add(Artifact("short", "pytest", score=0.1))
    composed = graph.compose("pytest", k=1, max_chars=20)
    assert [hit.artifact.artifact_id for hit in composed.hits] == ["short"]


def test_duplicate_ids_and_invalid_inputs_are_refused():
    graph = ArtifactGraph()
    graph.add(Artifact("a", "text"))
    with pytest.raises(ValueError):
        graph.add(Artifact("a", "other"))
    with pytest.raises(ValueError):
        Artifact("", "text")
    with pytest.raises(ValueError):
        Artifact("a", "")
    with pytest.raises(ValueError):
        Artifact("a", "text", kind="")
    with pytest.raises(ValueError):
        Artifact("a", "text", parents=("p", "p"))
    with pytest.raises(ValueError):
        graph.search("")
    with pytest.raises(ValueError):
        graph.search("text", k=0)
    with pytest.raises(ValueError):
        graph.neighbors("a", direction="sideways")
    with pytest.raises(KeyError):
        graph.children("missing")
    with pytest.raises(ValueError):
        ArtifactGraph(trust_weight=-1)
    with pytest.raises(ValueError):
        graph.compose("text", max_chars=0)
