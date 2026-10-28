"""Lineage must answer ancestry, impact, and cycle questions without guessing."""

import pytest

from agentdescent.lineage import (
    LineageGraph,
    LineageNode,
    build_lineage,
    lineage_report,
)


def _graph():
    graph = LineageGraph()
    graph.add_record("root", run_id="r0", score=0.1)
    graph.add_record("a", parent_ids=["root"], run_id="r1", score=0.2)
    graph.add_record("b", parent_ids=["root"], run_id="r2", score=0.3)
    graph.add_record("merge", parent_ids=["a", "b"], run_id="r3", score=0.4)
    graph.add_record("leaf", parent_ids=["merge"], run_id="r4", score=0.5)
    return graph


def test_ancestors_descendants_and_paths():
    graph = _graph()
    assert graph.parents("merge") == ("a", "b")
    assert graph.children("root") == ("a", "b")
    assert set(graph.ancestors("leaf")) == {"merge", "a", "b", "root"}
    assert set(graph.descendants("root")) == {"a", "b", "merge", "leaf"}
    assert graph.path("root", "leaf") == ("root", "a", "merge", "leaf") or \
        graph.path("root", "leaf") == ("root", "b", "merge", "leaf")
    assert graph.path("leaf", "root") is None
    assert graph.ancestors("leaf", depth=1) == ("merge",)
    assert graph.descendants("root", depth=1) == ("a", "b")


def test_roots_leaves_and_impact():
    graph = _graph()
    assert graph.roots() == ("root",)
    assert graph.leaves() == ("leaf",)
    assert graph.impact("root") == 4
    assert graph.impact("a") == 2
    assert graph.impact("leaf") == 0


def test_unknown_parents_are_external_not_fatal():
    graph = LineageGraph()
    graph.add_record("child", parent_ids=["not-in-this-store"])
    assert graph.roots() == ("child",)
    assert graph.known_parents("child") == ()
    assert graph.ancestors("child") == ("not-in-this-store",)


def test_cycles_are_reported_and_topology_refuses():
    graph = LineageGraph()
    graph.add_record("a", parent_ids=["b"])
    graph.add_record("b", parent_ids=["a"])
    cycles = graph.cycles()
    assert cycles
    assert cycles[0][0] == cycles[0][-1]
    assert graph.depth() == -1
    with pytest.raises(ValueError):
        graph.topological_order()
    report = lineage_report(graph)
    assert not report.healthy
    assert report.depth == -1


def test_topological_order_is_parents_first_and_deterministic():
    graph = _graph()
    order = graph.topological_order()
    assert order.index("root") < order.index("a") < order.index("merge") < order.index("leaf")
    assert order == graph.topological_order()
    assert graph.depth() == 3


def test_duplicate_and_malformed_nodes_are_refused():
    graph = LineageGraph()
    graph.add_record("a")
    with pytest.raises(ValueError):
        graph.add_record("a")
    with pytest.raises(ValueError):
        LineageNode("a", parent_ids=("a",))
    with pytest.raises(ValueError):
        LineageNode("a", parent_ids=("b", "b"))
    with pytest.raises(ValueError):
        LineageNode("a", score=float("nan"))
    with pytest.raises(KeyError):
        graph.children("missing")


def test_build_lineage_accepts_parents_alias_and_metadata():
    graph = build_lineage([
        {"artifact_id": "a", "run_id": "r1", "score": 0.2},
        {"artifact_id": "b", "parents": ["a"], "slot": "selection",
         "metadata": {"note": "merged"}},
    ])
    assert len(graph) == 2
    assert graph.parents("b") == ("a",)
    assert graph["b"].metadata["note"] == "merged"
    data = graph.to_dict()
    assert data["nodes"][1]["parent_ids"] == ["a"]


def test_missing_artifact_id_is_named():
    with pytest.raises(ValueError):
        build_lineage([{"run_id": "r1"}])


def test_report_counts_edges_and_largest_impact():
    report = lineage_report(_graph())
    assert report.nodes == 5
    assert report.edges == 5
    assert report.roots == ("root",)
    assert report.leaves == ("leaf",)
    assert report.max_impact == 4
    assert report.max_impact_id == "root"
    assert "depth=3" in report.detail


def test_path_is_exact_and_deterministic():
    graph = _graph()
    assert graph.path("root", "leaf") == ("root", "a", "merge", "leaf")


def test_child_added_before_parent_is_indexed_and_long_chains_do_not_recurse():
    import sys

    graph = LineageGraph()
    graph.add_record("child", parent_ids=["parent"])
    graph.add_record("parent")
    assert graph.children("parent") == ("child",)
    assert graph.roots() == ("parent",)

    chain = LineageGraph()
    length = sys.getrecursionlimit() + 200
    chain.add_record("n0")
    for index in range(1, length):
        chain.add_record(f"n{index}", parent_ids=[f"n{index - 1}"])
    assert chain.depth() == length - 1
    assert len(chain.ancestors(f"n{length - 1}")) == length - 1
    assert chain.cycles() == ()
    report = lineage_report(chain)
    assert report.nodes == length
    assert report.depth == length - 1
    assert report.max_impact == length - 1


def test_parent_coercion_is_consistent_and_strings_are_one_parent():
    graph = build_lineage([
        {"artifact_id": 1},
        {"artifact_id": 2, "parent_ids": [1]},
        {"artifact_id": "3", "parent_ids": "2"},
        {"artifact_id": "4", "parent_ids": []},
        {"artifact_id": "5", "parents": ["4"]},
    ])
    assert graph.parents("2") == ("1",)
    assert graph.children("1") == ("2",)
    assert graph.parents("3") == ("2",)
    assert graph.parents("4") == ()
    assert graph.parents("5") == ("4",)


def test_external_parents_are_reported_and_counted():
    graph = LineageGraph()
    graph.add_record("child", parent_ids=["external"])
    graph.add_record("other", parent_ids=["also-external"])
    assert graph.external_parents("child") == ("external",)
    assert graph.externals() == ("also-external", "external")
    report = lineage_report(graph)
    assert report.external == 2
    assert "external=2" in report.detail


def test_nodes_are_hashable_while_metadata_is_not_part_of_identity():
    graph = LineageGraph()
    graph.add_record("a", metadata={"note": "x"})
    first = graph["a"]
    second = LineageNode("a", metadata={"note": "y"})
    assert hash(first) == hash(second)
    assert first == second
