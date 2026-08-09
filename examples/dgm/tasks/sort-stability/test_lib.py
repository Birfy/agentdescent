from lib import sort_by_first

def test_distinct():
    assert sort_by_first([(2, "b"), (1, "a")]) == [(1, "a"), (2, "b")]

def test_stable():
    assert sort_by_first([(1, "z"), (1, "a")]) == [(1, "z"), (1, "a")]
