from lib import merge

def test_overlap():
    assert merge([(1, 3), (2, 4)]) == [(1, 4)]

def test_touching():
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]
