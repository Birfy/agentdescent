from lib import flatten

def test_one_level():
    assert flatten([1, [2, 3]]) == [1, 2, 3]

def test_deep():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]
