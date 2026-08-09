from lib import max_by

def test_values():
    assert max_by([1, 3, 2], lambda x: x) == 3

def test_empty():
    assert max_by([], lambda x: x) is None
