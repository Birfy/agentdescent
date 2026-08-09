from lib import mean

def test_values():
    assert mean([1, 2, 3]) == 2

def test_empty():
    assert mean([]) == 0.0
