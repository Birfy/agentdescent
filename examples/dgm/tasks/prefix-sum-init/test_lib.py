from lib import prefix_sums

def test_empty():
    assert prefix_sums([]) == [0]

def test_values():
    assert prefix_sums([1, 2]) == [0, 1, 3]
