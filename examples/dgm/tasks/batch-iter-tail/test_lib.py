from lib import batches

def test_exact():
    assert batches([1, 2], 2) == [[1, 2]]

def test_partial_tail():
    assert batches([1, 2, 3], 2) == [[1, 2], [3]]
