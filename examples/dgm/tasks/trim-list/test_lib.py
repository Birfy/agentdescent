from lib import trim

def test_head():
    assert trim([0, 1, 2], 0) == [1, 2]

def test_both_ends():
    assert trim([0, 1, 0], 0) == [1]
