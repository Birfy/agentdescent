from lib import expand

def test_single():
    assert expand("7") == [7]

def test_inclusive():
    assert expand("1-4") == [1, 2, 3, 4]
