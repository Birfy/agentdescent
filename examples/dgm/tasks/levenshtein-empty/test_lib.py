from lib import distance

def test_same_length():
    assert distance("cat", "cut") == 1

def test_empty():
    assert distance("", "abc") == 3
