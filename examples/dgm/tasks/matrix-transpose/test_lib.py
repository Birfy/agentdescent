from lib import transpose

def test_square():
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

def test_rectangular():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
