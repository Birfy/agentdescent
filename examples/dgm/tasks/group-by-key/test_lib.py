from lib import group_by

def test_distinct():
    assert group_by([1, 2], lambda x: x) == {1: [1], 2: [2]}

def test_collision():
    assert group_by([1, 3, 2], lambda x: x % 2) == {1: [1, 3], 0: [2]}
