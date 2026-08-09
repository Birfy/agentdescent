from lib import unique_by

def test_no_dupes():
    assert unique_by([1, 2], lambda x: x) == [1, 2]

def test_by_key():
    assert unique_by([1, 11, 2], lambda x: x % 10) == [1, 2]
