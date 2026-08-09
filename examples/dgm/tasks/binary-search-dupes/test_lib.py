from lib import first_index

def test_unique():
    assert first_index([1, 3, 5], 3) == 1

def test_duplicates():
    assert first_index([1, 2, 2, 2, 3], 2) == 1
