from lib import total

def test_flat():
    assert total({"a": 1, "b": 2}) == 3

def test_nested():
    assert total({"a": 1, "b": {"c": 2, "d": 3}}) == 6
