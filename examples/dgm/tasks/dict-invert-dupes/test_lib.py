from lib import invert

def test_unique():
    assert invert({"a": 1}) == {1: ["a"]}

def test_shared_value():
    assert invert({"a": 1, "b": 1}) == {1: ["a", "b"]}
