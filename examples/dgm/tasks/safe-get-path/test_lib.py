from lib import get_in

def test_present():
    assert get_in({"a": {"b": 1}}, "a.b") == 1

def test_missing():
    assert get_in({"a": {}}, "a.b", default=0) == 0
