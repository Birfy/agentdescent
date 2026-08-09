from lib import lookup

def test_exact():
    assert lookup({"a": 1}, "a") == 1

def test_different_case():
    assert lookup({"Accept": 1}, "accept") == 1
