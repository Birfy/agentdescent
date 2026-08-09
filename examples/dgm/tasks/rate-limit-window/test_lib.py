from lib import allowed

def test_under_limit():
    assert allowed([1.0], 2.0, 10.0, 3) is True

def test_old_calls_expire():
    assert allowed([1.0, 2.0, 3.0], 100.0, 10.0, 3) is True
