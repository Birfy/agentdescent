from lib import percent

def test_normal():
    assert percent(1, 4) == 25.0

def test_zero_total():
    assert percent(0, 0) == 0.0
