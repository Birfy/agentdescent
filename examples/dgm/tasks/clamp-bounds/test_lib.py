from lib import clamp

def test_inside():
    assert clamp(5, 0, 10) == 5

def test_at_upper():
    assert clamp(10, 0, 10) == 10
