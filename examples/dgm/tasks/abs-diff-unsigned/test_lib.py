from lib import diff

def test_positive():
    assert diff(5, 3) == 2

def test_negative():
    assert diff(3, 5) == 2
