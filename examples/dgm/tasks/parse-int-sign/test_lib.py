from lib import to_int

def test_plain():
    assert to_int("42") == 42

def test_negative():
    assert to_int("-7") == -7
