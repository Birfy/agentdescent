from lib import to_roman

def test_simple():
    assert to_roman(6) == "VI"

def test_subtractive():
    assert to_roman(4) == "IV"
