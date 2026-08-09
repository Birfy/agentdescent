from lib import plural

def test_regular():
    assert plural("cat") == "cats"

def test_consonant_y():
    assert plural("city") == "cities"
