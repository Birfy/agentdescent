from lib import ordinal

def test_first():
    assert ordinal(1) == "1st"

def test_eleventh():
    assert ordinal(11) == "11th"
