from lib import wrap

def test_short():
    assert wrap("ab", 5) == ["ab"]

def test_word_boundary():
    assert wrap("aa bb cc", 5) == ["aa bb", "cc"]
