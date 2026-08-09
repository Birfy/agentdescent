from lib import word_freq

def test_simple():
    assert word_freq("a b a") == {"a": 2, "b": 1}

def test_punctuation():
    assert word_freq("hi, hi!") == {"hi": 2}
