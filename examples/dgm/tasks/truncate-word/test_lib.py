from lib import truncate

def test_short():
    assert truncate("hi", 10) == "hi"

def test_word_boundary():
    assert truncate("hello big world", 11) == "hello big"
