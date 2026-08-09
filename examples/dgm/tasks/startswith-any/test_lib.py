from lib import starts_with_any

def test_first():
    assert starts_with_any("hello", ["he", "xx"]) is True

def test_second():
    assert starts_with_any("hello", ["xx", "he"]) is True
