from lib import join

def test_plain():
    assert join("a", "b") == "a/b"

def test_trailing_slash():
    assert join("a/", "b") == "a/b"
