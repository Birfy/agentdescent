from lib import normalise

def test_internal():
    assert normalise("a   b") == "a b"

def test_trims():
    assert normalise("  a b  ") == "a b"
