from lib import escape

def test_ampersand():
    assert escape("a & b") == "a &amp; b"

def test_angles_not_double_escaped():
    assert escape("<x>") == "&lt;x&gt;"
