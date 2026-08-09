from lib import indent

def test_single():
    assert indent("a") == "    a"

def test_blank_line():
    assert indent("a\n\nb") == "    a\n\n    b"
