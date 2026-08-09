from lib import dedent

def test_four_spaces():
    assert dedent("    a\n    b") == "a\nb"

def test_two_spaces():
    assert dedent("  a\n  b") == "a\nb"
