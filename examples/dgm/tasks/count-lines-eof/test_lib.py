from lib import count_lines

def test_trailing_newline():
    assert count_lines("a\nb\n") == 2

def test_no_trailing():
    assert count_lines("a\nb") == 2
