from lib import parse_row

def test_plain():
    assert parse_row("a,b,c") == ["a", "b", "c"]

def test_quoted_delimiter():
    assert parse_row('a,"b,c",d') == ["a", "b,c", "d"]
