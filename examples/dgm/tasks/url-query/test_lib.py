from lib import parse_query

def test_simple():
    assert parse_query("a=1&b=2") == {"a": ["1"], "b": ["2"]}

def test_repeated_key():
    assert parse_query("a=1&a=2") == {"a": ["1", "2"]}
