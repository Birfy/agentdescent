from lib import to_snake

def test_simple():
    assert to_snake("fooBar") == "foo_bar"

def test_acronym():
    assert to_snake("parseHTTPResponse") == "parse_http_response"
