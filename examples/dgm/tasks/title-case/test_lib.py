from lib import title_case

def test_plain():
    assert title_case("hello world") == "Hello World"

def test_apostrophe():
    assert title_case("o'brien's cat") == "O'brien's Cat"
