from lib import slugify

def test_simple():
    assert slugify("Hello") == "hello"

def test_collapse():
    assert slugify("a  b!") == "a-b"
