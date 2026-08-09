from lib import strip_prefix

def test_absent():
    assert strip_prefix("hello", "xyz") == "hello"

def test_char_set():
    assert strip_prefix("aabbcc", "ab") == "aabbcc"
