from lib import as_bool

def test_true():
    assert as_bool("true") is True

def test_false_string():
    assert as_bool("false") is False
