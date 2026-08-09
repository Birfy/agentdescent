from lib import split_args

def test_plain():
    assert split_args("a b") == ["a", "b"]

def test_quoted():
    assert split_args('a "b c"') == ["a", "b c"]
