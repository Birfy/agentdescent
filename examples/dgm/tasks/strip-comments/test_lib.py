from lib import strip_comment

def test_plain():
    assert strip_comment("x = 1  # set x") == "x = 1"

def test_hash_in_string():
    assert strip_comment('c = "#fff"  # colour') == 'c = "#fff"'
