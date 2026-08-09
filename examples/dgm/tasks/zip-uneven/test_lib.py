from lib import zip_fill

def test_equal():
    assert zip_fill([1, 2], ["a", "b"]) == [(1, "a"), (2, "b")]

def test_uneven():
    assert zip_fill([1, 2, 3], ["a"], fill="-") == [(1, "a"), (2, "-"), (3, "-")]
