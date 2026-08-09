from lib import join_parts

def test_all_present():
    assert join_parts(["a", "b"]) == "a, b"

def test_skips_empty():
    assert join_parts(["a", "", "b"]) == "a, b"
