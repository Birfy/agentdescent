from lib import sort_versions

def test_simple():
    assert sort_versions(["1.2", "1.1"]) == ["1.1", "1.2"]

def test_double_digit():
    assert sort_versions(["1.9", "1.10", "1.2"]) == ["1.2", "1.9", "1.10"]
