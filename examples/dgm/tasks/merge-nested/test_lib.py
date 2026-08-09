from lib import deep_merge

def test_flat():
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

def test_nested():
    assert deep_merge({"x": {"a": 1}}, {"x": {"b": 2}}) == {"x": {"a": 1, "b": 2}}
