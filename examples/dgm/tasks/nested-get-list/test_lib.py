from lib import get_path

def test_dict():
    assert get_path({"a": {"b": 1}}, ["a", "b"]) == 1

def test_list_index():
    assert get_path({"a": [10, 20]}, ["a", "1"]) == 20
