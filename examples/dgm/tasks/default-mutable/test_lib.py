from lib import add_item

def test_explicit():
    assert add_item(1, []) == [1]

def test_fresh_default():
    add_item(1)
    assert add_item(2) == [2]
