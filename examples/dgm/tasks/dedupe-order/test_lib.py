from lib import dedupe

def test_single():
    assert dedupe([70]) == [70]

def test_keeps_order():
    # Integers, not strings, and deliberately so: `list(set(...))` on strings
    # reorders according to `PYTHONHASHSEED`, so on roughly one seed in fifteen
    # the buggy implementation came out in the right order and this test passed.
    # CI drew such a seed. Small ints hash to themselves, so the wrong order here
    # is the same wrong order on every interpreter and every run.
    assert dedupe([40, 10, 30, 10, 20]) == [40, 10, 30, 20]
