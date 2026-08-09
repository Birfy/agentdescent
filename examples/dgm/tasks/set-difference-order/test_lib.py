from lib import without

def test_single():
    assert without([70], []) == [70]

def test_order():
    # Integers rather than strings: see the note in `dedupe-order`. With strings
    # this assertion passed on about one hash seed in five.
    assert without([40, 10, 30, 20], [10]) == [40, 30, 20]
