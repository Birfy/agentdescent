from kvtools.ranges import in_window, overlaps, window_len


def test_in_window_includes_lower_edge():
    assert in_window(0, 0, 5)


def test_in_window_inside():
    assert in_window(4, 0, 5)


def test_in_window_excludes_upper_edge():
    assert not in_window(5, 0, 5)


def test_in_window_outside():
    assert not in_window(-1, 0, 5)


def test_window_len_basic():
    assert window_len(2, 5) == 3


def test_window_len_empty():
    assert window_len(4, 4) == 0


def test_window_len_inverted_is_empty():
    assert window_len(6, 2) == 0


def test_overlaps_proper():
    assert overlaps(0, 5, 3, 8)


def test_overlaps_adjacent_is_disjoint():
    assert not overlaps(0, 5, 5, 9)


def test_overlaps_contained():
    assert overlaps(0, 10, 2, 3)


def test_overlaps_apart():
    assert not overlaps(0, 2, 4, 6)
