from kvtools.store import dig, first_or_none, lookup


def test_lookup_present():
    assert lookup({"a": 1}, "a") == 1


def test_lookup_missing_is_none():
    assert lookup({"a": 1}, "b") is None


def test_lookup_falsy_value_is_kept():
    assert lookup({"a": 0}, "a") == 0


def test_first_or_none_value():
    assert first_or_none([7, 8]) == 7


def test_first_or_none_empty():
    assert first_or_none([]) is None


def test_first_or_none_falsy_first():
    assert first_or_none([0, 1]) == 0


def test_dig_nested():
    assert dig({"a": {"b": 2}}, "a.b") == 2


def test_dig_missing_leaf():
    assert dig({"a": {"b": 2}}, "a.c") is None


def test_dig_through_non_mapping():
    assert dig({"a": 1}, "a.b") is None


def test_dig_missing_root():
    assert dig({}, "x") is None
