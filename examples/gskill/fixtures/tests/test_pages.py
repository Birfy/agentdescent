from kvtools.pages import bucket_count, chunk_count, pages_needed


def test_pages_needed_exact():
    assert pages_needed(4096, 4096) == 1


def test_pages_needed_partial_counts():
    assert pages_needed(4097, 4096) == 2


def test_pages_needed_one_byte():
    assert pages_needed(1, 4096) == 1


def test_pages_needed_zero():
    assert pages_needed(0, 4096) == 0


def test_chunk_count_exact():
    assert chunk_count(8, 4) == 2


def test_chunk_count_partial_counts():
    assert chunk_count(9, 4) == 3


def test_chunk_count_smaller_than_one_chunk():
    assert chunk_count(1, 4) == 1


def test_chunk_count_zero():
    assert chunk_count(0, 4) == 0


def test_bucket_count_exact():
    assert bucket_count(10, 5) == 2


def test_bucket_count_just_over_half_rounds_up():
    assert bucket_count(11, 5) == 3


def test_bucket_count_just_under_half_still_rounds_up():
    assert bucket_count(6, 5) == 2 and bucket_count(1, 5) == 1


def test_bucket_count_zero():
    assert bucket_count(0, 5) == 0
