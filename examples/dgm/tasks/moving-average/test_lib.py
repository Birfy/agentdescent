from lib import moving_avg

def test_window_one():
    assert moving_avg([1, 2, 3], 1) == [1.0, 2.0, 3.0]

def test_full_windows_only():
    assert moving_avg([1, 2, 3], 2) == [1.5, 2.5]
