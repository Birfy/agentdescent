def in_window(x, lo, hi):
    return lo <= x < hi


def window_len(lo, hi):
    return max(0, hi - lo)


def overlaps(a_lo, a_hi, b_lo, b_hi):
    return a_lo < b_hi and b_lo < a_hi
