def median(xs):
    """Median; the mean of the two middles when len is even."""
    s = sorted(xs)
    return s[len(s) // 2 - (1 if len(s) % 2 == 0 else 0)]
