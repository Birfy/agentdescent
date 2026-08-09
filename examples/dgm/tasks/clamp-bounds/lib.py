def clamp(x, lo, hi):
    """Clamp x into the inclusive range [lo, hi]."""
    if x < lo:
        return lo
    if x >= hi:
        return hi - 1
    return x
