def trim(xs, value):
    """Drop `value` from both ends of the list."""
    i = 0
    while i < len(xs) and xs[i] == value:
        i += 1
    return xs[i:]
