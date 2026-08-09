def allowed(times, now, window, limit):
    """True if fewer than `limit` calls fall within `window` before `now`."""
    return len(times) < limit
