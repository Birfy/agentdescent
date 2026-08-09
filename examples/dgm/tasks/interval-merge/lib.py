def merge(intervals):
    """Merge overlapping or touching [start, end] intervals."""
    out = []
    for s, e in sorted(intervals):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
