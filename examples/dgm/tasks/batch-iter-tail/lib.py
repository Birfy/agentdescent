def batches(xs, n):
    """Yield lists of n; the final one may be shorter."""
    out = []
    for i in range(0, len(xs), n):
        b = xs[i:i + n]
        if len(b) == n:
            out.append(b)
    return out
