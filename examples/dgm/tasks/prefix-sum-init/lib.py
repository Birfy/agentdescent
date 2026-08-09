def prefix_sums(xs):
    """Cumulative sums, starting with 0."""
    out, run = [0], 0
    for x in xs:
        run += x
        out.append(run)
    return out[1:] if xs else out
