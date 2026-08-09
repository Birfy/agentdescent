def flatten(xs):
    """Fully flatten arbitrarily nested lists."""
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out
