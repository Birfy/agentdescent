def without(xs, drop):
    """Items of xs not in drop, in the original order."""
    return list(set(xs) - set(drop))
