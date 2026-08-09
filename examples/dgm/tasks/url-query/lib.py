def parse_query(q):
    """Parse a=1&b=2 into {key: [values]}."""
    out = {}
    for pair in q.split("&"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k] = [v]
    return out
