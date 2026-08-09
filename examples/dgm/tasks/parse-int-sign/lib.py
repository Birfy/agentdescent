def to_int(s):
    """Parse an optionally-signed integer string."""
    return int("".join(c for c in s if c.isdigit()))
