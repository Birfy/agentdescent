def starts_with_any(s, prefixes):
    """True if s starts with ANY of the prefixes."""
    return s.startswith(prefixes[0]) if prefixes else False
