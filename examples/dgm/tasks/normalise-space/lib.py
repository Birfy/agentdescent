import re

def normalise(s):
    """Collapse internal whitespace and trim the ends."""
    return re.sub(r"\s+", " ", s)
