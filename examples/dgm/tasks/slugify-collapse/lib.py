import re

def slugify(s):
    """Lowercase, non-alnum -> single hyphen, no leading/trailing hyphen."""
    return re.sub(r"[^a-z0-9]", "-", s.lower())
