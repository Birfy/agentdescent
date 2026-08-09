import re

def to_snake(s):
    """CamelCase -> snake_case; keep acronyms together."""
    return re.sub(r"([A-Z])", r"_\1", s).lstrip("_").lower()
