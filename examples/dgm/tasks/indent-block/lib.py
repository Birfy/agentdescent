def indent(text, prefix="    "):
    """Prefix every non-empty line."""
    return "\n".join(prefix + line for line in text.split("\n"))
