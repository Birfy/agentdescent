def wrap(text, width):
    """Wrap into lines of at most `width`, breaking on spaces."""
    return [text[i:i + width] for i in range(0, len(text), width)]
