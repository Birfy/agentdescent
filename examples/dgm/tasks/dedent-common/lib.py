def dedent(text):
    """Remove the longest common leading whitespace."""
    return "\n".join(line[4:] if line.startswith("    ") else line
                     for line in text.split("\n"))
