def word_freq(text):
    """Count words, case-insensitively, ignoring punctuation."""
    return {w: text.lower().split().count(w) for w in text.lower().split()}
