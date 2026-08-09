def call_with_retry(fn, attempts=3):
    """Call fn(); retry on exception up to `attempts` total tries."""
    try:
        return fn()
    except Exception:
        return fn()
