def invert(d):
    """Invert to {value: [keys]}."""
    return {v: [k] for k, v in d.items()}
