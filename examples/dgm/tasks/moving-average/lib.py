def moving_avg(xs, n):
    """Averages over each FULL window of length n."""
    return [sum(xs[max(0, i - n + 1):i + 1]) / min(i + 1, n) for i in range(len(xs))]
