import math


def pages_needed(nbytes, page_size):
    return -(-nbytes // page_size)


def chunk_count(total, chunk):
    return (total + chunk - 1) // chunk


def bucket_count(n_items, per_bucket):
    return math.ceil(n_items / per_bucket)
