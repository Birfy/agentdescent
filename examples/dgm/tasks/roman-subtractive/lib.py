VALUES = [(1000,"M"),(500,"D"),(100,"C"),(50,"L"),(10,"X"),(5,"V"),(1,"I")]

def to_roman(n):
    """Integer -> Roman numeral, using subtractive pairs."""
    out = ""
    for v, sym in VALUES:
        while n >= v:
            out += sym
            n -= v
    return out
