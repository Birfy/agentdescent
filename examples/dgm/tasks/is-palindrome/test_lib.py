from lib import is_palindrome

def test_plain():
    assert is_palindrome("aba") is True

def test_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama") is True
