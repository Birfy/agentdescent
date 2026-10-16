def raises(exc_type, fn, *args):
    """True only if ``fn(*args)`` raises exactly an instance of ``exc_type``."""
    try:
        fn(*args)
    except exc_type:
        return True
    except Exception as error:  # noqa: BLE001 - the wrong type is a failure
        raise AssertionError(
            f"raised {type(error).__name__}, expected {exc_type.__name__}")
    raise AssertionError(f"did not raise {exc_type.__name__}")
