from ibis.util import cache_readonly


@cache_readonly
def my_cached_function():
    """Ye old docstring"""
    return 42


def test_cache_readonly_preserves_docstring():
    assert my_cached_function.__doc__ == 'Ye old docstring'
