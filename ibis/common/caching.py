from __future__ import annotations

import functools
from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize a function."""
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper
