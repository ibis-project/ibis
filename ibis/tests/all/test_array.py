import functools

import pytest

import ibis


def array_test(f):
    @functools.wraps(f)
    def wrapper(backend, *args, **kwargs):
        if not backend.supports_arrays:
            pytest.skip('Backend {} does not support arrays'.format(backend))
        return f(backend, *args, **kwargs)
    return wrapper


@array_test
def test_array_concat(backend, con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr)
    assert result == [1, 2, 3, 2, 1]


@array_test
def test_array_length(backend, con):
    expr = ibis.literal([1, 2, 3]).length()
    result = con.execute(expr)
    assert result == 3
