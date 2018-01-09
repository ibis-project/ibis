import functools

import pytest

import ibis
import ibis.tests.util as tu


def array_test(f):
    @functools.wraps(f)
    def wrapper(backend, *args, **kwargs):
        if not backend.supports_arrays:
            pytest.skip('Backend {} does not support arrays'.format(backend))
        return f(backend, *args, **kwargs)
    return wrapper


def direct_array_operation_test(f):
    @functools.wraps(array_test(f))
    def wrapper(backend, *args, **kwargs):
        if not backend.supports_arrays_outside_of_select:
            pytest.skip(
                'Backend {} does not support operations directly on '
                'arrays as column cells'.format(backend)
            )
        return f(backend, *args, **kwargs)
    return wrapper


@tu.skip_if_invalid_operation
@direct_array_operation_test
@pytest.mark.backend
def test_array_concat(backend, con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr)
    assert result == [1, 2, 3, 2, 1]


@tu.skip_if_invalid_operation
@direct_array_operation_test
@pytest.mark.backend
def test_array_length(backend, con):
    expr = ibis.literal([1, 2, 3]).length()
    result = con.execute(expr)
    assert result == 3
