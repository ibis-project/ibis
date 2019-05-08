import functools

import pytest

import ibis
import ibis.common as com
import ibis.util as util


def assert_equal(left, right):
    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right), 'Comparing schemas: \n%s !=\n%s' % (
            repr(left),
            repr(right),
        )
    else:
        assert left.equals(right), 'Objects unequal: {0}\nvs\n{1}'.format(
            repr(left), repr(right)
        )


def skipif_unsupported(f):
    @functools.wraps(f)
    def wrapper(backend, *args, **kwargs):
        try:
            return f(backend, *args, **kwargs)
        except (
            com.OperationNotDefinedError,
            com.UnsupportedOperationError,
            com.UnsupportedBackendType,
            NotImplementedError,
        ) as e:
            pytest.skip('{} using {}'.format(e, str(backend)))

    return wrapper


def skipif_backend(skip_backend):
    def wrapped(f):
        @functools.wraps(f)
        def wrapper(backend, *args, **kwargs):
            if isinstance(backend, skip_backend):
                pytest.skip('Skipping {} test'.format(str(backend)))
            else:
                return f(backend, *args, **kwargs)

        return wrapper

    return wrapped


def skipifnot_backend(skip_backend):
    def wrapped(f):
        @functools.wraps(f)
        def wrapper(backend, *args, **kwargs):
            if not isinstance(backend, skip_backend):
                pytest.skip('Skipping {} test'.format(str(backend)))
            else:
                return f(backend, *args, **kwargs)

        return wrapper

    return wrapped
