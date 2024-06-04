from __future__ import annotations

import hypothesis as h

import ibis.tests.strategies as its
from ibis.backends.datafusion import as_nullable


def is_nullable(dtype):
    if dtype.is_struct():
        return all(map(is_nullable, dtype.values()))
    elif dtype.is_array():
        return is_nullable(dtype.value_type)
    elif dtype.is_map():
        return is_nullable(dtype.key_type) and is_nullable(dtype.value_type)
    else:
        return dtype.nullable is True


@h.given(its.all_dtypes())
def test_as_nullable(dtype):
    nullable_dtype = as_nullable(dtype)
    assert nullable_dtype.nullable is True
    assert is_nullable(nullable_dtype)
