import pytest

import ibis.expr.datatypes as dt
import ibis.tests.util as tu


@pytest.mark.parametrize(('column', 'expected_dtype'), [
    pytest.mark.xfail(
        ('Unnamed: 0', 'int64'), reason='special characters in field name'
    ),
    ('id', 'int32'),
    ('bool_col', 'boolean'),
    ('tinyint_col', 'int8'),
    ('smallint_col', 'int16'),
    ('int_col', 'int32'),
    ('bigint_col', 'int64'),
    ('float_col', 'float32'),
    ('double_col', 'float64'),
    ('date_string_col', 'string'),
    ('string_col', 'string'),
    ('timestamp_col', 'timestamp')
])
@tu.skipif_unsupported
def test_exact_alltypes(backend, alltypes, df, column, expected_dtype):
    # datatype inferred by ibis or explicitly passed in backends.py
    expr = alltypes.limit(10)[column]
    dtype = expr.type()

    # pandas result to check correct ibis -> pandas conversion
    col = df[column]
    result = expr.execute()

    # a couple of backends don't support smaller datatypes
    if expected_dtype in backend.unsupported_datatypes:
        # so We check that the expected dtype is upcastable to the actual one
        assert dt.dtype(expected_dtype).castable(dtype)
    else:
        # otherwise We check for exact match
        assert dt.dtype(expected_dtype) == dtype

    # check that the resulting pandas dtype corresponds to the ibis one
    assert dtype.to_pandas() == col.dtype == result.dtype
