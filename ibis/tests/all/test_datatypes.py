import pytest

import numpy as np

import ibis.expr.datatypes as dt
import ibis.tests.util as tu


@pytest.mark.parametrize(('column', 'ibis_dtype', 'pandas_dtype'), [
    ('string_col', dt.String(nullable=True), np.object_),
    ('double_col', dt.Float64(nullable=True), np.float64),
])
@tu.skipif_unsupported
def test_columns_are_typecorrect(backend, alltypes, df, column, ibis_dtype,
                                 pandas_dtype):
    expr = alltypes.limit(10)[column]
    col = df[column]
    result = expr.execute()

    assert expr.type() == ibis_dtype
    assert col.dtype == pandas_dtype
    assert result.dtype == pandas_dtype
