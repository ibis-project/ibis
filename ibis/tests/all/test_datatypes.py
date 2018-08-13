import pytest

import numpy as np

import ibis.expr.datatypes as dt
import ibis.tests.util as tu


@pytest.mark.parametrize(('column', 'ibis_dtype', 'pandas_dtype'), [
    ('index', dt.Int64(nullable=True), np.int64),
    ('Unnamed: 0', dt.Int64(nullable=True), np.int64),
    ('id', dt.Int32(nullable=True), np.int32),
    ('bool_col', dt.Boolean(nullable=True), np.bool_),
    ('tinyint_col', dt.Int8(nullable=True), np.int8),
    ('smallint_col', dt.Int16(nullable=True), np.int16),
    ('int_col', dt.Int32(nullable=True), np.int32),
    ('bigint_col', dt.Int64(nullable=True), np.int64),
    ('float_col', dt.Float32(nullable=True), np.float32),
    ('double_col', dt.Float64(nullable=True), np.float64),
    ('date_string_col', dt.String(nullable=True), np.object_),
    ('string_col', dt.String(nullable=True), np.object_),
    ('timestamp_col', dt.Timestamp(nullable=True), np.dtype('datetime64[ns]'))
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
