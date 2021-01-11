import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from multipledispatch.conflict import ambiguities
from pandas.api.types import CategoricalDtype, DatetimeTZDtype

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir


def test_no_infer_ambiguities():
    assert not ambiguities(dt.infer.funcs)


@pytest.mark.parametrize(
    ('value', 'expected_dtype'),
    [
        # numpy types
        (np.int8(5), dt.int8),
        (np.int16(-1), dt.int16),
        (np.int32(2), dt.int32),
        (np.int64(-5), dt.int64),
        (np.uint8(5), dt.uint8),
        (np.uint16(50), dt.uint16),
        (np.uint32(500), dt.uint32),
        (np.uint64(5000), dt.uint64),
        (np.float32(5.5), dt.float32),
        (np.float32(5.5), dt.float),
        (np.float64(5.55), dt.float64),
        (np.float64(5.55), dt.double),
        (np.bool_(True), dt.boolean),
        (np.bool_(False), dt.boolean),
        (np.arange(5, dtype='int32'), dt.Array(dt.int32)),
        # pandas types
        (
            pd.Timestamp('2015-01-01 12:00:00', tz='US/Eastern'),
            dt.Timestamp('US/Eastern'),
        ),
        # TODO - add in Period/Interval/Int/Categorical
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


@pytest.mark.parametrize(
    ('numpy_dtype', 'ibis_dtype'),
    [
        (np.bool_, dt.boolean),
        (np.int8, dt.int8),
        (np.int16, dt.int16),
        (np.int32, dt.int32),
        (np.int64, dt.int64),
        (np.uint8, dt.uint8),
        (np.uint16, dt.uint16),
        (np.uint32, dt.uint32),
        (np.uint64, dt.uint64),
        (np.float16, dt.float16),
        (np.float32, dt.float32),
        (np.float64, dt.float64),
        (np.double, dt.double),
        (np.str_, dt.string),
        (np.datetime64, dt.timestamp),
        (np.timedelta64, dt.interval),
    ],
)
def test_numpy_dtype(numpy_dtype, ibis_dtype):
    assert dt.dtype(np.dtype(numpy_dtype)) == ibis_dtype


@pytest.mark.parametrize(
    ('dask_dtype', 'ibis_dtype'),
    [
        (
            DatetimeTZDtype(tz='US/Eastern', unit='ns'),
            dt.Timestamp('US/Eastern'),
        ),
        (CategoricalDtype(), dt.Category()),
    ],
)
def test_dask_dtype(dask_dtype, ibis_dtype):
    assert dt.dtype(dask_dtype) == ibis_dtype


@pytest.mark.xfail(TypeError, reason="TODO - as_value_expr - #2553")
def test_series_to_ibis_literal():
    values = [1, 2, 3, 4]
    s = dd.from_pandas(pd.Series(values), npartitions=1)

    expr = ir.as_value_expr(s)
    expected = ir.sequence(list(s))
    assert expr.equals(expected)


@pytest.mark.parametrize(
    ('col_data', 'schema_type'),
    [
        ([True, False, False], 'bool'),
        (np.int8([-3, 9, 17]), 'int8'),
        (np.int16([-5, 0, 12]), 'int16'),
        (np.int32([-12, 3, 25000]), 'int32'),
        (np.int64([102, 67228734, -0]), 'int64'),
        (np.float32([45e-3, -0.4, 99.0]), 'float'),
        (np.float64([-3e43, 43.0, 10000000.0]), 'double'),
        (np.uint8([3, 0, 16]), 'uint8'),
        (np.uint16([5569, 1, 33]), 'uint16'),
        (np.uint32([100, 0, 6]), 'uint32'),
        (np.uint64([666, 2, 3]), 'uint64'),
        (
            [
                pd.Timestamp('2010-11-01 00:01:00'),
                pd.Timestamp('2010-11-01 00:02:00.1000'),
                pd.Timestamp('2010-11-01 00:03:00.300000'),
            ],
            'timestamp',
        ),
        (
            [
                pd.Timedelta('1 days'),
                pd.Timedelta('-1 days 2 min 3us'),
                pd.Timedelta('-2 days +23:57:59.999997'),
            ],
            "interval('ns')",
        ),
        (['foo', 'bar', 'hello'], "string"),
        (pd.Series(['a', 'b', 'c', 'a']).astype('category'), dt.Category()),
    ],
)
def test_schema_infer(col_data, schema_type):
    df = dd.from_pandas(pd.DataFrame({'col': col_data}), npartitions=1)
    inferred = sch.infer(df)
    expected = ibis.schema([('col', schema_type)])
    assert inferred == expected
