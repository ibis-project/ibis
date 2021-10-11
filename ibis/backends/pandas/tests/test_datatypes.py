from datetime import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from multipledispatch.conflict import ambiguities
from pandas.api.types import CategoricalDtype, DatetimeTZDtype

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch


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
        # pandas types
        (
            pd.Timestamp('2015-01-01 12:00:00', tz='US/Eastern'),
            dt.Timestamp('US/Eastern'),
        ),
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


@pytest.mark.parametrize(
    ('value', 'expected_dtypes'),
    [
        # Explicitly-defined dtype
        (np.array([1, 2, 3], dtype='int8'), (dt.Array(dt.int8),)),
        (np.array([1, 2, 3], dtype='int16'), (dt.Array(dt.int16),)),
        (np.array([1, 2, 3], dtype='int32'), (dt.Array(dt.int32),)),
        (np.array([1, 2, 3], dtype='int64'), (dt.Array(dt.int64),)),
        (np.array([1, 2, 3], dtype='uint8'), (dt.Array(dt.uint8),)),
        (np.array([1, 2, 3], dtype='uint16'), (dt.Array(dt.uint16),)),
        (np.array([1, 2, 3], dtype='uint32'), (dt.Array(dt.uint32),)),
        (np.array([1, 2, 3], dtype='uint64'), (dt.Array(dt.uint64),)),
        (np.array([1.0, 2.0, 3.0], dtype='float32'), (dt.Array(dt.float32),)),
        (np.array([1.0, 2.0, 3.0], dtype='float64'), (dt.Array(dt.float64),)),
        (np.array([True, False, True], dtype='bool'), (dt.Array(dt.boolean),)),
        # Implicit dtype
        # Integer array could be inferred to int64 or int32 depending on system
        (np.array([1, 2, 3]), (dt.Array(dt.int64), dt.Array(dt.int32))),
        (np.array([1.0, 2.0, 3.0]), (dt.Array(dt.float64),)),
        (np.array([np.nan, np.nan, np.nan]), (dt.Array(dt.float64),)),
        (np.array([True, False, True]), (dt.Array(dt.boolean),)),
        (np.array(['1', '2', '3']), (dt.Array(dt.string),)),
        (
            np.array(
                [
                    pd.Timestamp('2015-01-01 12:00:00'),
                    pd.Timestamp('2015-01-02 12:00:00'),
                    pd.Timestamp('2015-01-03 12:00:00'),
                ]
            ),
            (dt.Array(dt.timestamp),),
        ),
        # Implied from object dtype
        (np.array([1, 2, 3], dtype=object), (dt.Array(dt.int64),)),
        (np.array([1.0, 2.0, 3.0], dtype=object), (dt.Array(dt.float64),)),
        (np.array([True, False, True], dtype=object), (dt.Array(dt.boolean),)),
        (np.array(['1', '2', '3'], dtype=object), (dt.Array(dt.string),)),
        (
            np.array(
                [
                    pd.Timestamp('2015-01-01 12:00:00'),
                    pd.Timestamp('2015-01-02 12:00:00'),
                    pd.Timestamp('2015-01-03 12:00:00'),
                ],
                dtype=object,
            ),
            (dt.Array(dt.timestamp),),
        ),
    ],
)
def test_infer_np_array(value, expected_dtypes):
    assert dt.infer(value) in expected_dtypes


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
    ('pandas_dtype', 'ibis_dtype'),
    [
        (
            DatetimeTZDtype(tz='US/Eastern', unit='ns'),
            dt.Timestamp('US/Eastern'),
        ),
        (CategoricalDtype(), dt.Category()),
    ],
)
def test_pandas_dtype(pandas_dtype, ibis_dtype):
    assert dt.dtype(pandas_dtype) == ibis_dtype


@pytest.mark.parametrize(
    ('col_data', 'schema_type'),
    [
        ([True, False, False], 'bool'),
        (np.array([-3, 9, 17], dtype='int8'), 'int8'),
        (np.array([-5, 0, 12], dtype='int16'), 'int16'),
        (np.array([-12, 3, 25000], dtype='int32'), 'int32'),
        (np.array([102, 67228734, -0], dtype='int64'), 'int64'),
        (np.array([45e-3, -0.4, 99.0], dtype='float32'), 'float'),
        (np.array([-3e43, 43.0, 10000000.0], dtype='float64'), 'double'),
        (np.array([3, 0, 16], dtype='uint8'), 'uint8'),
        (np.array([5569, 1, 33], dtype='uint16'), 'uint16'),
        (np.array([100, 0, 6], dtype='uint32'), 'uint32'),
        (np.array([666, 2, 3], dtype='uint64'), 'uint64'),
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
        (pd.Series([b'1', b'2', b'3']), dt.string),
        # mixed-integer
        (pd.Series([1, 2, '3']), dt.binary),
        # mixed-integer-float
        (pd.Series([1, 2, 3.0]), dt.float64),
        (
            pd.Series([Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]),
            dt.float64,
        ),
        # complex
        (pd.Series([1 + 1j, 1 + 2j, 1 + 3j], dtype=object), dt.binary),
        (
            pd.Series(
                [
                    pd.to_datetime('2010-11-01'),
                    pd.to_datetime('2010-11-02'),
                    pd.to_datetime('2010-11-03'),
                ]
            ),
            dt.timestamp,
        ),
        (pd.Series([time(1), time(2), time(3)]), dt.time),
        (
            pd.Series(
                [
                    pd.Period('2011-01'),
                    pd.Period('2011-02'),
                    pd.Period('2011-03'),
                ],
                dtype=object,
            ),
            dt.binary,
        ),
        # mixed
        (pd.Series([b'1', '2', 3.0]), dt.binary),
        # empty
        (pd.Series([], dtype='object'), dt.binary),
    ],
)
def test_schema_infer(col_data, schema_type):
    df = pd.DataFrame({'col': col_data})

    inferred = sch.infer(df)
    expected = ibis.schema([('col', schema_type)])
    assert inferred == expected
