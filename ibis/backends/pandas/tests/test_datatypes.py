from datetime import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as vparse
from pandas.api.types import CategoricalDtype, DatetimeTZDtype
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch


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
        (np.float64(5.55), dt.float64),
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
        param(np.array([1, 2, 3], dtype='int8'), (dt.Array(dt.int8),), id='int8'),
        param(np.array([1, 2, 3], dtype='int16'), (dt.Array(dt.int16),), id='int16'),
        param(np.array([1, 2, 3], dtype='int32'), (dt.Array(dt.int32),), id='int32'),
        param(np.array([1, 2, 3], dtype='int64'), (dt.Array(dt.int64),), id='int64'),
        param(np.array([1, 2, 3], dtype='uint8'), (dt.Array(dt.uint8),), id='uint8'),
        param(np.array([1, 2, 3], dtype='uint16'), (dt.Array(dt.uint16),), id='uint16'),
        param(np.array([1, 2, 3], dtype='uint32'), (dt.Array(dt.uint32),), id='uint32'),
        param(np.array([1, 2, 3], dtype='uint64'), (dt.Array(dt.uint64),), id='uint64'),
        param(
            np.array([1.0, 2.0, 3.0], dtype='float32'),
            (dt.Array(dt.float32),),
            id='float32',
        ),
        param(
            np.array([1.0, 2.0, 3.0], dtype='float64'),
            (dt.Array(dt.float64),),
            id='float64',
        ),
        param(
            np.array([True, False, True], dtype='bool'),
            (dt.Array(dt.boolean),),
            id='bool',
        ),
        # Implicit dtype
        # Integer array could be inferred to int64 or int32 depending on system
        param(
            np.array([1, 2, 3]),
            (dt.Array(dt.int64), dt.Array(dt.int32)),
            id='int_array',
        ),
        param(np.array([1.0, 2.0, 3.0]), (dt.Array(dt.float64),), id='float_array'),
        param(
            np.array([np.nan, np.nan, np.nan]), (dt.Array(dt.float64),), id='nan_array'
        ),
        param(np.array([True, False, True]), (dt.Array(dt.boolean),), id='bool_array'),
        param(np.array(['1', '2', '3']), (dt.Array(dt.string),), id='string_array'),
        param(
            np.array(
                [
                    pd.Timestamp('2015-01-01 12:00:00'),
                    pd.Timestamp('2015-01-02 12:00:00'),
                    pd.Timestamp('2015-01-03 12:00:00'),
                ]
            ),
            (dt.Array(dt.timestamp.copy(scale=6)),),
            marks=pytest.mark.xfail(raises=AssertionError),
            id='timestamp_array',
        ),
        # Implied from object dtype
        param(
            np.array([1, 2, 3], dtype=object),
            (dt.Array(dt.int64),),
            id='int_object_array',
        ),
        param(
            np.array([1.0, 2.0, 3.0], dtype=object),
            (dt.Array(dt.float64),),
            id='float_object_array',
        ),
        param(
            np.array([True, False, True], dtype=object),
            (dt.Array(dt.boolean),),
            id='bool_object_array',
        ),
        param(
            np.array(['1', '2', '3'], dtype=object),
            (dt.Array(dt.string),),
            id='string_object_array',
        ),
        param(
            np.array(
                [
                    pd.Timestamp('2015-01-01 12:00:00'),
                    pd.Timestamp('2015-01-02 12:00:00'),
                    pd.Timestamp('2015-01-03 12:00:00'),
                ],
                dtype=object,
            ),
            (dt.Array(dt.timestamp.copy(scale=6)),),
            marks=pytest.mark.xfail(raises=AssertionError),
            id='timestamp_object_array',
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
    ],
)
def test_numpy_dtype(numpy_dtype, ibis_dtype):
    assert dt.dtype(np.dtype(numpy_dtype)) == ibis_dtype


def test_numpy_dtype_timedelta():
    if vparse(pytest.importorskip("pyarrow").__version__) < vparse("9"):
        pytest.skip("pyarrow < 9 globally mutates the timedelta64 numpy dtype")

    assert dt.dtype(np.dtype(np.timedelta64)) == dt.interval


@pytest.mark.parametrize(
    ('pandas_dtype', 'ibis_dtype'),
    [
        (
            DatetimeTZDtype(tz='US/Eastern', unit='ns'),
            dt.Timestamp('US/Eastern'),
        ),
        (CategoricalDtype(), dt.String()),
        (pd.Series([], dtype="string").dtype, dt.String()),
    ],
)
def test_pandas_dtype(pandas_dtype, ibis_dtype):
    assert dt.dtype(pandas_dtype) == ibis_dtype


@pytest.mark.parametrize(
    ('col_data', 'schema_type'),
    [
        param([True, False, False], 'bool', id="bool"),
        param(np.array([-3, 9, 17], dtype='int8'), 'int8', id="int8"),
        param(np.array([-5, 0, 12], dtype='int16'), 'int16', id="int16"),
        param(np.array([-12, 3, 25000], dtype='int32'), 'int32', id="int32"),
        param(np.array([102, 67228734, -0], dtype='int64'), 'int64', id="int64"),
        param(np.array([45e-3, -0.4, 99.0], dtype='float32'), 'float32', id="float64"),
        param(np.array([45e-3, -0.4, 99.0], dtype='float64'), 'float64', id="float32"),
        param(
            np.array([-3e43, 43.0, 10000000.0], dtype='float64'), 'double', id="double"
        ),
        param(np.array([3, 0, 16], dtype='uint8'), 'uint8', id="uint8"),
        param(np.array([5569, 1, 33], dtype='uint16'), 'uint16', id="uint8"),
        param(np.array([100, 0, 6], dtype='uint32'), 'uint32', id="uint32"),
        param(np.array([666, 2, 3], dtype='uint64'), 'uint64', id="uint64"),
        param(
            [
                pd.Timestamp('2010-11-01 00:01:00'),
                pd.Timestamp('2010-11-01 00:02:00.1000'),
                pd.Timestamp('2010-11-01 00:03:00.300000'),
            ],
            'timestamp',
            id="timestamp",
        ),
        param(
            [
                pd.Timedelta('1 days'),
                pd.Timedelta('-1 days 2 min 3us'),
                pd.Timedelta('-2 days +23:57:59.999997'),
            ],
            "interval('ns')",
            id="interval_ns",
        ),
        param(['foo', 'bar', 'hello'], "string", id="string_list"),
        param(
            pd.Series(['a', 'b', 'c', 'a']).astype('category'),
            dt.String(),
            id="string_series",
        ),
        param(pd.Series([b'1', b'2', b'3']), dt.binary, id="string_binary"),
        # mixed-integer
        param(pd.Series([1, 2, '3']), dt.unknown, id="mixed_integer"),
        # mixed-integer-float
        param(pd.Series([1, 2, 3.0]), dt.float64, id="mixed_integer_float"),
        param(
            pd.Series([Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]),
            dt.Decimal(2, 1),
            id="decimal",
        ),
        # complex
        param(
            pd.Series([1 + 1j, 1 + 2j, 1 + 3j], dtype=object), dt.unknown, id="complex"
        ),
        param(
            pd.Series(
                [
                    pd.to_datetime('2010-11-01'),
                    pd.to_datetime('2010-11-02'),
                    pd.to_datetime('2010-11-03'),
                ]
            ),
            dt.timestamp,
            id="timestamp_to_datetime",
        ),
        param(pd.Series([time(1), time(2), time(3)]), dt.time, id="time"),
        param(
            pd.Series(
                [
                    pd.Period('2011-01'),
                    pd.Period('2011-02'),
                    pd.Period('2011-03'),
                ],
                dtype=object,
            ),
            dt.unknown,
            id="period",
        ),
        # mixed
        param(pd.Series([b'1', '2', 3.0]), dt.unknown, id="mixed"),
        # empty
        param(pd.Series([], dtype='object'), dt.null, id="empty_null"),
        param(pd.Series([], dtype="string"), dt.string, id="empty_string"),
        # array
        param(pd.Series([[1], [], None]), dt.Array(dt.int64), id="array_int64_first"),
        param(pd.Series([[], [1], None]), dt.Array(dt.int64), id="array_int64_second"),
    ],
)
def test_schema_infer(col_data, schema_type):
    df = pd.DataFrame({'col': col_data})

    inferred = sch.infer(df)
    expected = ibis.schema([('col', schema_type)])
    assert inferred == expected


def test_pyarrow_string():
    pytest.importorskip("pyarrow")

    s = pd.Series([], dtype="string[pyarrow]")
    assert dt.dtype(s.dtype) == dt.String()
