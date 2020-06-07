import numpy as np
import pandas as pd
import pytest
from multipledispatch.conflict import ambiguities

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.compat import CategoricalDtype, DatetimeTZDtype


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


def test_series_to_ibis_literal():
    values = [1, 2, 3, 4]
    s = pd.Series(values)

    expr = ir.as_value_expr(s)
    expected = ir.sequence(list(s))
    assert expr.equals(expected)


def test_dtype_bool():
    df = pd.DataFrame({'col': [True, False, False]})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'boolean')])
    assert inferred == expected


def test_dtype_int8():
    df = pd.DataFrame({'col': np.int8([-3, 9, 17])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'int8')])
    assert inferred == expected


def test_dtype_int16():
    df = pd.DataFrame({'col': np.int16([-5, 0, 12])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'int16')])
    assert inferred == expected


def test_dtype_int32():
    df = pd.DataFrame({'col': np.int32([-12, 3, 25000])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'int32')])
    assert inferred == expected


def test_dtype_int64():
    df = pd.DataFrame({'col': np.int64([102, 67228734, -0])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'int64')])
    assert inferred == expected


def test_dtype_float32():
    df = pd.DataFrame({'col': np.float32([45e-3, -0.4, 99.0])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'float')])
    assert inferred == expected


def test_dtype_float64():
    df = pd.DataFrame({'col': np.float64([-3e43, 43.0, 10000000.0])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'double')])
    assert inferred == expected


def test_dtype_uint8():
    df = pd.DataFrame({'col': np.uint8([3, 0, 16])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'uint8')])
    assert inferred == expected


def test_dtype_uint16():
    df = pd.DataFrame({'col': np.uint16([5569, 1, 33])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'uint16')])
    assert inferred == expected


def test_dtype_uint32():
    df = pd.DataFrame({'col': np.uint32([100, 0, 6])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'uint32')])
    assert inferred == expected


def test_dtype_uint64():
    df = pd.DataFrame({'col': np.uint64([666, 2, 3])})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'uint64')])
    assert inferred == expected


def test_dtype_datetime64():
    df = pd.DataFrame(
        {
            'col': [
                pd.Timestamp('2010-11-01 00:01:00'),
                pd.Timestamp('2010-11-01 00:02:00.1000'),
                pd.Timestamp('2010-11-01 00:03:00.300000'),
            ]
        }
    )
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'timestamp')])
    assert inferred == expected


def test_dtype_timedelta64():
    df = pd.DataFrame(
        {
            'col': [
                pd.Timedelta('1 days'),
                pd.Timedelta('-1 days 2 min 3us'),
                pd.Timedelta('-2 days +23:57:59.999997'),
            ]
        }
    )
    inferred = sch.infer(df)
    expected = ibis.schema([('col', "interval('ns')")])
    assert inferred == expected


def test_dtype_string():
    df = pd.DataFrame({'col': ['foo', 'bar', 'hello']})
    inferred = sch.infer(df)
    expected = ibis.schema([('col', 'string')])
    assert inferred == expected


def test_dtype_categorical():
    df = pd.DataFrame({'col': ['a', 'b', 'c', 'a']}, dtype='category')
    inferred = sch.infer(df)
    expected = ibis.schema([('col', dt.Category())])
    assert inferred == expected


def test_type_casting():
    df = pd.DataFrame(
        {
            'id_col': [123, 1234, 456, 0, 1],
            'date_col': [
                '2020-11-12',
                '1996-06-07',
                '1985-09-09',
                '1985-09-09',
                '1985-09-09',
            ],
            'string_col': [str(i) for i in range(5)],
        },
    )
    df_table = ibis.pandas.from_dataframe(df, "test")
    id_col = df_table.id_col
    date_col = df_table.date_col
    df_from_ibis = df_table.projection(
        [
            date_col.cast("date").name("date_col"),
            df_table.string_col.cast("string").name("string_col"),
            id_col.cast("int8").name("id_col_int8"),
            id_col.cast("int16").name("id_col_int16"),
            id_col.cast("int32").name("id_col_int32"),
            id_col.cast("int64").name("id_col_int64"),
            id_col.cast("float").name("id_col_float"),
            id_col.cast("float16").name("id_col_float16"),
            id_col.cast("float32").name("id_col_float32"),
            id_col.cast("float64").name("id_col_float64"),
            id_col.cast("bool").name("id_col_bool"),
            date_col.cast("timestamp").name("timestamp_col"),
        ],
    ).execute()
    df["id_col_int8"] = df["id_col"].astype("int8")
    df["id_col_int16"] = df["id_col"].astype("int16")
    df["id_col_int32"] = df["id_col"].astype("int32")
    df["id_col_int64"] = df["id_col"].astype("int64")
    df["id_col_float"] = df["id_col"].astype("float32")
    df["id_col_float16"] = df["id_col"].astype("float16")
    df["id_col_float32"] = df["id_col"].astype("float32")
    df["id_col_float64"] = df["id_col"].astype("float64")
    df["id_col_bool"] = df["id_col"].astype("bool")
    df["timestamp_col"] = df["date_col"].astype("datetime64")
    df["date_col"] = df["date_col"].astype("datetime64")
    df["string_col"] = df["string_col"].astype("object")
    df = df.drop(columns=["id_col"])
    pd.testing.assert_frame_equal(df_from_ibis, df)
