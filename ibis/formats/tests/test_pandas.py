import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats.pandas import (
    dtype_from_pandas,
    dtype_to_pandas,
    schema_from_pandas,
    schema_to_pandas,
)


@pytest.mark.parametrize(
    ("ibis_type", "pandas_type"),
    [
        (dt.string, np.dtype("object")),
        (dt.int8, np.dtype("int8")),
        (dt.int16, np.dtype("int16")),
        (dt.int32, np.dtype("int32")),
        (dt.int64, np.dtype("int64")),
        (dt.uint8, np.dtype("uint8")),
        (dt.uint16, np.dtype("uint16")),
        (dt.uint32, np.dtype("uint32")),
        (dt.uint64, np.dtype("uint64")),
        (dt.float16, np.dtype("float16")),
        (dt.float32, np.dtype("float32")),
        (dt.float64, np.dtype("float64")),
        (dt.boolean, np.dtype("bool")),
        (dt.date, np.dtype("datetime64[ns]")),
        (dt.time, np.dtype("timedelta64[ns]")),
        (dt.timestamp, np.dtype("datetime64[ns]")),
        (dt.Interval('s'), np.dtype("timedelta64[s]")),
        (dt.Interval('ms'), np.dtype("timedelta64[ms]")),
        (dt.Interval('us'), np.dtype("timedelta64[us]")),
        (dt.Interval('ns'), np.dtype("timedelta64[ns]")),
        (dt.Struct({"a": dt.int8, "b": dt.string}), np.dtype("object")),
    ],
)
def test_dtype_to_pandas(pandas_type, ibis_type):
    assert dtype_to_pandas(ibis_type) == pandas_type


@pytest.mark.parametrize(
    ("pandas_type", "ibis_type"),
    [
        ("string", dt.string),
        ("int8", dt.int8),
        ("int16", dt.int16),
        ("int32", dt.int32),
        ("int64", dt.int64),
        ("uint8", dt.uint8),
        ("uint16", dt.uint16),
        ("uint32", dt.uint32),
        ("uint64", dt.uint64),
        pytest.param(
            "list<item: string>",
            dt.Array(dt.string),
            marks=pytest.mark.xfail(
                reason="list repr in dtype Series argument doesn't work",
                raises=TypeError,
            ),
            id="list_string",
        ),
    ],
    ids=str,
)
def test_dtype_from_pandas_arrow_dtype(pandas_type, ibis_type):
    ser = pd.Series([], dtype=f"{pandas_type}[pyarrow]")
    assert dtype_from_pandas(ser.dtype) == ibis_type


def test_dtype_from_pandas_arrow_string_dtype():
    ser = pd.Series([], dtype="string[pyarrow]")
    assert dtype_from_pandas(ser.dtype) == dt.String()


def test_dtype_from_pandas_arrow_list_dtype():
    ser = pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.string())))
    assert dtype_from_pandas(ser.dtype) == dt.Array(dt.string)


@pytest.mark.parametrize(
    ("pandas_type", "ibis_type"),
    [
        (pd.StringDtype(), dt.string),
        (pd.Int8Dtype(), dt.int8),
        (pd.Int16Dtype(), dt.int16),
        (pd.Int32Dtype(), dt.int32),
        (pd.Int64Dtype(), dt.int64),
        (pd.UInt8Dtype(), dt.uint8),
        (pd.UInt16Dtype(), dt.uint16),
        (pd.UInt32Dtype(), dt.uint32),
        (pd.UInt64Dtype(), dt.uint64),
        (pd.BooleanDtype(), dt.boolean),
        (
            pd.DatetimeTZDtype(tz='US/Eastern', unit='ns'),
            dt.Timestamp('US/Eastern'),
        ),
        (pd.CategoricalDtype(), dt.String()),
        (pd.Series([], dtype="string").dtype, dt.String()),
    ],
    ids=str,
)
def test_dtype_from_nullable_extension_dtypes(pandas_type, ibis_type):
    assert dtype_from_pandas(pandas_type) == ibis_type


def test_schema_to_pandas():
    ibis_schema = sch.Schema(
        {
            'a': dt.int64,
            'b': dt.string,
            'c': dt.boolean,
            'd': dt.float64,
        }
    )
    pandas_schema = schema_to_pandas(ibis_schema)

    assert pandas_schema == [
        ('a', np.dtype('int64')),
        ('b', np.dtype('object')),
        ('c', np.dtype('bool')),
        ('d', np.dtype('float64')),
    ]


def test_schema_from_pandas():
    pandas_schema = [
        ('a', np.dtype('int64')),
        ('b', np.dtype('str')),
        ('c', np.dtype('bool')),
        ('d', np.dtype('float64')),
    ]

    ibis_schema = schema_from_pandas(pandas_schema)
    assert ibis_schema == sch.Schema(
        {
            'a': dt.int64,
            'b': dt.string,
            'c': dt.boolean,
            'd': dt.float64,
        }
    )
