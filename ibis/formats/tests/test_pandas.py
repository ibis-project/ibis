from __future__ import annotations

from datetime import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats.pandas import PandasData, PandasSchema, PandasType


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
        (dt.Interval("s"), np.dtype("timedelta64[s]")),
        (dt.Interval("ms"), np.dtype("timedelta64[ms]")),
        (dt.Interval("us"), np.dtype("timedelta64[us]")),
        (dt.Interval("ns"), np.dtype("timedelta64[ns]")),
        (dt.Struct({"a": dt.int8, "b": dt.string}), np.dtype("object")),
    ],
)
def test_dtype_to_pandas(pandas_type, ibis_type):
    assert PandasType.from_ibis(ibis_type) == pandas_type


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
    series = pd.Series([], dtype=f"{pandas_type}[pyarrow]")
    assert PandasType.to_ibis(series.dtype) == ibis_type


def test_dtype_from_pandas_arrow_string_dtype():
    series = pd.Series([], dtype="string[pyarrow]")
    assert PandasType.to_ibis(series.dtype) == dt.String()


def test_dtype_from_pandas_arrow_list_dtype():
    series = pd.Series([], dtype=pd.ArrowDtype(pa.list_(pa.string())))
    assert PandasType.to_ibis(series.dtype) == dt.Array(dt.string)


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
            pd.DatetimeTZDtype(tz="US/Eastern", unit="ns"),
            dt.Timestamp("US/Eastern"),
        ),
        (pd.CategoricalDtype(), dt.String()),
        (pd.Series([], dtype="string").dtype, dt.String()),
    ],
    ids=str,
)
def test_dtype_from_nullable_extension_dtypes(pandas_type, ibis_type):
    assert PandasType.to_ibis(pandas_type) == ibis_type


def test_schema_to_pandas():
    ibis_schema = sch.Schema(
        {
            "a": dt.int64,
            "b": dt.string,
            "c": dt.boolean,
            "d": dt.float64,
        }
    )
    pandas_schema = PandasSchema.from_ibis(ibis_schema)

    assert pandas_schema == [
        ("a", np.dtype("int64")),
        ("b", np.dtype("object")),
        ("c", np.dtype("bool")),
        ("d", np.dtype("float64")),
    ]


def test_schema_from_pandas():
    pandas_schema = [
        ("a", np.dtype("int64")),
        ("b", np.dtype("str")),
        ("c", np.dtype("bool")),
        ("d", np.dtype("float64")),
    ]

    ibis_schema = PandasSchema.to_ibis(pandas_schema)
    assert ibis_schema == sch.Schema(
        {
            "a": dt.int64,
            "b": dt.string,
            "c": dt.boolean,
            "d": dt.float64,
        }
    )


def test_schema_from_dataframe():
    df = pd.DataFrame(
        {
            "bigint_col": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="i8"),
            "bool_col": np.array(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    None,
                    True,
                    False,
                    True,
                    False,
                ],
                dtype=np.bool_,
            ),
            "bool_obj_col": np.array(
                [
                    True,
                    False,
                    np.nan,
                    False,
                    True,
                    np.nan,
                    True,
                    np.nan,
                    True,
                    False,
                ],
                dtype=np.object_,
            ),
            "date_string_col": [
                "11/01/10",
                None,
                "11/01/10",
                "11/01/10",
                "11/01/10",
                "11/01/10",
                "11/01/10",
                "11/01/10",
                "11/01/10",
                "11/01/10",
            ],
            "double_col": np.array(
                [
                    0.0,
                    10.1,
                    np.nan,
                    30.299999999999997,
                    40.399999999999999,
                    50.5,
                    60.599999999999994,
                    70.700000000000003,
                    80.799999999999997,
                    90.899999999999991,
                ],
                dtype=np.float64,
            ),
            "float_col": np.array(
                [
                    np.nan,
                    1.1000000238418579,
                    2.2000000476837158,
                    3.2999999523162842,
                    4.4000000953674316,
                    5.5,
                    6.5999999046325684,
                    7.6999998092651367,
                    8.8000001907348633,
                    9.8999996185302734,
                ],
                dtype="f4",
            ),
            "int_col": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="i4"),
            "month": [11, 11, 11, 11, 2, 11, 11, 11, 11, 11],
            "smallint_col": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="i2"),
            "string_col": [
                "0",
                "1",
                None,
                "double , whammy",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ],
            "timestamp_col": [
                pd.Timestamp("2010-11-01 00:00:00"),
                None,
                pd.Timestamp("2010-11-01 00:02:00.100000"),
                pd.Timestamp("2010-11-01 00:03:00.300000"),
                pd.Timestamp("2010-11-01 00:04:00.600000"),
                pd.Timestamp("2010-11-01 00:05:00.100000"),
                pd.Timestamp("2010-11-01 00:06:00.150000"),
                pd.Timestamp("2010-11-01 00:07:00.210000"),
                pd.Timestamp("2010-11-01 00:08:00.280000"),
                pd.Timestamp("2010-11-01 00:09:00.360000"),
            ],
            "tinyint_col": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="i1"),
            "year": [
                2010,
                2010,
                2010,
                2010,
                2010,
                2009,
                2009,
                2009,
                2009,
                2009,
            ],
        }
    )

    expected = sch.Schema.from_tuples(
        [
            ("bigint_col", dt.int64),
            ("bool_col", dt.boolean),
            ("bool_obj_col", dt.boolean),
            ("date_string_col", dt.string),
            ("double_col", dt.float64),
            ("float_col", dt.float32),
            ("int_col", dt.int32),
            ("month", dt.int64),
            ("smallint_col", dt.int16),
            ("string_col", dt.string),
            ("timestamp_col", dt.timestamp),
            ("tinyint_col", dt.int8),
            ("year", dt.int64),
        ]
    )

    assert PandasData.infer_table(df) == expected
    assert sch.infer(df) == expected


def test_schema_from_dataframe_with_array_column():
    df = pd.DataFrame(
        {
            # Columns containing np.arrays
            "int64_arr_col": [
                np.array([0, 1], dtype="int64"),
                np.array([3, 4], dtype="int64"),
            ],
            "string_arr_col": [np.array(["0", "1"]), np.array(["3", "4"])],
            # Columns containing pd.Series
            "int64_series_col": [
                pd.Series([0, 1], dtype="int64"),
                pd.Series([3, 4], dtype="int64"),
            ],
            "string_series_col": [
                pd.Series(["0", "1"]),
                pd.Series(["3", "4"]),
            ],
        }
    )

    expected = sch.Schema.from_tuples(
        [
            ("int64_arr_col", dt.Array(dt.int64)),
            ("string_arr_col", dt.Array(dt.string)),
            ("int64_series_col", dt.Array(dt.int64)),
            ("string_series_col", dt.Array(dt.string)),
        ]
    )

    assert PandasData.infer_table(df) == expected
    assert sch.infer(df) == expected


@pytest.mark.parametrize(
    ("col_data", "schema_type"),
    [
        param([True, False, False], "bool", id="bool"),
        param(np.array([-3, 9, 17], dtype="int8"), "int8", id="int8"),
        param(np.array([-5, 0, 12], dtype="int16"), "int16", id="int16"),
        param(np.array([-12, 3, 25000], dtype="int32"), "int32", id="int32"),
        param(np.array([102, 67228734, -0], dtype="int64"), "int64", id="int64"),
        param(np.array([45e-3, -0.4, 99.0], dtype="float32"), "float32", id="float64"),
        param(np.array([45e-3, -0.4, 99.0], dtype="float64"), "float64", id="float32"),
        param(
            np.array([-3e43, 43.0, 10000000.0], dtype="float64"), "double", id="double"
        ),
        param(np.array([3, 0, 16], dtype="uint8"), "uint8", id="uint8"),
        param(np.array([5569, 1, 33], dtype="uint16"), "uint16", id="uint8"),
        param(np.array([100, 0, 6], dtype="uint32"), "uint32", id="uint32"),
        param(np.array([666, 2, 3], dtype="uint64"), "uint64", id="uint64"),
        param(
            [
                pd.Timestamp("2010-11-01 00:01:00"),
                pd.Timestamp("2010-11-01 00:02:00.1000"),
                pd.Timestamp("2010-11-01 00:03:00.300000"),
            ],
            "timestamp",
            id="timestamp",
        ),
        param(
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("-1 days 2 min 3us"),
                pd.Timedelta("-2 days +23:57:59.999997"),
            ],
            "interval('ns')",
            id="interval_ns",
        ),
        param(["foo", "bar", "hello"], "string", id="string_list"),
        param(
            pd.Series(["a", "b", "c", "a"]).astype("category"),
            dt.String(),
            id="string_series",
        ),
        param(pd.Series([b"1", b"2", b"3"]), dt.binary, id="string_binary"),
        # mixed-integer
        param(pd.Series([1, 2, "3"]), dt.unknown, id="mixed_integer"),
        # mixed-integer-float
        param(pd.Series([1, 2, 3.0]), dt.float64, id="mixed_integer_float"),
        param(
            pd.Series([Decimal("1.0"), Decimal("2.0"), Decimal("3.0")]),
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
                    pd.to_datetime("2010-11-01"),
                    pd.to_datetime("2010-11-02"),
                    pd.to_datetime("2010-11-03"),
                ]
            ),
            dt.timestamp,
            id="timestamp_to_datetime",
        ),
        param(pd.Series([time(1), time(2), time(3)]), dt.time, id="time"),
        param(
            pd.Series(
                [
                    pd.Period("2011-01"),
                    pd.Period("2011-02"),
                    pd.Period("2011-03"),
                ],
                dtype=object,
            ),
            dt.unknown,
            id="period",
        ),
        # mixed
        param(pd.Series([b"1", "2", 3.0]), dt.unknown, id="mixed"),
        # empty
        param(pd.Series([], dtype="object"), dt.null, id="empty_null"),
        param(pd.Series([], dtype="string"), dt.string, id="empty_string"),
        # array
        param(pd.Series([[1], [], None]), dt.Array(dt.int64), id="array_int64_first"),
        param(pd.Series([[], [1], None]), dt.Array(dt.int64), id="array_int64_second"),
    ],
)
def test_schema_from_various_dataframes(col_data, schema_type):
    df = pd.DataFrame({"col": col_data})

    inferred = PandasData.infer_table(df)
    expected = sch.Schema({"col": schema_type})
    assert inferred == expected


def test_convert_dataframe_with_timezone():
    data = {"time": pd.date_range("2018-01-01", "2018-01-02", freq="H")}
    df = expected = pd.DataFrame(data).assign(
        time=lambda df: df.time.dt.tz_localize("EST")
    )
    desired_schema = ibis.schema(dict(time='timestamp("EST")'))
    result = PandasData.convert_table(df.copy(), desired_schema)
    tm.assert_frame_equal(expected, result)
