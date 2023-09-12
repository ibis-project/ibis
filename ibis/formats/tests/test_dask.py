from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.formats.pandas import DaskData

dask = pytest.importorskip("dask")
dd = pytest.importorskip("dask.dataframe")

dask.config.set({"dataframe.convert-string": False})

from dask.dataframe.utils import tm  # noqa: E402


@pytest.mark.parametrize(
    ("col_data", "schema_type"),
    [
        ([True, False, False], "bool"),
        (np.int8([-3, 9, 17]), "int8"),
        (np.int16([-5, 0, 12]), "int16"),
        (np.int32([-12, 3, 25000]), "int32"),
        (np.int64([102, 67228734, -0]), "int64"),
        (np.float32([45e-3, -0.4, 99.0]), "float32"),
        (np.float64([-3e43, 43.0, 10000000.0]), "double"),
        (np.uint8([3, 0, 16]), "uint8"),
        (np.uint16([5569, 1, 33]), "uint16"),
        (np.uint32([100, 0, 6]), "uint32"),
        (np.uint64([666, 2, 3]), "uint64"),
        (
            [
                pd.Timestamp("2010-11-01 00:01:00"),
                pd.Timestamp("2010-11-01 00:02:00.1000"),
                pd.Timestamp("2010-11-01 00:03:00.300000"),
            ],
            "timestamp",
        ),
        (
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("-1 days 2 min 3us"),
                pd.Timedelta("-2 days +23:57:59.999997"),
            ],
            "interval('ns')",
        ),
        (["foo", "bar", "hello"], "string"),
        (pd.Series(["a", "b", "c", "a"]).astype("category"), dt.String()),
    ],
)
def test_schema_infer_dataframe(col_data, schema_type):
    df = dd.from_pandas(pd.DataFrame({"col": col_data}), npartitions=1)
    inferred = DaskData.infer_table(df)
    expected = ibis.schema([("col", schema_type)])
    assert inferred == expected


def test_schema_infer_exhaustive_dataframe():
    npartitions = 2
    df = dd.from_pandas(
        pd.DataFrame(
            {
                "bigint_col": np.array(
                    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype="i8"
                ),
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
                    dtype=np.float32,
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
        ),
        npartitions=npartitions,
    )

    expected = [
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

    assert DaskData.infer_table(df) == ibis.schema(expected)


def test_convert_dataframe_with_timezone():
    data = {"time": pd.date_range("2018-01-01", "2018-01-02", freq="H")}
    df = dd.from_pandas(pd.DataFrame(data), npartitions=2)
    expected = df.assign(time=df.time.dt.tz_localize("EST"))
    desired_schema = ibis.schema([("time", 'timestamp("EST")')])
    result = DaskData.convert_table(df.copy(), desired_schema)
    tm.assert_frame_equal(result.compute(), expected.compute())
