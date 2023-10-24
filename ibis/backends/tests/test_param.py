from __future__ import annotations

import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import _

try:
    from google.api_core.exceptions import BadRequest as GoogleBadRequest
except ImportError:
    GoogleBadRequest = None


@pytest.mark.parametrize(
    ("column", "raw_value"),
    [
        ("double_col", 0.0),
        ("double_col", 10.1),
        ("float_col", 1.1),
        ("float_col", 2.2),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_floating_scalar_parameter(backend, alltypes, df, column, raw_value):
    value = ibis.param(dt.double)
    expr = (alltypes[column] + value).name("tmp")
    expected = df[column] + raw_value
    result = expr.execute(params={value: raw_value})
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("start_string", "end_string"),
    [("2009-03-01", "2010-07-03"), ("2014-12-01", "2017-01-05")],
)
@pytest.mark.notimpl(["datafusion", "mssql", "trino", "druid"])
@pytest.mark.broken(["oracle"], raises=sa.exc.DatabaseError)
def test_date_scalar_parameter(backend, alltypes, start_string, end_string):
    start, end = ibis.param(dt.date), ibis.param(dt.date)

    col = alltypes.timestamp_col.date()
    expr = col.between(start, end).name("output")
    expected_expr = col.between(start_string, end_string).name("output")

    result = expr.execute(params={start: start_string, end: end_string})
    expected = expected_expr.execute()

    backend.assert_series_equal(result, expected)


def test_timestamp_accepts_date_literals(alltypes):
    date_string = "2009-03-01"
    param = ibis.param(dt.timestamp)
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    assert expr.compile(params=params) is not None


@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "pandas", "pyspark", "druid", "oracle"]
)
@pytest.mark.never(
    ["mysql", "sqlite", "mssql"], reason="backend will never implement array types"
)
def test_scalar_param_array(con):
    value = [1, 2, 3]
    param = ibis.param(dt.Array(dt.int64))
    result = con.execute(param.length().name("tmp"), params={param: value})
    assert result == len(value)


@pytest.mark.notimpl(["datafusion", "impala", "postgres", "pyspark", "druid", "oracle"])
@pytest.mark.never(
    ["mysql", "sqlite", "mssql"],
    reason="mysql and sqlite will never implement struct types",
)
def test_scalar_param_struct(con):
    value = dict(a=1, b="abc", c=3.0)
    param = ibis.param("struct<a: int64, b: string, c: float64>")
    result = con.execute(param["a"], params={param: value})
    assert result == value["a"]


@pytest.mark.notimpl(["datafusion", "impala", "pyspark", "polars", "druid", "oracle"])
@pytest.mark.never(
    ["mysql", "sqlite", "mssql"],
    reason="mysql and sqlite will never implement map types",
)
@pytest.mark.notyet(["bigquery"])
def test_scalar_param_map(con):
    value = {"a": "ghi", "b": "def", "c": "abc"}
    param = ibis.param(dt.Map(dt.string, dt.string))
    result = con.execute(param["b"], params={param: value})
    assert result == value["b"]


@pytest.mark.parametrize(
    ("value", "dtype", "col"),
    [
        param("0", "string", "string_col", id="string"),
        param(0, "int64", "int_col", id="int"),
        param(0.0, "float64", "double_col", id="double"),
        param(
            True,
            "bool",
            "bool_col",
            id="bool",
            marks=[pytest.mark.notimpl(["druid"])],
        ),
        param(
            "2009-01-20 01:02:03",
            "timestamp",
            "timestamp_col",
            id="string_timestamp",
            marks=[
                pytest.mark.notimpl(["druid"]),
                pytest.mark.broken(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="No matching for operator = for argument types: DATETIME, TIMESTAMP",
                ),
            ],
        ),
        param(
            datetime.date(2009, 1, 20),
            "timestamp",
            "timestamp_col",
            id="date_timestamp",
            marks=[
                pytest.mark.notimpl(["druid"]),
                pytest.mark.broken(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="No matching for operator = for argument types: DATETIME, TIMESTAMP",
                ),
            ],
        ),
        param(
            datetime.datetime(2009, 1, 20, 1, 2, 3),
            "timestamp",
            "timestamp_col",
            id="datetime_timestamp",
            marks=[
                pytest.mark.notimpl(["druid"]),
                pytest.mark.broken(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="No matching for operator = for argument types: DATETIME, TIMESTAMP",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_scalar_param(alltypes, df, value, dtype, col):
    param = ibis.param(dtype)
    expr = alltypes.filter([_[col] == param])

    result = (
        expr.execute(params={param: value}).sort_values("id").reset_index(drop=True)
    )
    expected = df.loc[df[col] == value].sort_values("id").reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    ["2009-01-20", datetime.date(2009, 1, 20), datetime.datetime(2009, 1, 20)],
    ids=["string", "date", "datetime"],
)
@pytest.mark.notimpl(["datafusion", "druid", "oracle"])
@pytest.mark.notyet(["impala"], reason="impala doesn't support dates")
def test_scalar_param_date(backend, alltypes, value):
    param = ibis.param("date")
    ds_col = alltypes.date_string_col
    month = ds_col[:2]
    day = ds_col[3:5]
    year = "20" + ds_col[6:8]
    date_col = (year + "-" + month + "-" + day).cast(param.type())

    base = alltypes.mutate(date_col=date_col)
    expr = (
        alltypes.mutate(date_col=date_col)
        .filter(lambda t: t.date_col == param)
        .drop("date_col")
    )

    result = (
        expr.execute(params={param: value}).sort_values("id").reset_index(drop=True)
    )
    df = base.execute()
    expected = (
        df.loc[df.date_col.dt.normalize() == pd.Timestamp(value).normalize()]
        .sort_values("id")
        .reset_index(drop=True)
        .drop(columns=["date_col"])
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(["mysql"], reason="no struct support")
@pytest.mark.notimpl(
    [
        "postgres",
        "datafusion",
        "clickhouse",
        "polars",
        "duckdb",
        "sqlite",
        "snowflake",
        "impala",
        "oracle",
        "pyspark",
        "mssql",
        "druid",
    ]
)
def test_scalar_param_nested(con):
    param = ibis.param("struct<x: array<struct<y: array<double>>>>")
    value = OrderedDict([("x", [OrderedDict([("y", [1.0, 2.0, 3.0])])])])
    result = con.execute(param, {param: value})
    assert pytest.approx(result["x"][0]["y"]) == np.array([1.0, 2.0, 3.0])
