import operator

import numpy as np
import pandas as pd
import pytest


@pytest.mark.notimpl(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
    ]
)
def test_rowid(con):
    t = con.table('functional_alltypes')
    result = t[t.rowid()].execute()
    first_value = 1
    expected = pd.Series(
        range(first_value, first_value + len(result)),
        dtype=np.int64,
        name='rowid',
    )
    pd.testing.assert_series_equal(result.iloc[:, 0], expected)


@pytest.mark.notimpl(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "pandas",
        "postgres",
        "pyspark",
    ]
)
def test_named_rowid(con):
    t = con.table('functional_alltypes')
    result = t[t.rowid().name('number')].execute()
    first_value = 1
    expected = pd.Series(
        range(first_value, first_value + len(result)),
        dtype=np.int64,
        name='number',
    )
    pd.testing.assert_series_equal(result.iloc[:, 0], expected)


@pytest.mark.parametrize(
    "column",
    ["string_col", "double_col", "date_string_col", "timestamp_col"],
)
@pytest.mark.notimpl(["datafusion"])
def test_distinct_column(alltypes, df, column):
    expr = alltypes[[column]].distinct()
    result = expr.execute()
    expected = df[[column]].drop_duplicates()
    assert set(result) == set(expected)


@pytest.mark.parametrize(
    ("opname", "expected"),
    [
        ("year", {2009, 2010}),
        ("month", set(range(1, 13))),
        ("day", set(range(1, 32))),
    ],
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.notyet(["impala"])
def test_date_extract_field(con, opname, expected):
    op = operator.methodcaller(opname)
    t = con.table("functional_alltypes")
    expr = t[op(t.timestamp_col.cast("date")).name("date")].distinct()
    result = expr.execute()["date"].astype(int)
    assert set(result) == expected
