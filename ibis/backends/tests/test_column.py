import operator

import pytest


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "snowflake",
        "trino",
    ]
)
def test_rowid(con):
    t = con.table('functional_alltypes')
    result = t.rowid().execute()
    # Only guarantee is that the values are unique integers
    assert result.is_unique

    # Can be named
    result = t.rowid().name("myrowid").execute()
    assert result.is_unique
    assert result.name == "myrowid"


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
