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
        "druid",
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
