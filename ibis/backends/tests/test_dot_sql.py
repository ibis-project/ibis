import pandas as pd
import pytest

import ibis
from ibis import util

dot_sql_notimpl = pytest.mark.notimpl(
    ["clickhouse", "datafusion", "impala", "sqlite"]
)
dot_sql_never = pytest.mark.never(
    ["dask", "pandas"],
    reason="dask and pandas do not accept SQL",
)


@dot_sql_notimpl
@dot_sql_never
@pytest.mark.parametrize(
    "dialect1",
    [
        pytest.param(dialect, id=f"{dialect}_dialect")
        for dialect in (None, "mysql", "duckdb", "pyspark", "postgres")
    ],
)
@pytest.mark.parametrize(
    "dialect2",
    [
        pytest.param(dialect, id=f"{dialect}_dialect")
        for dialect in (None, "mysql", "duckdb", "pyspark", "postgres")
    ],
)
def test_dot_sql(backend, con, dialect1, dialect2):
    alltypes = con.table("functional_alltypes")
    t = (
        alltypes.sql(
            """
            SELECT
                string_col as s,
                double_col + 1.0 AS new_col
            FROM functional_alltypes
            """,
            dialect=dialect1,
        )
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql(
            "SELECT fancy_af AS yas FROM awesome_t ORDER BY fancy_af",
            dialect=dialect2,
        )
    )

    alltypes_df = alltypes.execute()
    result = t.execute()["yas"]
    expected = (
        alltypes_df.assign(
            s=alltypes_df.string_col, new_col=alltypes_df.double_col + 1.0
        )
        .groupby("s")
        .new_col.mean()
        .rename("yas")
        .reset_index()
        .yas
    )
    backend.assert_series_equal(result, expected)


@dot_sql_notimpl
@dot_sql_never
@pytest.mark.parametrize(
    "dialect1",
    [
        pytest.param(dialect, id=f"{dialect}_dialect")
        for dialect in (None, "mysql", "duckdb", "pyspark", "postgres")
    ],
)
@pytest.mark.parametrize(
    "dialect2",
    [
        pytest.param(dialect, id=f"{dialect}_dialect")
        for dialect in (None, "mysql", "duckdb", "pyspark", "postgres")
    ],
)
def test_dot_sql_with_join(backend, con, dialect1, dialect2):
    alltypes = con.table("functional_alltypes")
    t = (
        alltypes.sql(
            """
            SELECT
                string_col as s,
                double_col + 1.0 AS new_col
            FROM functional_alltypes
            """,
            dialect=dialect1,
        )
        .alias("ft")
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql(
            """
            SELECT
                l.fancy_af AS yas,
                r.s
            FROM awesome_t AS l
            LEFT JOIN ft AS r
            ON l.s = r.s
            """,
            dialect=dialect2,
        )
        .sort_by(["s", "yas"])
    )

    alltypes_df = alltypes.execute()
    result = t.execute()

    ft = alltypes_df.assign(
        s=alltypes_df.string_col, new_col=alltypes_df.double_col + 1.0
    )
    expected = pd.merge(
        ft.groupby("s").new_col.mean().rename("yas").reset_index(),
        ft[["s"]],
        on=["s"],
        how="left",
    )[["yas", "s"]].sort_values(["s", "yas"])
    backend.assert_frame_equal(result, expected)


@dot_sql_notimpl
@dot_sql_never
def test_dot_sql_repr(con):
    alltypes = con.table("functional_alltypes")
    t = (
        alltypes.sql(
            """
            SELECT
                string_col as s,
                double_col + 1.0 AS new_col
            FROM functional_alltypes
            """
        )
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql("SELECT fancy_af AS yas FROM awesome_t ORDER BY fancy_af")
    )

    assert repr(t)


@dot_sql_notimpl
@dot_sql_never
def test_dot_sql_does_not_clobber_existing_tables(con):
    name = f"ibis_{util.guid()}"
    con.create_table(name, schema=ibis.schema(dict(a="string")))
    try:
        expr = con.table(name).sql("SELECT 1 as x FROM functional_alltypes")
        with pytest.raises(ValueError):
            expr.alias(name)
    finally:
        con.drop_table(name, force=True)
        assert name not in con.list_tables()
