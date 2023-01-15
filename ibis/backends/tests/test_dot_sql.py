import pandas as pd
import pytest
from pytest import param

import ibis
from ibis import _, util

table_dot_sql_notimpl = pytest.mark.notimpl(["bigquery", "clickhouse", "impala"])
dot_sql_notimpl = pytest.mark.notimpl(["datafusion"])
dot_sql_notyet = pytest.mark.notyet(
    ["snowflake"],
    reason="snowflake column names are case insensitive",
)
dot_sql_never = pytest.mark.never(
    ["dask", "pandas", "polars"],
    reason="dask and pandas do not accept SQL",
)

pytestmark = pytest.mark.xdist_group("dot_sql")


@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.parametrize(
    "schema",
    [
        param(None, id="implicit_schema"),
        param({"s": "string", "new_col": "double"}, id="explicit_schema"),
    ],
)
def test_con_dot_sql(backend, con, schema):
    alltypes = con.table("functional_alltypes")
    # pull out the quoted name
    name = alltypes.op().name
    t = (
        con.sql(
            f"""
            SELECT
                string_col as s,
                double_col + 1.0 AS new_col
            FROM {name}
            """,
            schema=schema,
        )
        .group_by("s")  # group by a column from SQL
        .aggregate(yas=lambda t: t.new_col.max())
        .order_by("yas")
    )

    alltypes_df = alltypes.execute()
    result = t.execute()["yas"]
    expected = (
        alltypes_df.assign(
            s=alltypes_df.string_col, new_col=alltypes_df.double_col + 1.0
        )
        .groupby("s")
        .new_col.max()
        .rename("yas")
        .sort_values()
        .reset_index(drop=True)
    )
    backend.assert_series_equal(result, expected)


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.notimpl(["trino"])
def test_table_dot_sql(backend, con):
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
        .sql("SELECT fancy_af AS yas FROM awesome_t")
        .order_by(_.yas)
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


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.notimpl(["trino"])
def test_table_dot_sql_with_join(backend, con):
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
            """
        )
        .order_by(["s", "yas"])
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


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.notimpl(["trino"])
def test_table_dot_sql_repr(con):
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


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_never
def test_table_dot_sql_does_not_clobber_existing_tables(con):
    name = f"ibis_{util.guid()}"
    con.create_table(name, schema=ibis.schema(dict(a="string")))
    try:
        expr = con.table(name).sql("SELECT 1 as x FROM functional_alltypes")
        with pytest.raises(ValueError):
            expr.alias(name)
    finally:
        con.drop_table(name, force=True)
        assert name not in con.list_tables()


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.notimpl(["trino"])
def test_dot_sql_alias_with_params(backend, alltypes, df):
    t = alltypes
    x = t.select(x=t.string_col + " abc").alias("foo")
    result = x.execute()
    expected = df.string_col.add(" abc").rename("x")
    backend.assert_series_equal(result.x, expected)
