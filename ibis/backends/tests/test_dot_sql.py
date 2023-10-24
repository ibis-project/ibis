from __future__ import annotations

import pandas as pd
import pytest
import sqlglot as sg
from pytest import param

import ibis
from ibis import _
from ibis.backends.base import _IBIS_TO_SQLGLOT_DIALECT, _get_backend_names

try:
    from polars.exceptions import ComputeError as PolarsComputeError
except ImportError:
    PolarsComputeError = None

table_dot_sql_notimpl = pytest.mark.notimpl(["bigquery", "impala", "druid"])
dot_sql_notimpl = pytest.mark.notimpl(["datafusion"])
dot_sql_notyet = pytest.mark.notyet(
    ["snowflake", "oracle"],
    reason="snowflake and oracle column names are case insensitive",
)
dot_sql_never = pytest.mark.never(
    ["dask", "pandas"], reason="dask and pandas do not accept SQL"
)

pytestmark = [pytest.mark.xdist_group("dot_sql")]

_NAMES = {
    "bigquery": "ibis_gbq_testing.functional_alltypes",
}

try:
    from clickhouse_connect.driver.exceptions import DatabaseError
except ImportError:
    DatabaseError = None


@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.parametrize(
    "schema",
    [
        param(
            None,
            id="implicit_schema",
            marks=[
                pytest.mark.notimpl(["druid"]),
                pytest.mark.notyet(["polars"], raises=PolarsComputeError),
            ],
        ),
        param({"s": "string", "new_col": "double"}, id="explicit_schema"),
    ],
)
def test_con_dot_sql(backend, con, schema):
    alltypes = con.table("functional_alltypes")
    # pull out the quoted name
    name = _NAMES.get(con.name, alltypes.op().name)
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
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
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
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
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
                r.s AS s
            FROM awesome_t AS l
            LEFT JOIN ft AS r
            ON l.s = r.s
            """  # clickhouse needs the r.s AS s, otherwise the column name is returned as r.s
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
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
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
@pytest.mark.notimpl(["oracle"])
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
def test_table_dot_sql_does_not_clobber_existing_tables(con, temp_table):
    t = con.create_table(temp_table, schema=ibis.schema(dict(a="string")))
    expr = t.sql("SELECT 1 as x FROM functional_alltypes")
    with pytest.raises(ValueError):
        expr.alias(temp_table)


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_never
@pytest.mark.notimpl(["oracle"])
def test_dot_sql_alias_with_params(backend, alltypes, df):
    t = alltypes
    x = t.select(x=t.string_col + " abc").alias("foo")
    result = x.execute()
    expected = df.string_col.add(" abc").rename("x")
    backend.assert_series_equal(result.x, expected)


@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_never
@pytest.mark.notimpl(["oracle"])
def test_dot_sql_reuse_alias_with_different_types(backend, alltypes, df):
    foo1 = alltypes.select(x=alltypes.string_col).alias("foo")
    foo2 = alltypes.select(x=alltypes.bigint_col).alias("foo")
    expected1 = df.string_col.rename("x")
    expected2 = df.bigint_col.rename("x")
    backend.assert_series_equal(foo1.x.execute(), expected1)
    backend.assert_series_equal(foo2.x.execute(), expected2)


_NO_SQLGLOT_DIALECT = {"pandas", "dask", "druid", "flink"}
no_sqlglot_dialect = sorted(
    param(backend, marks=pytest.mark.xfail) for backend in _NO_SQLGLOT_DIALECT
)


@pytest.mark.parametrize(
    "dialect",
    [
        *sorted(_get_backend_names() - _NO_SQLGLOT_DIALECT),
        *no_sqlglot_dialect,
    ],
)
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@table_dot_sql_notimpl
@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
def test_table_dot_sql_transpile(backend, alltypes, dialect, df):
    name = "foo2"
    foo = alltypes.select(x=_.int_col + 1).alias(name)
    expr = sg.select("x").from_(sg.table(name, quoted=True))
    dialect = _IBIS_TO_SQLGLOT_DIALECT.get(dialect, dialect)
    sqlstr = expr.sql(dialect=dialect, pretty=True)
    dot_sql_expr = foo.sql(sqlstr, dialect=dialect)
    result = dot_sql_expr.execute()
    expected = df.int_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@pytest.mark.parametrize(
    "dialect",
    [
        *sorted(_get_backend_names() - {"pyspark", *_NO_SQLGLOT_DIALECT}),
        *no_sqlglot_dialect,
    ],
)
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@pytest.mark.notyet(["druid"], raises=ValueError)
@pytest.mark.notyet(["snowflake", "bigquery"])
@pytest.mark.notyet(
    ["oracle"], strict=False, reason="only works with backends that quote everything"
)
@dot_sql_notimpl
@dot_sql_never
def test_con_dot_sql_transpile(backend, con, dialect, df):
    t = sg.table("functional_alltypes")
    foo = sg.select(sg.alias(sg.column("int_col") + 1, "x")).from_(t)
    dialect = _IBIS_TO_SQLGLOT_DIALECT.get(dialect, dialect)
    sqlstr = foo.sql(dialect=dialect, pretty=True)
    expr = con.sql(sqlstr, dialect=dialect)
    result = expr.execute()
    expected = df.int_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@dot_sql_notimpl
@dot_sql_never
@pytest.mark.notimpl(["druid", "flink", "impala", "polars", "pyspark"])
@pytest.mark.notyet(["snowflake"], reason="snowflake column names are case insensitive")
def test_order_by_no_projection(backend):
    con = backend.connection
    astronauts = con.table("astronauts")
    expr = (
        astronauts.group_by("name")
        .agg(nbr_missions=_.count())
        .order_by(_.nbr_missions.desc())
    )

    result = con.sql(ibis.to_sql(expr)).execute().name.iloc[:2]
    assert set(result) == {"Ross, Jerry L.", "Chang-Diaz, Franklin R."}


@dot_sql_notimpl
@dot_sql_notyet
@dot_sql_never
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
def test_dot_sql_limit(con):
    expr = con.sql("SELECT 'abc' ts").limit(1)
    assert expr.execute().equals(pd.DataFrame({"ts": ["abc"]}))
