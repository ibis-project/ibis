from __future__ import annotations

import contextlib

import pandas as pd
import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis import _
from ibis.backends import _get_backend_names

# import here to load the dialect in to sqlglot so we can use it for transpilation
from ibis.backends.sql.dialects import (  # noqa: F401
    MSSQL,
    DataFusion,
    Druid,
    Exasol,
    Flink,
    Impala,
    Polars,
    PySpark,
    RisingWave,
)
from ibis.backends.tests.errors import (
    ExaQueryError,
    GoogleBadRequest,
    OracleDatabaseError,
    PolarsComputeError,
)

dot_sql_never = pytest.mark.never(
    ["dask", "pandas"], reason="dask and pandas do not accept SQL"
)

_NAMES = {
    "bigquery": "ibis_gbq_testing.functional_alltypes",
    "exasol": '"functional_alltypes"',
}


@pytest.mark.notyet(["oracle"], reason="table quoting behavior")
@dot_sql_never
@pytest.mark.parametrize(
    "schema",
    [
        param(None, id="implicit_schema", marks=[pytest.mark.notimpl(["druid"])]),
        param({"s": "string", "new_col": "double"}, id="explicit_schema"),
    ],
)
def test_con_dot_sql(backend, con, schema):
    alltypes = backend.functional_alltypes
    # pull out the quoted name
    name = _NAMES.get(con.name, "functional_alltypes")
    quoted = True
    cols = [
        sg.column("string_col", quoted=quoted).as_("s", quoted=quoted).sql(con.dialect),
        (sg.column("double_col", quoted=quoted) + 1.0)
        .as_("new_col", quoted=quoted)
        .sql(con.dialect),
    ]
    t = (
        con.sql(
            f"SELECT {', '.join(cols)} FROM {name}",
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
    backend.assert_series_equal(result.astype(expected.dtype), expected)


@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
@pytest.mark.notyet(
    ["druid"], raises=com.IbisTypeError, reason="druid does not preserve case"
)
@dot_sql_never
def test_table_dot_sql(backend):
    alltypes = backend.functional_alltypes
    t = (
        alltypes.sql(
            """
            SELECT
              "string_col" AS "s",
              "double_col" + CAST(1.0 AS DOUBLE) AS "new_col"
            FROM "functional_alltypes"
            """,
            dialect="duckdb",
        )
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql('SELECT "fancy_af" AS "yas" FROM "awesome_t"', dialect="duckdb")
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
    assert pytest.approx(result) == expected


@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
@pytest.mark.notyet(
    ["druid"], raises=com.IbisTypeError, reason="druid does not preserve case"
)
@pytest.mark.notimpl(
    ["oracle"],
    OracleDatabaseError,
    reason="oracle doesn't know which of the tables in the join to sort from",
)
@dot_sql_never
def test_table_dot_sql_with_join(backend):
    alltypes = backend.functional_alltypes
    t = (
        alltypes.sql(
            """
            SELECT
              "string_col" AS "s",
              "double_col" + CAST(1.0 AS DOUBLE) AS "new_col"
            FROM "functional_alltypes"
            """,
            dialect="duckdb",
        )
        .alias("ft")
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql(
            """
            SELECT
              "l"."fancy_af" AS "yas",
              "r"."s" AS "s"
            FROM "awesome_t" AS "l"
            LEFT JOIN "ft" AS "r"
            ON "l"."s" = "r"."s"
            """,  # clickhouse needs the r.s AS s, otherwise the column name is returned as r.s
            dialect="duckdb",
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


@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@pytest.mark.notyet(["druid"], reason="druid doesn't respect column name case")
@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
@dot_sql_never
def test_table_dot_sql_repr(backend):
    alltypes = backend.functional_alltypes
    t = (
        alltypes.sql(
            """
            SELECT
              "string_col" AS "s",
              "double_col" + CAST(1.0 AS DOUBLE) AS "new_col"
            FROM "functional_alltypes"
            """,
            dialect="duckdb",
        )
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql(
            'SELECT "fancy_af" AS "yas" FROM "awesome_t" ORDER BY "fancy_af"',
            dialect="duckdb",
        )
    )

    assert repr(t)


@dot_sql_never
def test_dot_sql_alias_with_params(backend, alltypes, df):
    t = alltypes
    x = t.select(x=t.string_col + " abc").alias("foo")
    result = x.execute()
    expected = df.string_col.add(" abc").rename("x")
    backend.assert_series_equal(result.x, expected)


@dot_sql_never
def test_dot_sql_reuse_alias_with_different_types(backend, alltypes, df):
    foo1 = alltypes.select(x=alltypes.string_col).alias("foo")
    foo2 = alltypes.select(x=alltypes.bigint_col).alias("foo")
    expected1 = df.string_col.rename("x")
    expected2 = df.bigint_col.rename("x")
    backend.assert_series_equal(foo1.x.execute(), expected1)
    backend.assert_series_equal(foo2.x.execute(), expected2)


_NO_SQLGLOT_DIALECT = ("pandas", "dask")
no_sqlglot_dialect = [
    param(dialect, marks=pytest.mark.xfail) for dialect in sorted(_NO_SQLGLOT_DIALECT)
]
dialects = sorted(_get_backend_names(exclude=_NO_SQLGLOT_DIALECT)) + no_sqlglot_dialect


@pytest.mark.parametrize("dialect", dialects)
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
@dot_sql_never
@pytest.mark.notyet(["druid"], reason="druid doesn't respect column name case")
def test_table_dot_sql_transpile(backend, alltypes, dialect, df):
    name = "foo2"
    foo = alltypes.select(x=_.bigint_col + 1).alias(name)
    expr = sg.select(sg.column("x", quoted=True)).from_(sg.table(name, quoted=True))
    sqlstr = expr.sql(dialect=dialect, pretty=True)
    dot_sql_expr = foo.sql(sqlstr, dialect=dialect)
    result = dot_sql_expr.execute()
    expected = df.bigint_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@pytest.mark.parametrize("dialect", dialects)
@pytest.mark.notyet(
    ["druid"], raises=AttributeError, reason="druid doesn't respect column names"
)
@pytest.mark.notyet(["bigquery"])
@dot_sql_never
def test_con_dot_sql_transpile(backend, con, dialect, df):
    t = sg.table("functional_alltypes", quoted=True)
    foo = sg.select(
        sg.alias(sg.column("bigint_col", quoted=True) + 1, "x", quoted=True)
    ).from_(t)
    sqlstr = foo.sql(dialect=dialect, pretty=True)
    expr = con.sql(sqlstr, dialect=dialect)
    result = expr.execute()
    expected = df.bigint_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@dot_sql_never
@pytest.mark.notimpl(["druid", "polars"])
@pytest.mark.notimpl(
    ["exasol"],
    raises=ExaQueryError,
    reason="loading the test data is a pain because of embedded commas",
)
def test_order_by_no_projection(backend):
    con = backend.connection
    expr = (
        backend.astronauts.group_by("name")
        .agg(nbr_missions=_.count())
        .order_by(_.nbr_missions.desc())
    )

    result = con.sql(ibis.to_sql(expr)).execute().name.iloc[:2]
    assert set(result) == {"Ross, Jerry L.", "Chang-Diaz, Franklin R."}


@dot_sql_never
@pytest.mark.notyet(["polars"], raises=PolarsComputeError)
def test_dot_sql_limit(con):
    expr = con.sql('SELECT * FROM (SELECT \'abc\' "ts") "x"', dialect="duckdb").limit(1)
    result = expr.execute()

    assert len(result) == 1
    assert len(result.columns) == 1
    assert result.columns[0].lower() == "ts"
    assert result.iat[0, 0] == "abc"


@pytest.fixture(scope="module")
def mem_t(con):
    if con.name == "druid":
        pytest.xfail("druid does not support create_table")

    name = ibis.util.gen_name(con.name)

    # flink only supports memtables if `temp` is True, seems like we should
    # address that for users
    con.create_table(
        name, ibis.memtable({"a": list("def")}), temp=con.name == "flink" or None
    )
    yield name
    with contextlib.suppress(NotImplementedError):
        con.drop_table(name, force=True)


@dot_sql_never
@pytest.mark.notyet(["polars"], raises=NotImplementedError)
def test_cte(con, mem_t):
    t = con.table(mem_t)
    foo = t.alias("foo")
    assert foo.schema() == t.schema()
    assert foo.count().execute() == t.count().execute()

    expr = foo.sql('SELECT count(*) "x" FROM "foo"', dialect="duckdb")
    assert expr.execute().iat[0, 0] == t.count().execute()
