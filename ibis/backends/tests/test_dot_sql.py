from __future__ import annotations

import getpass
import sqlite3

import pytest
import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from pytest import param

import ibis
import ibis.backends.sql.compilers as sc
import ibis.backends.sql.dialects  # to load dialects
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis import _
from ibis.backends import _get_backend_names
from ibis.backends.conftest import _get_backends_to_test
from ibis.backends.tests.base import PYTHON_SHORT_VERSION
from ibis.backends.tests.errors import (
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    OracleDatabaseError,
    PolarsSQLInterfaceError,
    PsycoPg2InternalError,
    PsycoPgSyntaxError,
    Py4JJavaError,
    PyAthenaOperationalError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    SingleStoreDBOperationalError,
)

pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")

_NAMES = {
    "bigquery": f"ibis_gbq_testing_{getpass.getuser()}_{PYTHON_SHORT_VERSION}.functional_alltypes",
}


@pytest.fixture(scope="module")
def ftname_raw(con):
    return _NAMES.get(con.name, "functional_alltypes")


@pytest.fixture(scope="module")
def ftname(con, ftname_raw):
    table = sg.parse_one(ftname_raw, into=sge.Table)
    table = sg.table(table.name, db=table.db, catalog=table.catalog, quoted=True)
    return table.sql(con.dialect)


@pytest.mark.parametrize(
    "schema",
    [
        param(
            None,
            id="implicit_schema",
            marks=[pytest.mark.notimpl(["druid"])],
        ),
        param(
            {"s": "string", "new_col": "double"},
            id="explicit_schema",
        ),
    ],
)
def test_con_dot_sql(backend, con, schema, ftname):
    alltypes = backend.functional_alltypes
    # pull out the quoted name
    quoted = True
    cols = [
        sg.column("string_col", quoted=quoted).as_("s", quoted=quoted).sql(con.dialect),
        (sg.column("double_col", quoted=quoted) + 1.0)
        .as_("new_col", quoted=quoted)
        .sql(con.dialect),
    ]
    t = (
        con.sql(
            f"SELECT {', '.join(cols)} FROM {ftname}",
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


@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
@pytest.mark.notyet(
    ["druid"], raises=com.IbisTypeError, reason="druid does not preserve case"
)
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
        .alias("ft2")
        .group_by("s")  # group by a column from SQL
        .aggregate(fancy_af=lambda t: t.new_col.mean())
        .alias("awesome_t")  # create a name for the aggregate
        .sql(
            """
            SELECT
              "l"."fancy_af" AS "yas",
              "r"."s" AS "s"
            FROM "awesome_t" AS "l"
            LEFT JOIN "ft2" AS "r"
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


@pytest.mark.notyet(["druid"], reason="druid doesn't respect column name case")
@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
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


def test_dot_sql_alias_with_params(backend, alltypes, df):
    t = alltypes
    x = t.select(x=t.string_col + " abc").alias("foo")
    result = x.execute()
    expected = df.string_col.add(" abc").rename("x")
    backend.assert_series_equal(result.x, expected)


def test_dot_sql_reuse_alias_with_different_types(backend, alltypes, df):
    foo1 = alltypes.select(x=alltypes.string_col).alias("foo")
    foo2 = alltypes.select(x=alltypes.bigint_col).alias("foo")
    expected1 = df.string_col.rename("x")
    expected2 = df.bigint_col.rename("x")
    backend.assert_series_equal(foo1.x.execute(), expected1)
    backend.assert_series_equal(foo2.x.execute(), expected2)


dialects = sorted(_get_backend_names())

# The "db2" sqlglot dialect is only registered when the optional
# `db2-sqlglot-dialect` package (the `db2` extra) is installed.
DB2_DIALECT_REASON = "db2 sqlglot dialect requires the db2 extra (db2-sqlglot-dialect)"

# Map backend names to SQLGlot dialect names when they differ
BACKEND_TO_SQLGLOT_DIALECT = {
    "singlestoredb": "singlestore",
}


@pytest.mark.parametrize("dialect", dialects)
@pytest.mark.notyet(["druid"], reason="druid doesn't respect column name case")
def test_table_dot_sql_transpile(backend, alltypes, dialect, df):
    if dialect == "db2":
        pytest.importorskip("db2_sqlglot", reason=DB2_DIALECT_REASON)
    sqlglot_dialect = BACKEND_TO_SQLGLOT_DIALECT.get(dialect, dialect)
    name = "foo2"
    foo = alltypes.select(x=_.bigint_col + 1).alias(name)
    expr = sg.select(sg.column("x", quoted=True)).from_(sg.table(name, quoted=True))
    sqlstr = expr.sql(dialect=sqlglot_dialect, pretty=True)
    dot_sql_expr = foo.sql(sqlstr, dialect=sqlglot_dialect)
    result = dot_sql_expr.execute()
    expected = df.bigint_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@pytest.mark.parametrize("dialect", dialects)
@pytest.mark.notyet(
    ["druid"], raises=AttributeError, reason="druid doesn't respect column names"
)
@pytest.mark.notyet(["bigquery"])
def test_con_dot_sql_transpile(backend, con, dialect, df):
    if dialect == "db2":
        pytest.importorskip("db2_sqlglot", reason=DB2_DIALECT_REASON)
    sqlglot_dialect = BACKEND_TO_SQLGLOT_DIALECT.get(dialect, dialect)
    t = sg.table("functional_alltypes", quoted=True)
    foo = sg.select(
        sg.alias(sg.column("bigint_col", quoted=True) + 1, "x", quoted=True)
    ).from_(t)
    sqlstr = foo.sql(dialect=sqlglot_dialect, pretty=True)
    expr = con.sql(sqlstr, dialect=sqlglot_dialect)
    result = expr.execute()
    expected = df.bigint_col.add(1).rename("x")
    backend.assert_series_equal(result.x, expected)


@pytest.mark.notyet(
    ["mysql", "singlestoredb"],
    raises=AssertionError,
    reason="https://github.com/tobymao/sqlglot/issues/8626",
)
@pytest.mark.notyet(
    ["sqlite"], raises=sqlite3.OperationalError, reason="sqlite has no STDDEV function"
)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="STDDEV requires the druid-stats extension",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="only stddev_samp and stddev_pop are supported",
)
@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="requires a qualified name"
)
def test_dot_sql_transpile_stddev(con, df):
    expr = con.sql(
        'SELECT STDDEV("double_col") AS "s" FROM "functional_alltypes" WHERE "id" < 5',
        dialect="duckdb",
    )
    result = expr.execute().s.iat[0]
    expected = df.loc[df.id < 5, "double_col"].std()
    assert result == pytest.approx(expected)


@pytest.mark.notimpl(["druid", "polars"])
def test_order_by_no_projection(backend):
    con = backend.connection
    expr = (
        backend.astronauts.group_by("name")
        .agg(nbr_missions=_.count())
        .order_by(_.nbr_missions.desc())
    )

    result = con.sql(ibis.to_sql(expr)).execute().name.iloc[:2]
    assert set(result) == {"Ross, Jerry L.", "Chang-Diaz, Franklin R."}


def test_dot_sql_limit(con):
    expr = con.sql('SELECT * FROM (SELECT \'abc\' "ts") "x"', dialect="duckdb").limit(1)
    result = expr.execute()

    assert len(result) == 1
    assert len(result.columns) == 1
    assert result.columns[0].lower() == "ts"
    assert result.iat[0, 0] == "abc"


@pytest.mark.notyet(
    ["druid"],
    raises=KeyError,
    reason="upstream does not preserve column names in schema inference",
)
def test_cte(alltypes, df):
    expr = alltypes.alias("ft").sql(
        'SELECT "string_col", CAST(COUNT(*) AS BIGINT) "n" FROM "ft" GROUP BY "string_col"',
        dialect="duckdb",
    )
    result = expr.to_pandas().set_index("string_col").sort_index()

    expected = (
        df.groupby("string_col")
        .size()
        .reset_index(name="n")
        .set_index("string_col")
        .sort_index()
    )

    tm.assert_frame_equal(result, expected)


def test_bare_minimum(alltypes, df, ftname_raw):
    """Test that a backend that supports dot sql can do the most basic thing."""

    expr = alltypes.sql(f'SELECT COUNT(*) AS "n" FROM "{ftname_raw}"', dialect="duckdb")
    assert expr.to_pandas().iat[0, 0] == len(df)


def test_embedded_cte(alltypes, ftname_raw):
    sql = f'WITH "x" AS (SELECT * FROM "{ftname_raw}") SELECT * FROM "x"'
    expr = alltypes.sql(sql, dialect="duckdb")
    result = expr.head(1).execute()
    assert len(result) == 1


def test_embedded_cte_with_alias_simple(con):
    expr = con.sql('SELECT * FROM (SELECT \'abc\' "ts") "x"', dialect="duckdb")
    alias = "alias"
    expr = expr.alias(alias)
    expr = expr.sql(
        f'WITH "x" AS (SELECT * FROM "{alias}") SELECT * FROM "x"', dialect="duckdb"
    )
    result = expr.head(1).execute()
    assert len(result) == 1
    assert result["ts"][0] == "abc"


@pytest.mark.notyet(
    ["oracle"], raises=OracleDatabaseError, reason="Oracle doesn't allow embedding CTEs"
)
@pytest.mark.notyet(
    ["mssql"],
    raises=PyODBCProgrammingError,
    reason="MS SQL doesn't allow embedding CTEs",
)
def test_embedded_cte_with_alias_nested(con):
    expr = con.sql(
        'WITH "second_alias" as (SELECT * FROM (SELECT \'abc\' "ts") "x") SELECT * FROM "second_alias"',
        dialect="duckdb",
    )
    alias = "alias"
    expr = expr.alias(alias)
    expr = expr.sql(
        f'WITH "x" AS (SELECT * FROM "{alias}") SELECT * FROM "x"', dialect="duckdb"
    )
    result = expr.head(1).execute()
    assert len(result) == 1
    assert result["ts"][0] == "abc"


@pytest.mark.never(["exasol"], raises=ExaQueryError, reason="backend requires aliasing")
@pytest.mark.never(
    ["oracle"], raises=OracleDatabaseError, reason="backend requires aliasing"
)
@pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError)
def test_unnamed_columns(con):
    sql = "SELECT 'a', 1 AS \"col42\""
    sgexpr = sg.parse_one(sql, read="duckdb")
    expr = con.sql(sgexpr.sql(con.dialect))

    schema = expr.schema()
    names = schema.names
    types = schema.types

    assert len(names) == 2

    assert names[0]
    assert names[1] == "col42"

    assert types[0].is_string()
    assert types[1].is_integer()


def test_scalar_dot_sql(con):
    sql = sg.select(sge.convert(1).as_("a")).sql(con.dialect)
    expr = con.sql(sql).as_scalar()
    assert expr.type().is_numeric()


RECURSIVE_COUNTER = """
WITH RECURSIVE "counter" AS (
  SELECT 1 AS "n"
  UNION ALL
  SELECT "n" + 1 FROM "counter" WHERE "n" < 3
)
SELECT "n" FROM "counter"
"""

# Same counter, but the anchor row comes from a table named "t", so it can be
# passed to `Table.sql` on a table aliased as "t"
RECURSIVE_COUNTER_FROM_T = """
WITH RECURSIVE "counter" AS (
  SELECT "x" AS "n" FROM "t"
  UNION ALL
  SELECT "n" + 1 FROM "counter" WHERE "n" < 3
)
SELECT "n" FROM "counter"
"""

no_recursive_cte = toolz.compose(
    pytest.mark.notyet(
        ["polars"],
        raises=PolarsSQLInterfaceError,
        reason="Polars SQL doesn't support recursive CTEs",
    ),
    pytest.mark.notyet(
        ["exasol"], raises=ExaQueryError, reason="Exasol doesn't support WITH RECURSIVE"
    ),
    pytest.mark.notyet(
        ["risingwave"],
        raises=PsycoPg2InternalError,
        reason="RisingWave doesn't support recursive CTEs in batch queries",
    ),
    pytest.mark.notyet(
        ["druid"],
        raises=PyDruidProgrammingError,
        reason="Druid doesn't support recursive CTEs",
    ),
    pytest.mark.notyet(
        ["flink"],
        raises=Py4JJavaError,
        reason="Flink doesn't support recursive CTEs; sqlglot drops RECURSIVE",
    ),
    pytest.mark.notyet(
        ["impala"],
        raises=ImpalaHiveServer2Error,
        reason="Impala doesn't support recursive CTEs; sqlglot drops RECURSIVE",
    ),
    pytest.mark.notyet(
        ["materialize"],
        raises=PsycoPgSyntaxError,
        reason="Materialize only supports WITH MUTUALLY RECURSIVE",
    ),
    pytest.mark.notyet(
        ["oracle"],
        raises=OracleDatabaseError,
        reason="sqlglot emits WITH RECURSIVE, which Oracle doesn't accept",
    ),
    pytest.mark.notyet(
        ["singlestoredb"],
        raises=SingleStoreDBOperationalError,
        reason="unsupported recursive common table expression query shape",
    ),
    pytest.mark.notyet(
        ["pyspark"],
        reason="Spark < 4.1 doesn't support recursive CTEs (3.5 fails to parse, 4.0 to plan)",
    ),
)
nested_recursive_cte = pytest.mark.notyet(
    ["bigquery"],
    raises=GoogleBadRequest,
    reason="BigQuery rejects WITH RECURSIVE nested in a subquery or CTE",
)


@no_recursive_cte
def test_con_dot_sql_recursive_cte(backend, con):
    expr = con.sql(RECURSIVE_COUNTER, dialect="duckdb")
    result = expr.execute()["n"].sort_values().reset_index(drop=True)
    expected = pd.Series([1, 2, 3], name="n")
    backend.assert_series_equal(result.astype(expected.dtype), expected)


@no_recursive_cte
@nested_recursive_cte
def test_con_dot_sql_recursive_cte_filtered(backend, con):
    expr = con.sql(RECURSIVE_COUNTER, dialect="duckdb").filter(_.n > 1)
    result = expr.execute()["n"].sort_values().reset_index(drop=True)
    expected = pd.Series([2, 3], name="n")
    backend.assert_series_equal(result.astype(expected.dtype), expected)


@no_recursive_cte
def test_table_dot_sql_recursive_query(backend, con):
    t = con.sql('SELECT 1 AS "x"', dialect="duckdb").alias("t")
    expr = t.sql(RECURSIVE_COUNTER_FROM_T, dialect="duckdb")
    result = expr.execute()["n"].sort_values().reset_index(drop=True)
    expected = pd.Series([1, 2, 3], name="n")
    backend.assert_series_equal(result.astype(expected.dtype), expected)


@no_recursive_cte
@nested_recursive_cte
def test_table_dot_sql_recursive_parent(backend, con):
    t = con.sql(RECURSIVE_COUNTER, dialect="duckdb").alias("t")
    expr = t.sql('SELECT "n" FROM "t"', dialect="duckdb")
    result = expr.execute()["n"].sort_values().reset_index(drop=True)
    expected = pd.Series([1, 2, 3], name="n")
    backend.assert_series_equal(result.astype(expected.dtype), expected)


def _sql_query_result(query, compiler):
    """Build what `con.sql(query, dialect="duckdb")` builds, without a connection."""
    query = sg.transpile(query, read="duckdb", write=compiler.dialect)[0]
    return ops.SQLQueryResult(query, ibis.schema({"n": "int64"}), None).to_expr()


no_recursive_keyword = toolz.compose(
    pytest.mark.never(
        ["mssql", "db2"],
        raises=AssertionError,
        reason="T-SQL and Db2 have no RECURSIVE keyword",
    ),
    pytest.mark.notyet(
        ["flink", "impala"],
        raises=AssertionError,
        reason="recursive CTEs aren't supported, so sqlglot drops RECURSIVE",
    ),
    pytest.mark.xfail_version(
        pyspark=["sqlglot<30"],
        databricks=["sqlglot<30"],
        raises=AssertionError,
        reason="sqlglot drops RECURSIVE for Spark before 30.0",
    ),
)


@no_recursive_keyword
@pytest.mark.parametrize("backend_name", _get_backends_to_test(discard=("polars",)))
def test_recursive_cte_con_sql_compiles(backend_name):
    compiler = getattr(sc, backend_name).compiler
    expr = _sql_query_result(RECURSIVE_COUNTER, compiler)
    assert "WITH RECURSIVE" in compiler.to_sqlglot(expr).sql(compiler.dialect)


@no_recursive_keyword
@pytest.mark.parametrize(
    ("make_table", "query"),
    [
        param(
            lambda _compiler: ibis.table({"x": "int64"}, name="src"),
            RECURSIVE_COUNTER_FROM_T,
            id="recursive_query",
        ),
        param(
            lambda compiler: _sql_query_result(RECURSIVE_COUNTER, compiler),
            'SELECT "n" FROM "t"',
            id="recursive_parent",
        ),
    ],
)
@pytest.mark.parametrize("backend_name", _get_backends_to_test(discard=("polars",)))
def test_recursive_cte_table_sql_compiles(backend_name, make_table, query):
    compiler = getattr(sc, backend_name).compiler
    query = sg.transpile(query, read="duckdb", write=compiler.dialect)[0]
    sql = compiler.add_query_to_expr(name="t", table=make_table(compiler), query=query)
    assert "WITH RECURSIVE" in sql
