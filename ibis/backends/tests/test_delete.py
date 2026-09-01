"""Tests for `Backend.delete`."""

from __future__ import annotations

import contextlib

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis.backends.tests.conftest import NO_DELETE_SUPPORT, combine_marks
from ibis.backends.tests.errors import ClickHouseDatabaseError
from ibis.util import gen_name

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

# These backends cannot create the scratch tables this module stages its
# tests on -- a limitation of table creation, not of DELETE. `raises` is
# deliberately omitted: they convert any failure, mirroring the unconditional
# `pytest.xfail()` calls they replace.
CANNOT_CREATE_TEST_TABLES_MARKS = [
    pytest.mark.notimpl(["druid"], reason="doesn't implement create_table"),
    pytest.mark.notimpl(
        ["flink"],
        reason="doesn't implement create_table from schema without additional arguments",
    ),
    pytest.mark.notyet(
        ["athena"], reason="create table must specify external location"
    ),
]
CANNOT_CREATE_TEST_TABLES = combine_marks(CANNOT_CREATE_TEST_TABLES_MARKS)

# Subquery predicates compile to an aliased DELETE (`DELETE FROM t AS t0 ...`)
# so that correlated references can name the enclosing scope; these backends
# cannot run that statement.
NO_DELETE_ALIAS_SUPPORT_MARKS = [
    pytest.mark.notyet(
        ["clickhouse"],
        raises=ClickHouseDatabaseError,
        reason="ClickHouse DELETE does not accept a table alias",
    ),
    pytest.mark.notyet(
        ["risingwave"],
        raises=com.UnsupportedOperationError,
        reason="RisingWave DELETE does not accept a table alias; "
        "ibis refuses the statement client-side",
    ),
    pytest.mark.notyet(
        ["datafusion"],
        raises=ValueError,
        reason="DataFusion DELETE does not bind the target table alias, so "
        "alias-qualified column references fail with a schema error",
    ),
]
NO_DELETE_ALIAS_SUPPORT = combine_marks(NO_DELETE_ALIAS_SUPPORT_MARKS)


def employee_data_1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )


def employee_data_2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "first_name": ["X", "Y", "Z"],
            "last_name": ["A", "B", "C"],
            "department_name": ["XX", "YY", "ZZ"],
            "salary": [400.0, 500.0, 600.0],
        }
    )


@pytest.fixture
def test_employee_schema() -> ibis.schema:
    return ibis.schema(
        {
            "first_name": "string",
            "last_name": "string",
            "department_name": "string",
            "salary": "float64",
        }
    )


@pytest.fixture
def employee_table(con, test_employee_schema):
    """Return a factory that creates an employee table with the given rows.

    Creation runs inside the test call phase, NOT fixture setup: ibis
    translates `notimpl`/`notyet` marks into xfails in `pytest_runtest_call`,
    which never runs when a fixture errors during setup. Creating the table
    from the test body lets create_table limitations (druid, flink, athena)
    surface where the marks can convert them.
    """
    created = []

    def make(data) -> str:
        table_name = gen_name("employee")
        # Register for cleanup before creating so a partial failure is still
        # dropped; drop_table(force=True) tolerates names that never existed.
        created.append(table_name)
        con.create_table(
            table_name, obj=ibis.memtable(data, schema=test_employee_schema)
        )
        return table_name

    try:
        yield make
    finally:
        for name in created:
            con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("make_where", "remaining"),
    [
        param(lambda _: ibis._.salary > 200, ["A", "B"], id="deferred"),
        param(lambda _: lambda x: x.salary > 200, ["A", "B"], id="callable"),
        param(lambda t: t.salary > 200, ["A", "B"], id="bound"),
        param(lambda _: ibis._.salary > 1000, ["A", "B", "C"], id="no-matching-rows"),
    ],
)
@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
def test_delete_simple_predicates(backend, con, employee_table, make_where, remaining):
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    con.delete(table_name, make_where(target))

    result = target.execute().sort_values("first_name").reset_index(drop=True)
    data = employee_data_1()
    expected = (
        data[data.first_name.isin(remaining)]
        .sort_values("first_name")
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
@NO_DELETE_ALIAS_SUPPORT
@pytest.mark.parametrize(
    ("negate", "expected_count"),
    [
        # The salary sets ({100, 200, 300} vs {400, 500, 600}) are disjoint:
        # EXISTS matches no row (delete nothing); NOT EXISTS matches every row
        # (delete everything). Regression test: a correlated subquery must
        # never collapse into a tautology that deletes the wrong rows.
        param(False, 3, id="exists"),
        param(True, 0, id="not-exists"),
    ],
)
def test_delete_correlated_subquery(con, employee_table, negate, expected_count):
    target_name = employee_table(employee_data_1())
    source_name = employee_table(employee_data_2())
    target = con.table(target_name)
    source = con.table(source_name)

    predicate = (source.salary == target.salary).any()

    con.delete(target_name, where=~predicate if negate else predicate)

    assert target.count().execute() == expected_count


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
@NO_DELETE_ALIAS_SUPPORT
def test_delete_correlated_subquery_compound(con, employee_table):
    # Compound predicate: a correlated EXISTS AND a simple predicate. Because
    # no salary matches, the EXISTS branch is false for every row, so the whole
    # predicate is false and NO rows should be deleted.
    target_name = employee_table(employee_data_1())
    source_name = employee_table(employee_data_2())
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(
        target_name,
        where=(source.salary == target.salary).any() & (target.department_name == "BB"),
    )

    assert target.count().execute() == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
@NO_DELETE_ALIAS_SUPPORT
def test_delete_uncorrelated_subquery(con, employee_table):
    # An uncorrelated subquery. The source contains exactly one salary that
    # overlaps the target (200); deleting target rows whose salary is in the
    # source removes only the 200 row.
    source_data = pd.DataFrame(
        {
            "first_name": ["X", "Y", "Z", "M"],
            "last_name": ["A", "B", "C", "M"],
            "department_name": ["XX", "YY", "ZZ", "MM"],
            "salary": [400.0, 500.0, 600.0, 200.0],
        }
    )
    target_name = employee_table(employee_data_1())
    source_name = employee_table(source_data)
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(target_name, where=target.salary.isin(source.salary))

    assert target.count().execute() == 2


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
@NO_DELETE_ALIAS_SUPPORT
def test_delete_scalar_subquery_predicate(con, employee_table):
    # An aggregate over the target table compiles to a scalar subquery that
    # scans the table being deleted from. This is also the rewrite the
    # window-predicate error message recommends. Salaries are {100, 200, 300}
    # (mean 200), so only the 300 row is deleted.
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    con.delete(table_name, target.salary > target.salary.mean())

    assert target.count().execute() == 2


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
def test_delete_null_predicate_semantics(con, employee_table):
    # SQL three-valued logic: a DELETE only removes rows where the predicate
    # is TRUE. Rows where it evaluates to NULL survive. The data is a pyarrow
    # table because pandas would turn the NULL salary into NaN, which several
    # drivers (pyexasol, pyodbc) refuse to insert.
    data = pa.table(
        {
            "first_name": ["A", "B", "C", "N"],
            "last_name": ["D", "E", "F", "O"],
            "department_name": ["AA", "BB", "CC", "NN"],
            "salary": [100.0, 200.0, 300.0, None],
        }
    )
    table_name = employee_table(data)
    target = con.table(table_name)

    con.delete(table_name, ibis._.salary > 150)

    result = target.execute()
    assert len(result) == 2  # the 100 row and the NULL-salary row
    assert result.salary.isna().sum() == 1


@pytest.mark.notimpl(
    ["polars"], raises=AttributeError, reason="`delete` method not implemented"
)
@CANNOT_CREATE_TEST_TABLES
def test_delete_where_none_raises(con, employee_table):
    # No DELETE DML marks (materialize, pyspark): the error is raised
    # client-side before any DELETE statement is sent to the backend.
    table_name = employee_table(employee_data_1())

    with pytest.raises(com.IbisInputError, match="truncate_table"):
        con.delete(table_name, where=None)


@pytest.mark.notimpl(
    ["polars"], raises=AttributeError, reason="`delete` method not implemented"
)
@CANNOT_CREATE_TEST_TABLES
def test_delete_literal_bool_raises(con, employee_table):
    # A literal `True` would delete every row (that is `truncate_table`'s job)
    # and a literal `False` would delete nothing, so both are rejected
    # client-side before any DELETE statement is sent.
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.IbisInputError, match="literal bool"):
        con.delete(table_name, where=True)

    with pytest.raises(com.IbisInputError, match="literal bool"):
        con.delete(table_name, where=False)

    assert target.count().execute() == 3


@pytest.mark.notimpl(
    ["polars"], raises=AttributeError, reason="`delete` method not implemented"
)
@CANNOT_CREATE_TEST_TABLES
def test_delete_window_predicate_raises(con, employee_table):
    # Window functions cannot appear in a DELETE statement's WHERE clause
    # (they compile to QUALIFY); ibis rejects them with a typed error rather
    # than emitting invalid SQL or crashing. No DELETE DML marks (materialize,
    # pyspark): the error is raised client-side before any DELETE is sent.
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.UnsupportedOperationError, match=r"[Ww]indow"):
        con.delete(
            table_name,
            where=target.salary > target.salary.mean().over(ibis.window()),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(
    ["polars"], raises=AttributeError, reason="`delete` method not implemented"
)
@CANNOT_CREATE_TEST_TABLES
def test_delete_multiple_predicates_raises(con, employee_table):
    # Unlike `filter`, `delete` takes a single predicate; a tuple is rejected
    # with a clear message pointing at `&` instead of a cryptic internal error.
    # No DELETE DML marks (materialize, pyspark): the error is raised
    # client-side before any DELETE statement is sent.
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.IbisInputError, match="single boolean predicate"):
        con.delete(
            table_name,
            where=(target.salary > 100, target.salary < 300),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(
    ["polars"], raises=AttributeError, reason="`delete` method not implemented"
)
@CANNOT_CREATE_TEST_TABLES
def test_delete_non_boolean_predicate_raises(con, employee_table):
    # A non-boolean `where` (e.g. a table or a column name) is rejected with a
    # clear message instead of an internal unpacking error. No DELETE DML marks
    # (materialize, pyspark): the error is raised client-side before any
    # DELETE statement is sent.
    table_name = employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(table_name, where=target)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(table_name, where="salary")

    assert target.count().execute() == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEST_TABLES
def test_delete_with_database_param(con_create_database, test_employee_schema):
    # Delete from a table that lives in an explicitly created database, passing
    # `database=` to resolve it.
    con = con_create_database
    database = gen_name("test_delete_db")
    con.create_database(database)
    try:
        table_name = gen_name("employee_db")
        con.create_table(
            table_name,
            obj=ibis.memtable(employee_data_1(), schema=test_employee_schema),
            database=database,
        )
        try:
            target = con.table(table_name, database=database)
            assert target.count().execute() == 3

            con.delete(table_name, ibis._.salary > 200, database=database)

            assert target.count().execute() == 2
        finally:
            con.drop_table(table_name, database=database, force=True)
    finally:
        con.drop_database(database, force=True)


@contextlib.contextmanager
def _create_and_destroy_catalog_db(con):
    catalog = gen_name("test_delete_catalog")
    con.create_catalog(catalog)
    try:
        database = gen_name("test_delete_database")
        con.create_database(database, catalog=catalog)
        try:
            yield catalog, database
        finally:
            con.drop_database(database, catalog=catalog)
    finally:
        con.drop_catalog(catalog)


@NO_DELETE_SUPPORT
@pytest.mark.notyet(["datafusion"], reason="cannot list or drop catalogs")
def test_delete_with_database_tuple(con_create_catalog_database, test_employee_schema):
    con = con_create_catalog_database
    with _create_and_destroy_catalog_db(con) as (catalog, database):
        table_name = gen_name("employee_catalog_db")
        con.create_table(
            table_name,
            obj=ibis.memtable(employee_data_1(), schema=test_employee_schema),
            database=(catalog, database),
        )
        try:
            target = con.table(table_name, database=(catalog, database))
            assert target.count().execute() == 3

            con.delete(table_name, ibis._.salary > 200, database=(catalog, database))

            assert target.count().execute() == 2
        finally:
            con.drop_table(table_name, database=(catalog, database), force=True)


@pytest.mark.usefixtures("con")
def test_delete_alias_stripping_dialect_detection():
    # Pins the sqlglot behavior `delete` guards against: presto-family
    # generators cannot express an aliased DELETE target, so they drop the
    # alias and unqualify every column, silently collapsing a correlated
    # predicate into a tautology. `_delete_preserves_alias` must detect the
    # stripping so `delete` raises instead of removing the wrong rows.
    import sqlglot as sg

    from ibis.backends.sql import SQLBackend

    stmt = sg.parse_one(
        'DELETE FROM "tgt" AS "t0" WHERE '
        'EXISTS(SELECT 1 FROM "src" AS "t1" WHERE "t1"."s" = "t0"."s")',
        read="duckdb",
    )

    for dialect in ("trino", "presto", "athena"):
        assert not SQLBackend._delete_preserves_alias(stmt, "t0", dialect)

    for dialect in ("duckdb", "postgres", "mysql", "sqlite", "bigquery"):
        assert SQLBackend._delete_preserves_alias(stmt, "t0", dialect)
