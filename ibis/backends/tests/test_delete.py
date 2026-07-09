"""Tests for `Backend.delete`."""

from __future__ import annotations

import contextlib

import pytest

import ibis
import ibis.common.exceptions as com
from ibis.backends.tests.conftest import NO_DELETE_SUPPORT, combine_marks
from ibis.util import gen_name

pd = pytest.importorskip("pandas")

# These backends cannot create the scratch tables this module stages its
# tests on -- a limitation of table creation, not of DELETE. `raises` is
# deliberately omitted: they convert any failure, mirroring the unconditional
# `pytest.xfail()` calls they replace.
CANNOT_CREATE_TEMP_TABLES_MARKS = [
    pytest.mark.notimpl(["druid"], reason="doesn't implement create_table"),
    pytest.mark.notimpl(
        ["flink"],
        reason="doesn't implement create_table from schema without additional arguments",
    ),
    pytest.mark.notyet(
        ["athena"], reason="create table must specify external location"
    ),
]
CANNOT_CREATE_TEMP_TABLES = combine_marks(CANNOT_CREATE_TEMP_TABLES_MARKS)


def _create_temp_table_with_schema(backend, con, temp_table_name, schema, data=None):
    temporary = con.create_table(temp_table_name, schema=schema)
    assert temporary.to_pandas().empty

    if data is not None and isinstance(data, pd.DataFrame):
        assert not data.empty
        tmp = con.create_table(temp_table_name, data, overwrite=True)
        result = tmp.to_pandas()
        assert len(result) == len(data.index)
        backend.assert_frame_equal(
            result.sort_values(result.columns[0]).reset_index(drop=True),
            data.sort_values(result.columns[0]).reset_index(drop=True),
        )
        return tmp

    return temporary


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
def temp_employee_table(backend, con, test_employee_schema):
    """Return a factory that creates a temporary employee table.

    Creation runs inside the test call phase, NOT fixture setup: ibis
    translates `notimpl`/`notyet` marks into xfails in `pytest_runtest_call`,
    which never runs when a fixture errors during setup. Creating the table
    from the test body lets create_table limitations (druid, flink, athena)
    surface where the marks can convert them.
    """
    created = []

    def make(data: pd.DataFrame) -> str:
        temp_table_name = gen_name("temp_employee")
        # Register for cleanup before creating: if creation fails partway
        # through (the helper creates, then validates), the table is still
        # dropped; drop_table(force=True) tolerates names that never existed.
        created.append(temp_table_name)
        _create_temp_table_with_schema(
            backend, con, temp_table_name, test_employee_schema, data=data
        )
        return temp_table_name

    try:
        yield make
    finally:
        for name in created:
            con.drop_table(name, force=True)


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_with_where(backend, con, temp_employee_table):
    table_name = temp_employee_table(employee_data_1())
    temporary = con.table(table_name)

    con.delete(table_name, ibis._.salary > 200)

    result = temporary.execute()
    assert len(result) == 2
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        pd.DataFrame(
            {
                "first_name": ["A", "B"],
                "last_name": ["D", "E"],
                "department_name": ["AA", "BB"],
                "salary": [100.0, 200.0],
            }
        ),
    )


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_with_callable(backend, con, temp_employee_table):
    table_name = temp_employee_table(employee_data_1())
    temporary = con.table(table_name)

    con.delete(table_name, lambda t: t.salary > 200)

    result = temporary.execute()
    assert len(result) == 2
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        pd.DataFrame(
            {
                "first_name": ["A", "B"],
                "last_name": ["D", "E"],
                "department_name": ["AA", "BB"],
                "salary": [100.0, 200.0],
            }
        ),
    )


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_no_matching_rows(con, temp_employee_table):
    table_name = temp_employee_table(employee_data_1())
    temporary = con.table(table_name)

    con.delete(table_name, ibis._.salary > 1000)

    result = temporary.execute()
    assert len(result) == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_with_bound_predicate(con, temp_employee_table):
    # A predicate already bound to the table, as opposed to the `Deferred` and
    # callable forms covered above.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    con.delete(table_name, target.salary > 200)

    assert target.count().execute() == 2


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_literal_boolean_predicates(con, temp_employee_table):
    # Literal booleans are valid predicates: `False` deletes nothing, and, as
    # called out in the `delete` docstring, `True` is NOT caught by the
    # `where=None` safety check and deletes every row.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    con.delete(table_name, False)
    assert target.count().execute() == 3

    con.delete(table_name, True)
    assert target.count().execute() == 0


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_correlated_subquery_exists(con, temp_employee_table):
    # Delete rows whose salary matches a salary in the other table. The salary
    # sets ({100, 200, 300} vs {400, 500, 600}) are disjoint, so NO rows match
    # and NO rows should be deleted. Regression test: a correlated subquery
    # must never collapse into a tautology that deletes every row.
    target_name = temp_employee_table(employee_data_1())
    source_name = temp_employee_table(employee_data_2())
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(target_name, where=(source.salary == target.salary).any())

    assert target.count().execute() == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_correlated_subquery_not_exists(con, temp_employee_table):
    # Delete rows whose salary does NOT match any salary in the other table.
    # The salary sets are disjoint, so ALL rows fail to match and ALL rows
    # should be deleted.
    target_name = temp_employee_table(employee_data_1())
    source_name = temp_employee_table(employee_data_2())
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(target_name, where=~(source.salary == target.salary).any())

    assert target.count().execute() == 0


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_correlated_subquery_compound(con, temp_employee_table):
    # Compound predicate: a correlated EXISTS AND a simple predicate. Because
    # no salary matches, the EXISTS branch is false for every row, so the whole
    # predicate is false and NO rows should be deleted.
    target_name = temp_employee_table(employee_data_1())
    source_name = temp_employee_table(employee_data_2())
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(
        target_name,
        where=(source.salary == target.salary).any() & (target.department_name == "BB"),
    )

    assert target.count().execute() == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_uncorrelated_subquery(con, temp_employee_table):
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
    target_name = temp_employee_table(employee_data_1())
    source_name = temp_employee_table(source_data)
    target = con.table(target_name)
    source = con.table(source_name)

    con.delete(target_name, where=target.salary.isin(source.salary))

    assert target.count().execute() == 2


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_scalar_subquery_predicate(con, temp_employee_table):
    # An aggregate over the target table compiles to a scalar subquery that
    # scans the table being deleted from. This is also the rewrite the
    # window-predicate error message recommends. Salaries are {100, 200, 300}
    # (mean 200), so only the 300 row is deleted.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    con.delete(table_name, target.salary > target.salary.mean())

    assert target.count().execute() == 2


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_null_predicate_semantics(con, temp_employee_table):
    # SQL three-valued logic: a DELETE only removes rows where the predicate
    # is TRUE. Rows where it evaluates to NULL survive.
    data = pd.DataFrame(
        {
            "first_name": ["A", "B", "C", "N"],
            "last_name": ["D", "E", "F", "O"],
            "department_name": ["AA", "BB", "CC", "NN"],
            "salary": [100.0, 200.0, 300.0, None],
        }
    )
    table_name = temp_employee_table(data)
    target = con.table(table_name)

    con.delete(table_name, ibis._.salary > 150)

    result = target.execute()
    assert len(result) == 2  # the 100 row and the NULL-salary row
    assert result.salary.isna().sum() == 1


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@CANNOT_CREATE_TEMP_TABLES
def test_delete_where_none_raises(con, temp_employee_table):
    # No DELETE DML marks (datafusion, materialize): the error is raised
    # client-side before any DELETE statement is sent to the backend.
    table_name = temp_employee_table(employee_data_1())

    with pytest.raises(com.IbisInputError, match="truncate_table"):
        con.delete(table_name, where=None)


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@CANNOT_CREATE_TEMP_TABLES
def test_delete_window_predicate_raises(con, temp_employee_table):
    # Window functions cannot appear in a DELETE statement's WHERE clause
    # (they compile to QUALIFY); ibis rejects them with a typed error rather
    # than emitting invalid SQL or crashing. No DELETE DML marks (datafusion,
    # materialize): the error is raised client-side before any DELETE is sent.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.UnsupportedOperationError, match=r"[Ww]indow"):
        con.delete(
            table_name,
            where=target.salary > target.salary.mean().over(ibis.window()),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@CANNOT_CREATE_TEMP_TABLES
def test_delete_multiple_predicates_raises(con, temp_employee_table):
    # Unlike `filter`, `delete` takes a single predicate; a tuple is rejected
    # with a clear message pointing at `&` instead of a cryptic internal error.
    # No DELETE DML marks (datafusion, materialize): the error is raised
    # client-side before any DELETE statement is sent.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.IbisInputError, match="single boolean predicate"):
        con.delete(
            table_name,
            where=(target.salary > 100, target.salary < 300),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@CANNOT_CREATE_TEMP_TABLES
def test_delete_non_boolean_predicate_raises(con, temp_employee_table):
    # A non-boolean `where` (e.g. a table or a column name) is rejected with a
    # clear message instead of an internal unpacking error. No DELETE DML marks
    # (datafusion, materialize): the error is raised client-side before any
    # DELETE statement is sent.
    table_name = temp_employee_table(employee_data_1())
    target = con.table(table_name)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(table_name, where=target)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(table_name, where="salary")

    assert target.count().execute() == 3


@NO_DELETE_SUPPORT
@CANNOT_CREATE_TEMP_TABLES
def test_delete_with_database_param(con_create_database, test_employee_schema):
    # Delete from a table that lives in an explicitly created database, passing
    # `database=` to resolve it.
    con = con_create_database
    database = gen_name("test_delete_db")
    con.create_database(database)
    try:
        table_name = gen_name("temp_employee_db")
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
def test_delete_with_database_tuple(con_create_catalog_database, test_employee_schema):
    con = con_create_catalog_database
    with _create_and_destroy_catalog_db(con) as (catalog, database):
        table_name = gen_name("temp_employee_catalog_db")
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
