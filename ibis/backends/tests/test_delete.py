"""Tests for `Backend.delete`."""

from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
from ibis.util import gen_name

pd = pytest.importorskip("pandas")


def _create_temp_table_with_schema(backend, con, temp_table_name, schema, data=None):
    if con.name == "druid":
        pytest.xfail("druid doesn't implement create_table")
    elif con.name == "flink":
        pytest.xfail(
            "flink doesn't implement create_table from schema without additional arguments"
        )
    elif con.name == "athena":
        pytest.xfail("create table must specific external location")

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
def employee_data_1_temp_table(backend, con, test_employee_schema):
    test_employee_data_1 = pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )

    temp_table_name = gen_name("temp_employee_data_1")
    _create_temp_table_with_schema(
        backend, con, temp_table_name, test_employee_schema, data=test_employee_data_1
    )
    assert temp_table_name in con.list_tables()
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.fixture
def employee_data_2_temp_table(backend, con, test_employee_schema):
    test_employee_data_2 = pd.DataFrame(
        {
            "first_name": ["X", "Y", "Z"],
            "last_name": ["A", "B", "C"],
            "department_name": ["XX", "YY", "ZZ"],
            "salary": [400.0, 500.0, 600.0],
        }
    )

    temp_table_name = gen_name("temp_employee_data_2")
    _create_temp_table_with_schema(
        backend, con, temp_table_name, test_employee_schema, data=test_employee_data_2
    )
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_with_where(backend, con, employee_data_1_temp_table):
    temporary = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, ibis._.salary > 200)

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


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_with_callable(backend, con, employee_data_1_temp_table):
    temporary = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, lambda t: t.salary > 200)

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


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_no_matching_rows(con, employee_data_1_temp_table):
    temporary = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, ibis._.salary > 1000)

    result = temporary.execute()
    assert len(result) == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
def test_delete_where_none_raises(con, employee_data_1_temp_table):
    with pytest.raises(com.IbisInputError, match="truncate_table"):
        con.delete(employee_data_1_temp_table, where=None)


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_with_bound_predicate(con, employee_data_1_temp_table):
    # A predicate already bound to the table, as opposed to the `Deferred` and
    # callable forms covered above.
    target = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, target.salary > 200)

    assert target.count().execute() == 2


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_literal_boolean_predicates(con, employee_data_1_temp_table):
    # Literal booleans are valid predicates: `False` deletes nothing, and, as
    # called out in the `delete` docstring, `True` is NOT caught by the
    # `where=None` safety check and deletes every row.
    target = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, False)
    assert target.count().execute() == 3

    con.delete(employee_data_1_temp_table, True)
    assert target.count().execute() == 0


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_correlated_subquery_exists(
    con, employee_data_1_temp_table, employee_data_2_temp_table
):
    # Delete rows whose salary matches a salary in the other table. The salary
    # sets ({100, 200, 300} vs {400, 500, 600}) are disjoint, so NO rows match
    # and NO rows should be deleted. Regression test: a correlated subquery
    # must never collapse into a tautology that deletes every row.
    target = con.table(employee_data_1_temp_table)
    source = con.table(employee_data_2_temp_table)

    con.delete(
        employee_data_1_temp_table,
        where=(source.salary == target.salary).any(),
    )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_correlated_subquery_not_exists(
    con, employee_data_1_temp_table, employee_data_2_temp_table
):
    # Delete rows whose salary does NOT match any salary in the other table.
    # The salary sets are disjoint, so ALL rows fail to match and ALL rows
    # should be deleted.
    target = con.table(employee_data_1_temp_table)
    source = con.table(employee_data_2_temp_table)

    con.delete(
        employee_data_1_temp_table,
        where=~(source.salary == target.salary).any(),
    )

    assert target.count().execute() == 0


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_correlated_subquery_compound(
    con, employee_data_1_temp_table, employee_data_2_temp_table
):
    # Compound predicate: a correlated EXISTS AND a simple predicate. Because
    # no salary matches, the EXISTS branch is false for every row, so the whole
    # predicate is false and NO rows should be deleted.
    target = con.table(employee_data_1_temp_table)
    source = con.table(employee_data_2_temp_table)

    con.delete(
        employee_data_1_temp_table,
        where=(source.salary == target.salary).any() & (target.department_name == "BB"),
    )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_uncorrelated_subquery(
    con, employee_data_1_temp_table, employee_data_2_temp_table
):
    # An uncorrelated subquery. Insert a known-overlapping salary into the
    # source, then delete target rows whose salary is in the source. Only the
    # 200 row should be removed.
    target = con.table(employee_data_1_temp_table)

    con.insert(employee_data_2_temp_table, [("M", "M", "MM", 200.0)])
    source = con.table(employee_data_2_temp_table)

    con.delete(employee_data_1_temp_table, where=target.salary.isin(source.salary))

    assert target.count().execute() == 2


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_scalar_subquery_predicate(con, employee_data_1_temp_table):
    # An aggregate over the target table compiles to a scalar subquery that
    # scans the table being deleted from. This is also the rewrite the
    # window-predicate error message recommends. Salaries are {100, 200, 300}
    # (mean 200), so only the 300 row is deleted.
    target = con.table(employee_data_1_temp_table)

    con.delete(employee_data_1_temp_table, target.salary > target.salary.mean())

    assert target.count().execute() == 2


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
def test_delete_null_predicate_semantics(con, employee_data_1_temp_table):
    # SQL three-valued logic: a DELETE only removes rows where the predicate
    # is TRUE. Rows where it evaluates to NULL survive.
    target = con.table(employee_data_1_temp_table)

    con.insert(
        employee_data_1_temp_table,
        ibis.memtable(
            {
                "first_name": ["N"],
                "last_name": ["O"],
                "department_name": ["NN"],
                "salary": [None],
            },
            schema=target.schema(),
        ),
    )

    con.delete(employee_data_1_temp_table, ibis._.salary > 150)

    result = target.execute()
    assert len(result) == 2  # the 100 row and the NULL-salary row
    assert result.salary.isna().sum() == 1


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
def test_delete_window_predicate_raises(con, employee_data_1_temp_table):
    # Window functions cannot appear in a DELETE statement's WHERE clause
    # (they compile to QUALIFY); ibis rejects them with a typed error rather
    # than emitting invalid SQL or crashing.
    target = con.table(employee_data_1_temp_table)

    with pytest.raises(com.UnsupportedOperationError, match=r"[Ww]indow"):
        con.delete(
            employee_data_1_temp_table,
            where=target.salary > target.salary.mean().over(ibis.window()),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
def test_delete_multiple_predicates_raises(con, employee_data_1_temp_table):
    # Unlike `filter`, `delete` takes a single predicate; a tuple is rejected
    # with a clear message pointing at `&` instead of a cryptic internal error.
    target = con.table(employee_data_1_temp_table)

    with pytest.raises(com.IbisInputError, match="single boolean predicate"):
        con.delete(
            employee_data_1_temp_table,
            where=(target.salary > 100, target.salary < 300),
        )

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
def test_delete_non_boolean_predicate_raises(con, employee_data_1_temp_table):
    # A non-boolean `where` (e.g. a table or a column name) is rejected with a
    # clear message instead of an internal unpacking error.
    target = con.table(employee_data_1_temp_table)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(employee_data_1_temp_table, where=target)

    with pytest.raises(com.IbisInputError, match="boolean predicate"):
        con.delete(employee_data_1_temp_table, where="salary")

    assert target.count().execute() == 3


@pytest.mark.notimpl(["polars"], reason="`delete` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["materialize"],
    raises=Exception,
    reason="Materialize restricts DML within transaction blocks",
)
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
            obj=ibis.memtable(
                {
                    "first_name": ["A", "B", "C"],
                    "last_name": ["D", "E", "F"],
                    "department_name": ["AA", "BB", "CC"],
                    "salary": [100.0, 200.0, 300.0],
                },
                schema=test_employee_schema,
            ),
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
