from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops

pytest.importorskip("black")

catalog = {
    "employee": {"first_name": "string", "last_name": "string", "id": "int64"},
    "call": {
        "start_time": "timestamp",
        "end_time": "timestamp",
        "employee_id": "int64",
        "call_outcome_id": "int64",
        "call_attempts": "int64",
    },
    "call_outcome": {"outcome_text": "string", "id": "int64"},
}


def test_parse_sql_basic_projection(snapshot):
    sql = "SELECT *, first_name as first FROM employee WHERE id < 5 ORDER BY id DESC"
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


@pytest.mark.parametrize("how", ["right", "left", "inner"])
def test_parse_sql_basic_join(how, snapshot):
    sql = f"""
SELECT
  *,
  first_name as first
FROM employee {how.upper()}
JOIN call ON
  employee.id = call.employee_id
WHERE
  id < 5
ORDER BY
  id DESC"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_limited_join(snapshot):
    sql = """
SELECT
  *,
  first_name as first
FROM employee
JOIN call ON
  employee.id = call.employee_id
LIMIT 3"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_multiple_joins(snapshot):
    sql = """
SELECT employee.*, call.*, call_outcome.outcome_text, call_outcome.id as id_right
FROM employee
JOIN call
  ON employee.id = call.employee_id
JOIN call_outcome
  ON call.call_outcome_id = call_outcome.id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_basic_aggregation(snapshot):
    sql = """
SELECT
  employee_id,
  sum(call_attempts) AS attempts
FROM call
GROUP BY employee_id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_basic_aggregation_with_join(snapshot):
    sql = """
SELECT
  id,
  sum(call_attempts) AS attempts
FROM employee
LEFT JOIN call
  ON employee.id = call.employee_id
GROUP BY id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_aggregation_with_multiple_joins(snapshot):
    sql = """
SELECT
  t.employee_id,
  AVG(t.call_attempts) AS avg_attempts
FROM (
  SELECT * FROM employee JOIN call ON employee.id = call.employee_id
  JOIN call_outcome ON call.call_outcome_id = call_outcome.id
) AS t
GROUP BY t.employee_id"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_simple_reduction(snapshot):
    sql = """SELECT AVG(call_attempts) AS mean FROM call"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_scalar_subquery(snapshot):
    sql = """
SELECT *
FROM call
WHERE call_attempts > (
  SELECT avg(call_attempts) AS mean
  FROM call
)"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_join_subquery(snapshot):
    sql = """
SELECT *
FROM call
INNER JOIN (
  SELECT
    AVG(call.call_attempts) AS mean
  FROM call
) AS subq
  ON subq.mean < call.call_attempts"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_simple_select_count(snapshot):
    sql = """SELECT COUNT(first_name) FROM employee"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_table_alias(snapshot):
    sql = """SELECT e.* FROM employee AS e"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_join_with_filter(snapshot):
    sql = """
SELECT *, first_name as first FROM employee
LEFT JOIN call ON employee.id = call.employee_id
WHERE id < 5
ORDER BY id DESC"""
    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


def test_parse_sql_in_clause(snapshot):
    sql = """
SELECT first_name FROM employee
WHERE first_name IN ('Graham', 'John', 'Terry', 'Eric', 'Michael')"""

    expr = ibis.parse_sql(sql, catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "decompiled.py")


tables = {
    "t": {"a": "int64", "b": "string", "c": "float64"},
    "u": {"a": "int64", "x": "string"},
    "v": {"a": "int64", "y": "string"},
}


def test_comma_join_conditions_become_join_keys():
    expr = ibis.parse_sql("SELECT t.a FROM t, u WHERE t.a = u.a AND t.c > 1", tables)
    (chain,) = expr.op().find(ops.JoinChain)
    (link,) = chain.rest
    assert link.how == "inner"
    (predicate,) = link.predicates
    assert isinstance(predicate, ops.Equals)
    # the condition on one table stays a filter, as written
    (filt,) = expr.op().find(ops.Filter)
    (predicate,) = filt.predicates
    assert isinstance(predicate, ops.Greater)


def test_comma_join_conditions_stay_filters_before_a_right_join():
    sql = "SELECT t.a FROM t, u RIGHT JOIN v ON u.a = v.a WHERE t.a = u.a"
    expr = ibis.parse_sql(sql, tables)
    (chain,) = expr.op().find(ops.JoinChain)
    first, second = chain.rest
    assert first.how == "cross"
    assert not first.predicates
    assert second.how == "right"
    (filt,) = expr.op().find(ops.Filter)
    (predicate,) = filt.predicates
    assert isinstance(predicate, ops.Equals)


def test_dropped_join_key_resolves_after_a_filter():
    sql = "SELECT u.a, u.x FROM t JOIN u ON t.a = u.a WHERE t.c > 1"
    expr = ibis.parse_sql(sql, tables)
    assert expr.columns == ("a", "x")
    assert "WHERE" in ibis.to_sql(expr)


def test_second_scan_of_a_table_is_a_view():
    expr = ibis.parse_sql(
        "SELECT t1.a FROM t AS t1 JOIN t AS t2 ON t1.a = t2.a", tables
    )
    assert expr.op().find(ops.SelfReference)


def test_correlated_scalar_subquery_over_a_derived_table_is_anchored():
    derived = "FROM (SELECT a, c FROM t WHERE a > 1) AS s"
    scalar = ibis.parse_sql(
        f"SELECT s.a {derived} WHERE s.c > (SELECT AVG(u.a) FROM u WHERE u.a = s.a)",
        tables,
    )
    # the derived table keeps its alias in the compiled SQL
    assert scalar.op().find(ops.SelfReference)
    exists = ibis.parse_sql(
        f"SELECT s.a {derived} WHERE EXISTS (SELECT 1 FROM u WHERE u.a = s.a)", tables
    )
    assert not exists.op().find(ops.SelfReference)


def test_duplicate_output_names_are_suffixed():
    expr = ibis.parse_sql("SELECT a, a FROM t", tables)
    assert expr.columns == ("a", "a_1")


def test_identity_projection_is_elided():
    expr = ibis.parse_sql("SELECT * FROM t WHERE a > 1", tables)
    assert isinstance(expr.op(), ops.Filter)


def test_join_projection_with_a_window_is_not_the_chain_output():
    joined = "FROM t JOIN u ON t.a = u.a"
    plain = ibis.parse_sql(f"SELECT t.a, u.x {joined}", tables)
    assert isinstance(plain.op(), ops.JoinChain)
    window = ibis.parse_sql(
        f"SELECT t.a, ROW_NUMBER() OVER (ORDER BY t.a) AS rn {joined}", tables
    )
    assert isinstance(window.op(), ops.Project)


def test_column_outside_group_by_is_rejected():
    with pytest.raises(com.IbisInputError, match="GROUP BY"):
        ibis.parse_sql("SELECT a, b FROM t GROUP BY a", tables)


def test_concat_null_handling_follows_the_dialect():
    sql = "SELECT CONCAT(b, 'x') AS bx FROM t"
    # CONCAT skips NULL arguments in DuckDB and propagates them in MySQL
    assert "COALESCE" in ibis.to_sql(ibis.parse_sql(sql, tables, dialect="duckdb"))
    assert "COALESCE" not in ibis.to_sql(ibis.parse_sql(sql, tables, dialect="mysql"))


def test_from_less_select_keeps_its_order_by():
    expr = ibis.parse_sql("SELECT 2 AS two, 1 AS one ORDER BY one", tables)
    assert isinstance(expr.op(), ops.Sort)
