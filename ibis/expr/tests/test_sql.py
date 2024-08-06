from __future__ import annotations

import pytest

import ibis

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


def test_parse_sql_multiple_joins(snapshot):
    sql = """
SELECT *
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


tpch_catalog = {
    "lineitem": {
        "l_orderkey": "int32",
        "l_partkey": "int32",
        "l_suppkey": "int32",
        "l_linenumber": "int32",
        "l_quantity": "decimal(15, 2)",
        "l_extendedprice": "decimal(15, 2)",
        "l_discount": "decimal(15, 2)",
        "l_tax": "decimal(15, 2)",
        "l_returnflag": "string",
        "l_linestatus": "string",
        "l_shipdate": "date",
        "l_commitdate": "date",
        "l_receiptdate": "date",
        "l_shipinstruct": "string",
        "l_shipmode": "string",
        "l_comment": "string",
    }
}


def test_parse_sql_tpch1(snapshot):
    sql = """
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) AS sum_qty,
    sum(l_extendedprice) AS sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    avg(l_quantity) AS avg_qty,
    avg(l_extendedprice) AS avg_price,
    avg(l_discount) AS avg_disc,
    count(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
    """

    expr = ibis.parse_sql(sql, tpch_catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "ibis_tpch1.py")
