"""`parse_sql` on everyday SQL, checked against DuckDB running the same SQL."""

from __future__ import annotations

import decimal

import duckdb
import pandas as pd
import pytest

import ibis
from ibis.tests.util import assert_same_result, decompile_roundtrip, expose_sort_keys

pytest.importorskip("duckdb")

CATALOG = {
    "t": {"a": "int64", "b": "string", "c": "float64", "d": "date", "ts": "timestamp"},
    "u": {"a": "int64", "x": "string"},
}

T_ROWS = """
VALUES
  (1, 'foo', 1.5, DATE '2024-01-01', TIMESTAMP '2024-01-01 10:00:00'),
  (2, 'bar', 2.5, DATE '2024-02-01', TIMESTAMP '2024-02-02 11:00:00'),
  (3, NULL, NULL, DATE '2024-03-01', TIMESTAMP '2024-03-03 12:00:00'),
  (2, 'baz', 4.0, DATE '2024-04-01', TIMESTAMP '2024-04-04 13:00:00')
"""
U_ROWS = "VALUES (1, 'p'), (2, 'q'), (9, 'r')"


@pytest.fixture(scope="module")
def duck():
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM ({T_ROWS}) v(a, b, c, d, ts)")
    con.execute(f"CREATE TABLE u AS SELECT * FROM ({U_ROWS}) v(a, x)")
    return con


CASES = {
    "select_where": "SELECT a, b FROM t WHERE a > 1",
    "count_star": "SELECT COUNT(*) AS n FROM t",
    "count_star_group_by": "SELECT b, COUNT(*) AS n FROM t GROUP BY b",
    "count_constant": "SELECT b, COUNT(1) AS n, COUNT(1) FILTER (WHERE a > 1) AS m, COUNT(TRUE) AS k FROM t GROUP BY b",
    "ranking_empty_window": "SELECT a, RANK() OVER () AS r, ROW_NUMBER() OVER () > 0 AS pos FROM t",
    "distinct": "SELECT DISTINCT a FROM t",
    "having": "SELECT a, SUM(c) AS s FROM t GROUP BY a HAVING SUM(c) > 2",
    "order_limit_offset": "SELECT a FROM t ORDER BY a LIMIT 2 OFFSET 1",
    "case_when": "SELECT CASE WHEN a > 1 THEN 'big' ELSE 'small' END AS sz FROM t",
    "not_is_null": "SELECT a FROM t WHERE NOT (b IS NULL)",
    "is_null": "SELECT a FROM t WHERE b IS NULL",
    "is_true_false": "SELECT a, (c > 2) IS TRUE AS yes, (c > 2) IS FALSE AS no FROM t",
    "is_not_true_where": "SELECT a FROM t WHERE (c > 2) IS NOT TRUE",
    "is_not_false_where": "SELECT a FROM t WHERE (c > 2) IS NOT FALSE",
    "like": "SELECT a FROM t WHERE b LIKE 'ba%'",
    "not_like": "SELECT a FROM t WHERE b NOT LIKE 'ba%'",
    "not_ilike": "SELECT a FROM t WHERE b NOT ILIKE 'BA%' AND b ILIKE '%A%'",
    "not_like_join_on": "SELECT t.a, u.x FROM t LEFT JOIN u ON t.a = u.a AND u.x NOT LIKE 'q%'",
    "between": "SELECT a FROM t WHERE a BETWEEN 1 AND 2",
    "coalesce": "SELECT COALESCE(b, 'none') AS b2 FROM t",
    "concat_ws": "SELECT CONCAT_WS('-', b, 'x') AS s, CONCAT_WS('-', b, NULL, 'y', b) AS s2, CONCAT_WS('-', NULL, NULL) AS s3 FROM t",
    "string_functions": "SELECT UPPER(b) AS ub, LENGTH(b) AS lb, SUBSTRING(b, 1, 2) AS sb FROM t",
    "date_functions": "SELECT EXTRACT(year FROM d) AS y, DATE_TRUNC('month', ts) AS m FROM t",
    "date_add_interval": "SELECT DATE_ADD(d, INTERVAL 1 DAY) AS d1, DATE_ADD(ts, INTERVAL 2 HOUR) AS t2, DATE_ADD(d, INTERVAL '3' MONTH) AS m3 FROM t",
    "date_arithmetic": "SELECT a FROM t WHERE d >= DATE '2024-02-01' AND d < DATE '2024-02-01' + INTERVAL '1' MONTH",
    "cast": "SELECT CAST(a AS VARCHAR) AS s FROM t",
    "decimal_literal": "SELECT CAST(1.10 AS VARCHAR) AS s, 1.50 AS x, 0.1 + 0.2 = 0.3 AS eq, 1e2 AS e FROM t",
    "decimal_literal_arithmetic": "SELECT c * 0.5 AS h, c / 0.25 AS q, a * 1.25 AS p, c + 0.1 AS plus, 0.1 + c AS plus2 FROM t",
    "decimal_literal_scale": "SELECT SUM(a) * 0.0001000000 AS x, SUM(a) * 100.0000 AS y FROM t",
    "decimal_column_literal_scale": "SELECT CAST(c AS DECIMAL(10, 2)) * 1.00000 AS z, CAST(c AS DECIMAL(10, 2)) + 0.0001 AS w FROM t",
    "arithmetic_alias": "SELECT a * 2 + 1 AS z, c / 2 AS h FROM t",
    "window_row_number": "SELECT a, ROW_NUMBER() OVER (PARTITION BY b ORDER BY a) AS rn FROM t",
    "window_cumulative_sum": "SELECT a, SUM(c) OVER (ORDER BY a) AS cs FROM t",
    "window_rows_frame": "SELECT a, SUM(c) OVER (ORDER BY a ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t",
    "window_range_frame": "SELECT a, SUM(c) OVER (PARTITION BY b ORDER BY a RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM t",
    "window_lag_first_value": "SELECT a, LAG(a) OVER (ORDER BY a) AS p, FIRST_VALUE(b) OVER (PARTITION BY b ORDER BY a) AS f FROM t",
    "window_rank": "SELECT a, RANK() OVER (ORDER BY c DESC) AS r, DENSE_RANK() OVER (ORDER BY a) AS d FROM t",
    "window_ntile": "SELECT a, NTILE(2) OVER (ORDER BY a) AS n FROM t",
    "window_over_aggregate": "SELECT a, SUM(c) AS s, AVG(SUM(c)) OVER () AS avg_s FROM t GROUP BY a",
    "cte": "WITH x AS (SELECT a, c FROM t WHERE a > 1) SELECT a, SUM(c) AS s FROM x GROUP BY a",
    "cte_with_union": "WITH x AS (SELECT a FROM t UNION ALL SELECT a FROM u) SELECT a, COUNT(*) AS n FROM x GROUP BY a",
    "cte_referenced_twice": "WITH x AS (SELECT a, c FROM t WHERE a > 1) SELECT x1.a, x2.c FROM x AS x1 JOIN x AS x2 ON x1.a = x2.a",
    "derived_table": "SELECT s.a FROM (SELECT a FROM t WHERE a > 1) AS s",
    "derived_union": "SELECT s.a FROM (SELECT a FROM t UNION SELECT a FROM u) AS s WHERE s.a > 1",
    "union_all": "SELECT a FROM t UNION ALL SELECT a FROM u",
    "union": "SELECT a FROM t UNION SELECT a FROM u",
    "union_order_limit": "SELECT a FROM t UNION ALL SELECT a FROM u ORDER BY a LIMIT 3",
    "except": "SELECT a FROM t EXCEPT SELECT a FROM u",
    "intersect": "SELECT a FROM t INTERSECT SELECT a FROM u",
    "in_subquery": "SELECT a FROM t WHERE a IN (SELECT a FROM u)",
    "exists": "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)",
    "not_exists": "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a)",
    "correlated_scalar_subquery": "SELECT a FROM t WHERE c > (SELECT AVG(c) FROM t AS t2 WHERE t2.a = t.a)",
    "scalar_subquery": "SELECT a FROM t WHERE c > (SELECT AVG(c) FROM t)",
    "join": "SELECT t.a, u.x FROM t JOIN u ON t.a = u.a",
    "left_join_anti": "SELECT t.a, u.x FROM t LEFT JOIN u ON t.a = u.a WHERE u.x IS NULL",
    "self_join": "SELECT t1.a, t2.b FROM t AS t1 JOIN t AS t2 ON t1.a = t2.a",
    "cross_join": "SELECT t.a, u.x FROM t CROSS JOIN u",
    "count_distinct": "SELECT COUNT(DISTINCT a) AS n FROM t",
    "reductions": "SELECT AVG(c) AS m, MIN(a) AS lo, MAX(a) AS hi FROM t",
    "stddev_variance": "SELECT STDDEV(c) AS s, VAR_POP(c) AS v FROM t",
    "or": "SELECT a FROM t WHERE a = 1 OR b = 'baz'",
    "select_without_from": "SELECT 1 AS one, 'x' AS s",
    "order_by_desc_nulls_last": "SELECT a, c FROM t ORDER BY c DESC NULLS LAST",
    "group_by_expression": "SELECT a % 2 AS parity, COUNT(*) AS n FROM t GROUP BY a % 2",
    "aggregate_of_expression": "SELECT SUM(c * 2) AS s FROM t",
    "math_functions": "SELECT ROUND(c, 1) AS r, ABS(a - 2) AS ab, GREATEST(a, 2) AS g FROM t",
    "concat": "SELECT b || '_' || b AS bb, CONCAT(b, 'x') AS bx FROM t",
    "derived_computed": "SELECT s.z FROM (SELECT a + c AS z FROM t) AS s",
    "derived_where_alias": "SELECT s.aa FROM (SELECT a AS aa FROM t) AS s WHERE s.aa > 1",
    "nested_derived": "SELECT q.z FROM (SELECT s.z + 1 AS z FROM (SELECT a AS z FROM t) AS s) AS q",
    "derived_with_limit": "SELECT s.a FROM (SELECT a FROM t ORDER BY a LIMIT 2) AS s",
    "scalar_sub_select": "SELECT a, (SELECT MAX(a) FROM u) AS m FROM t",
    "corr_scalar_select": "SELECT a, (SELECT MAX(u.a) FROM u WHERE u.a = t.a) AS m FROM t",
    "corr_scalar_select_after_where": "SELECT a, (SELECT COUNT(*) FROM u WHERE u.a = t.a) AS n FROM t WHERE c > 1",
    "count_star_arith": "SELECT SUM(c) / COUNT(*) AS r FROM t",
    "order_by_nonprojected": "SELECT a FROM t ORDER BY c DESC",
    "order_by_alias": "SELECT a + 1 AS x FROM t ORDER BY x",
    "order_by_ordinal": "SELECT a, b FROM t ORDER BY 2, 1",
    "order_by_agg_not_selected": "SELECT a FROM t GROUP BY a ORDER BY SUM(c)",
    "having_alias": "SELECT a, SUM(c) AS s FROM t GROUP BY a HAVING s > 2",
    "having_count_not_selected": "SELECT b FROM t GROUP BY b HAVING COUNT(*) > 1",
    "having_corr_sub": "SELECT a, SUM(c) AS s FROM t GROUP BY a HAVING SUM(c) > (SELECT AVG(u.a) FROM u WHERE u.a = t.a)",
    "select_corr_after_group": "SELECT a, (SELECT COUNT(*) FROM u WHERE u.a = t.a) AS n FROM t GROUP BY a",
    "group_by_expression_alias": "SELECT a % 2 AS parity, COUNT(*) AS n FROM t GROUP BY parity",
    "group_by_ordinal": "SELECT a % 2, COUNT(*) FROM t GROUP BY 1",
    "dup_names": "SELECT a, a FROM t",
    "dup_alias_select": "SELECT a, COUNT(*) AS a FROM t GROUP BY a",
    "eq_null": "SELECT a FROM t WHERE b = NULL",
    "null_arith": "SELECT a + NULL AS z FROM t",
    "try_cast": "SELECT TRY_CAST(b AS INTEGER) AS i FROM t",
    "using": "SELECT a, x FROM t JOIN u USING (a)",
    "semi": "SELECT a FROM t SEMI JOIN u ON t.a = u.a",
    "anti": "SELECT a FROM t ANTI JOIN u ON t.a = u.a",
    "right_join": "SELECT t.a, u.x FROM t RIGHT JOIN u ON t.a = u.a",
    "full_join": "SELECT t.a, u.x FROM t FULL OUTER JOIN u ON t.a = u.a",
    "select_star_join": "SELECT * FROM t JOIN u ON t.a = u.a",
    "window_filter_derived": "SELECT * FROM (SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn FROM t) AS s WHERE rn = 1",
    "window_order_by_dropped": "SELECT a, SUM(c) OVER (PARTITION BY b) AS s FROM t ORDER BY b",
    "subq_in_join_source": "SELECT t.a, s.m FROM t JOIN (SELECT a, MAX(c) AS m FROM t GROUP BY a) AS s ON t.a = s.a WHERE s.m > 1",
    "join_on_subquery_eq": "SELECT t.a FROM t JOIN u ON t.a = u.a AND t.c > (SELECT AVG(c) FROM t)",
    "three_way_comma": "SELECT t.a, t2.b FROM t, u, t AS t2 WHERE t.a = u.a AND u.a = t2.a AND t2.c > 1",
    "comma_join_where_or": "SELECT t.a FROM t, u WHERE (t.a = u.a AND t.c > 1) OR (t.a = u.a + 1 AND t.c < 1)",
    "left_join_filter_right": "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a WHERE u.x = 'p'",
    "named_window": "SELECT a, SUM(c) OVER w AS s FROM t WINDOW w AS (PARTITION BY b ORDER BY a)",
    "named_window_extended": "SELECT a, SUM(c) OVER (w ORDER BY a) AS s, SUM(c) OVER (w2 ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s2 FROM t WINDOW w AS (PARTITION BY b), w2 AS (PARTITION BY b ORDER BY a)",
    "window_frame_start_only": "SELECT a, SUM(c) OVER (ORDER BY a ROWS 1 PRECEDING) AS s1, SUM(c) OVER (ORDER BY a RANGE UNBOUNDED PRECEDING) AS s2, SUM(c) OVER (ORDER BY a ROWS CURRENT ROW) AS s3 FROM t",
    "window_filter_default_frame": "SELECT a, SUM(c) FILTER (WHERE a > 1) OVER (ORDER BY a) AS s FROM t",
    "count_star_window_group": "SELECT a, COUNT(*) OVER () AS n FROM t GROUP BY a",
    "filter_where_agg": "SELECT SUM(c) FILTER (WHERE a > 1) AS s FROM t",
    "any_subquery": "SELECT a FROM t WHERE a = ANY (SELECT a FROM u)",
    "distinct_order_by": "SELECT DISTINCT a FROM t ORDER BY a",
    "cte_corr_scalar": "WITH x AS (SELECT a, c FROM t WHERE a > 1) SELECT a FROM x WHERE c > (SELECT AVG(u.a) FROM u WHERE u.a = x.a)",
    "exists_join_inner": "SELECT t.a FROM t WHERE EXISTS (SELECT 1 FROM u JOIN t AS t2 ON u.a = t2.a WHERE t2.c = t.c)",
    "rank_over_sum_group": "SELECT a, RANK() OVER (ORDER BY SUM(c) DESC) AS r FROM t GROUP BY a",
    "extract_day_of_week": "SELECT EXTRACT(dow FROM d) AS dow, EXTRACT(isodow FROM d) AS iso, EXTRACT(dow FROM DATE '2024-01-07') AS sun, EXTRACT(isodow FROM DATE '2024-01-07') AS isosun, EXTRACT(doy FROM d) AS doy FROM t",
    "extract_more_units": "SELECT EXTRACT(epoch FROM ts) AS e, EXTRACT(week FROM d) AS w, EXTRACT(quarter FROM d) AS q, EXTRACT(millisecond FROM ts) AS ms FROM t",
    "date_part": "SELECT DATE_PART('year', d) AS y, DATE_PART('dow', d) AS dow, DATE_PART('days', d) AS dd FROM t",
    "is_distinct_from": "SELECT a FROM t WHERE b IS DISTINCT FROM 'foo'",
    "is_not_distinct_from_null": "SELECT a FROM t WHERE b IS NOT DISTINCT FROM NULL",
    "join_is_not_distinct_from": "SELECT t.a, u.x FROM t JOIN u ON t.a IS NOT DISTINCT FROM u.a",
    "string_agg": "SELECT STRING_AGG(b, ',' ORDER BY a, b) AS s FROM t",
    "string_agg_group": "SELECT a, STRING_AGG(b, '|' ORDER BY b) AS s, STRING_AGG(b ORDER BY b) AS t FROM t GROUP BY a",
    "bool_or_and": "SELECT BOOL_OR(a > 1) AS any_big, BOOL_AND(a > 1) AS all_big FROM t",
    "int_div": "SELECT a // 2 AS h, a % 2 AS m, -a // 2 AS nh, -a % 2 AS nm, a // -2 AS hn FROM t",
    "date_diff": "SELECT DATE_DIFF('day', d, DATE '2024-06-01') AS days, DATE_DIFF('month', d, DATE '2024-06-01') AS months, DATE_DIFF('hour', ts, TIMESTAMP '2024-06-01 00:00:00') AS hours FROM t",
    "current_date": "SELECT a FROM t WHERE d < CURRENT_DATE AND ts < CURRENT_TIMESTAMP AND ts < NOW() AND d < TODAY()",
    "exp_log": "SELECT EXP(c) AS e, LOG(c) AS l10, LOG(2, c) AS l2, LN(c) AS l FROM t",
    "position": "SELECT POSITION('a' IN b) AS p, STRPOS(b, 'z') AS z FROM t",
    "split_part": "SELECT SPLIT_PART(b, 'a', 1) AS p1, SPLIT_PART(b, 'a', 2) AS p2 FROM t",
    "regexp_matches": "SELECT a FROM t WHERE REGEXP_MATCHES(b, '^ba')",
    "select_without_from_order_by": "SELECT 2 AS two, 1 AS one ORDER BY one",
    "trim_sides": "SELECT LTRIM('  x ') AS l, RTRIM('  x ') AS r, TRIM(LEADING FROM ' y ') AS tl, TRIM(TRAILING FROM ' y ') AS tt, TRIM(b) AS b FROM t",
    "fetch_first": "SELECT a FROM t ORDER BY a FETCH FIRST 2 ROWS ONLY",
    "window_partition_by_aggregate": "SELECT a, SUM(c) AS s, COUNT(*) OVER (PARTITION BY SUM(c) > 2) AS n FROM t GROUP BY a",
    "having_without_group_by": "SELECT SUM(c) AS s FROM t HAVING SUM(c) > 1",
    "star_exclude_replace": "SELECT * EXCLUDE (ts) REPLACE (a + 1 AS a) FROM t",
    "exists_uncorrelated": "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)",
    "except_all": "SELECT a FROM t EXCEPT ALL SELECT a FROM u",
    "intersect_all": "SELECT a FROM t INTERSECT ALL SELECT a FROM t",
    "aliased_negation": "SELECT a, NOT (c > 2) AS x, b IS NOT NULL AS nn, a NOT IN (1, 2) AS ni, -(-a) AS k, (c > 2) IS NOT TRUE AS nt, -2.5 AS neg, NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a) AS ne FROM t",
    "cte_self_exists": "WITH x AS (SELECT * FROM t WHERE a > 1) SELECT a FROM x WHERE EXISTS (SELECT 1 FROM x AS x2 WHERE x2.a = x.a AND x2.b <> x.b)",
    "nested_exists": "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND EXISTS (SELECT 1 FROM t AS t2 WHERE t2.a = t.a AND t2.c > 2))",
    "nullif_interval": "SELECT NULLIF(b, 'foo') AS n, d + INTERVAL (a) DAY AS dd FROM t",
    "exists_uncorrelated_rescan": "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM t AS t2)",
}


# the cases the planner-based ``parse_sql`` translates today; every other case
# is a strict xfail until the converter that replaces the planner lands
TRANSLATED_CASES = frozenset(
    {
        "aggregate_of_expression",
        "arithmetic_alias",
        "between",
        "cast",
        "comma_join_where_or",
        "count_star",
        "count_star_group_by",
        "cross_join",
        "cte",
        "date_arithmetic",
        "decimal_literal_arithmetic",
        "decimal_literal_scale",
        "derived_computed",
        "derived_table",
        "derived_where_alias",
        "join",
        "join_on_subquery_eq",
        "left_join_filter_right",
        "nested_derived",
        "or",
        "order_by_agg_not_selected",
        "order_by_desc_nulls_last",
        "order_by_nonprojected",
        "order_by_ordinal",
        "reductions",
        "right_join",
        "scalar_subquery",
        "select_where",
        "semi",
        "star_exclude_replace",
        "subq_in_join_source",
    }
)


def _params(table, translated):
    """Parametrize over ``table``; a case not in ``translated`` is a strict xfail."""
    return [
        pytest.param(
            value,
            id=name,
            marks=[]
            if name in translated
            else [
                pytest.mark.xfail(
                    reason="parse_sql cannot translate this yet", strict=True
                )
            ],
        )
        for name, value in table.items()
    ]


def assert_matches_duckdb(duck, expr, sql):
    """Assert that ``expr`` computes what DuckDB computes for ``sql``."""
    ordered_sql, sort_keys = expose_sort_keys(sql, dialect="duckdb")
    expected = duck.execute(ordered_sql).df()
    result = ibis.duckdb.from_connection(duck).to_pandas(expr)
    assert_same_result(result, expected, sort_keys=sort_keys)


@pytest.mark.parametrize("sql", _params(CASES, TRANSLATED_CASES))
def test_parse_sql_matches_duckdb(duck, sql):
    expr = ibis.parse_sql(sql, CATALOG, dialect="duckdb")
    assert_matches_duckdb(duck, expr, sql)


@pytest.mark.parametrize("sql", _params(CASES, TRANSLATED_CASES))
def test_decompile_roundtrip_matches_duckdb(duck, sql):
    """Parsed SQL decompiles to Python that rebuilds an equivalent expression."""
    expr = decompile_roundtrip(ibis.parse_sql(sql, CATALOG, dialect="duckdb"))
    assert_matches_duckdb(duck, expr, sql)


def test_comparator_checks_row_order(duck):
    sql, sort_keys = expose_sort_keys("SELECT a, b FROM t ORDER BY a", "duckdb")
    expected = duck.execute(sql).df()
    assert list(expected.columns) == ["a", "b", "_sort_key_0"]
    rows = expected[["a", "b"]]
    # the two rows with a = 2 tie on the key and may come in either order
    assert_same_result(rows.iloc[[0, 2, 1, 3]], expected, sort_keys=sort_keys)
    with pytest.raises(AssertionError, match="differs at row 0"):
        assert_same_result(rows.iloc[::-1], expected, sort_keys=sort_keys)
    # without a top-level ORDER BY the rows are a multiset
    assert_same_result(rows.iloc[::-1], rows)


@pytest.mark.parametrize(
    ("sql", "ascending"),
    [
        ("SELECT a, b FROM t ORDER BY 2 DESC, 1", False),
        ("SELECT a + 1 AS x, b FROM t ORDER BY x DESC", False),
        # the output column named a, not the source column
        ("SELECT b AS a, a AS b FROM t ORDER BY a", True),
        ("SELECT * FROM t ORDER BY 2 DESC, c", False),
        ("SELECT a FROM t GROUP BY a ORDER BY SUM(c) DESC", False),
        ("SELECT a FROM t UNION ALL SELECT a FROM u ORDER BY 1 DESC", False),
    ],
)
def test_comparator_resolves_sort_keys(duck, sql, ascending):
    ordered_sql, sort_keys = expose_sort_keys(sql, "duckdb")
    expected = duck.execute(ordered_sql).df()
    key = expected.iloc[:, sort_keys[0]].dropna()
    assert key.is_monotonic_increasing if ascending else key.is_monotonic_decreasing
    result = duck.execute(sql).df()
    assert_same_result(result, expected, sort_keys=sort_keys)
    with pytest.raises(AssertionError, match="differs"):
        assert_same_result(result.iloc[::-1], expected, sort_keys=sort_keys)


def test_comparator_checks_column_names(duck):
    expected = duck.execute("SELECT a AS x, COUNT(*) FROM t GROUP BY a").df()
    assert list(expected.columns) == ["x", "count_star()"]
    # case does not count, and an unaliased expression is named by the engine
    assert_same_result(expected.set_axis(["X", "_col_1"], axis=1), expected)
    with pytest.raises(AssertionError, match="column names differ"):
        assert_same_result(expected.set_axis(["y", "_col_1"], axis=1), expected)


def test_comparator_checks_value_kinds(duck):
    expected = duck.execute("SELECT a FROM t").df()
    with pytest.raises(AssertionError, match="holds string values"):
        assert_same_result(expected.astype(str), expected)


def test_comparator_admits_narrow_decimals(duck):
    expected = duck.execute("SELECT AVG(c) AS m FROM t").df()
    assert expected["m"].iloc[0] == pytest.approx(2.6666666)
    rounded = pd.DataFrame({"m": [decimal.Decimal("2.67")]})
    with pytest.raises(AssertionError, match="narrower decimal scale"):
        assert_same_result(rounded, expected)
    assert_same_result(rounded, expected, narrow_decimals=True)
    exact = pd.DataFrame({"m": [decimal.Decimal(str(expected["m"].iloc[0]))]})
    with pytest.raises(AssertionError, match="drop `narrow_decimals`"):
        assert_same_result(exact, expected, narrow_decimals=True)
