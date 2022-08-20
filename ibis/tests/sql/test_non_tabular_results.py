import pandas as pd

import ibis
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import get_query


def test_simple_scalar_aggregates(con):
    # Things like table.column.{sum, mean, ...}()
    table = con.table('alltypes')

    expr = table[table.c > 0].f.sum()

    query = get_query(expr)

    sql_query = query.compile()
    expected = """\
SELECT sum(`f`) AS `sum`
FROM alltypes
WHERE `c` > 0"""

    assert sql_query == expected

    # Maybe the result handler should act on the cursor. Not sure.
    handler = query.result_handler
    output = pd.DataFrame({'sum': [5]})
    assert handler(output) == 5


def test_scalar_aggregates_multiple_tables(con):
    # #740
    table = ibis.table([('flag', 'string'), ('value', 'double')], 'tbl')

    flagged = table[table.flag == '1']
    unflagged = table[table.flag == '0']

    expr = flagged.value.mean() / unflagged.value.mean() - 1

    result = Compiler.to_sql(expr)
    expected = """\
SELECT (t0.`mean` / t1.`mean`) - 1 AS `tmp`
FROM (
  SELECT avg(`value`) AS `mean`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) AS `mean`
    FROM tbl
    WHERE `flag` = '0'
  ) t1"""
    assert result == expected

    fv = flagged.value
    uv = unflagged.value

    expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())
    result = Compiler.to_sql(expr)
    expected = """\
SELECT t0.`tmp` - t1.`tmp` AS `tmp`
FROM (
  SELECT avg(`value`) / sum(`value`) AS `tmp`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) / sum(`value`) AS `tmp`
    FROM tbl
    WHERE `flag` = '0'
  ) t1"""
    assert result == expected


def test_table_column_unbox(alltypes):
    table = alltypes
    m = table.f.sum().name('total')
    agged = table[table.c > 0].group_by('g').aggregate([m])
    expr = agged.g

    query = get_query(expr)

    sql_query = query.compile()
    expected = """\
SELECT `g`
FROM (
  SELECT `g`, sum(`f`) AS `total`
  FROM alltypes
  WHERE `c` > 0
  GROUP BY 1
) t0"""

    assert sql_query == expected

    # Maybe the result handler should act on the cursor. Not sure.
    handler = query.result_handler
    output = pd.DataFrame({'g': ['foo', 'bar', 'baz']})
    assert (handler(output) == output['g']).all()


def test_complex_array_expr_projection(alltypes):
    table = alltypes
    # May require finding the base table and forming a projection.
    expr = table.group_by('g').aggregate([table.count().name('count')])
    expr2 = expr.g.cast('double')

    query = Compiler.to_sql(expr2)
    expected = """\
SELECT CAST(`g` AS double) AS `cast(g, float64)`
FROM (
  SELECT `g`, count(*) AS `count`
  FROM alltypes
  GROUP BY 1
) t0"""
    assert query == expected


def test_scalar_exprs_no_table_refs(con):
    expr1 = ibis.now()
    expected1 = "SELECT now() AS `tmp`"

    expr2 = ibis.literal(1) + ibis.literal(2)
    expected2 = "SELECT 1 + 2 AS `tmp`"

    cases = [(expr1, expected1), (expr2, expected2)]

    for expr, expected in cases:
        result = Compiler.to_sql(expr)
        assert result == expected


def test_isnull_case_expr_rewrite_failure(alltypes):
    # #172, case expression that was not being properly converted into an
    # aggregation
    reduction = alltypes.g.isnull().ifelse(1, 0).sum()

    result = Compiler.to_sql(reduction)
    expected = """\
SELECT sum(if(`g` IS NULL, 1, 0)) AS `sum`
FROM alltypes"""
    assert result == expected
