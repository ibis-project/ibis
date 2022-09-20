import pandas as pd

import ibis
from ibis.tests.sql.conftest import get_query, sqlgolden, sqlgoldens


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


@sqlgoldens
def test_scalar_aggregates_multiple_tables(con):
    # #740
    table = ibis.table([('flag', 'string'), ('value', 'double')], 'tbl')

    flagged = table[table.flag == '1']
    unflagged = table[table.flag == '0']

    yield flagged.value.mean() / unflagged.value.mean() - 1

    fv = flagged.value
    uv = unflagged.value

    fv_stat = (fv.mean() / fv.sum()).name('fv')
    uv_stat = (uv.mean() / uv.sum()).name('uv')

    yield (fv_stat - uv_stat).name('tmp')


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


@sqlgolden
def test_complex_array_expr_projection(alltypes):
    table = alltypes
    # May require finding the base table and forming a projection.
    expr = table.group_by('g').aggregate([table.count().name('count')])
    expr2 = expr.g.cast('double')

    return expr2


@sqlgoldens
def test_scalar_exprs_no_table_refs(con):
    yield ibis.now().name('tmp')
    yield ibis.literal(1) + ibis.literal(2)


@sqlgolden
def test_isnull_case_expr_rewrite_failure(alltypes):
    # #172, case expression that was not being properly converted into an
    # aggregation
    return alltypes.g.isnull().ifelse(1, 0).sum()
