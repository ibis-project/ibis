import pandas as pd
import pytest
from pytest import param

import ibis
from ibis.tests.sql.conftest import get_query, to_sql


def test_simple_scalar_aggregates(con, snapshot):
    # Things like table.column.{sum, mean, ...}()
    table = con.table('alltypes')

    expr = table[table.c > 0].f.sum()

    query = get_query(expr)

    sql_query = query.compile()

    snapshot.assert_match(sql_query, "out.sql")

    # Maybe the result handler should act on the cursor. Not sure.
    handler = query.result_handler
    output = pd.DataFrame({'sum': [5]})
    assert handler(output) == 5


def test_scalar_aggregates_multiple_tables(snapshot):
    # #740
    table = ibis.table([('flag', 'string'), ('value', 'double')], 'tbl')

    flagged = table[table.flag == '1']
    unflagged = table[table.flag == '0']

    expr = flagged.value.mean() / unflagged.value.mean() - 1
    snapshot.assert_match(to_sql(expr), "mean.sql")

    fv = flagged.value
    uv = unflagged.value

    fv_stat = (fv.mean() / fv.sum()).name('fv')
    uv_stat = (uv.mean() / uv.sum()).name('uv')

    expr = (fv_stat - uv_stat).name('tmp')
    snapshot.assert_match(to_sql(expr), "mean_sum.sql")


def test_table_column_unbox(alltypes, snapshot):
    table = alltypes
    m = table.f.sum().name('total')
    agged = table[table.c > 0].group_by('g').aggregate([m])
    expr = agged.g

    query = get_query(expr)
    snapshot.assert_match(query.compile(), "out.sql")

    # Maybe the result handler should act on the cursor. Not sure.
    handler = query.result_handler
    output = pd.DataFrame({'g': ['foo', 'bar', 'baz']})
    assert (handler(output) == output['g']).all()


def test_complex_array_expr_projection(alltypes, snapshot):
    table = alltypes
    # May require finding the base table and forming a projection.
    expr = table.group_by('g').aggregate([table.count().name('count')])
    expr2 = expr.g.cast('double')
    snapshot.assert_match(to_sql(expr2), "out.sql")


@pytest.mark.parametrize(
    "expr",
    [
        param(ibis.now().name('tmp'), id="now"),
        param(ibis.literal(1) + ibis.literal(2), id="add"),
    ],
)
def test_scalar_exprs_no_table_refs(expr, snapshot):
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_isnull_case_expr_rewrite_failure(alltypes, snapshot):
    # #172, case expression that was not being properly converted into an
    # aggregation
    expr = alltypes.g.isnull().ifelse(1, 0).sum()
    snapshot.assert_match(to_sql(expr), "out.sql")
