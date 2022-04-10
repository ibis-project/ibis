import pytest

import ibis.expr.operations as ops
from ibis.tests.sql.conftest import get_query

pytest.importorskip('sqlalchemy')


def test_ast_with_projection_join_filter(con):
    table = con.table('test1')
    table2 = con.table('test2')

    filter_pred = table['f'] > 0

    table3 = table[filter_pred]

    join_pred = table3['g'] == table2['key']

    joined = table2.inner_join(table3, [join_pred])
    result = joined[[table3, table2['value']]]

    stmt = get_query(result)

    def foo():
        table3 = table[filter_pred]
        joined = table2.inner_join(table3, [join_pred])
        result = joined[[table3, table2['value']]]
        return result

    assert len(stmt.select_set) == 2

    # #790, make sure the filter stays put
    assert len(stmt.where) == 0

    # Check that the joined tables are not altered
    tbl = stmt.table_set
    tbl_node = tbl.op()
    assert isinstance(tbl_node, ops.InnerJoin)
    assert tbl_node.left is table2
    assert tbl_node.right is table3


def test_ast_with_aggregation_join_filter(con):
    table = con.table('test1')
    table2 = con.table('test2')

    filter_pred = table['f'] > 0
    table3 = table[filter_pred]
    join_pred = table3['g'] == table2['key']

    joined = table2.inner_join(table3, [join_pred])

    met1 = (table3['f'] - table2['value']).mean().name('foo')
    result = joined.aggregate(
        [met1, table3['f'].sum().name('bar')],
        by=[table3['g'], table2['key']],
    )

    stmt = get_query(result)

    # #790, this behavior was different before
    ex_pred = [table3['g'] == table2['key']]
    expected_table_set = table2.inner_join(table3, ex_pred)
    assert stmt.table_set.equals(expected_table_set)

    # Check various exprs
    ex_metrics = [
        (table3['f'] - table2['value']).mean().name('foo'),
        table3['f'].sum().name('bar'),
    ]
    ex_by = [table3['g'], table2['key']]
    for res, ex in zip(stmt.select_set, ex_by + ex_metrics):
        assert res.equals(ex)

    for res, ex in zip(stmt.group_by, ex_by):
        assert stmt.select_set[res].equals(ex)

    # The filter is in the joined subtable
    assert len(stmt.where) == 0
