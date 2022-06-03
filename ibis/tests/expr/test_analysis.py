import time

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as L
import ibis.expr.operations as ops
from ibis.tests.util import assert_equal

# TODO: test is_reduction
# TODO: test is_scalar_reduction

# Place to collect esoteric expression analysis bugs and tests


def test_rewrite_join_projection_without_other_ops(con):
    # See #790, predicate pushdown in joins not supported

    # Star schema with fact table
    table = con.table('star1')
    table2 = con.table('star2')
    table3 = con.table('star3')

    filtered = table[table['f'] > 0]

    pred1 = table['foo_id'] == table2['foo_id']
    pred2 = filtered['bar_id'] == table3['bar_id']

    j1 = filtered.left_join(table2, [pred1])
    j2 = j1.inner_join(table3, [pred2])

    # Project out the desired fields
    view = j2[[filtered, table2['value1'], table3['value2']]]

    # Construct the thing we expect to obtain
    ex_pred2 = table['bar_id'] == table3['bar_id']
    ex_expr = table.left_join(table2, [pred1]).inner_join(table3, [ex_pred2])

    rewritten_proj = L.substitute_parents(view)
    op = rewritten_proj.op()

    assert not op.table.equals(ex_expr)


def test_multiple_join_deeper_reference():
    # Join predicates down the chain might reference one or more root
    # tables in the hierarchy.
    table1 = ibis.table(
        {'key1': 'string', 'key2': 'string', 'value1': 'double'}
    )
    table2 = ibis.table({'key3': 'string', 'value2': 'double'})
    table3 = ibis.table({'key4': 'string', 'value3': 'double'})

    joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
    joined2 = joined.inner_join(table3, [table1['key2'] == table3['key4']])

    # it works, what more should we test here?
    repr(joined2)


def test_filter_on_projected_field(con):
    # See #173. Impala and other SQL engines do not allow filtering on a
    # just-created alias in a projection
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')

    fields_of_interest = [
        customer,
        region.r_name.name('region'),
        orders.o_totalprice.name('amount'),
        orders.o_orderdate.cast('timestamp').name('odate'),
    ]

    all_join = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)
    )

    tpch = all_join[fields_of_interest]

    # Correlated subquery, yikes!
    t2 = tpch.view()
    conditional_avg = t2[(t2.region == tpch.region)].amount.mean()

    # `amount` is part of the projection above as an aliased field
    amount_filter = tpch.amount > conditional_avg

    result = tpch.filter([amount_filter])

    # Now then! Predicate pushdown here is inappropriate, so we check that
    # it didn't occur.
    assert isinstance(result.op(), ops.Selection)
    assert result.op().table is tpch


def test_join_predicate_from_derived_raises():
    # Join predicate references a derived table, but we can salvage and
    # rewrite it to get the join semantics out
    # see ibis #74
    table = ibis.table(
        [('c', 'int32'), ('f', 'double'), ('g', 'string')], 'foo_table'
    )

    table2 = ibis.table([('key', 'string'), ('value', 'double')], 'bar_table')

    filter_pred = table['f'] > 0
    table3 = table[filter_pred]

    with pytest.raises(com.ExpressionError):
        table.inner_join(table2, [table3['g'] == table2['key']])


def test_bad_join_predicate_raises():
    table = ibis.table(
        [('c', 'int32'), ('f', 'double'), ('g', 'string')], 'foo_table'
    )

    table2 = ibis.table([('key', 'string'), ('value', 'double')], 'bar_table')

    table3 = ibis.table([('key', 'string'), ('value', 'double')], 'baz_table')

    with pytest.raises(com.ExpressionError):
        table.inner_join(table2, [table['g'] == table3['key']])


def test_filter_self_join():
    # GH #667
    purchases = ibis.table(
        [
            ('region', 'string'),
            ('kind', 'string'),
            ('user', 'int64'),
            ('amount', 'double'),
        ],
        'purchases',
    )

    metric = purchases.amount.sum().name('total')
    agged = purchases.group_by(['region', 'kind']).aggregate(metric)

    left = agged[agged.kind == 'foo']
    right = agged[agged.kind == 'bar']

    cond = left.region == right.region
    joined = left.join(right, cond)

    metric = (left.total - right.total).name('diff')
    what = [left.region, metric]
    projected = joined.projection(what)

    proj_exprs = projected.op().selections

    # proj exprs unaffected by analysis
    assert_equal(proj_exprs[0], left.region)
    assert_equal(proj_exprs[1], metric)


def test_no_rewrite(con):
    table = con.table('test1')
    table4 = table[['c', (table['c'] * 2).name('foo')]]
    expr = table4['c'] == table4['foo']
    result = L.substitute_parents(expr)
    expected = expr
    assert result.equals(expected)


def test_join_table_choice():
    # GH807
    x = ibis.table(ibis.schema([('n', 'int64')]), 'x')
    t = x.aggregate(cnt=x.n.count())
    predicate = t.cnt > 0
    assert L.sub_for(predicate, [(t, t.op().table)]).equals(predicate)


def test_is_ancestor_analytic():
    x = ibis.table(ibis.schema([('col', 'int32')]), 'x')
    with_filter_col = x[x.columns + [ibis.null().name('filter')]]
    filtered = with_filter_col[with_filter_col['filter'].isnull()]
    subquery = filtered[filtered.columns]

    with_analytic = subquery[
        subquery.columns + [subquery.count().name('analytic')]
    ]

    assert not subquery.op().equals(with_analytic.op())


# Pr 2635
def test_mutation_fusion_no_overwrite():
    """Test fusion with chained mutation that doesn't overwrite existing
    columns.
    """
    t = ibis.table(ibis.schema([('col', 'int32')]), 't')

    result = t
    result = result.mutate(col1=t['col'] + 1)
    result = result.mutate(col2=t['col'] + 2)
    result = result.mutate(col3=t['col'] + 3)

    first_selection = result

    assert len(result.op().selections) == 4
    assert (
        first_selection.op().selections[1].equals((t['col'] + 1).name('col1'))
    )
    assert (
        first_selection.op().selections[2].equals((t['col'] + 2).name('col2'))
    )
    assert (
        first_selection.op().selections[3].equals((t['col'] + 3).name('col3'))
    )


# Pr 2635
def test_mutation_fusion_overwrite():
    """Test fusion with chained mutation that overwrites existing columns."""
    t = ibis.table(ibis.schema([('col', 'int32')]), 't')

    result = t

    result = result.mutate(col1=t['col'] + 1)
    result = result.mutate(col2=t['col'] + 2)
    result = result.mutate(col3=t['col'] + 3)
    result = result.mutate(col=t['col'] - 1)
    result = result.mutate(col4=t['col'] + 4)

    second_selection = result
    first_selection = second_selection.op().table

    assert len(first_selection.op().selections) == 4
    assert (
        first_selection.op().selections[1].equals((t['col'] + 1).name('col1'))
    )
    assert (
        first_selection.op().selections[2].equals((t['col'] + 2).name('col2'))
    )
    assert (
        first_selection.op().selections[3].equals((t['col'] + 3).name('col3'))
    )

    # Since the second selection overwrites existing columns, it will
    # not have the Table as the first selection
    assert len(second_selection.op().selections) == 5
    assert (
        second_selection.op().selections[0].equals((t['col'] - 1).name('col'))
    )
    assert second_selection.op().selections[1].equals(first_selection['col1'])
    assert second_selection.op().selections[2].equals(first_selection['col2'])
    assert second_selection.op().selections[3].equals(first_selection['col3'])
    assert (
        second_selection.op().selections[4].equals((t['col'] + 4).name('col4'))
    )


# Pr 2635
def test_select_filter_mutate_fusion():
    """Test fusion with filter followed by mutation on the same input."""

    t = ibis.table(ibis.schema([('col', 'float32')]), 't')

    result = t[['col']]
    result = result[result['col'].isnan()]
    result = result.mutate(col=result['col'].cast('int32'))

    second_selection = result
    first_selection = second_selection.op().table

    assert len(second_selection.op().selections) == 1
    assert (
        second_selection.op()
        .selections[0]
        .equals(first_selection['col'].cast('int32').name('col'))
    )

    # we don't look past the projection when a filter is encountered, so the
    # number of selections in the first projection (`first_selection`) is 0
    #
    # previously we did, but this was buggy when executing against the pandas
    # backend
    #
    # eventually we will bring this back, but we're trading off the ability
    # to remove materialize for some performance in the short term
    assert len(first_selection.op().selections) == 0
    assert len(first_selection.op().predicates) == 1


def test_no_filter_means_no_selection():
    t = ibis.table(dict(a="string"))
    proj = t.filter([])
    assert proj.equals(t)


def test_mutate_overwrites_existing_column():
    t = ibis.table(dict(a="string"))
    mut = t.mutate(a=42).select(["a"])
    sel = mut.op().selections[0].op().table.op().selections[0].op().arg
    assert isinstance(sel.op(), ops.Literal)
    assert sel.op().value == 42


def test_agg_selection_does_not_share_roots():
    t = ibis.table(dict(a="string"), name="t")
    s = ibis.table(dict(b="float64"), name="s")
    gb = t.group_by("a")
    n = s.count()

    with pytest.raises(com.RelationError, match="Selection expressions"):
        gb.aggregate(n=n)


@pytest.mark.parametrize("num_joins", [1, 10])
def test_large_compile(num_joins):
    num_columns = 20
    table = ibis.table(
        {f"col_{i:d}": "string" for i in range(num_columns)}, name="t"
    )
    for _ in range(num_joins):
        start = time.time()
        table = table.mutate(dummy=ibis.literal(""))
        stop = time.time()
        assert stop - start < 1.0

        start = time.time()
        table = table.left_join(table, ["dummy"])[[table]]
        stop = time.time()
        assert stop - start < 1.0
