import datetime
import pickle
import re

import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.config as config
import ibis.expr.analysis as L
import ibis.expr.api as api
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.exceptions import ExpressionError, RelationError
from ibis.expr.types import ColumnExpr, TableExpr
from ibis.tests.util import assert_equal


@pytest.fixture
def set_ops_schema_top():
    return [('key', 'string'), ('value', 'double')]


@pytest.fixture
def set_ops_schema_bottom():
    return [('key', 'string'), ('key2', 'string'), ('value', 'double')]


@pytest.fixture
def setops_table_foo(set_ops_schema_top):
    return ibis.table(set_ops_schema_top, 'foo')


@pytest.fixture
def setops_table_bar(set_ops_schema_top):
    return ibis.table(set_ops_schema_top, 'bar')


@pytest.fixture
def setops_table_baz(set_ops_schema_bottom):
    return ibis.table(set_ops_schema_bottom, 'baz')


@pytest.fixture
def setops_relation_error_message():
    return 'Table schemas must be equal for set operations'


def test_empty_schema():
    table = api.table([], 'foo')
    assert not table.schema()


def test_columns(con):
    t = con.table('alltypes')
    result = t.columns
    expected = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    assert result == expected


def test_view_new_relation(table):
    # For assisting with self-joins and other self-referential operations
    # where we need to be able to treat instances of the same TableExpr as
    # semantically distinct
    #
    # This thing is not exactly a projection, since it has no semantic
    # meaning when it comes to execution
    tview = table.view()

    roots = tview._root_tables()
    assert len(roots) == 1
    assert roots[0] is tview.op()


def test_get_type(table, schema_dict):
    for k, v in schema_dict.items():
        assert table._get_type(k) == dt.dtype(v)


def test_getitem_column_select(table, schema_dict):
    for k, v in schema_dict.items():
        col = table[k]

        # Make sure it's the right type
        assert isinstance(col, ColumnExpr)

        # Ensure we have a field selection with back-reference to the table
        parent = col.parent()
        assert isinstance(parent, ops.TableColumn)
        assert parent.parent() is table


def test_getitem_attribute(table):
    result = table.a
    assert_equal(result, table['a'])

    assert 'a' in dir(table)

    # Project and add a name that conflicts with a TableExpr built-in
    # attribute
    view = table[[table, table['a'].name('schema')]]
    assert not isinstance(view.schema, ColumnExpr)


def test_projection(table):
    cols = ['f', 'a', 'h']

    proj = table[cols]
    assert isinstance(proj, TableExpr)
    assert isinstance(proj.op(), ops.Selection)

    assert proj.schema().names == cols
    for c in cols:
        expr = proj[c]
        assert isinstance(expr, type(table[c]))


def test_projection_no_list(table):
    expr = (table.f * 2).name('bar')
    result = table.select(expr)
    expected = table.projection([expr])
    assert_equal(result, expected)


def test_projection_with_exprs(table):
    # unnamed expr to test
    mean_diff = (table['a'] - table['c']).mean()

    col_exprs = [table['b'].log().name('log_b'), mean_diff.name('mean_diff')]

    proj = table[col_exprs + ['g']]
    schema = proj.schema()
    assert schema.names == ['log_b', 'mean_diff', 'g']
    assert schema.types == [dt.double, dt.double, dt.string]

    # Test with unnamed expr
    with pytest.raises(ExpressionError):
        table.projection(['g', table['a'] - table['c']])


def test_projection_duplicate_names(table):
    with pytest.raises(com.IntegrityError):
        table.projection([table.c, table.c])


def test_projection_invalid_root(table):
    schema1 = {'foo': 'double', 'bar': 'int32'}

    left = api.table(schema1, name='foo')
    right = api.table(schema1, name='bar')

    exprs = [right['foo'], right['bar']]
    with pytest.raises(RelationError):
        left.projection(exprs)


def test_projection_unnamed_literal_interactive_blowup(con):
    # #147 and #153 alike
    table = con.table('functional_alltypes')

    with config.option_context('interactive', True):
        try:
            table.select([table.bigint_col, ibis.literal(5)])
        except Exception as e:
            assert 'named' in e.args[0]


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_projection_of_aggregated(table):
    # Fully-formed aggregations "block"; in a projection, column
    # expressions referencing table expressions below the aggregation are
    # invalid.
    assert False


def test_projection_with_star_expr(table):
    new_expr = (table['a'] * 5).name('bigger_a')

    t = table

    # it lives!
    proj = t[t, new_expr]
    repr(proj)

    ex_names = table.schema().names + ['bigger_a']
    assert proj.schema().names == ex_names

    # cannot pass an invalid table expression
    t2 = t.aggregate([t['a'].sum().name('sum(a)')], by=['g'])
    with pytest.raises(RelationError):
        t[[t2]]
    # TODO: there may be some ways this can be invalid


def test_projection_convenient_syntax(table):
    proj = table[table, table['a'].name('foo')]
    proj2 = table[[table, table['a'].name('foo')]]
    assert_equal(proj, proj2)


def test_projection_mutate_analysis_bug(con):
    # GH #549

    t = con.table('airlines')

    filtered = t[t.depdelay.notnull()]
    leg = ibis.literal('-').join([t.origin, t.dest])
    mutated = filtered.mutate(leg=leg)

    # it works!
    mutated['year', 'month', 'day', 'depdelay', 'leg']


def test_projection_self(table):
    result = table[table]
    expected = table.projection(table)

    assert_equal(result, expected)


def test_projection_array_expr(table):
    result = table[table.a]
    expected = table[[table.a]]
    assert_equal(result, expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_add_column_aggregate_crossjoin(table):
    # A new column that depends on a scalar value produced by this or some
    # other table.
    #
    # For example:
    # SELECT *, b - VAL
    # FROM table1
    #
    # Here, VAL could be something produced by aggregating table1 or any
    # other table for that matter.
    assert False


def test_mutate(table):
    one = table.f * 2
    foo = (table.a + table.b).name('foo')

    expr = table.mutate(foo, one=one, two=2)
    expected = table[table, foo, one.name('one'), ibis.literal(2).name('two')]
    assert_equal(expr, expected)


def test_mutate_alter_existing_columns(table):
    new_f = table.f * 2
    foo = table.d * 2
    expr = table.mutate(f=new_f, foo=foo)

    expected = table[
        'a',
        'b',
        'c',
        'd',
        'e',
        new_f.name('f'),
        'g',
        'h',
        'i',
        'j',
        'k',
        foo.name('foo'),
    ]

    assert_equal(expr, expected)


def test_replace_column(table):
    tb = api.table([('a', 'int32'), ('b', 'double'), ('c', 'string')])

    expr = tb.b.cast('int32')
    tb2 = tb.set_column('b', expr)
    expected = tb[tb.a, expr.name('b'), tb.c]

    assert_equal(tb2, expected)


def test_filter_no_list(table):
    pred = table.a > 5

    result = table.filter(pred)
    expected = table[pred]
    assert_equal(result, expected)


def test_add_predicate(table):
    pred = table['a'] > 5
    result = table[pred]
    assert isinstance(result.op(), ops.Selection)


def test_invalid_predicate(table, schema):
    # a lookalike
    table2 = api.table(schema, name='bar')
    predicate = table2.a > 5
    with pytest.raises(RelationError):
        table.filter(predicate)


def test_add_predicate_coalesce(table):
    # Successive predicates get combined into one rather than nesting. This
    # is mainly to enhance readability since we could handle this during
    # expression evaluation anyway.
    pred1 = table['a'] > 5
    pred2 = table['b'] > 0

    result = table[pred1][pred2]
    expected = table.filter([pred1, pred2])
    assert_equal(result, expected)

    # 59, if we are not careful, we can obtain broken refs
    interm = table[pred1]
    result = interm.filter([interm['b'] > 0])
    assert_equal(result, expected)


def test_repr_same_but_distinct_objects(con):
    t = con.table('test1')
    t_copy = con.table('test1')
    table2 = t[t_copy['f'] > 0]

    result = repr(table2)
    assert result.count('DatabaseTable') == 1


def test_filter_fusion_distinct_table_objects(con):
    t = con.table('test1')
    tt = con.table('test1')

    expr = t[t.f > 0][t.c > 0]
    expr2 = t[t.f > 0][tt.c > 0]
    expr3 = t[tt.f > 0][tt.c > 0]
    expr4 = t[tt.f > 0][t.c > 0]

    assert_equal(expr, expr2)
    assert repr(expr) == repr(expr2)
    assert_equal(expr, expr3)
    assert_equal(expr, expr4)


def test_column_relabel(table):
    # GH #551. Keeping the test case very high level to not presume that
    # the relabel is necessarily implemented using a projection
    types = ['int32', 'string', 'double']
    table = api.table(zip(['foo', 'bar', 'baz'], types))
    result = table.relabel({'foo': 'one', 'baz': 'three'})

    schema = result.schema()
    ex_schema = api.schema(zip(['one', 'bar', 'three'], types))
    assert_equal(schema, ex_schema)


def test_limit(table):
    limited = table.limit(10, offset=5)
    assert limited.op().n == 10
    assert limited.op().offset == 5


def test_sort_by(table):
    # Commit to some API for ascending and descending
    #
    # table.sort_by(['g', expr1, desc(expr2), desc(expr3)])
    #
    # Default is ascending for anything coercable to an expression,
    # and we'll have ascending/descending wrappers to help.
    result = table.sort_by(['f'])

    sort_key = result.op().sort_keys[0].op()

    assert_equal(sort_key.expr, table.f)
    assert sort_key.ascending

    # non-list input. per #150
    result2 = table.sort_by('f')
    assert_equal(result, result2)

    result2 = table.sort_by([('f', False)])
    result3 = table.sort_by([('f', 'descending')])
    result4 = table.sort_by([('f', 0)])

    key2 = result2.op().sort_keys[0].op()
    key3 = result3.op().sort_keys[0].op()
    key4 = result4.op().sort_keys[0].op()

    assert not key2.ascending
    assert not key3.ascending
    assert not key4.ascending
    assert_equal(result2, result3)


def test_sort_by_desc_deferred_sort_key(table):
    result = table.group_by('g').size().sort_by(ibis.desc('count'))

    tmp = table.group_by('g').size()
    expected = tmp.sort_by((tmp['count'], False))
    expected2 = tmp.sort_by(ibis.desc(tmp['count']))

    assert_equal(result, expected)
    assert_equal(result, expected2)


def test_slice_convenience(table):
    expr = table[:5]
    expr2 = table[:5:1]
    assert_equal(expr, table.limit(5))
    assert_equal(expr, expr2)

    expr = table[2:7]
    expr2 = table[2:7:1]
    assert_equal(expr, table.limit(5, offset=2))
    assert_equal(expr, expr2)

    with pytest.raises(ValueError):
        table[2:15:2]

    with pytest.raises(ValueError):
        table[5:]

    with pytest.raises(ValueError):
        table[:-5]

    with pytest.raises(ValueError):
        table[-10:-5]


def test_table_count(table):
    result = table.count()
    assert isinstance(result, ir.IntegerScalar)
    assert isinstance(result.op(), ops.Count)
    assert result.get_name() == 'count'


def test_len_raises_expression_error(table):
    with pytest.raises(com.ExpressionError):
        len(table)


def test_sum_expr_basics(table, int_col):
    # Impala gives bigint for all integer types
    ex_class = ir.IntegerScalar
    result = table[int_col].sum()
    assert isinstance(result, ex_class)
    assert isinstance(result.op(), ops.Sum)


def test_sum_expr_basics_floats(table, float_col):
    # Impala gives double for all floating point types
    ex_class = ir.FloatingScalar
    result = table[float_col].sum()
    assert isinstance(result, ex_class)
    assert isinstance(result.op(), ops.Sum)


def test_mean_expr_basics(table, numeric_col):
    result = table[numeric_col].mean()
    assert isinstance(result, ir.FloatingScalar)
    assert isinstance(result.op(), ops.Mean)


def test_aggregate_no_keys(table):
    metrics = [
        table['a'].sum().name('sum(a)'),
        table['c'].mean().name('mean(c)'),
    ]

    # A TableExpr, which in SQL at least will yield a table with a single
    # row
    result = table.aggregate(metrics)
    assert isinstance(result, TableExpr)


def test_aggregate_keys_basic(table):
    metrics = [
        table['a'].sum().name('sum(a)'),
        table['c'].mean().name('mean(c)'),
    ]

    # A TableExpr, which in SQL at least will yield a table with a single
    # row
    result = table.aggregate(metrics, by=['g'])
    assert isinstance(result, TableExpr)

    # it works!
    repr(result)


def test_aggregate_non_list_inputs(table):
    # per #150
    metric = table.f.sum().name('total')
    by = 'g'
    having = table.c.sum() > 10

    result = table.aggregate(metric, by=by, having=having)
    expected = table.aggregate([metric], by=[by], having=[having])
    assert_equal(result, expected)


def test_aggregate_keywords(table):
    t = table

    expr = t.aggregate(foo=t.f.sum(), bar=lambda x: x.f.mean(), by='g')
    expr2 = t.group_by('g').aggregate(foo=t.f.sum(), bar=lambda x: x.f.mean())
    expected = t.aggregate(
        [t.f.mean().name('bar'), t.f.sum().name('foo')], by='g'
    )

    assert_equal(expr, expected)
    assert_equal(expr2, expected)


def test_groupby_alias(table):
    t = table

    result = t.groupby('g').size()
    expected = t.group_by('g').size()
    assert_equal(result, expected)


def test_summary_expand_list(table):
    summ = table.f.summary()

    metric = table.g.group_concat().name('bar')
    result = table.aggregate([metric, summ])
    expected = table.aggregate([metric] + summ.exprs())
    assert_equal(result, expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_aggregate_invalid(table):
    # Pass a non-aggregation or non-scalar expr
    assert False


def test_filter_aggregate_pushdown_predicate(table):
    # In the case where we want to add a predicate to an aggregate
    # expression after the fact, rather than having to backpedal and add it
    # before calling aggregate.
    #
    # TODO (design decision): This could happen automatically when adding a
    # predicate originating from the same root table; if an expression is
    # created from field references from the aggregated table then it
    # becomes a filter predicate applied on top of a view

    pred = table.f > 0
    metrics = [table.a.sum().name('total')]
    agged = table.aggregate(metrics, by=['g'])
    filtered = agged.filter([pred])
    expected = table[pred].aggregate(metrics, by=['g'])
    assert_equal(filtered, expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_filter_aggregate_partial_pushdown(table):
    assert False


def test_aggregate_post_predicate(table):
    # Test invalid having clause
    metrics = [table.f.sum().name('total')]
    by = ['g']

    invalid_having_cases = [table.f.sum(), table.f > 2]
    for case in invalid_having_cases:
        with pytest.raises(com.ExpressionError):
            table.aggregate(metrics, by=by, having=[case])


def test_group_by_having_api(table):
    # #154, add a HAVING post-predicate in a composable way
    metric = table.f.sum().name('foo')
    postp = table.d.mean() > 1

    expr = table.group_by('g').having(postp).aggregate(metric)

    expected = table.aggregate(metric, by='g', having=postp)
    assert_equal(expr, expected)


def test_group_by_kwargs(table):
    t = table
    expr = t.group_by(['f', t.h], z='g', z2=t.d).aggregate(
        t.d.mean().name('foo')
    )
    expected = t.group_by(['f', t.h, t.g.name('z'), t.d.name('z2')]).aggregate(
        t.d.mean().name('foo')
    )
    assert_equal(expr, expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_aggregate_root_table_internal(table):
    assert False


def test_compound_aggregate_expr(table):
    # See ibis #24
    compound_expr = (table['a'].sum() / table['a'].mean()).name('foo')
    assert L.is_reduction(compound_expr)

    # Validates internally
    table.aggregate([compound_expr])


def test_groupby_convenience(table):
    metrics = [table.f.sum().name('total')]

    expr = table.group_by('g').aggregate(metrics)
    expected = table.aggregate(metrics, by=['g'])
    assert_equal(expr, expected)

    group_expr = table.g.cast('double').name('g')
    expr = table.group_by(group_expr).aggregate(metrics)
    expected = table.aggregate(metrics, by=[group_expr])
    assert_equal(expr, expected)


def test_group_by_count_size(table):
    # #148, convenience for interactive use, and so forth
    result1 = table.group_by('g').size()
    result2 = table.group_by('g').count()

    expected = table.group_by('g').aggregate([table.count().name('count')])

    assert_equal(result1, expected)
    assert_equal(result2, expected)

    result = table.group_by('g').count('foo')
    expected = table.group_by('g').aggregate([table.count().name('foo')])
    assert_equal(result, expected)


def test_group_by_column_select_api(table):
    grouped = table.group_by('g')

    result = grouped.f.sum()
    expected = grouped.aggregate(table.f.sum().name('sum(f)'))
    assert_equal(result, expected)

    supported_functions = ['sum', 'mean', 'count', 'size', 'max', 'min']

    # make sure they all work
    for fn in supported_functions:
        getattr(grouped.f, fn)()


def test_value_counts_convenience(table):
    # #152
    result = table.g.value_counts()
    expected = table.group_by('g').aggregate(table.count().name('count'))

    assert_equal(result, expected)


def test_isin_value_counts(table):
    # #157, this code path was untested before
    bool_clause = table.g.notin(['1', '4', '7'])
    # it works!
    bool_clause.name('notin').value_counts()


def test_value_counts_unnamed_expr(con):
    nation = con.table('tpch_nation')

    expr = nation.n_name.lower().value_counts()
    expected = nation.n_name.lower().name('unnamed').value_counts()
    assert_equal(expr, expected)


def test_aggregate_unnamed_expr(con):
    nation = con.table('tpch_nation')
    expr = nation.n_name.lower().left(1)

    with pytest.raises(com.ExpressionError):
        nation.group_by(expr).aggregate(nation.count().name('metric'))


def test_default_reduction_names(table):
    d = table.f
    cases = [
        (d.count(), 'count'),
        (d.sum(), 'sum'),
        (d.mean(), 'mean'),
        (d.approx_nunique(), 'approx_nunique'),
        (d.approx_median(), 'approx_median'),
        (d.min(), 'min'),
        (d.max(), 'max'),
    ]

    for expr, ex_name in cases:
        assert expr.get_name() == ex_name


def test_join_no_predicate_list(con):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    pred = region.r_regionkey == nation.n_regionkey
    joined = region.inner_join(nation, pred)
    expected = region.inner_join(nation, [pred])
    assert_equal(joined, expected)


def test_asof_join():
    left = ibis.table([('time', 'int32'), ('value', 'double')])
    right = ibis.table([('time', 'int32'), ('value2', 'double')])
    joined = api.asof_join(left, right, 'time')
    pred = joined.op().predicates[0].op()
    assert pred.left.op().name == pred.right.op().name == 'time'


def test_asof_join_with_by():
    left = ibis.table(
        [('time', 'int32'), ('key', 'int32'), ('value', 'double')]
    )
    right = ibis.table(
        [('time', 'int32'), ('key', 'int32'), ('value2', 'double')]
    )
    joined = api.asof_join(left, right, 'time', by='key')
    by = joined.op().by[0].op()
    assert by.left.op().name == by.right.op().name == 'key'


@pytest.mark.parametrize(
    ('ibis_interval', 'timedelta_interval'),
    [
        [ibis.interval(days=2), pd.Timedelta('2 days')],
        [ibis.interval(days=2), datetime.timedelta(days=2)],
        [ibis.interval(hours=5), pd.Timedelta('5 hours')],
        [ibis.interval(hours=5), datetime.timedelta(hours=5)],
        [ibis.interval(minutes=7), pd.Timedelta('7 minutes')],
        [ibis.interval(minutes=7), datetime.timedelta(minutes=7)],
        [ibis.interval(seconds=9), pd.Timedelta('9 seconds')],
        [ibis.interval(seconds=9), datetime.timedelta(seconds=9)],
        [ibis.interval(milliseconds=11), pd.Timedelta('11 milliseconds')],
        [ibis.interval(milliseconds=11), datetime.timedelta(milliseconds=11)],
        [ibis.interval(microseconds=15), pd.Timedelta('15 microseconds')],
        [ibis.interval(microseconds=15), datetime.timedelta(microseconds=15)],
        [ibis.interval(nanoseconds=17), pd.Timedelta('17 nanoseconds')],
    ],
)
def test_asof_join_with_tolerance(ibis_interval, timedelta_interval):
    left = ibis.table(
        [('time', 'int32'), ('key', 'int32'), ('value', 'double')]
    )
    right = ibis.table(
        [('time', 'int32'), ('key', 'int32'), ('value2', 'double')]
    )

    joined = api.asof_join(left, right, 'time', tolerance=ibis_interval)
    tolerance = joined.op().tolerance
    assert_equal(tolerance, ibis_interval)

    joined = api.asof_join(left, right, 'time', tolerance=timedelta_interval)
    tolerance = joined.op().tolerance
    assert isinstance(tolerance, ir.IntervalScalar)
    assert isinstance(tolerance.op(), ops.Literal)


def test_equijoin_schema_merge():
    table1 = ibis.table([('key1', 'string'), ('value1', 'double')])
    table2 = ibis.table([('key2', 'string'), ('stuff', 'int32')])

    pred = table1['key1'] == table2['key2']
    join_types = ['inner_join', 'left_join', 'outer_join']

    ex_schema = api.Schema(
        ['key1', 'value1', 'key2', 'stuff'],
        ['string', 'double', 'string', 'int32'],
    )

    for fname in join_types:
        f = getattr(table1, fname)
        joined = f(table2, [pred]).materialize()
        assert_equal(joined.schema(), ex_schema)


def test_join_combo_with_projection(table):
    # Test a case where there is column name overlap, but the projection
    # passed makes it a non-issue. Highly relevant with self-joins
    #
    # For example, where left/right have some field names in common:
    # SELECT left.*, right.a, right.b
    # FROM left join right on left.key = right.key
    t = table
    t2 = t.mutate(foo=t.f * 2, bar=t.f * 4)

    # this works
    joined = t.left_join(t2, [t['g'] == t2['g']])
    proj = joined.projection([t, t2['foo'], t2['bar']])
    repr(proj)


def test_join_getitem_projection(con):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    pred = region.r_regionkey == nation.n_regionkey
    joined = region.inner_join(nation, pred)

    result = joined[nation]
    expected = joined.projection(nation)
    assert_equal(result, expected)


def test_self_join(table):
    # Self-joins are problematic with this design because column
    # expressions may reference either the left or right  For example:
    #
    # SELECT left.key, sum(left.value - right.value) as total_deltas
    # FROM table left
    #  INNER JOIN table right
    #    ON left.current_period = right.previous_period + 1
    # GROUP BY 1
    #
    # One way around the self-join issue is to force the user to add
    # prefixes to the joined fields, then project using those. Not that
    # satisfying, though.
    left = table
    right = table.view()
    metric = (left['a'] - right['b']).mean().name('metric')

    joined = left.inner_join(right, [right['g'] == left['g']])
    # basic check there's no referential problems
    result_repr = repr(joined)
    assert 'ref_0' in result_repr
    assert 'ref_1' in result_repr

    # Cannot be immediately materialized because of the schema overlap
    with pytest.raises(RelationError):
        joined.materialize()

    # Project out left table schema
    proj = joined[[left]]
    assert_equal(proj.schema(), left.schema())

    # Try aggregating on top of joined
    aggregated = joined.aggregate([metric], by=[left['g']])
    ex_schema = api.Schema(['g', 'metric'], ['string', 'double'])
    assert_equal(aggregated.schema(), ex_schema)


def test_self_join_no_view_convenience(table):
    # #165, self joins ought to be possible when the user specifies the
    # column names to join on rather than referentially-valid expressions

    result = table.join(table, [('g', 'g')])

    t2 = table.view()
    expected = table.join(t2, table.g == t2.g)
    assert_equal(result, expected)


def test_materialized_join_reference_bug(con):
    # GH#403
    orders = con.table('tpch_orders')
    customer = con.table('tpch_customer')
    lineitem = con.table('tpch_lineitem')

    items = (
        orders.join(lineitem, orders.o_orderkey == lineitem.l_orderkey)[
            lineitem, orders.o_custkey, orders.o_orderpriority
        ]
        .join(customer, [('o_custkey', 'c_custkey')])
        .materialize()
    )
    items['o_orderpriority'].value_counts()


def test_join_project_after(table):
    # e.g.
    #
    # SELECT L.foo, L.bar, R.baz, R.qux
    # FROM table1 L
    #   INNER JOIN table2 R
    #     ON L.key = R.key
    #
    # or
    #
    # SELECT L.*, R.baz
    # ...
    #
    # The default for a join is selecting all fields if possible
    table1 = ibis.table([('key1', 'string'), ('value1', 'double')])
    table2 = ibis.table([('key2', 'string'), ('stuff', 'int32')])

    pred = table1['key1'] == table2['key2']

    joined = table1.left_join(table2, [pred])
    projected = joined.projection([table1, table2['stuff']])
    assert projected.schema().names == ['key1', 'value1', 'stuff']

    projected = joined.projection([table2, table1['key1']])
    assert projected.schema().names == ['key2', 'stuff', 'key1']


def test_semi_join_schema(table):
    # A left semi join discards the schema of the right table
    table1 = ibis.table([('key1', 'string'), ('value1', 'double')])
    table2 = ibis.table([('key2', 'string'), ('stuff', 'double')])

    pred = table1['key1'] == table2['key2']
    semi_joined = table1.semi_join(table2, [pred]).materialize()

    result_schema = semi_joined.schema()
    assert_equal(result_schema, table1.schema())


def test_cross_join(table):
    metrics = [
        table['a'].sum().name('sum_a'),
        table['b'].mean().name('mean_b'),
    ]
    scalar_aggs = table.aggregate(metrics)

    joined = table.cross_join(scalar_aggs).materialize()
    agg_schema = api.Schema(['sum_a', 'mean_b'], ['int64', 'double'])
    ex_schema = table.schema().append(agg_schema)
    assert_equal(joined.schema(), ex_schema)


def test_cross_join_multiple(table):
    a = table['a', 'b', 'c']
    b = table['d', 'e']
    c = table['f', 'h']

    joined = ibis.cross_join(a, b, c)
    expected = a.cross_join(b.cross_join(c))
    assert joined.equals(expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_join_compound_boolean_predicate(table):
    # The user might have composed predicates through logical operations
    assert False


def test_filter_join_unmaterialized(table):
    table1 = ibis.table(
        {'key1': 'string', 'key2': 'string', 'value1': 'double'}
    )
    table2 = ibis.table({'key3': 'string', 'value2': 'double'})

    # It works!
    joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
    filtered = joined.filter([table1.value1 > 0])
    repr(filtered)


def test_join_overlapping_column_names(table):
    t1 = ibis.table(
        [('foo', 'string'), ('bar', 'string'), ('value1', 'double')]
    )
    t2 = ibis.table(
        [('foo', 'string'), ('bar', 'string'), ('value2', 'double')]
    )

    joined = t1.join(t2, 'foo')
    expected = t1.join(t2, t1.foo == t2.foo)
    assert_equal(joined, expected)

    joined = t1.join(t2, ['foo', 'bar'])
    expected = t1.join(t2, [t1.foo == t2.foo, t1.bar == t2.bar])
    assert_equal(joined, expected)


def test_join_key_alternatives(con):
    t1 = con.table('star1')
    t2 = con.table('star2')

    # Join with tuples
    joined = t1.inner_join(t2, [('foo_id', 'foo_id')])
    joined2 = t1.inner_join(t2, [(t1.foo_id, t2.foo_id)])

    # Join with single expr
    joined3 = t1.inner_join(t2, t1.foo_id == t2.foo_id)

    expected = t1.inner_join(t2, [t1.foo_id == t2.foo_id])

    assert_equal(joined, expected)
    assert_equal(joined2, expected)
    assert_equal(joined3, expected)

    with pytest.raises(com.ExpressionError):
        t1.inner_join(t2, [('foo_id', 'foo_id', 'foo_id')])


def test_join_invalid_refs(con):
    t1 = con.table('star1')
    t2 = con.table('star2')
    t3 = con.table('star3')

    predicate = t1.bar_id == t3.bar_id
    with pytest.raises(com.RelationError):
        t1.inner_join(t2, [predicate])


def test_join_invalid_expr_type(con):
    left = con.table('star1')
    invalid_right = left.foo_id
    join_key = ['bar_id']

    with pytest.raises(TypeError, match=type(invalid_right).__name__):
        left.inner_join(invalid_right, join_key)


def test_join_non_boolean_expr(con):
    t1 = con.table('star1')
    t2 = con.table('star2')

    # oops
    predicate = t1.f * t2.value1
    with pytest.raises(com.ExpressionError):
        t1.inner_join(t2, [predicate])


def test_unravel_compound_equijoin(table):
    t1 = ibis.table(
        [
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value1', 'double'),
        ],
        'foo_table',
    )

    t2 = ibis.table(
        [
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value2', 'double'),
        ],
        'bar_table',
    )

    p1 = t1.key1 == t2.key1
    p2 = t1.key2 == t2.key2
    p3 = t1.key3 == t2.key3

    joined = t1.inner_join(t2, [p1 & p2 & p3])
    expected = t1.inner_join(t2, [p1, p2, p3])
    assert_equal(joined, expected)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_join_add_prefixes(table):
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_join_nontrivial_exprs(table):
    assert False


def test_union(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.union(setops_table_bar)
    assert isinstance(result.op(), ops.Union)
    assert not result.op().distinct

    result = setops_table_foo.union(setops_table_bar, distinct=True)
    assert result.op().distinct

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.union(setops_table_baz)


def test_intersection(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.intersect(setops_table_bar)
    assert isinstance(result.op(), ops.Intersection)

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.intersect(setops_table_baz)


def test_difference(
    setops_table_foo,
    setops_table_bar,
    setops_table_baz,
    setops_relation_error_message,
):
    result = setops_table_foo.difference(setops_table_bar)
    assert isinstance(result.op(), ops.Difference)

    with pytest.raises(RelationError, match=setops_relation_error_message):
        setops_table_foo.difference(setops_table_baz)


def test_column_ref_on_projection_rename(con):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')

    joined = region.inner_join(
        nation, [region.r_regionkey == nation.n_regionkey]
    ).inner_join(customer, [customer.c_nationkey == nation.n_nationkey])

    proj_exprs = [
        customer,
        nation.n_name.name('nation'),
        region.r_name.name('region'),
    ]
    joined = joined.projection(proj_exprs)

    metrics = [joined.c_acctbal.sum().name('metric')]

    # it works!
    joined.aggregate(metrics, by=['region'])


@pytest.fixture
def t1():
    return ibis.table(
        [('key1', 'string'), ('key2', 'string'), ('value1', 'double')], 'foo'
    )


@pytest.fixture
def t2():
    return ibis.table([('key1', 'string'), ('key2', 'string')], 'bar')


def test_simple_existence_predicate(t1, t2):
    cond = (t1.key1 == t2.key1).any()

    assert isinstance(cond, ir.BooleanColumn)
    op = cond.op()
    assert isinstance(op, ops.Any)

    # it works!
    expr = t1[cond]
    assert isinstance(expr.op(), ops.Selection)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_cannot_use_existence_expression_in_join(table):
    # Join predicates must consist only of comparisons
    assert False


def test_not_exists_predicate(t1, t2):
    cond = -((t1.key1 == t2.key1).any())
    assert isinstance(cond.op(), ops.NotAny)


def test_aggregate_metrics(table):

    functions = [
        lambda x: x.e.sum().name('esum'),
        lambda x: x.f.sum().name('fsum'),
    ]
    exprs = [table.e.sum().name('esum'), table.f.sum().name('fsum')]

    result = table.aggregate(functions[0])
    expected = table.aggregate(exprs[0])
    assert_equal(result, expected)

    result = table.aggregate(functions)
    expected = table.aggregate(exprs)
    assert_equal(result, expected)


def test_group_by_keys(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    expr = m.group_by(lambda x: x.foo).size()
    expected = m.group_by('foo').size()
    assert_equal(expr, expected)

    expr = m.group_by([lambda x: x.foo, lambda x: x.bar]).size()
    expected = m.group_by(['foo', 'bar']).size()
    assert_equal(expr, expected)


def test_having(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    expr = m.group_by('foo').having(lambda x: x.foo.sum() > 10).size()
    expected = m.group_by('foo').having(m.foo.sum() > 10).size()

    assert_equal(expr, expected)


def test_filter(table):
    m = table.mutate(foo=table.f * 2, bar=table.e / 2)

    result = m.filter(lambda x: x.foo > 10)
    result2 = m[lambda x: x.foo > 10]
    expected = m[m.foo > 10]

    assert_equal(result, expected)
    assert_equal(result2, expected)

    result = m.filter([lambda x: x.foo > 10, lambda x: x.bar < 0])
    expected = m.filter([m.foo > 10, m.bar < 0])
    assert_equal(result, expected)


def test_sort_by2(table):
    m = table.mutate(foo=table.e + table.f)

    result = m.sort_by(lambda x: -x.foo)
    expected = m.sort_by(-m.foo)
    assert_equal(result, expected)

    result = m.sort_by(lambda x: ibis.desc(x.foo))
    expected = m.sort_by(ibis.desc('foo'))
    assert_equal(result, expected)

    result = m.sort_by(ibis.desc(lambda x: x.foo))
    expected = m.sort_by(ibis.desc('foo'))
    assert_equal(result, expected)


def test_projection2(table):
    m = table.mutate(foo=table.f * 2)

    def f(x):
        return (x.foo * 2).name('bar')

    result = m.projection([f, 'f'])
    result2 = m[f, 'f']
    expected = m.projection([f(m), 'f'])
    assert_equal(result, expected)
    assert_equal(result2, expected)


def test_mutate2(table):
    m = table.mutate(foo=table.f * 2)

    def g(x):
        return x.foo * 2

    def h(x):
        return x.bar * 2

    result = m.mutate(bar=g).mutate(baz=h)

    m2 = m.mutate(bar=g(m))
    expected = m2.mutate(baz=h(m2))

    assert_equal(result, expected)


def test_groupby_mutate(table):
    t = table

    g = t.group_by('g').order_by('f')
    expr = g.mutate(foo=lambda x: x.f.lag(), bar=lambda x: x.f.rank())
    expected = g.mutate(foo=t.f.lag(), bar=t.f.rank())

    assert_equal(expr, expected)


def test_groupby_projection(table):
    t = table

    g = t.group_by('g').order_by('f')
    expr = g.projection(
        [lambda x: x.f.lag().name('foo'), lambda x: x.f.rank().name('bar')]
    )
    expected = g.projection([t.f.lag().name('foo'), t.f.rank().name('bar')])

    assert_equal(expr, expected)


def test_set_column(table):
    def g(x):
        return x.f * 2

    result = table.set_column('f', g)
    expected = table.set_column('f', table.f * 2)
    assert_equal(result, expected)


def test_pickle_table_expr():
    schema = [('time', 'timestamp'), ('key', 'string'), ('value', 'double')]
    t0 = ibis.table(schema, name='t0')
    raw = pickle.dumps(t0, protocol=2)
    t1 = pickle.loads(raw)
    assert t1.equals(t0)


def test_group_by_key_function():
    t = ibis.table([('a', 'timestamp'), ('b', 'string'), ('c', 'double')])
    expr = t.groupby(new_key=lambda t: t.b.length()).aggregate(foo=t.c.mean())
    assert expr.columns == ['new_key', 'foo']


def test_unbound_table_name():
    t = ibis.table([('a', 'timestamp')])
    name = t.op().name
    match = re.match(r'^unbound_table_\d+$', name)
    assert match is not None


def test_mutate_chain():
    one = ibis.table([('a', 'string'), ('b', 'string')], name='t')
    two = one.mutate(b=lambda t: t.b.fillna('Short Term'))
    three = two.mutate(a=lambda t: t.a.fillna('Short Term'))
    a, b = three.op().selections

    # we can't fuse these correctly yet
    assert isinstance(a.op(), ops.IfNull)
    assert isinstance(b.op(), ops.TableColumn)
    assert isinstance(b.op().table.op().selections[1].op(), ops.IfNull)
