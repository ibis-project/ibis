import ibis
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import get_query, sql_golden_check, sqlgolden, sqlgoldens


def test_select_sql(alltypes, star1, star2):
    tests = dict(
        agg_explicit_column=lambda t=star1: t.aggregate(
            [t['f'].sum().name('total')], [t['foo_id']]
        ),
        agg_string_columns=lambda t=star1: t.aggregate(
            [t['f'].sum().name('total')], ['foo_id', 'bar_id']
        ),
        single_column=lambda t=star1: t.order_by("f"),
        mixed_columns_ascending=lambda t=star1: t.order_by(["c", ("f", 0)]),
        limit_simple=lambda t=star1: t.limit(10),
        limit_with_offset=lambda t=star1: t.limit(10, offset=5),
        filter_then_limit=lambda t=star1: t[t.f > 0].limit(10),
        limit_then_filter=lambda t=star1: t.limit(10)[lambda x: x.f > 0],
        aggregate_table_count_metric=lambda t=star1: t.count(),
        self_reference_simple=lambda t=star1: t.view(),
        test_physical_table_reference_translate=lambda t=alltypes: t,
    )

    for k, f in tests.items():
        sql_golden_check(k, f())


@sqlgolden
def test_nameless_table():
    # Generate a unique table name when we haven't passed on
    nameless = ibis.table([('key', 'string')])
    assert Compiler.to_sql(nameless) == f'SELECT *\nFROM {nameless.op().name}'

    return ibis.table([('key', 'string')], name='baz')


@sqlgoldens
def test_simple_joins(star1, star2):
    t1 = star1
    t2 = star2

    pred = t1['foo_id'] == t2['foo_id']
    pred2 = t1['bar_id'] == t2['foo_id']
    yield t1.inner_join(t2, [pred])[[t1]]
    yield t1.left_join(t2, [pred])[[t1]]
    yield t1.outer_join(t2, [pred])[[t1]]
    yield t1.inner_join(t2, [pred, pred2])[[t1]]


@sqlgolden
def test_multiple_joins(con, star1, star2, star3):
    t1 = star1
    t2 = star2
    t3 = star3

    predA = t1['foo_id'] == t2['foo_id']
    predB = t1['bar_id'] == t3['bar_id']

    return (
        t1.left_join(t2, [predA])
        .inner_join(t3, [predB])
        .projection([t1, t2['value1'], t3['value2']])
    )


@sqlgolden
def test_join_between_joins():
    t1 = ibis.table(
        [('key1', 'string'), ('key2', 'string'), ('value1', 'double')],
        'first',
    )

    t2 = ibis.table([('key1', 'string'), ('value2', 'double')], 'second')
    t3 = ibis.table(
        [('key2', 'string'), ('key3', 'string'), ('value3', 'double')],
        'third',
    )
    t4 = ibis.table([('key3', 'string'), ('value4', 'double')], 'fourth')
    left = t1.inner_join(t2, [('key1', 'key1')])[t1, t2.value2]
    right = t3.inner_join(t4, [('key3', 'key3')])[t3, t4.value4]

    joined = left.inner_join(right, [('key2', 'key2')])

    # At one point, the expression simplification was resulting in bad refs
    # here (right.value3 referencing the table inside the right join)
    exprs = [left, right.value3, right.value4]
    return joined.projection(exprs)


@sqlgolden
def test_join_just_materialized(con, nation, region, customer):
    t1 = nation
    t2 = region
    t3 = customer

    # GH #491
    return t1.inner_join(t2, t1.n_regionkey == t2.r_regionkey).inner_join(
        t3, t1.n_nationkey == t3.c_nationkey
    )


@sqlgolden
def test_semi_join(star1, star2):
    return star1.semi_join(star2, [star1.foo_id == star2.foo_id])[[star1]]


@sqlgolden
def test_anti_join(star1, star2):
    return star1.anti_join(star2, [star1.foo_id == star2.foo_id])[[star1]]


@sqlgolden
def test_where_no_pushdown_possible(star1, star2):
    t1 = star1
    t2 = star2

    joined = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[
        t1, (t1.f - t2.value1).name('diff')
    ]

    return joined[joined.diff > 1]


@sqlgolden
def test_where_with_between(alltypes):
    t = alltypes

    return t.filter([t.a > 0, t.f.between(0, 1)])


@sqlgolden
def test_where_analyze_scalar_op(functional_alltypes):
    # root cause of #310
    table = functional_alltypes

    return table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp('2010-01-01') + ibis.interval(months=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()


@sqlgolden
def test_bug_duplicated_where(airlines):
    # GH #539
    table = airlines

    t = table['arrdelay', 'dest']
    expr = t.group_by('dest').mutate(
        dest_avg=t.arrdelay.mean(), dev=t.arrdelay - t.arrdelay.mean()
    )

    tmp1 = expr[expr.dev.notnull()]
    tmp2 = tmp1.order_by(ibis.desc('dev'))
    return tmp2.limit(10)


@sqlgoldens
def test_aggregate_having(star1):
    # Filtering post-aggregation predicate
    t1 = star1

    total = t1.f.sum().name('total')
    metrics = [total]

    e1 = t1.aggregate(metrics, by=['foo_id'], having=[total > 10])
    e2 = t1.aggregate(metrics, by=['foo_id'], having=[t1.count() > 100])

    return e1, e2


@sqlgolden
def test_aggregate_count_joined(con):
    # count on more complicated table
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    return (
        region.inner_join(nation, region.r_regionkey == nation.n_regionkey)
        .select([nation, region.r_name.name('region')])
        .count()
    )


def test_no_aliases_needed():
    table = ibis.table([('key1', 'string'), ('key2', 'string'), ('value', 'double')])

    expr = table.aggregate([table['value'].sum().name('total')], by=['key1', 'key2'])

    query = get_query(expr)
    context = query.context
    assert not context.need_aliases()


@sqlgoldens
def test_fuse_projections():
    table = ibis.table(
        [('foo', 'int32'), ('bar', 'int64'), ('value', 'double')],
        name='tbl',
    )

    # Cases where we project in both cases using the base table reference
    f1 = (table['foo'] + table['bar']).name('baz')
    pred = table['value'] > 0

    table2 = table[table, f1]
    table2_filtered = table2[pred]

    f2 = (table2['foo'] * 2).name('qux')

    table3 = table2.projection([table2, f2])

    # fusion works even if there's a filter
    table3_filtered = table2_filtered.projection([table2, f2])

    return table3, table3_filtered


@sqlgolden
def test_projection_filter_fuse(projection_fuse_filter):
    expr1, expr2, expr3 = projection_fuse_filter

    sql1 = Compiler.to_sql(expr1)
    sql2 = Compiler.to_sql(expr2)

    assert sql1 == sql2

    # ideally sql1 == sql3 but the projection logic has been a mess for a long
    # time and causes bugs like
    #
    # https://github.com/ibis-project/ibis/issues/4003
    #
    # so we're conservative in fusing projections and filters
    #
    # even though it may seem obvious what to do, it's not
    return expr3


@sqlgolden
def test_bug_project_multiple_times(customer, nation, region):
    # GH: 108
    joined = customer.inner_join(
        nation, [customer.c_nationkey == nation.n_nationkey]
    ).inner_join(region, [nation.n_regionkey == region.r_regionkey])
    proj1 = [customer, nation.n_name, region.r_name]
    step1 = joined[proj1]

    topk_by = step1.c_acctbal.cast('double').sum()
    pred = step1.n_name.topk(10, by=topk_by)

    proj_exprs = [step1.c_name, step1.r_name, step1.n_name]
    step2 = step1[pred]
    return step2.projection(proj_exprs)


@sqlgoldens
def test_aggregate_projection_subquery(alltypes):
    t = alltypes

    proj = t[t.f > 0][t, (t.a + t.b).name('foo')]

    def agg(x):
        return x.aggregate([x.foo.sum().name('foo total')], by=['g'])

    # predicate gets pushed down
    filtered = proj[proj.g == 'bar']

    # Pushdown is not possible (in Impala, Postgres, others)
    return proj, filtered, agg(filtered), agg(proj[proj.foo < 10])


@sqlgolden
def test_double_nested_subquery_no_aliases():
    # We don't require any table aliasing anywhere
    t = ibis.table(
        [
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value', 'double'),
        ],
        'foo_table',
    )

    agg1 = t.aggregate([t.value.sum().name('total')], by=['key1', 'key2', 'key3'])
    agg2 = agg1.aggregate([agg1.total.sum().name('total')], by=['key1', 'key2'])
    return agg2.aggregate([agg2.total.sum().name('total')], by=['key1'])


@sqlgolden
def test_aggregate_projection_alias_bug(star1, star2):
    # Observed in use
    t1 = star1
    t2 = star2

    what = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[[t1, t2.value1]]

    # TODO: Not fusing the aggregation with the projection yet
    return what.aggregate([what.value1.sum().name('total')], by=[what.foo_id])


@sqlgolden
def test_subquery_in_union(alltypes):
    t = alltypes

    expr1 = t.group_by(['a', 'g']).aggregate(t.f.sum().name('metric'))
    expr2 = expr1.view()

    join1 = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]
    join2 = join1.view()

    return join1.union(join2)


@sqlgolden
def test_limit_with_self_join(functional_alltypes):
    t = functional_alltypes
    t2 = t.view()

    return t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()


@sqlgolden
def test_topk_predicate_pushdown_bug(nation, customer, region):
    # Observed on TPCH data
    cplusgeo = customer.inner_join(
        nation, [customer.c_nationkey == nation.n_nationkey]
    ).inner_join(region, [nation.n_regionkey == region.r_regionkey])[
        customer, nation.n_name, region.r_name
    ]

    pred = cplusgeo.n_name.topk(10, by=cplusgeo.c_acctbal.sum())
    return cplusgeo.filter([pred])


@sqlgolden
def test_topk_analysis_bug():
    # GH #398
    airlines = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    dests = ('ORD', 'JFK', 'SFO')
    delay_filter = airlines.dest.topk(10, by=airlines.arrdelay.mean())
    t = airlines[airlines.dest.isin(dests)]
    return t[delay_filter].group_by('origin').size()


@sqlgolden
def test_topk_to_aggregate():
    t = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    return t.dest.topk(10, by=t.arrdelay.mean())


@sqlgolden
def test_bool_bool():
    t = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    x = ibis.literal(True)
    return t[(t.dest.cast('int64') == 0) == x]


@sqlgolden
def test_case_in_projection(alltypes):
    t = alltypes

    expr = t.g.case().when('foo', 'bar').when('baz', 'qux').else_('default').end()

    expr2 = ibis.case().when(t.g == 'foo', 'bar').when(t.g == 'baz', t.g).end()

    return t[expr.name('col1'), expr2.name('col2'), t]


@sqlgolden
def test_identifier_quoting():
    data = ibis.table([('date', 'int32'), ('explain', 'string')], 'table')

    return data[data.date.name('else'), data.explain.name('join')]


@sqlgolden
def test_scalar_subquery_different_table(foo, bar):
    return foo[foo.y > bar.x.max()]


def test_exists_subquery_repr(t1, t2):
    # GH #660

    cond = t1.key1 == t2.key1
    expr = t1[cond.any()]
    stmt = get_query(expr)

    repr(stmt.where[0])


@sqlgolden
def test_filter_inside_exists():
    events = ibis.table(
        [
            ('session_id', 'int64'),
            ('user_id', 'int64'),
            ('event_type', 'int32'),
            ('ts', 'timestamp'),
        ],
        'events',
    )

    purchases = ibis.table(
        [
            ('item_id', 'int64'),
            ('user_id', 'int64'),
            ('price', 'double'),
            ('ts', 'timestamp'),
        ],
        'purchases',
    )
    filt = purchases.ts > '2015-08-15'
    cond = (events.user_id == purchases[filt].user_id).any()
    return events[cond]


@sqlgolden
def test_order_by_on_limit_yield_subquery(functional_alltypes):
    # x.limit(...).order_by(...)
    #   is semantically different from
    # x.order_by(...).limit(...)
    #   and will often yield different results
    t = functional_alltypes
    return (
        t.group_by('string_col')
        .aggregate([t.count().name('nrows')])
        .limit(5)
        .order_by('string_col')
    )


@sqlgolden
def test_join_with_limited_table(star1, star2):
    limited = star1.limit(100)
    return limited.inner_join(star2, [limited.foo_id == star2.foo_id])[[limited]]


@sqlgolden
def test_multiple_limits(functional_alltypes):
    t = functional_alltypes

    expr = t.limit(20).limit(10)
    stmt = get_query(expr)

    assert stmt.limit.n == 10
    return expr


@sqlgolden
def test_join_filtered_tables_no_pushdown():
    # #790, #781
    tbl_a = ibis.table(
        [
            ('year', 'int32'),
            ('month', 'int32'),
            ('day', 'int32'),
            ('value_a', 'double'),
        ],
        'a',
    )

    tbl_b = ibis.table(
        [
            ('year', 'int32'),
            ('month', 'int32'),
            ('day', 'int32'),
            ('value_b', 'double'),
        ],
        'b',
    )

    tbl_a_filter = tbl_a.filter([tbl_a.year == 2016, tbl_a.month == 2, tbl_a.day == 29])

    tbl_b_filter = tbl_b.filter([tbl_b.year == 2016, tbl_b.month == 2, tbl_b.day == 29])

    joined = tbl_a_filter.left_join(tbl_b_filter, ['year', 'month', 'day'])
    result = joined[tbl_a_filter.value_a, tbl_b_filter.value_b].op()

    join_op = result.table
    assert join_op.left == tbl_a_filter.op()
    assert join_op.right == tbl_b_filter.op()

    return result


@sqlgolden
def test_loj_subquery_filter_handling():
    # #781
    left = ibis.table([('id', 'int32'), ('desc', 'string')], 'foo')
    right = ibis.table([('id', 'int32'), ('desc', 'string')], 'bar')
    left = left[left.id < 2]
    right = right[right.id < 3]

    joined = left.left_join(right, ['id', 'desc'])
    return joined[
        [left[name].name('left_' + name) for name in left.columns]
        + [right[name].name('right_' + name) for name in right.columns]
    ]


@sqlgolden
def test_startswith(startswith):
    return startswith.name('tmp')


@sqlgolden
def test_endswith(endswith):
    return endswith.name('tmp')


@sqlgolden
def test_filter_predicates():
    table = ibis.table([("color", "string")], name="t")
    predicates = [
        lambda x: x.color.lower().like('%de%'),
        lambda x: x.color.lower().contains('de'),
        lambda x: x.color.lower().rlike('.*ge.*'),
    ]

    expr = table
    for pred in predicates:
        filtered = expr.filter(pred(expr))
        projected = filtered.projection([expr])
        expr = projected

    return expr
