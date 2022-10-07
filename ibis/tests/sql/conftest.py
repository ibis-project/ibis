import functools
from pathlib import Path

import pytest

import ibis
from ibis.backends.base.sql.compiler import Compiler, QueryContext
from ibis.tests.expr.mocks import MockBackend


def sql_golden_check(funcname, sql):
    if not isinstance(sql, str):
        sql = Compiler.to_sql(sql) + '\n'

    p = Path(ibis.__file__).parent / "tests" / "sqlgolden" / f"{funcname}.sql"
    if p.exists():
        assert sql == p.read_text()
    else:
        p.write_text(sql)


def sqlgolden(func):  # decorator
    @functools.wraps(func)
    def _decorated(*args, **kwargs):
        sql_golden_check(func.__name__, func(*args, **kwargs))

    return _decorated


def sqlgoldens(func):  # decorator
    @functools.wraps(func)
    def _decorated(*args, **kwargs):
        for i, expr in enumerate(func(*args, **kwargs)):
            sql_golden_check(f"{func.__name__}{i}", expr)

    return _decorated


@pytest.fixture(scope="module")
def con():
    return MockBackend()


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("alltypes")


@pytest.fixture(scope="module")
def functional_alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def star1(con):
    return con.table("star1")


@pytest.fixture(scope="module")
def star2(con):
    return con.table("star2")


@pytest.fixture(scope="module")
def star3(con):
    return con.table("star3")


@pytest.fixture(scope="module")
def nation(con):
    return con.table("tpch_nation")


@pytest.fixture(scope="module")
def region(con):
    return con.table("tpch_region")


@pytest.fixture(scope="module")
def customer(con):
    return con.table("tpch_customer")


@pytest.fixture(scope="module")
def airlines(con):
    return con.table("airlines")


@pytest.fixture(scope="module")
def foo_t(con):
    return con.table("foo_t")


@pytest.fixture(scope="module")
def bar_t(con):
    return con.table("bar_t")


def get_query(expr):
    ast = Compiler.to_ast(expr, QueryContext(compiler=Compiler))
    return ast.queries[0]


@sqlgolden
def test_multiple_joins(con, star1, star2, star3):
    t1 = star1
    t2 = star2
    t3 = star3

    predA = t1['foo_id'] == t2['foo_id']
    predB = t1['bar_id'] == t3['bar_id']

    what = (
        t1.left_join(t2, [predA])
        .inner_join(t3, [predB])
        .projection([t1, t2['value1'], t3['value2']])
    )
    return what


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
def test_join_projection_subquery_bug(nation, region, customer):
    # From an observed bug, derived from tpch tables
    geo = nation.inner_join(region, [('n_regionkey', 'r_regionkey')])[
        nation.n_nationkey,
        nation.n_name.name('nation'),
        region.r_name.name('region'),
    ]

    return geo.inner_join(customer, [('n_nationkey', 'c_nationkey')])[
        customer,
        geo,
    ]


@sqlgolden
def test_where_with_join(con, star1, star2):
    t1 = star1
    t2 = star2

    # This also tests some cases of predicate pushdown
    e1 = (
        t1.inner_join(t2, [t1.foo_id == t2.foo_id])
        .projection([t1, t2.value1, t2.value3])
        .filter([t1.f > 0, t2.value3 < 1000])
    )

    # e2 = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
    #       .filter([t1.f > 0, t2.value3 < 1000])
    #       .projection([t1, t2.value1, t2.value3]))

    # return e1, e2

    return e1


@sqlgolden
def test_subquery_used_for_self_join(con):
    # There could be cases that should look in SQL like
    # WITH t0 as (some subquery)
    # select ...
    # from t0 t1
    #   join t0 t2
    #     on t1.kind = t2.subkind
    # ...
    # However, the Ibis code will simply have an expression (projection or
    # aggregation, say) built on top of the subquery expression, so we need
    # to extract the subquery unit (we see that it appears multiple times
    # in the tree).
    t = con.table('alltypes')

    agged = t.aggregate([t.f.sum().name('total')], by=['g', 'a', 'b'])
    view = agged.view()
    metrics = [(agged.total - view.total).max().name('metric')]
    expr = agged.inner_join(view, [agged.a == view.b]).aggregate(metrics, by=[agged.g])

    return expr


@sqlgolden
def test_subquery_factor_correlated_subquery(con):
    # #173, #183 and other issues
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

    tpch = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    )

    # Self-reference + correlated subquery complicates things
    t2 = tpch.view()
    conditional_avg = t2[t2.region == tpch.region].amount.mean()
    amount_filter = tpch.amount > conditional_avg

    return tpch[amount_filter].limit(10)


@sqlgolden
def self_join_subquery_distinct_equal(con):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    j1 = region.join(nation, region.r_regionkey == nation.n_regionkey)[region, nation]

    j2 = region.join(nation, region.r_regionkey == nation.n_regionkey)[
        region, nation
    ].view()

    expr = j1.join(j2, j1.r_regionkey == j2.r_regionkey)[j1.r_name, j2.n_name]

    return expr


@sqlgolden
def test_tpch_self_join_failure(con):
    # duplicating the integration test here

    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')

    fields_of_interest = [
        region.r_name.name('region'),
        nation.n_name.name('nation'),
        orders.o_totalprice.name('amount'),
        orders.o_orderdate.cast('timestamp').name('odate'),
    ]

    joined_all = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    )

    year = joined_all.odate.year().name('year')
    total = joined_all.amount.sum().cast('double').name('total')
    annual_amounts = joined_all.group_by(['region', year]).aggregate(total)

    current = annual_amounts
    prior = annual_amounts.view()

    yoy_change = (current.total - prior.total).name('yoy_change')
    yoy = current.join(prior, current.year == (prior.year - 1))[
        current.region, current.year, yoy_change
    ]
    Compiler.to_sql(yoy)  # fail


@sqlgoldens
def test_subquery_in_filter_predicate(con, star1):
    # E.g. comparing against some scalar aggregate value. See Ibis #43
    t1 = star1

    pred = t1.f > t1.f.mean()
    expr = t1[pred]

    # This brought out another expression rewriting bug, since the filtered
    # table isn't found elsewhere in the expression.
    pred2 = t1.f > t1[t1.foo_id == 'foo'].f.mean()
    expr2 = t1[pred2]

    return expr, expr2


@sqlgoldens
def test_filter_subquery_derived_reduction(con, star1):
    t1 = star1

    # Reduction can be nested inside some scalar expression
    pred3 = t1.f > t1[t1.foo_id == 'foo'].f.mean().log()
    pred4 = t1.f > (t1[t1.foo_id == 'foo'].f.mean().log() + 1)

    expr3 = t1[pred3]
    expr4 = t1[pred4]

    return expr3, expr4


@sqlgoldens
def test_topk_operation(con):
    # TODO: top K with filter in place

    table = ibis.table(
        [
            ('foo', 'string'),
            ('bar', 'string'),
            ('city', 'string'),
            ('v1', 'double'),
            ('v2', 'double'),
        ],
        'tbl',
    )

    what = table.city.topk(10, by=table.v2.mean())
    e1 = table[what]

    # Test the default metric (count)
    what = table.city.topk(10)
    e2 = table[what]

    return e1, e2


@pytest.fixture(scope="module")
def foo(con):
    return con.table("foo")


@pytest.fixture(scope="module")
def bar(con):
    return con.table("bar")


@pytest.fixture(scope="module")
def t1(con):
    return con.table("t1")


@pytest.fixture(scope="module")
def t2(con):
    return con.table("t2")


@pytest.fixture(scope="module")
def where_uncorrelated_subquery(foo, bar):
    return foo[foo.job.isin(bar.job)]


@pytest.fixture(scope="module")
def not_exists(foo_t, bar_t):
    return foo_t[-(foo_t.key1 == bar_t.key1).any()]


@pytest.fixture(scope="module")
def union(con):
    table = con.table('functional_alltypes')

    t1 = table[table.int_col > 0][
        table.string_col.name('key'),
        table.float_col.cast('double').name('value'),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name('key'), table.double_col.name('value')
    ]

    return t1.union(t2, distinct=True)


@pytest.fixture(scope="module")
def union_all(con):
    table = con.table('functional_alltypes')

    t1 = table[table.int_col > 0][
        table.string_col.name('key'),
        table.float_col.cast('double').name('value'),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name('key'), table.double_col.name('value')
    ]

    return t1.union(t2, distinct=False)


@pytest.fixture(scope="module")
def intersect(con):
    table = con.table('functional_alltypes')

    t1 = table[table.int_col > 0][
        table.string_col.name('key'),
        table.float_col.cast('double').name('value'),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name('key'), table.double_col.name('value')
    ]

    return t1.intersect(t2)


@pytest.fixture(scope="module")
def difference(con):
    table = con.table('functional_alltypes')

    t1 = table[table.int_col > 0][
        table.string_col.name('key'),
        table.float_col.cast('double').name('value'),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name('key'), table.double_col.name('value')
    ]

    return t1.difference(t2)


@pytest.fixture(scope="module")
def simple_case(con):
    t = con.table('alltypes')
    return t.g.case().when('foo', 'bar').when('baz', 'qux').else_('default').end()


@pytest.fixture(scope="module")
def search_case(con):
    t = con.table('alltypes')
    return ibis.case().when(t.f > 0, t.d * 2).when(t.c < 0, t.a * 2).end()


@sqlgolden
def self_reference_limit_exists(con):
    alltypes = con.table('functional_alltypes')
    t = alltypes.limit(100)
    t2 = t.view()
    return t[-((t.string_col == t2.string_col).any())]


@sqlgolden
def test_limit_cte_extract(con):
    alltypes = con.table('functional_alltypes')
    t = alltypes.limit(100)
    t2 = t.view()
    return t.join(t2).projection(t)


@sqlgolden
def test_filter_self_join_analysis_bug(con):
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

    joined = left.join(right, left.region == right.region)
    result = joined[left.region, (left.total - right.total).name('diff')]

    return result, purchases


@pytest.fixture(scope="module")
def projection_fuse_filter(con):
    # Probably test this during the evaluation phase. In SQL, "fusable"
    # table operations will be combined together into a single select
    # statement
    #
    # see ibis #71 for more on this

    t = ibis.table(
        [
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float32'),
            ('f', 'float64'),
            ('g', 'string'),
            ('h', 'boolean'),
        ],
        'foo',
    )

    proj = t['a', 'b', 'c']

    # Rewrite a little more aggressively here
    expr1 = proj[t.a > 0]

    # at one point these yielded different results
    filtered = t[t.a > 0]

    expr2 = filtered[t.a, t.b, t.c]
    expr3 = filtered.projection(['a', 'b', 'c'])

    return expr1, expr2, expr3


@pytest.fixture(scope="module")
def startswith(con, star1):
    t1 = star1
    return t1.foo_id.startswith('foo')


@pytest.fixture(scope="module")
def endswith(con, star1):
    t1 = star1
    return t1.foo_id.endswith('foo')
