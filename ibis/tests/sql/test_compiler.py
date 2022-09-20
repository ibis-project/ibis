import datetime

import ibis
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import sqlgolden


@sqlgolden
def test_union(union):
    return union


@sqlgolden
def test_union_project_column(union_all):
    # select a column, get a subquery
    return union_all[[union_all.key]]


@sqlgolden
def test_table_intersect(intersect):
    return intersect


@sqlgolden
def test_table_difference(difference):
    return difference


@sqlgolden
def test_intersect_project_column(intersect):
    # select a column, get a subquery
    return intersect[[intersect.key]]


@sqlgolden
def test_difference_project_column(difference):
    # select a column, get a subquery
    return difference[[difference.key]]


@sqlgolden
def test_table_distinct(con):
    t = con.table('functional_alltypes')

    return t[t.string_col, t.int_col].distinct()


@sqlgolden
def test_column_distinct(con):
    t = con.table('functional_alltypes')
    return t[t.string_col].distinct()


@sqlgolden
def test_count_distinct(con):
    t = con.table('functional_alltypes')

    metric = t.int_col.nunique().name('nunique')
    return t[t.bigint_col > 0].group_by('string_col').aggregate([metric])


@sqlgolden
def test_multiple_count_distinct(con):
    # Impala and some other databases will not execute multiple
    # count-distincts in a single aggregation query. This error reporting
    # will be left to the database itself, for now.
    t = con.table('functional_alltypes')
    metrics = [
        t.int_col.nunique().name('int_card'),
        t.smallint_col.nunique().name('smallint_card'),
    ]

    return t.group_by('string_col').aggregate(metrics)


@sqlgolden
def test_pushdown_with_or():
    t = ibis.table(
        [
            ('double_col', 'float64'),
            ('string_col', 'string'),
            ('int_col', 'int32'),
            ('float_col', 'float32'),
        ],
        'functional_alltypes',
    )
    subset = t[(t.double_col > 3.14) & t.string_col.contains('foo')]
    return subset[(subset.int_col - 1 == 0) | (subset.float_col <= 1.34)]


@sqlgolden
def test_having_size():
    t = ibis.table(
        [
            ('double_col', 'double'),
            ('string_col', 'string'),
            ('int_col', 'int32'),
            ('float_col', 'float'),
        ],
        'functional_alltypes',
    )
    return t.group_by(t.string_col).having(t.double_col.max() == 1).size()


@sqlgolden
def test_having_from_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], 't')
    filt = t[t.b == 'm']
    gb = filt.group_by(filt.b)
    having = gb.having(filt.a.max() == 2)
    return having.aggregate(filt.a.sum().name('sum'))


@sqlgolden
def test_simple_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='my_table')
    filt = t[t.a < 100]
    return filt[filt.a == filt.a.max()]


@sqlgolden
def test_agg_and_non_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='my_table')
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    return expr[expr.b == 'a']


@sqlgolden
def test_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'int64')], name='my_table')
    t = t.mutate(b2=t.b * 2)
    t = t[['a', 'b2']]
    filt = t[t.a < 100]
    return filt[filt.a == filt.a.max().name('blah')]


@sqlgolden
def test_agg_filter_with_alias():
    t = ibis.table([('a', 'int64'), ('b', 'int64')], name='my_table')
    t = t.mutate(b2=t.b * 2)
    t = t[['a', 'b2']]
    filt = t[t.a < 100]
    return filt[filt.a.name('A') == filt.a.max().name('blah')]


@sqlgolden
def test_table_drop_with_filter():
    left = ibis.table(
        [('a', 'int64'), ('b', 'string'), ('c', 'timestamp')], name='t'
    ).relabel({'c': 'C'})
    left = left.filter(left.C == datetime.datetime(2018, 1, 1))
    left = left.drop('C')
    left = left.mutate(the_date=datetime.datetime(2018, 1, 1))

    right = ibis.table([('b', 'string')], name='s')
    joined = left.join(right, left.b == right.b)
    joined = joined[left.a]
    return joined.filter(joined.a < 1.0)


def test_table_drop_consistency():
    # GH2829
    t = ibis.table(
        [('a', 'int64'), ('b', 'string'), ('c', 'timestamp')], name='t'
    )

    expected = t.projection(["a", "c"])
    result = t.drop("b")

    assert expected.schema() == result.schema()
    assert set(result.columns) == {"a", "c"}


@sqlgolden
def test_subquery_where_location():
    t = ibis.table(
        [
            ("float_col", "float32"),
            ("timestamp_col", "timestamp"),
            ("int_col", "int32"),
            ("string_col", "string"),
        ],
        name="alltypes",
    )
    param = ibis.param("timestamp").name("my_param")
    expr = (
        t[["float_col", "timestamp_col", "int_col", "string_col"]][
            lambda t: t.timestamp_col < param
        ]
        .groupby("string_col")
        .aggregate(foo=lambda t: t.float_col.sum())
        .foo.count()
    )
    return Compiler.to_sql(expr, params={param: "20140101"})


@sqlgolden
def test_column_expr_retains_name():
    t = ibis.table(
        [
            ('int_col', 'int32'),
        ],
        'int_col_table',
    )
    return (t.int_col + 4).name('foo')


@sqlgolden
def test_column_expr_default_name():
    t = ibis.table(
        [
            ('int_col', 'int32'),
        ],
        'int_col_table',
    )
    return t.int_col + 4
