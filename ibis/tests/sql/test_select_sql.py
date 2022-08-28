import pytest

import ibis
from ibis.backends.base.sql.compiler import Compiler
from ibis.tests.sql.conftest import get_query


def test_nameless_table():
    # Generate a unique table name when we haven't passed on
    nameless = ibis.table([('key', 'string')])
    assert Compiler.to_sql(nameless) == 'SELECT *\nFROM {}'.format(
        nameless.op().name
    )

    with_name = ibis.table([('key', 'string')], name='baz')
    result = Compiler.to_sql(with_name)
    assert result == 'SELECT *\nFROM baz'


def test_physical_table_reference_translate(alltypes):
    result = Compiler.to_sql(alltypes)
    expected = "SELECT *\nFROM alltypes"
    assert result == expected


def test_simple_joins(star1, star2):
    t1 = star1
    t2 = star2

    pred = t1['foo_id'] == t2['foo_id']
    pred2 = t1['bar_id'] == t2['foo_id']
    cases = [
        (
            t1.inner_join(t2, [pred])[[t1]],
            """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`""",
        ),
        (
            t1.left_join(t2, [pred])[[t1]],
            """SELECT t0.*
FROM star1 t0
  LEFT OUTER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`""",
        ),
        (
            t1.outer_join(t2, [pred])[[t1]],
            """SELECT t0.*
FROM star1 t0
  FULL OUTER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`""",
        ),
        # multiple predicates
        (
            t1.inner_join(t2, [pred, pred2])[[t1]],
            """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON (t0.`foo_id` = t1.`foo_id`) AND
       (t0.`bar_id` = t1.`foo_id`)""",
        ),
    ]

    for expr, expected_sql in cases:
        result_sql = Compiler.to_sql(expr)
        assert result_sql == expected_sql


def test_multiple_joins(multiple_joins):
    what = multiple_joins

    result = Compiler.to_sql(what)
    expected = """\
SELECT *, `value1`, t1.`value2`
FROM (
  SELECT t2.`c`, t2.`f`, t2.`foo_id` AS `foo_id_x`, t2.`bar_id`,
         t3.`foo_id` AS `foo_id_y`, t3.`value1`, t3.`value3`
  FROM star1 t2
    LEFT OUTER JOIN star2 t3
      ON t2.`foo_id` = t3.`foo_id`
) t0
  INNER JOIN star3 t1
    ON `bar_id` = t1.`bar_id`"""
    assert result == expected


def test_join_between_joins(join_between_joins):
    projected = join_between_joins

    result = Compiler.to_sql(projected)
    expected = """\
SELECT t0.*, t1.`value3`, t1.`value4`
FROM (
  SELECT t2.*, t3.`value2`
  FROM `first` t2
    INNER JOIN second t3
      ON t2.`key1` = t3.`key1`
) t0
  INNER JOIN (
    SELECT t2.*, t3.`value4`
    FROM third t2
      INNER JOIN fourth t3
        ON t2.`key3` = t3.`key3`
  ) t1
    ON t0.`key2` = t1.`key2`"""
    assert result == expected


def test_join_just_materialized(join_just_materialized):
    joined = join_just_materialized
    result = Compiler.to_sql(joined)
    expected = """SELECT *
FROM tpch_nation t0
  INNER JOIN tpch_region t1
    ON t0.`n_regionkey` = t1.`r_regionkey`
  INNER JOIN tpch_customer t2
    ON t0.`n_nationkey` = t2.`c_nationkey`"""
    assert result == expected

    result = Compiler.to_sql(joined)
    assert result == expected


def test_semi_anti_joins(semi_anti_joins):
    sj, aj = semi_anti_joins

    result = Compiler.to_sql(sj)
    expected = """SELECT t0.*
FROM star1 t0
  LEFT SEMI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
    assert result == expected

    result = Compiler.to_sql(aj)
    expected = """SELECT t0.*
FROM star1 t0
  LEFT ANTI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
    assert result == expected


def test_self_reference_simple(self_reference_simple):
    expr = self_reference_simple

    result_sql = Compiler.to_sql(expr)
    expected_sql = "SELECT *\nFROM star1"
    assert result_sql == expected_sql


def test_join_self_reference(self_reference_join):
    result = self_reference_join

    result_sql = Compiler.to_sql(result)
    expected_sql = """SELECT t0.*
FROM star1 t0
  INNER JOIN star1 t1
    ON t0.`foo_id` = t1.`bar_id`"""
    assert result_sql == expected_sql


def test_join_projection_subquery_broken_alias(join_projection_subquery_bug):
    expr = join_projection_subquery_bug

    result = Compiler.to_sql(expr)
    expected = """SELECT t1.*, t0.*
FROM (
  SELECT t2.`n_nationkey`, t2.`n_name` AS `nation`, t3.`r_name` AS `region`
  FROM tpch_nation t2
    INNER JOIN tpch_region t3
      ON t2.`n_regionkey` = t3.`r_regionkey`
) t0
  INNER JOIN tpch_customer t1
    ON t0.`n_nationkey` = t1.`c_nationkey`"""
    assert result == expected


def test_where_simple_comparisons(where_simple_comparisons):
    what = where_simple_comparisons
    result = Compiler.to_sql(what)
    expected = """SELECT *
FROM star1
WHERE (`f` > 0) AND
      (`c` < (`f` * 2))"""
    assert result == expected


def test_where_with_join(where_with_join):
    e1 = where_with_join

    expected_sql = """SELECT t0.*, t1.`value1`, t1.`value3`
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
WHERE (t0.`f` > 0) AND
      (t1.`value3` < 1000)"""

    result_sql = Compiler.to_sql(e1)
    assert result_sql == expected_sql


def test_where_no_pushdown_possible(star1, star2):
    t1 = star1
    t2 = star2

    joined = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[
        t1, (t1.f - t2.value1).name('diff')
    ]

    filtered = joined[joined.diff > 1]

    expected_sql = """\
SELECT t0.*
FROM (
  SELECT t1.*, t1.`f` - t2.`value1` AS `diff`
  FROM star1 t1
    INNER JOIN star2 t2
      ON t1.`foo_id` = t2.`foo_id`
) t0
WHERE t0.`diff` > 1"""

    result_sql = Compiler.to_sql(filtered)
    assert result_sql == expected_sql


def test_where_with_between(alltypes):
    t = alltypes

    what = t.filter([t.a > 0, t.f.between(0, 1)])
    result = Compiler.to_sql(what)
    expected = """SELECT *
FROM alltypes
WHERE (`a` > 0) AND
      (`f` BETWEEN 0 AND 1)"""
    assert result == expected


def test_where_analyze_scalar_op(functional_alltypes):
    # root cause of #310
    table = functional_alltypes

    expr = table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp('2010-01-01') + ibis.interval(months=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()

    result = Compiler.to_sql(expr)
    expected = """\
SELECT count(*) AS `count`
FROM functional_alltypes
WHERE (`timestamp_col` < date_add(cast({!r} as timestamp), INTERVAL 3 MONTH)) AND
      (`timestamp_col` < date_add(cast(now() as timestamp), INTERVAL 10 DAY))"""  # noqa: E501
    assert result == expected.format("2010-01-01T00:00:00")


def test_bug_duplicated_where(airlines):
    # GH #539
    table = airlines

    t = table['arrdelay', 'dest']
    expr = t.group_by('dest').mutate(
        dest_avg=t.arrdelay.mean(), dev=t.arrdelay - t.arrdelay.mean()
    )

    tmp1 = expr[expr.dev.notnull()]
    tmp2 = tmp1.sort_by(ibis.desc('dev'))
    worst = tmp2.limit(10)

    result = Compiler.to_sql(worst)

    expected = """\
SELECT *
FROM (
  SELECT t1.*
  FROM (
    SELECT *, avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dest_avg`,
           `arrdelay` - avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dev`
    FROM (
      SELECT `arrdelay`, `dest`
      FROM airlines
    ) t3
  ) t1
  WHERE t1.`dev` IS NOT NULL
) t0
ORDER BY `dev` DESC
LIMIT 10"""
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.aggregate([t['f'].sum().name('total')], [t['foo_id']]),
            """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1""",
            id="explicit_column",
        ),
        pytest.param(
            lambda t: t.aggregate(
                [t['f'].sum().name('total')], ['foo_id', 'bar_id']
            ),
            """SELECT `foo_id`, `bar_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1, 2""",
            id="string_columns",
        ),
    ],
)
def test_simple_aggregate_query(star1, expr_fn, expected):
    expr = expr_fn(star1)
    result = Compiler.to_sql(expr)
    assert result == expected


def test_aggregate_having(aggregate_having):
    e1, e2 = aggregate_having

    result = Compiler.to_sql(e1)
    expected = """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1
HAVING sum(`f`) > 10"""
    assert result == expected

    result = Compiler.to_sql(e2)
    expected = """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1
HAVING count(*) > 100"""
    assert result == expected


def test_aggregate_table_count_metric(star1):
    expr = star1.count()

    result = Compiler.to_sql(expr)
    expected = """SELECT count(*) AS `count`
FROM star1"""
    assert result == expected


def test_aggregate_count_joined(aggregate_count_joined):
    expr = aggregate_count_joined

    result = Compiler.to_sql(expr)
    expected = """\
SELECT count(*) AS `count`
FROM (
  SELECT t2.*, t1.`r_name` AS `region`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
) t0"""
    assert result == expected


def test_no_aliases_needed():
    table = ibis.table(
        [('key1', 'string'), ('key2', 'string'), ('value', 'double')]
    )

    expr = table.aggregate(
        [table['value'].sum().name('total')], by=['key1', 'key2']
    )

    query = get_query(expr)
    context = query.context
    assert not context.need_aliases()


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

    ex_sql = """\
SELECT *, `foo` * 2 AS `qux`
FROM (
  SELECT *, `foo` + `bar` AS `baz`
  FROM tbl
) t0"""

    ex_sql2 = """\
SELECT *, `foo` * 2 AS `qux`
FROM (
  SELECT *, `foo` + `bar` AS `baz`
  FROM tbl
  WHERE `value` > 0
) t0"""

    table3_sql = Compiler.to_sql(table3)
    table3_filt_sql = Compiler.to_sql(table3_filtered)

    assert table3_sql == ex_sql
    assert table3_filt_sql == ex_sql2


def test_projection_filter_fuse(projection_fuse_filter):
    expr1, expr2, expr3 = projection_fuse_filter

    sql1 = Compiler.to_sql(expr1)
    sql2 = Compiler.to_sql(expr2)
    sql3 = Compiler.to_sql(expr3)

    assert sql1 == sql2

    # ideally sql1 == sql3 but the projection logic has been a mess for a long
    # time and causes bugs like
    #
    # https://github.com/ibis-project/ibis/issues/4003
    #
    # so we're conservative in fusing projections and filters
    #
    # even though it may seem obvious what to do, it's not
    expected_sql3 = """\
SELECT `a`, `b`, `c`
FROM (
  SELECT *
  FROM foo
  WHERE `a` > 0
) t0"""
    assert sql3 == expected_sql3


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
    expr = step2.projection(proj_exprs)

    # it works!
    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT t2.*, t3.`n_name`, t4.`r_name`
  FROM tpch_customer t2
    INNER JOIN tpch_nation t3
      ON t2.`c_nationkey` = t3.`n_nationkey`
    INNER JOIN tpch_region t4
      ON t3.`n_regionkey` = t4.`r_regionkey`
)
SELECT t0.`c_name`, t0.`r_name`, t0.`n_name`
FROM t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `n_name`, sum(CAST(`c_acctbal` AS double)) AS `sum`
      FROM t0
      GROUP BY 1
    ) t2
    ORDER BY `sum` DESC
    LIMIT 10
  ) t1
    ON t0.`n_name` = t1.`n_name`"""
    assert result == expected


def test_aggregate_projection_subquery(alltypes):
    t = alltypes

    proj = t[t.f > 0][t, (t.a + t.b).name('foo')]

    result = Compiler.to_sql(proj)
    expected = """SELECT *, `a` + `b` AS `foo`
FROM alltypes
WHERE `f` > 0"""
    assert result == expected

    def agg(x):
        return x.aggregate([x.foo.sum().name('foo total')], by=['g'])

    # predicate gets pushed down
    filtered = proj[proj.g == 'bar']

    result = Compiler.to_sql(filtered)
    expected = """SELECT *, `a` + `b` AS `foo`
FROM alltypes
WHERE (`f` > 0) AND
      (`g` = 'bar')"""
    assert result == expected

    agged = agg(filtered)
    result = Compiler.to_sql(agged)
    expected = """SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE (`f` > 0) AND
        (`g` = 'bar')
) t0
GROUP BY 1"""
    assert result == expected

    # Pushdown is not possible (in Impala, Postgres, others)
    agged2 = agg(proj[proj.foo < 10])

    result = Compiler.to_sql(agged2)
    expected = """SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE `f` > 0
) t0
WHERE `foo` < 10
GROUP BY 1"""
    assert result == expected


def test_subquery_aliased(subquery_aliased):
    expected = """SELECT t0.*, t1.`value1`
FROM (
  SELECT `foo_id`, sum(`f`) AS `total`
  FROM star1
  GROUP BY 1
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
    result = Compiler.to_sql(subquery_aliased)
    assert result == expected


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

    agg1 = t.aggregate(
        [t.value.sum().name('total')], by=['key1', 'key2', 'key3']
    )
    agg2 = agg1.aggregate(
        [agg1.total.sum().name('total')], by=['key1', 'key2']
    )
    agg3 = agg2.aggregate([agg2.total.sum().name('total')], by=['key1'])

    result = Compiler.to_sql(agg3)
    expected = """\
SELECT `key1`, sum(`total`) AS `total`
FROM (
  SELECT `key1`, `key2`, sum(`total`) AS `total`
  FROM (
    SELECT `key1`, `key2`, `key3`, sum(`value`) AS `total`
    FROM foo_table
    GROUP BY 1, 2, 3
  ) t1
  GROUP BY 1, 2
) t0
GROUP BY 1"""
    assert result == expected


def test_aggregate_projection_alias_bug(star1, star2):
    # Observed in use
    t1 = star1
    t2 = star2

    what = t1.inner_join(t2, [t1.foo_id == t2.foo_id])[[t1, t2.value1]]

    what = what.aggregate([what.value1.sum().name('total')], by=[what.foo_id])

    # TODO: Not fusing the aggregation with the projection yet
    result = Compiler.to_sql(what)
    expected = """SELECT `foo_id`, sum(`value1`) AS `total`
FROM (
  SELECT t1.*, t2.`value1`
  FROM star1 t1
    INNER JOIN star2 t2
      ON t1.`foo_id` = t2.`foo_id`
) t0
GROUP BY 1"""
    assert result == expected


def test_subquery_used_for_self_join(subquery_used_for_self_join):
    expr = subquery_used_for_self_join

    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT `g`, `a`, `b`, sum(`f`) AS `total`
  FROM alltypes
  GROUP BY 1, 2, 3
)
SELECT t0.`g`, max(t0.`total` - `total`) AS `metric`
FROM (
  SELECT t0.`g` AS `g_x`, t0.`a` AS `a_x`, t0.`b` AS `b_x`,
         t0.`total` AS `total_x`, t3.`g` AS `g_y`, t3.`a` AS `a_y`,
         t3.`b` AS `b_y`, t3.`total` AS `total_y`
  FROM t0
    INNER JOIN t0 t3
      ON t0.`a` = t3.`b`
) t1
GROUP BY 1"""
    assert result == expected


def test_subquery_in_union(alltypes):
    t = alltypes

    expr1 = t.group_by(['a', 'g']).aggregate(t.f.sum().name('metric'))
    expr2 = expr1.view()

    join1 = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]
    join2 = join1.view()

    expr = join1.union(join2)
    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT `a`, `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1, 2
),
t1 AS (
  SELECT t0.*
  FROM t0
    INNER JOIN t0 t3
      ON t0.`g` = t3.`g`
)
SELECT *
FROM t1
UNION ALL
SELECT t0.*
FROM t0
  INNER JOIN t0 t3
    ON t0.`g` = t3.`g`"""
    assert result == expected


def test_subquery_factor_correlated_subquery(
    subquery_factor_correlated_subquery,
):
    # #173, #183 and other issues

    expr = subquery_factor_correlated_subquery

    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT t6.*, t1.`r_name` AS `region`, t3.`o_totalprice` AS `amount`,
         CAST(t3.`o_orderdate` AS timestamp) AS `odate`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
    INNER JOIN tpch_customer t6
      ON t6.`c_nationkey` = t2.`n_nationkey`
    INNER JOIN tpch_orders t3
      ON t3.`o_custkey` = t6.`c_custkey`
)
SELECT t0.*
FROM t0
WHERE t0.`amount` > (
  SELECT avg(t4.`amount`) AS `mean`
  FROM t0 t4
  WHERE t4.`region` = t0.`region`
)
LIMIT 10"""
    assert result == expected


def test_self_join_subquery_distinct_equal(self_join_subquery_distinct_equal):
    expr = self_join_subquery_distinct_equal

    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT t2.*, t3.*
  FROM tpch_region t2
    INNER JOIN tpch_nation t3
      ON t2.`r_regionkey` = t3.`n_regionkey`
)
SELECT t0.`r_name`, t1.`n_name`
FROM t0
  INNER JOIN t0 t1
    ON t0.`r_regionkey` = t1.`r_regionkey`"""

    assert result == expected


def test_limit_with_self_join(functional_alltypes):
    t = functional_alltypes
    t2 = t.view()

    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

    # it works
    result = Compiler.to_sql(expr)
    expected = """\
SELECT count(*) AS `count`
FROM (
  SELECT t1.`id` AS `id_x`, t1.`bool_col` AS `bool_col_x`,
         t1.`tinyint_col` AS `tinyint_col_x`,
         t1.`smallint_col` AS `smallint_col_x`,
         t1.`int_col` AS `int_col_x`, t1.`bigint_col` AS `bigint_col_x`,
         t1.`float_col` AS `float_col_x`,
         t1.`double_col` AS `double_col_x`,
         t1.`date_string_col` AS `date_string_col_x`,
         t1.`string_col` AS `string_col_x`,
         t1.`timestamp_col` AS `timestamp_col_x`, t1.`year` AS `year_x`,
         t1.`month` AS `month_x`, t2.`id` AS `id_y`,
         t2.`bool_col` AS `bool_col_y`,
         t2.`tinyint_col` AS `tinyint_col_y`,
         t2.`smallint_col` AS `smallint_col_y`,
         t2.`int_col` AS `int_col_y`, t2.`bigint_col` AS `bigint_col_y`,
         t2.`float_col` AS `float_col_y`,
         t2.`double_col` AS `double_col_y`,
         t2.`date_string_col` AS `date_string_col_y`,
         t2.`string_col` AS `string_col_y`,
         t2.`timestamp_col` AS `timestamp_col_y`, t2.`year` AS `year_y`,
         t2.`month` AS `month_y`
  FROM functional_alltypes t1
    INNER JOIN functional_alltypes t2
      ON t1.`tinyint_col` < extract(t2.`timestamp_col`, 'minute')
) t0"""
    assert result == expected


def test_cte_factor_distinct_but_equal(cte_factor_distinct_but_equal):
    expr = cte_factor_distinct_but_equal

    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1
    ON t0.`g` = t1.`g`"""

    assert result == expected


def test_tpch_self_join_failure(tpch_self_join_failure):
    Compiler.to_sql(tpch_self_join_failure)


def test_subquery_in_filter_predicate(subquery_in_filter_predicate):
    expr, expr2 = subquery_in_filter_predicate

    result = Compiler.to_sql(expr)
    expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`) AS `mean`
  FROM star1
)"""
    assert result == expected

    result = Compiler.to_sql(expr2)
    expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`) AS `mean`
  FROM star1
  WHERE `foo_id` = 'foo'
)"""
    assert result == expected


def test_filter_subquery_derived_reduction(filter_subquery_derived_reduction):
    expr3, expr4 = filter_subquery_derived_reduction

    result = Compiler.to_sql(expr3)
    expected = """\
SELECT *
FROM star1
WHERE `f` > ln((
  SELECT avg(`f`) AS `mean`
  FROM star1
  WHERE `foo_id` = 'foo'
))"""
    assert result == expected

    result = Compiler.to_sql(expr4)
    expected = """\
SELECT *
FROM star1
WHERE `f` > (ln((
  SELECT avg(`f`) AS `mean`
  FROM star1
  WHERE `foo_id` = 'foo'
)) + 1)"""
    assert result == expected


def test_topk_operation(topk_operation):
    filtered, filtered2 = topk_operation

    query = Compiler.to_sql(filtered)
    expected = """SELECT t0.*
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `city`, avg(`v2`) AS `mean`
      FROM tbl
      GROUP BY 1
    ) t2
    ORDER BY `mean` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`"""

    assert query == expected

    query = Compiler.to_sql(filtered2)
    expected = """\
SELECT t0.*
FROM tbl t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `city`, count(`city`) AS `count`
      FROM tbl
      GROUP BY 1
    ) t2
    ORDER BY `count` DESC
    LIMIT 10
  ) t1
    ON t0.`city` = t1.`city`"""
    assert query == expected


def test_topk_predicate_pushdown_bug(nation, customer, region):
    # Observed on TPCH data
    cplusgeo = customer.inner_join(
        nation, [customer.c_nationkey == nation.n_nationkey]
    ).inner_join(region, [nation.n_regionkey == region.r_regionkey])[
        customer, nation.n_name, region.r_name
    ]

    pred = cplusgeo.n_name.topk(10, by=cplusgeo.c_acctbal.sum())
    expr = cplusgeo.filter([pred])

    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT t2.*, t3.`n_name`, t4.`r_name`
  FROM tpch_customer t2
    INNER JOIN tpch_nation t3
      ON t2.`c_nationkey` = t3.`n_nationkey`
    INNER JOIN tpch_region t4
      ON t3.`n_regionkey` = t4.`r_regionkey`
)
SELECT t0.*
FROM t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `n_name`, sum(`c_acctbal`) AS `sum`
      FROM t0
      GROUP BY 1
    ) t2
    ORDER BY `sum` DESC
    LIMIT 10
  ) t1
    ON t0.`n_name` = t1.`n_name`"""
    assert result == expected


def test_topk_analysis_bug():
    # GH #398
    airlines = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    dests = {'ORD', 'JFK', 'SFO'}
    dests_formatted = repr(tuple(dests))
    delay_filter = airlines.dest.topk(10, by=airlines.arrdelay.mean())
    t = airlines[airlines.dest.isin(dests)]
    expr = t[delay_filter].group_by('origin').size()

    result = Compiler.to_sql(expr)
    expected = f"""\
SELECT `origin`, count(*) AS `count`
FROM (
  SELECT t1.*
  FROM (
    SELECT *
    FROM airlines
    WHERE `dest` IN {dests_formatted}
  ) t1
    LEFT SEMI JOIN (
      SELECT *
      FROM (
        SELECT `dest`, avg(`arrdelay`) AS `mean`
        FROM airlines
        GROUP BY 1
      ) t3
      ORDER BY `mean` DESC
      LIMIT 10
    ) t2
      ON `dest` = t2.`dest`
) t0
GROUP BY 1"""

    assert result == expected


def test_topk_to_aggregate():
    t = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    top = t.dest.topk(10, by=t.arrdelay.mean())

    result = Compiler.to_sql(top)
    expected = Compiler.to_sql(top.to_aggregation())
    assert result == expected


def test_bool_bool():
    import ibis
    from ibis.backends.base.sql.compiler import Compiler

    t = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    x = ibis.literal(True)
    top = t[(t.dest.cast('int64') == 0) == x]

    result = Compiler.to_sql(top)
    expected = """\
SELECT *
FROM airlines
WHERE (CAST(`dest` AS bigint) = 0) = TRUE"""
    assert result == expected


def test_case_in_projection(alltypes):
    t = alltypes

    expr = (
        t.g.case().when('foo', 'bar').when('baz', 'qux').else_('default').end()
    )

    expr2 = ibis.case().when(t.g == 'foo', 'bar').when(t.g == 'baz', t.g).end()

    proj = t[expr.name('col1'), expr2.name('col2'), t]

    result = Compiler.to_sql(proj)
    expected = """\
SELECT
  CASE `g`
    WHEN 'foo' THEN 'bar'
    WHEN 'baz' THEN 'qux'
    ELSE 'default'
  END AS `col1`,
  CASE
    WHEN `g` = 'foo' THEN 'bar'
    WHEN `g` = 'baz' THEN `g`
    ELSE CAST(NULL AS string)
  END AS `col2`, *
FROM alltypes"""
    assert result == expected


def test_identifier_quoting():
    data = ibis.table([('date', 'int32'), ('explain', 'string')], 'table')

    expr = data[data.date.name('else'), data.explain.name('join')]

    result = Compiler.to_sql(expr)
    expected = """SELECT `date` AS `else`, `explain` AS `join`
FROM `table`"""
    assert result == expected


def test_scalar_subquery_different_table(foo, bar):
    t1, t2 = foo, bar
    expr = t1[t1.y > t2.x.max()]

    result = Compiler.to_sql(expr)
    expected = """\
SELECT *
FROM foo
WHERE `y` > (
  SELECT max(`x`) AS `max`
  FROM bar
)"""
    assert result == expected


def test_where_uncorrelated_subquery(where_uncorrelated_subquery):
    expr = where_uncorrelated_subquery

    result = Compiler.to_sql(expr)
    expected = """SELECT *
FROM foo
WHERE `job` IN (
  SELECT `job`
  FROM bar
)"""
    assert result == expected


def test_where_correlated_subquery(where_correlated_subquery):
    expr = where_correlated_subquery
    result = Compiler.to_sql(expr)
    expected = """SELECT t0.*
FROM foo t0
WHERE t0.`y` > (
  SELECT avg(t1.`y`) AS `mean`
  FROM foo t1
  WHERE t0.`dept_id` = t1.`dept_id`
)"""
    assert result == expected


def test_exists(exists):
    e1, e2 = exists

    result = Compiler.to_sql(e1)
    expected = """\
SELECT t0.*
FROM foo_t t0
WHERE EXISTS (
  SELECT 1
  FROM bar_t t1
  WHERE t0.`key1` = t1.`key1`
)"""
    assert result == expected

    result = Compiler.to_sql(e2)
    expected = """\
SELECT t0.*
FROM foo_t t0
WHERE EXISTS (
  SELECT 1
  FROM bar_t t1
  WHERE (t0.`key1` = t1.`key1`) AND
        (t1.`key2` = 'foo')
)"""
    assert result == expected


def test_exists_subquery_repr(t1, t2):
    # GH #660

    cond = t1.key1 == t2.key1
    expr = t1[cond.any()]
    stmt = get_query(expr)

    repr(stmt.where[0])


def test_not_exists(not_exists):
    expr = not_exists
    result = Compiler.to_sql(expr)
    expected = """\
SELECT t0.*
FROM foo_t t0
WHERE NOT EXISTS (
  SELECT 1
  FROM bar_t t1
  WHERE t0.`key1` = t1.`key1`
)"""
    assert result == expected


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
    expr = events[cond]

    result = Compiler.to_sql(expr)
    expected = """\
SELECT t0.*
FROM events t0
WHERE EXISTS (
  SELECT 1
  FROM (
    SELECT *
    FROM purchases
    WHERE `ts` > '2015-08-15'
  ) t1
  WHERE t0.`user_id` = t1.`user_id`
)"""

    assert result == expected


def test_self_reference_in_exists(self_reference_in_exists):
    semi, anti = self_reference_in_exists

    result = Compiler.to_sql(semi)
    expected = """\
SELECT t0.*
FROM functional_alltypes t0
WHERE EXISTS (
  SELECT 1
  FROM functional_alltypes t1
  WHERE t0.`string_col` = t1.`string_col`
)"""
    assert result == expected

    result = Compiler.to_sql(anti)
    expected = """\
SELECT t0.*
FROM functional_alltypes t0
WHERE NOT EXISTS (
  SELECT 1
  FROM functional_alltypes t1
  WHERE t0.`string_col` = t1.`string_col`
)"""
    assert result == expected


def test_self_reference_limit_exists(self_reference_limit_exists):
    case = self_reference_limit_exists

    expected = """\
WITH t0 AS (
  SELECT *
  FROM functional_alltypes
  LIMIT 100
)
SELECT *
FROM t0
WHERE NOT EXISTS (
  SELECT 1
  FROM t0 t1
  WHERE t0.`string_col` = t1.`string_col`
)"""
    result = Compiler.to_sql(case)
    assert result == expected


def test_limit_cte_extract(limit_cte_extract):
    case = limit_cte_extract
    result = Compiler.to_sql(case)

    expected = """\
WITH t0 AS (
  SELECT *
  FROM functional_alltypes
  LIMIT 100
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1"""

    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.sort_by("f"),
            """\
SELECT *
FROM star1
ORDER BY `f`""",
            id="single_column",
        ),
        pytest.param(
            lambda t: t.sort_by(("f", 0)),
            """\
SELECT *
FROM star1
ORDER BY `f` DESC""",
            id="single_column_explicit_ascending",
        ),
        pytest.param(
            lambda t: t.sort_by(["c", ("f", 0)]),
            """\
SELECT *
FROM star1
ORDER BY `c`, `f` DESC""",
            id="mixed_columns_ascending",
        ),
    ],
)
def test_sort_by(star1, expr_fn, expected):
    expr = expr_fn(star1)
    result = Compiler.to_sql(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.limit(10),
            """\
SELECT *
FROM star1
LIMIT 10""",
            id="simple",
        ),
        pytest.param(
            lambda t: t.limit(10, offset=5),
            """\
SELECT *
FROM star1
LIMIT 10 OFFSET 5""",
            id="limit_with_offset",
        ),
        pytest.param(
            lambda t: t[t.f > 0].limit(10),
            """\
SELECT *
FROM star1
WHERE `f` > 0
LIMIT 10""",
            id="filter_then_limit",
        ),
        pytest.param(
            lambda t: t.limit(10)[lambda x: x.f > 0],
            """\
SELECT *
FROM (
  SELECT *
  FROM star1
  LIMIT 10
) t0
WHERE `f` > 0""",
            id="limit_then_filter",
        ),
    ],
)
def test_limit(star1, expr_fn, expected):
    expr = expr_fn(star1)
    result = Compiler.to_sql(expr)
    assert result == expected


def test_join_with_limited_table(join_with_limited_table):
    joined = join_with_limited_table

    result = Compiler.to_sql(joined)
    expected = """\
SELECT t0.*
FROM (
  SELECT *
  FROM star1
  LIMIT 100
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""

    assert result == expected


def test_sort_by_on_limit_yield_subquery(functional_alltypes):
    # x.limit(...).sort_by(...)
    #   is semantically different from
    # x.sort_by(...).limit(...)
    #   and will often yield different results
    t = functional_alltypes
    expr = (
        t.group_by('string_col')
        .aggregate([t.count().name('nrows')])
        .limit(5)
        .sort_by('string_col')
    )

    result = Compiler.to_sql(expr)
    expected = """SELECT *
FROM (
  SELECT `string_col`, count(*) AS `nrows`
  FROM functional_alltypes
  GROUP BY 1
  LIMIT 5
) t0
ORDER BY `string_col`"""
    assert result == expected


def test_multiple_limits(functional_alltypes):
    t = functional_alltypes

    expr = t.limit(20).limit(10)
    stmt = get_query(expr)

    assert stmt.limit.n == 10


def test_self_join_filter_analysis_bug(filter_self_join_analysis_bug):
    expr, _ = filter_self_join_analysis_bug

    expected = """\
WITH t0 AS (
  SELECT `region`, `kind`, sum(`amount`) AS `total`
  FROM purchases
  GROUP BY 1, 2
)
SELECT t1.`region`, t1.`total` - t2.`total` AS `diff`
FROM (
  SELECT *
  FROM t0
  WHERE `kind` = 'foo'
) t1
  INNER JOIN (
    SELECT *
    FROM t0
    WHERE `kind` = 'bar'
  ) t2
    ON t1.`region` = t2.`region`"""
    result = Compiler.to_sql(expr)
    assert result == expected


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

    tbl_a_filter = tbl_a.filter(
        [tbl_a.year == 2016, tbl_a.month == 2, tbl_a.day == 29]
    )

    tbl_b_filter = tbl_b.filter(
        [tbl_b.year == 2016, tbl_b.month == 2, tbl_b.day == 29]
    )

    joined = tbl_a_filter.left_join(tbl_b_filter, ['year', 'month', 'day'])
    result = joined[tbl_a_filter.value_a, tbl_b_filter.value_b]

    join_op = result.op().table.op()
    assert join_op.left.equals(tbl_a_filter)
    assert join_op.right.equals(tbl_b_filter)

    result_sql = Compiler.to_sql(result)
    expected_sql = """\
SELECT t0.`value_a`, t1.`value_b`
FROM (
  SELECT *
  FROM a
  WHERE (`year` = 2016) AND
        (`month` = 2) AND
        (`day` = 29)
) t0
  LEFT OUTER JOIN (
    SELECT *
    FROM b
    WHERE (`year` = 2016) AND
          (`month` = 2) AND
          (`day` = 29)
  ) t1
    ON (t0.`year` = t1.`year`) AND
       (t0.`month` = t1.`month`) AND
       (t0.`day` = t1.`day`)"""

    assert result_sql == expected_sql


def test_loj_subquery_filter_handling():
    # #781
    left = ibis.table([('id', 'int32'), ('desc', 'string')], 'foo')
    right = ibis.table([('id', 'int32'), ('desc', 'string')], 'bar')
    left = left[left.id < 2]
    right = right[right.id < 3]

    joined = left.left_join(right, ['id', 'desc'])
    joined = joined[
        [left[name].name('left_' + name) for name in left.columns]
        + [right[name].name('right_' + name) for name in right.columns]
    ]

    result = Compiler.to_sql(joined)
    expected = """\
SELECT t0.`id` AS `left_id`, t0.`desc` AS `left_desc`, t1.`id` AS `right_id`,
       t1.`desc` AS `right_desc`
FROM (
  SELECT *
  FROM foo
  WHERE `id` < 2
) t0
  LEFT OUTER JOIN (
    SELECT *
    FROM bar
    WHERE `id` < 3
  ) t1
    ON (t0.`id` = t1.`id`) AND
       (t0.`desc` = t1.`desc`)"""

    assert result == expected


def test_startswith(startswith):
    expr = startswith
    expected = """\
SELECT `foo_id` like concat('foo', '%') AS `tmp`
FROM star1"""
    assert Compiler.to_sql(expr) == expected


def test_endswith(endswith):
    expr = endswith
    expected = """\
SELECT `foo_id` like concat('%', 'foo') AS `tmp`
FROM star1"""
    assert Compiler.to_sql(expr) == expected


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

    expected = """\
SELECT *
FROM (
  SELECT *
  FROM (
    SELECT *
    FROM t
    WHERE (lower(`color`) LIKE '%de%') AND
          (locate('de', lower(`color`)) - 1 >= 0)
  ) t1
  WHERE regexp_like(lower(`color`), '.*ge.*')
) t0"""

    result = Compiler.to_sql(expr)
    assert result == expected
