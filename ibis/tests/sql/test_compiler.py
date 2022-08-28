import datetime
import textwrap

import ibis
from ibis.backends.base.sql.compiler import Compiler

QUERY = """\
SELECT `string_col` AS `key`, CAST(`float_col` AS double) AS `value`
FROM functional_alltypes
WHERE `int_col` > 0
{}
SELECT `string_col` AS `key`, `double_col` AS `value`
FROM functional_alltypes
WHERE `int_col` <= 0"""

SELECT_QUERY = f"SELECT `key`\nFROM (\n{textwrap.indent(QUERY, '  ')}\n) t0"


def test_union(union):
    result = Compiler.to_sql(union)
    expected = QUERY.format('UNION')
    assert result == expected


def test_union_project_column(union_all):
    # select a column, get a subquery
    expr = union_all[[union_all.key]]
    result = Compiler.to_sql(expr)
    expected = SELECT_QUERY.format("UNION ALL")
    assert result == expected


def test_table_intersect(intersect):
    result = Compiler.to_sql(intersect)
    expected = QUERY.format('INTERSECT')
    assert result == expected


def test_table_difference(difference):
    result = Compiler.to_sql(difference)
    expected = QUERY.format('EXCEPT')
    assert result == expected


def test_intersect_project_column(intersect):
    # select a column, get a subquery
    expr = intersect[[intersect.key]]
    result = Compiler.to_sql(expr)
    expected = SELECT_QUERY.format('INTERSECT')
    assert result == expected


def test_difference_project_column(difference):
    # select a column, get a subquery
    expr = difference[[difference.key]]
    result = Compiler.to_sql(expr)
    expected = SELECT_QUERY.format('EXCEPT')
    assert result == expected


def test_table_distinct(con):
    t = con.table('functional_alltypes')

    expr = t[t.string_col, t.int_col].distinct()

    result = Compiler.to_sql(expr)
    expected = """SELECT DISTINCT `string_col`, `int_col`
FROM functional_alltypes"""
    assert result == expected


def test_column_distinct(con):
    t = con.table('functional_alltypes')
    expr = t[t.string_col].distinct()

    result = Compiler.to_sql(expr)
    expected = """SELECT DISTINCT `string_col`
FROM functional_alltypes"""
    assert result == expected


def test_count_distinct(con):
    t = con.table('functional_alltypes')

    metric = t.int_col.nunique().name('nunique')
    expr = t[t.bigint_col > 0].group_by('string_col').aggregate([metric])

    result = Compiler.to_sql(expr)
    expected = """\
SELECT `string_col`, count(DISTINCT `int_col`) AS `nunique`
FROM functional_alltypes
WHERE `bigint_col` > 0
GROUP BY 1"""
    assert result == expected


def test_multiple_count_distinct(con):
    # Impala and some other databases will not execute multiple
    # count-distincts in a single aggregation query. This error reporting
    # will be left to the database itself, for now.
    t = con.table('functional_alltypes')
    metrics = [
        t.int_col.nunique().name('int_card'),
        t.smallint_col.nunique().name('smallint_card'),
    ]

    expr = t.group_by('string_col').aggregate(metrics)

    result = Compiler.to_sql(expr)
    expected = """\
SELECT `string_col`, count(DISTINCT `int_col`) AS `int_card`,
       count(DISTINCT `smallint_col`) AS `smallint_card`
FROM functional_alltypes
GROUP BY 1"""
    assert result == expected


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
    filt = subset[(subset.int_col - 1 == 0) | (subset.float_col <= 1.34)]
    result = Compiler.to_sql(filt)
    expected = """\
SELECT *
FROM functional_alltypes
WHERE ((`double_col` > 3.14) AND locate('foo', `string_col`) - 1 >= 0) AND
      (((`int_col` - 1) = 0) OR (`float_col` <= 1.34))"""
    assert result == expected


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
    expr = t.group_by(t.string_col).having(t.double_col.max() == 1).size()
    result = Compiler.to_sql(expr)
    assert (
        result
        == """\
SELECT `string_col`, count(*) AS `count`
FROM functional_alltypes
GROUP BY 1
HAVING max(`double_col`) = 1"""
    )


def test_having_from_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], 't')
    filt = t[t.b == 'm']
    gb = filt.group_by(filt.b)
    having = gb.having(filt.a.max() == 2)
    agg = having.aggregate(filt.a.sum().name('sum'))
    result = Compiler.to_sql(agg)
    expected = """\
SELECT `b`, sum(`a`) AS `sum`
FROM t
WHERE `b` = 'm'
GROUP BY 1
HAVING max(`a`) = 2"""
    assert result == expected


def test_simple_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='my_table')
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    result = Compiler.to_sql(expr)
    expected = """\
SELECT *
FROM (
  SELECT *
  FROM my_table
  WHERE `a` < 100
) t0
WHERE `a` = (
  SELECT max(`a`) AS `max`
  FROM my_table
  WHERE `a` < 100
)"""
    assert result == expected


def test_agg_and_non_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='my_table')
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    expr = expr[expr.b == 'a']
    result = Compiler.to_sql(expr)
    expected = """\
SELECT *
FROM (
  SELECT *
  FROM my_table
  WHERE `a` < 100
) t0
WHERE (`a` = (
  SELECT max(`a`) AS `max`
  FROM my_table
  WHERE `a` < 100
)) AND
      (`b` = 'a')"""
    assert result == expected


def test_agg_filter():
    t = ibis.table([('a', 'int64'), ('b', 'int64')], name='my_table')
    t = t.mutate(b2=t.b * 2)
    t = t[['a', 'b2']]
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max().name('blah')]
    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT *, `b` * 2 AS `b2`
  FROM my_table
),
t1 AS (
  SELECT t0.`a`, t0.`b2`
  FROM t0
  WHERE t0.`a` < 100
)
SELECT t1.*
FROM t1
WHERE t1.`a` = (
  SELECT max(`a`) AS `blah`
  FROM t1
)"""
    assert result == expected


def test_agg_filter_with_alias():
    t = ibis.table([('a', 'int64'), ('b', 'int64')], name='my_table')
    t = t.mutate(b2=t.b * 2)
    t = t[['a', 'b2']]
    filt = t[t.a < 100]
    expr = filt[filt.a.name('A') == filt.a.max().name('blah')]
    result = Compiler.to_sql(expr)
    expected = """\
WITH t0 AS (
  SELECT *, `b` * 2 AS `b2`
  FROM my_table
),
t1 AS (
  SELECT t0.`a`, t0.`b2`
  FROM t0
  WHERE t0.`a` < 100
)
SELECT t1.*
FROM t1
WHERE t1.`a` = (
  SELECT max(`a`) AS `blah`
  FROM t1
)"""
    assert result == expected


def test_table_drop_with_filter():
    left = ibis.table(
        [('a', 'int64'), ('b', 'string'), ('c', 'timestamp')], name='t'
    ).relabel({'c': 'C'})
    left = left.filter(left.C == datetime.datetime(2018, 1, 1))
    left = left.drop(['C'])
    left = left.mutate(the_date=datetime.datetime(2018, 1, 1))

    right = ibis.table([('b', 'string')], name='s')
    joined = left.join(right, left.b == right.b)
    joined = joined[left.a]
    joined = joined.filter(joined.a < 1.0)
    result = Compiler.to_sql(joined)
    expected = """\
SELECT t0.*
FROM (
  SELECT t2.`a`
  FROM (
    SELECT `a`, `b`, '2018-01-01T00:00:00' AS `the_date`
    FROM (
      SELECT *
      FROM (
        SELECT `a`, `b`, `c` AS `C`
        FROM t
      ) t5
      WHERE `C` = '2018-01-01T00:00:00'
    ) t4
  ) t2
    INNER JOIN s t1
      ON t2.`b` = t1.`b`
) t0
WHERE t0.`a` < 1.0"""
    assert result == expected


def test_table_drop_consistency():
    # GH2829
    t = ibis.table(
        [('a', 'int64'), ('b', 'string'), ('c', 'timestamp')], name='t'
    )

    expected = t.projection(["a", "c"])
    result_1 = t.drop(["b"])
    result_2 = t.drop("b")

    assert expected.schema() == result_1.schema()
    assert expected.schema() == result_2.schema()

    assert expected.schema() != t.schema()

    assert "b" not in expected.columns
    assert "a" in result_1.columns
    assert "c" in result_2.columns


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
    result = Compiler.to_sql(expr, params={param: "20140101"})
    expected = """\
SELECT count(`foo`) AS `count`
FROM (
  SELECT `string_col`, sum(`float_col`) AS `foo`
  FROM (
    SELECT `float_col`, `timestamp_col`, `int_col`, `string_col`
    FROM alltypes
    WHERE `timestamp_col` < '20140101'
  ) t1
  GROUP BY 1
) t0"""
    assert result == expected


def test_column_expr_retains_name():
    t = ibis.table(
        [
            ('int_col', 'int32'),
        ],
        'int_col_table',
    )
    named_derived_expr = (t.int_col + 4).name('foo')
    result = Compiler.to_sql(named_derived_expr)
    expected = 'SELECT `int_col` + 4 AS `foo`\nFROM int_col_table'

    assert result == expected


def test_column_expr_default_name():
    t = ibis.table(
        [
            ('int_col', 'int32'),
        ],
        'int_col_table',
    )
    named_derived_expr = t.int_col + 4
    result = Compiler.to_sql(named_derived_expr)
    expected = 'SELECT `int_col` + 4 AS `tmp`\nFROM int_col_table'

    assert result == expected
