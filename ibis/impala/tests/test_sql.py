import operator
import unittest

import pytest

import ibis

pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala.compiler import to_sql  # noqa: E402

pytestmark = pytest.mark.impala


class TestImpalaSQL(unittest.TestCase):

    def test_relabel_projection(self):
        # GH #551
        types = ['int32', 'string', 'double']
        table = ibis.table(zip(['foo', 'bar', 'baz'], types), name='table')
        relabeled = table.relabel({'foo': 'one', 'baz': 'three'})

        result = to_sql(relabeled)
        expected = """\
SELECT `foo` AS `one`, `bar`, `baz` AS `three`
FROM `table`"""
        assert result == expected


def test_nested_join_base():
    t = ibis.table([('uuid', 'string'), ('ts', 'timestamp')], name='t')
    counts = t.group_by('uuid').size()
    max_counts = counts.group_by('uuid').aggregate(
        max_count=lambda x: x['count'].max()
    )
    result = max_counts.left_join(counts, 'uuid').projection([counts])
    compiled_result = to_sql(result)

    expected = """\
WITH t0 AS (
  SELECT `uuid`, count(*) AS `count`
  FROM t
  GROUP BY 1
)
SELECT t0.*
FROM (
  SELECT `uuid`, max(`count`) AS `max_count`
  FROM t0
  GROUP BY 1
) t1
  LEFT OUTER JOIN t0
    ON t1.`uuid` = t0.`uuid`"""
    assert compiled_result == expected


def test_nested_joins_single_cte():
    t = ibis.table([('uuid', 'string'), ('ts', 'timestamp')], name='t')

    counts = t.group_by('uuid').size()

    last_visit = t.group_by('uuid').aggregate(last_visit=t.ts.max())

    max_counts = counts.group_by('uuid').aggregate(
        max_count=counts['count'].max()
    )

    main_kw = max_counts.left_join(
        counts,
        ['uuid', max_counts.max_count == counts['count']]
    ).projection([counts])

    result = main_kw.left_join(last_visit, 'uuid').projection([
        main_kw,
        last_visit.last_visit,
    ])

    expected = """\
WITH t0 AS (
  SELECT `uuid`, count(*) AS `count`
  FROM t
  GROUP BY 1
)
SELECT t1.*, t2.`last_visit`
FROM (
  SELECT t0.*
  FROM (
    SELECT `uuid`, max(`count`) AS `max_count`
    FROM t0
    GROUP BY 1
  ) t3
    LEFT OUTER JOIN t0
      ON (t3.`uuid` = t0.`uuid`) AND
         (t3.`max_count` = t0.`count`)
) t1
  LEFT OUTER JOIN (
    SELECT `uuid`, max(`ts`) AS `last_visit`
    FROM t
    GROUP BY 1
  ) t2
    ON t1.`uuid` = t2.`uuid`"""
    compiled_result = to_sql(result)
    assert compiled_result == expected


def test_nested_join_multiple_ctes():
    ratings = ibis.table(
        [
            ('userid', 'int64'),
            ('movieid', 'int64'),
            ('rating', 'int8'),
            ('timestamp', 'string'),
        ],
        name='ratings'
    )
    movies = ibis.table(
        [
            ('movieid', 'int64'),
            ('title', 'string'),
        ],
        name='movies'
    )

    expr = ratings.timestamp.cast('timestamp')
    ratings2 = ratings['userid', 'movieid', 'rating', expr.name('datetime')]
    joined2 = ratings2.join(movies, ['movieid'])[ratings2, movies['title']]
    joined3 = joined2.filter([
        joined2.userid == 118205,
        joined2.datetime.year() > 2001
    ])
    top_user_old_movie_ids = joined3.filter([
        joined3.userid == 118205,
        joined3.datetime.year() < 2009
    ])[joined3.movieid]
    cond = joined3.movieid.isin(top_user_old_movie_ids.movieid)
    result = joined3[cond]

    expected = """\
WITH t0 AS (
  SELECT `userid`, `movieid`, `rating`,
         CAST(`timestamp` AS timestamp) AS `datetime`
  FROM ratings
),
t1 AS (
  SELECT t0.*, t5.`title`
  FROM t0
    INNER JOIN movies t5
      ON t0.`movieid` = t5.`movieid`
),
t2 AS (
  SELECT t1.*
  FROM t1
  WHERE (t1.`userid` = 118205) AND
        (extract(t1.`datetime`, 'year') > 2001)
)
SELECT t2.*
FROM t2
WHERE t2.`movieid` IN (
  SELECT `movieid`
  FROM (
    SELECT t1.*
    FROM t1
    WHERE (t1.`userid` = 118205) AND
          (extract(t1.`datetime`, 'year') > 2001) AND
          (t1.`userid` = 118205) AND
          (extract(t1.`datetime`, 'year') < 2009)
  ) t4
)"""
    compiled_result = to_sql(result)
    assert compiled_result == expected


def test_logically_negate_complex_boolean_expr():
    t = ibis.table(
        [
            ('a', 'string'),
            ('b', 'double'),
            ('c', 'int64'),
            ('d', 'string'),
        ],
        name='t'
    )

    def f(t):
        return t.a.isin(['foo']) & t.c.notnull()

    expr = f(t)
    result = to_sql(~expr)
    expected = """\
SELECT NOT (`a` IN ('foo') AND (`c` IS NOT NULL)) AS `tmp`
FROM t"""
    assert result == expected


def test_join_with_nested_or_condition():
    t1 = ibis.table([('a', 'string'),
                     ('b', 'string')], 't')
    t2 = t1.view()

    joined = t1.join(t2, [t1.a == t2.a, (t1.a != t2.b) | (t1.b != t2.a)])
    expr = joined[t1]

    expected = """\
SELECT t0.*
FROM t t0
  INNER JOIN t t1
    ON (t0.`a` = t1.`a`) AND
       ((t0.`a` != t1.`b`) OR (t0.`b` != t1.`a`))"""
    assert to_sql(expr) == expected


def test_join_with_nested_xor_condition():
    t1 = ibis.table([('a', 'string'),
                     ('b', 'string')], 't')
    t2 = t1.view()

    joined = t1.join(t2, [t1.a == t2.a, (t1.a != t2.b) ^ (t1.b != t2.a)])
    expr = joined[t1]

    expected = """\
SELECT t0.*
FROM t t0
  INNER JOIN t t1
    ON (t0.`a` = t1.`a`) AND
       (((t0.`a` != t1.`b`) OR (t0.`b` != t1.`a`)) AND NOT ((t0.`a` != t1.`b`) AND (t0.`b` != t1.`a`)))"""  # noqa: E501
    assert to_sql(expr) == expected


@pytest.mark.parametrize(
    ('method', 'sql'),
    [
        ('isnull', 'IS'),
        ('notnull', 'IS NOT'),
    ]
)
def test_is_parens(method, sql):
    t = ibis.table([('a', 'string'), ('b', 'string')], 'table')
    func = operator.methodcaller(method)
    expr = t[func(t.a) == func(t.b)]

    result = to_sql(expr)
    expected = """\
SELECT *
FROM `table`
WHERE (`a` {sql} NULL) = (`b` {sql} NULL)""".format(sql=sql)
    assert result == expected


def test_is_parens_identical_to():
    t = ibis.table([('a', 'string'), ('b', 'string')], 'table')
    expr = t[t.a.identical_to(None) == t.b.identical_to(None)]

    result = to_sql(expr)
    expected = """\
SELECT *
FROM `table`
WHERE (`a` IS NOT DISTINCT FROM NULL) = (`b` IS NOT DISTINCT FROM NULL)"""
    assert result == expected


def test_join_aliasing():
    test = ibis.table(
        [
            ('a', 'int64'),
            ('b', 'int64'),
            ('c', 'int64'),
        ],
        name='test_table'
    )
    test = test.mutate(d=test.a + 20)
    test2 = test[test.d, test.c]
    idx = (test2.d / 15).cast('int64').name('idx')
    test3 = (
        test2.groupby([test2.d, idx, test2.c])
             .aggregate(row_count=test2.count())
    )
    test3_totals = test3.groupby(test3.d).aggregate(
        total=test3.row_count.sum())
    test4 = test3.join(
        test3_totals, test3.d == test3_totals.d)[test3, test3_totals.total]
    test5 = test4[test4.row_count < test4.total / 2]
    agg = test.groupby([test.d, test.b]).aggregate(
        count=test.count(), unique=test.c.nunique()).view()
    joined = agg.join(test5, agg.d == test5.d)[agg, test5.total]
    result = joined
    result = to_sql(result)
    expected = """\
WITH t0 AS (
  SELECT *, `a` + 20 AS `d`
  FROM test_table
),
t1 AS (
  SELECT `d`, `c`
  FROM t0
),
t2 AS (
  SELECT `d`, CAST(`d` / 15 AS bigint) AS `idx`, `c`, count(*) AS `row_count`
  FROM t1
  GROUP BY 1, 2, 3
)
SELECT t3.*, t4.`total`
FROM (
  SELECT `d`, `b`, count(*) AS `count`, count(DISTINCT `c`) AS `unique`
  FROM t0
  GROUP BY 1, 2
) t3
  INNER JOIN (
    SELECT t5.*
    FROM (
      SELECT t2.*, t8.`total`
      FROM t2
        INNER JOIN (
          SELECT `d`, sum(`row_count`) AS `total`
          FROM t2
          GROUP BY 1
        ) t8
          ON t2.`d` = t8.`d`
    ) t5
    WHERE t5.`row_count` < (t5.`total` / 2)
  ) t4
    ON t3.`d` = t4.`d`"""
    assert result == expected


def test_multiple_filters():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='t0')
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    result = to_sql(expr)
    expected = """\
SELECT *
FROM (
  SELECT *
  FROM t0
  WHERE `a` < 100
) t0
WHERE `a` = (
  SELECT max(`a`) AS `max`
  FROM t0
  WHERE `a` < 100
)"""
    assert result == expected


def test_multiple_filters2():
    t = ibis.table([('a', 'int64'), ('b', 'string')], name='t0')
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    expr = expr[expr.b == 'a']
    result = to_sql(expr)
    expected = """\
SELECT *
FROM (
  SELECT *
  FROM t0
  WHERE `a` < 100
) t0
WHERE (`a` = (
  SELECT max(`a`) AS `max`
  FROM t0
  WHERE `a` < 100
)) AND
      (`b` = 'a')"""
    assert result == expected
