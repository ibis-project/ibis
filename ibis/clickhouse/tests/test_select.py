import sys
import pytest
import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.common as com
from ibis.clickhouse.compiler import to_sql
from ibis.expr.tests.mocks import MockConnection


@pytest.fixture(scope='module')
def table():
    con = MockConnection()
    return con.table('alltypes')


@pytest.fixture(scope='module')
def batting(con):
    return con.table('batting')


@pytest.fixture(scope='module')
def awards_players(con):
    return con.table('awards_players')


def test_timestamp_extract_field(con, alltypes):
    t = alltypes.timestamp_col
    expr = alltypes[t.year().name('year'),
                    t.month().name('month'),
                    t.day().name('day'),
                    t.hour().name('hour'),
                    t.minute().name('minute'),
                    t.second().name('second')]

    result = to_sql(expr)

    expected = """\
SELECT toYear(`timestamp_col`) AS `year`, toMonth(`timestamp_col`) AS `month`,
       toDayOfMonth(`timestamp_col`) AS `day`,
       toHour(`timestamp_col`) AS `hour`,
       toMinute(`timestamp_col`) AS `minute`,
       toSecond(`timestamp_col`) AS `second`
FROM ibis_testing.`functional_alltypes`"""
    assert result == expected


def test_isin_notin_in_select(con, alltypes, translate):
    filtered = alltypes[alltypes.string_col.isin(['foo', 'bar'])]
    result = to_sql(filtered)
    expected = """SELECT *
FROM ibis_testing.`functional_alltypes`
WHERE `string_col` IN ('foo', 'bar')"""
    assert result == expected

    filtered = alltypes[alltypes.string_col.notin(['foo', 'bar'])]
    result = to_sql(filtered)
    expected = """SELECT *
FROM ibis_testing.`functional_alltypes`
WHERE `string_col` NOT IN ('foo', 'bar')"""
    assert result == expected


def test_subquery(alltypes, df):
    t = alltypes

    expr = (t.mutate(d=t.double_col)
            .limit(1000)
            .group_by('string_col')
            .size())
    result = expr.execute()

    result = result.sort_values('string_col').reset_index(drop=True)
    expected = (df.assign(d=df.double_col.fillna(0))
                  .head(1000)
                  .groupby('string_col')
                  .string_col.count()
                  .reset_index(name='count')
                  .sort_values('string_col')
                  .reset_index(drop=True))

    result['count'] = result['count'].astype('int64')

    check_column_type = sys.version_info.major >= 3
    tm.assert_frame_equal(result, expected,
                          check_column_type=check_column_type)


def test_simple_scalar_aggregates(alltypes):
    # Things like table.column.{sum, mean, ...}()
    table = alltypes

    expr = table[table.int_col > 0].float_col.sum()

    sql_query = to_sql(expr)
    expected = """SELECT sum(`float_col`) AS `sum`
FROM ibis_testing.`functional_alltypes`
WHERE `int_col` > 0"""

    assert sql_query == expected


# def test_scalar_aggregates_multiple_tables(alltypes):
#     # #740
#     table = ibis.table([('flag', 'string'),
#                         ('value', 'double')],
#                        'tbl')

#     flagged = table[table.flag == '1']
#     unflagged = table[table.flag == '0']

#     expr = flagged.value.mean() / unflagged.value.mean() - 1

#     result = to_sql(expr)
#     expected = """\
# SELECT (t0.`mean` / t1.`mean`) - 1 AS `tmp`
# FROM (
#   SELECT avg(`value`) AS `mean`
#   FROM tbl
#   WHERE `flag` = '1'
# ) t0
#   CROSS JOIN (
#     SELECT avg(`value`) AS `mean`
#     FROM tbl
#     WHERE `flag` = '0'
#   ) t1"""
#     assert result == expected

#     fv = flagged.value
#     uv = unflagged.value

#     expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())
#     result = to_sql(expr)
#     expected = """\
# SELECT t0.`tmp` - t1.`tmp` AS `tmp`
# FROM (
#   SELECT avg(`value`) / sum(`value`) AS `tmp`
#   FROM tbl
#   WHERE `flag` = '1'
# ) t0
#   CROSS JOIN (
#     SELECT avg(`value`) / sum(`value`) AS `tmp`
#     FROM tbl
#     WHERE `flag` = '0'
#   ) t1"""
#     assert result == expected


# TODO use alltypes
def test_table_column_unbox(table):
    m = table.f.sum().name('total')
    agged = table[table.c > 0].group_by('g').aggregate([m])
    expr = agged.g

    sql_query = to_sql(expr)
    expected = """\
SELECT `g`
FROM (
  SELECT `g`, sum(`f`) AS `total`
  FROM alltypes
  WHERE `c` > 0
  GROUP BY `g`
) t0"""

    assert sql_query == expected

    # Maybe the result handler should act on the cursor. Not sure.
    # handler = query.result_handler
    # output = DataFrame({'g': ['foo', 'bar', 'baz']})
    # assert (handler(output) == output['g']).all()


# TODO: use alltypes
def test_complex_array_expr_projection(table):
    # May require finding the base table and forming a projection.
    expr = (table.group_by('g')
            .aggregate([table.count().name('count')]))
    expr2 = expr.g.cast('double')

    query = to_sql(expr2)
    expected = """SELECT CAST(`g` AS Float64) AS `tmp`
FROM (
  SELECT `g`, count(*) AS `count`
  FROM alltypes
  GROUP BY `g`
) t0"""
    assert query == expected


@pytest.mark.parametrize(('expr', 'expected'), [
    (ibis.now(), 'SELECT now() AS `tmp`'),
    (ibis.literal(1) + ibis.literal(2), 'SELECT 1 + 2 AS `tmp`')
])
def test_scalar_exprs_no_table_refs(expr, expected):
    assert to_sql(expr) == expected


def test_expr_list_no_table_refs():
    exlist = ibis.api.expr_list([ibis.literal(1).name('a'),
                                 ibis.now().name('b'),
                                 ibis.literal(2).log().name('c')])
    result = to_sql(exlist)
    expected = """\
SELECT 1 AS `a`, now() AS `b`, log(2) AS `c`"""
    assert result == expected


# TODO: use alltypes
def test_isnull_case_expr_rewrite_failure(table):
    # #172, case expression that was not being properly converted into an
    # aggregation
    reduction = table.g.isnull().ifelse(1, 0).sum()

    result = to_sql(reduction)
    expected = """\
SELECT sum(CASE WHEN isNull(`g`) THEN 1 ELSE 0 END) AS `sum`
FROM alltypes"""
    assert result == expected


# def test_nameless_table(con):
#     # Generate a unique table name when we haven't passed on
#     nameless = con.table([('key', 'string')])
#     assert to_sql(nameless) == 'SELECT *\nFROM {}'.format(
#         nameless.op().name
#     )

#     with_name = con.table([('key', 'string')], name='baz')
#     result = to_sql(with_name)
#     assert result == 'SELECT *\nFROM baz'


def test_physical_table_reference_translate(alltypes):
    # If an expression's table leaves all reference database tables, verify
    # we translate correctlys
    sql_string = to_sql(alltypes)
    expected = "SELECT *\nFROM ibis_testing.`functional_alltypes`"
    assert sql_string == expected


def test_join_with_predicate_raises(con, batting, awards_players):
    t1 = batting
    t2 = awards_players

    pred = t1['playerID'] == t2['playerID']
    expr = t1.inner_join(t2, [pred])[[t1]]

    with pytest.raises(com.TranslationError):
        to_sql(expr)


def test_simple_joins(con, batting, awards_players):
    t1, t2 = batting, awards_players

    expr = t1.any_inner_join(t2, ['playerID'])[[t1]]
    con.execute(expr)

    # cases = [
#         (t1.inner_join(t2, [pred])[[t1]],
#          """SELECT t0.*
# FROM star1 t0
#   INNER JOIN star2 t1
#     ON t0.`foo_id` = t1.`foo_id`"""),
#         (t1.left_join(t2, [pred])[[t1]],
#          """SELECT t0.*
# FROM star1 t0
#   LEFT OUTER JOIN star2 t1
#     ON t0.`foo_id` = t1.`foo_id`"""),
#         (t1.outer_join(t2, [pred])[[t1]],
#          """SELECT t0.*
# FROM star1 t0
#   FULL OUTER JOIN star2 t1
#     ON t0.`foo_id` = t1.`foo_id`"""),
#         # multiple predicates
#         (t1.inner_join(t2, [pred, pred2])[[t1]],
#          """SELECT t0.*
# FROM star1 t0
#   INNER JOIN star2 t1
#     ON t0.`foo_id` = t1.`foo_id` AND
#        t0.`bar_id` = t1.`foo_id`"""),
#     ]

#     for expr, expected_sql in cases:
#         result_sql = to_sql(expr)
#         assert result_sql == expected_sql


# def test_semi_join(t, s):
#     t_a, s_a = t.op().sqla_table.alias('t0'), s.op().sqla_table.alias('t1')
#     expr = t.semi_join(s, t.id == s.id)
#     result = expr.compile().compile(compile_kwargs=dict(literal_binds=True))
#     base = sa.select([t_a.c.id, t_a.c.name]).where(
#         sa.exists(sa.select([1]).where(t_a.c.id == s_a.c.id))
#     )
#     expected = sa.select([base.c.id, base.c.name])
#     assert str(result) == str(expected)


# def test_anti_join(t, s):
#     t_a, s_a = t.op().sqla_table.alias('t0'), s.op().sqla_table.alias('t1')
#     expr = t.anti_join(s, t.id == s.id)
#     result = expr.compile().compile(compile_kwargs=dict(literal_binds=True))
#     expected = sa.select([sa.column('id'), sa.column('name')]).select_from(
#         sa.select([t_a.c.id, t_a.c.name]).where(
#             ~sa.exists(sa.select([1]).where(t_a.c.id == s_a.c.id))
#         )
#     )
#     assert str(result) == str(expected)


# def test_union(alltypes):
#     t = alltypes

#     expr = (t.group_by('string_col')
#             .aggregate(t.double_col.sum().name('foo'))
#             .sort_by('string_col'))

#     t1 = expr.limit(4)
#     t2 = expr.limit(4, offset=4)
#     t3 = expr.limit(8)

#     result = t1.union(t2).execute()
#     expected = t3.execute()
#     tm.assert_frame_equal(result, expected)
