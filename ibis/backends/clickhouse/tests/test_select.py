import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com

driver = pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


@pytest.fixture(scope='module')
def diamonds(con):
    return con.table('diamonds')


@pytest.fixture(scope='module')
def batting(con):
    return con.table('batting')


@pytest.fixture(scope='module')
def awards_players(con):
    return con.table('awards_players')


def test_timestamp_extract_field(con, db, alltypes):
    t = alltypes.timestamp_col
    expr = alltypes[
        t.year().name('year'),
        t.month().name('month'),
        t.day().name('day'),
        t.hour().name('hour'),
        t.minute().name('minute'),
        t.second().name('second'),
    ]

    result = ibis.clickhouse.compile(expr)

    expected = """\
SELECT toYear(`timestamp_col`) AS `year`, toMonth(`timestamp_col`) AS `month`,
       toDayOfMonth(`timestamp_col`) AS `day`,
       toHour(`timestamp_col`) AS `hour`,
       toMinute(`timestamp_col`) AS `minute`,
       toSecond(`timestamp_col`) AS `second`
FROM {0}.`functional_alltypes`"""
    assert result == expected.format(db.name)


def test_isin_notin_in_select(con, db, alltypes, translate):
    values = ['foo', 'bar']
    filtered = alltypes[alltypes.string_col.isin(values)]
    result = ibis.clickhouse.compile(filtered)
    expected = """SELECT *
FROM {}.`functional_alltypes`
WHERE `string_col` IN {}"""
    assert result == expected.format(db.name, tuple(set(values)))

    filtered = alltypes[alltypes.string_col.notin(values)]
    result = ibis.clickhouse.compile(filtered)
    expected = """SELECT *
FROM {}.`functional_alltypes`
WHERE `string_col` NOT IN {}"""
    assert result == expected.format(db.name, tuple(set(values)))


def test_head(alltypes):
    result = alltypes.head().execute()
    expected = alltypes.limit(5).execute()
    tm.assert_frame_equal(result, expected)


def test_limit_offset(alltypes):
    expected = alltypes.execute()

    tm.assert_frame_equal(alltypes.limit(4).execute(), expected.head(4))
    tm.assert_frame_equal(alltypes.limit(8).execute(), expected.head(8))
    tm.assert_frame_equal(
        alltypes.limit(4, offset=4).execute(),
        expected.iloc[4:8].reset_index(drop=True),
    )


def test_subquery(alltypes, df):
    t = alltypes

    expr = t.mutate(d=t.double_col).limit(1000).group_by('string_col').size()
    result = expr.execute()

    result = result.sort_values('string_col').reset_index(drop=True)
    expected = (
        df.assign(d=df.double_col.fillna(0))
        .head(1000)
        .groupby('string_col')
        .string_col.count()
        .reset_index(name='count')
        .sort_values('string_col')
        .reset_index(drop=True)
    )

    result['count'] = result['count'].astype('int64')
    tm.assert_frame_equal(result, expected)


def test_simple_scalar_aggregates(db, alltypes):
    # Things like table.column.{sum, mean, ...}()
    table = alltypes

    expr = table[table.int_col > 0].float_col.sum()

    sql_query = ibis.clickhouse.compile(expr)
    expected = """SELECT sum(`float_col`) AS `sum`
FROM {0}.`functional_alltypes`
WHERE `int_col` > 0"""

    assert sql_query == expected.format(db.name)


# def test_scalar_aggregates_multiple_tables(alltypes):
#     # #740
#     table = ibis.table([('flag', 'string'),
#                         ('value', 'double')],
#                        'tbl')

#     flagged = table[table.flag == '1']
#     unflagged = table[table.flag == '0']

#     expr = flagged.value.mean() / unflagged.value.mean() - 1

#     result = ibis.clickhouse.compile(expr)
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
#     result = ibis.clickhouse.compile(expr)
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
def test_table_column_unbox(db, alltypes):
    m = alltypes.float_col.sum().name('total')
    agged = (
        alltypes[alltypes.int_col > 0].group_by('string_col').aggregate([m])
    )
    expr = agged.string_col

    sql_query = ibis.clickhouse.compile(expr)
    expected = """\
SELECT `string_col`
FROM (
  SELECT `string_col`, sum(`float_col`) AS `total`
  FROM {0}.`functional_alltypes`
  WHERE `int_col` > 0
  GROUP BY `string_col`
) t0"""

    assert sql_query == expected.format(db.name)


def test_complex_array_expr_projection(db, alltypes):
    # May require finding the base table and forming a projection.
    expr = alltypes.group_by('string_col').aggregate(
        [alltypes.count().name('count')]
    )
    expr2 = expr.string_col.cast('double')

    query = ibis.clickhouse.compile(expr2)
    expected = """SELECT CAST(`string_col` AS Float64) AS `tmp`
FROM (
  SELECT `string_col`, count(*) AS `count`
  FROM {0}.`functional_alltypes`
  GROUP BY `string_col`
) t0"""
    assert query == expected.format(db.name)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.now(), 'SELECT now() AS `tmp`'),
        (ibis.literal(1) + ibis.literal(2), 'SELECT 1 + 2 AS `tmp`'),
    ],
)
def test_scalar_exprs_no_table_refs(expr, expected):
    assert ibis.clickhouse.compile(expr) == expected


def test_expr_list_no_table_refs():
    exlist = ibis.api.expr_list(
        [
            ibis.literal(1).name('a'),
            ibis.now().name('b'),
            ibis.literal(2).log().name('c'),
        ]
    )
    result = ibis.clickhouse.compile(exlist)
    expected = """\
SELECT 1 AS `a`, now() AS `b`, log(2) AS `c`"""
    assert result == expected


# TODO: use alltypes
def test_isnull_case_expr_rewrite_failure(db, alltypes):
    # #172, case expression that was not being properly converted into an
    # aggregation
    reduction = alltypes.string_col.isnull().ifelse(1, 0).sum()

    result = ibis.clickhouse.compile(reduction)
    expected = """\
SELECT sum(CASE WHEN isNull(`string_col`) THEN 1 ELSE 0 END) AS `sum`
FROM {0}.`functional_alltypes`"""
    assert result == expected.format(db.name)


# def test_nameless_table(con):
#     # Generate a unique table name when we haven't passed on
#     nameless = con.table([('key', 'string')])
#     assert ibis.clickhouse.compile(nameless) == 'SELECT *\nFROM {}'.format(
#         nameless.op().name
#     )

#     with_name = con.table([('key', 'string')], name='baz')
#     result = ibis.clickhouse.compile(with_name)
#     assert result == 'SELECT *\nFROM baz'


def test_physical_table_reference_translate(db, alltypes):
    # If an expression's table leaves all reference database tables, verify
    # we translate correctlys
    sql_string = ibis.clickhouse.compile(alltypes)
    expected = "SELECT *\nFROM {0}.`functional_alltypes`"
    assert sql_string == expected.format(db.name)


def test_non_equijoin(alltypes):
    t = alltypes.limit(100)
    t2 = t.view()
    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

    with pytest.raises(com.TranslationError):
        expr.execute()


def test_join_with_predicate_on_different_columns_raises(
    con, batting, awards_players
):
    t1 = batting
    t2 = awards_players

    pred = t1['playerID'] == t2['awardID']
    expr = t1.inner_join(t2, [pred])[[t1]]

    with pytest.raises(com.TranslationError):
        ibis.clickhouse.compile(expr)


@pytest.mark.parametrize(
    ('join_type', 'join_clause'),
    [
        ('any_inner_join', 'ANY INNER JOIN'),
        ('inner_join', 'ALL INNER JOIN'),
        ('any_left_join', 'ANY LEFT JOIN'),
        ('left_join', 'ALL LEFT JOIN'),
    ],
)
def test_simple_joins(
    con, db, batting, awards_players, join_type, join_clause
):
    t1, t2 = batting, awards_players
    expr = getattr(t1, join_type)(t2, ['playerID'])[[t1]]

    expected = """SELECT t0.*
FROM {0}.`batting` t0
  {join_clause} {0}.`awards_players` t1
    USING `playerID`""".format(
        db.name, join_clause=join_clause
    )

    assert ibis.clickhouse.compile(expr) == expected
    assert len(con.execute(expr))


def test_self_reference_simple(con, db, alltypes):
    expr = alltypes.view()
    result_sql = ibis.clickhouse.compile(expr)
    expected_sql = "SELECT *\nFROM {0}.`functional_alltypes`"
    assert result_sql == expected_sql.format(db.name)
    assert len(con.execute(expr))


def test_join_self_reference(con, db, alltypes):
    t1 = alltypes
    t2 = t1.view()
    expr = t1.any_inner_join(t2, ['id'])[[t1]]

    result_sql = ibis.clickhouse.compile(expr)
    expected_sql = """SELECT t0.*
FROM {0}.`functional_alltypes` t0
  ANY INNER JOIN {0}.`functional_alltypes` t1
    USING `id`"""
    assert result_sql == expected_sql.format(db.name)
    assert len(con.execute(expr))


def test_where_simple_comparisons(con, db, alltypes):
    t1 = alltypes
    expr = t1.filter([t1.float_col > 0, t1.int_col < t1.float_col * 2])

    result = ibis.clickhouse.compile(expr)
    expected = """SELECT *
FROM {0}.`functional_alltypes`
WHERE (`float_col` > 0) AND
      (`int_col` < (`float_col` * 2))"""
    assert result == expected.format(db.name)
    assert len(con.execute(expr))


def test_where_with_between(con, db, alltypes):
    t = alltypes

    expr = t.filter([t.int_col > 0, t.float_col.between(0, 1)])
    result = ibis.clickhouse.compile(expr)
    expected = """SELECT *
FROM {0}.`functional_alltypes`
WHERE (`int_col` > 0) AND
      (`float_col` BETWEEN 0 AND 1)"""
    assert result == expected.format(db.name)
    con.execute(expr)


def test_where_use_if(con, alltypes, translate):
    expr = ibis.where(
        alltypes.float_col > 0, alltypes.int_col, alltypes.bigint_col
    )

    result = translate(expr)
    expected = "if(`float_col` > 0, `int_col`, `bigint_col`)"
    assert result == expected
    con.execute(expr)


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


# def test_unions_with_ctes(con, alltypes):
#     t = alltypes

#     expr1 = (t.group_by(['tinyint_col', 'string_col'])
#              .aggregate(t.double_col.sum().name('metric')))
#     expr2 = expr1.view()

#     join1 = (expr1.join(expr2, expr1.string_col == expr2.string_col)
#              [[expr1]])
#     join2 = join1.view()

#     expr = join1.union(join2)
#     con.execute(expr)
#     con.explain(expr)


@pytest.mark.xfail(
    raises=com.RelationError, reason='Expression equality is broken'
)
def test_filter_predicates(diamonds):
    predicates = [
        lambda x: x.color.lower().like('%de%'),
        # lambda x: x.color.lower().contains('de'),
        lambda x: x.color.lower().rlike('.*ge.*'),
    ]

    expr = diamonds
    for pred in predicates:
        expr = expr[pred(expr)].projection([expr])

    expr.execute()


def test_where_with_timestamp():
    t = ibis.table(
        [('uuid', 'string'), ('ts', 'timestamp'), ('search_level', 'int64')],
        name='t',
    )
    expr = t.group_by(t.uuid).aggregate(
        min_date=t.ts.min(where=t.search_level == 1)
    )
    result = ibis.clickhouse.compile(expr)
    expected = """\
SELECT `uuid`, minIf(`ts`, `search_level` = 1) AS `min_date`
FROM t
GROUP BY `uuid`"""
    assert result == expected


def test_timestamp_scalar_in_filter(alltypes, translate):
    table = alltypes

    expr = table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp('2010-01-01') + ibis.interval(weeks=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()
    expr.execute()


def test_named_from_filter_groupby():
    t = ibis.table([('key', 'string'), ('value', 'double')], name='t0')
    gb = t.filter(t.value == 42).groupby(t.key)
    sum_expr = lambda t: (t.value + 1 + 2 + 3).sum()  # noqa: E731
    expr = gb.aggregate(abc=sum_expr)
    expected = """\
SELECT `key`, sum(((`value` + 1) + 2) + 3) AS `abc`
FROM t0
WHERE `value` = 42
GROUP BY `key`"""
    assert ibis.clickhouse.compile(expr) == expected

    expr = gb.aggregate(foo=sum_expr)
    expected = """\
SELECT `key`, sum(((`value` + 1) + 2) + 3) AS `foo`
FROM t0
WHERE `value` = 42
GROUP BY `key`"""
    assert ibis.clickhouse.compile(expr) == expected


def test_join_with_external_table_errors(con, alltypes, df):
    external_table = ibis.table(
        [('a', 'string'), ('b', 'int64'), ('c', 'string')], name='external'
    )

    alltypes = alltypes.mutate(b=alltypes.tinyint_col)
    expr = alltypes.inner_join(external_table, ['b'])[
        external_table.a, external_table.c, alltypes.id
    ]

    with pytest.raises(driver.errors.ServerException):
        expr.execute()

    with pytest.raises(TypeError):
        expr.execute(external_tables={'external': []})


def test_join_with_external_table(con, alltypes, df):
    external_df = pd.DataFrame(
        [('alpha', 1, 'first'), ('beta', 2, 'second'), ('gamma', 3, 'third')],
        columns=['a', 'b', 'c'],
    )
    external_df['b'] = external_df['b'].astype('int8')

    external_table = ibis.table(
        [('a', 'string'), ('b', 'int64'), ('c', 'string')], name='external'
    )

    alltypes = alltypes.mutate(b=alltypes.tinyint_col)
    expr = alltypes.inner_join(external_table, ['b'])[
        external_table.a, external_table.c, alltypes.id
    ]

    result = expr.execute(external_tables={'external': external_df})
    expected = df.assign(b=df.tinyint_col).merge(external_df, on='b')[
        ['a', 'c', 'id']
    ]

    result = result.sort_values('id').reset_index(drop=True)
    expected = expected.sort_values('id').reset_index(drop=True)

    tm.assert_frame_equal(result, expected, check_column_type=False)
