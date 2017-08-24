import sys
import pandas.util.testing as tm
from ibis.clickhouse.compiler import to_sql


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
