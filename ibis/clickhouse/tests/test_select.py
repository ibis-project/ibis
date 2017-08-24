# from ibis.clickhouse.compiler import to_sql


# # TODO
# def test_timestamp_extract_field(self):
#     fields = ['year', 'month', 'day', 'hour', 'minute',
#               'second']
#     expected = ['toYear', 'toMonth', 'toDayOfMonth', 'toHour', 'toMinute']

#     cases = [(getattr(self.table.i, field)(), "{0}(`i`)".format(exp))
#              for field, exp in zip(fields, expected)]
#     self._check_expr_cases(cases)

#     # integration with SQL translation
#     expr = self.table[self.table.i.year().name('year'),
#                       self.table.i.month().name('month'),
#                       self.table.i.day().name('day')]

#     result = to_sql(expr)
#     expected = """\
# SELECT toYear(`i`) AS `year`, toMonth(`i`) AS `month`,
#        toDayOfMonth(`i`) AS `day`
# FROM alltypes"""
#     assert result == expected



# def test_isin_notin_in_select(con, alltypes, translate):
#     filtered = alltypes[alltypes.string_col.isin(['foo', 'bar'])]
#     result = to_sql(filtered)
#     expected = """SELECT *
# FROM functional_alltypes
# WHERE `string_col` IN ('foo', 'bar')"""
#     assert result == expected

#     filtered = alltypes[alltypes.string_col.notin(['foo', 'bar'])]
#     result = to_sql(filtered)
#     expected = """SELECT *
# FROM functional_alltypes
# WHERE `string_col` NOT IN ('foo', 'bar')"""
#     assert result == expected


# def test_subquery(alltypes, df):
#     t = alltypes

#     expr = (t.mutate(d=t.double_col.fillna(0))
#             .limit(1000)
#             .group_by('string_col')
#             .size())
#     result = expr.execute().sort_values('string_col').reset_index(drop=True)
#     expected = df.assign(
#         d=df.double_col.fillna(0)
#     ).head(1000).groupby('string_col').string_col.count().reset_index(
#         name='count'
#     ).sort_values('string_col').reset_index(drop=True)
#     tm.assert_frame_equal(
#         result,
#         expected,

#         # Python 2 + pandas inferred type here is 'mixed' because of SQLAlchemy
#         # string type subclasses
#         check_column_type=sys.version_info.major >= 3
#     )


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
