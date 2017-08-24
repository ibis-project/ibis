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
