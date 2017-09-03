# import unittest

# import pytest

# import ibis
# import ibis.expr.types as ir
# import ibis.expr.api as api

# from ibis import literal as L
# from ibis.expr.datatypes import Category

# from ibis.expr.tests.mocks import MockConnection

# from ibis.clickhouse.compiler import ClickhouseExprTranslator  # noqa: E402
# from ibis.clickhouse.compiler import to_sql  # noqa: E402
# from ibis.clickhouse.compiler import ClickhouseQueryContext  # noqa: E402
# from ibis.sql.tests.test_compiler import ExprTestCases  # noqa: E402
# from ibis.clickhouse.tests.common import ClickhouseE2E  # noqa: E402


# # TODO: move these cases to other test files

# def approx_equal(a, b, eps):
#     return abs(a - b) < eps


# class ExprSQLTest(object):

#     def _check_expr_cases(self, cases, context=None, named=False):
#         for expr, expected in cases:
#             repr(expr)
#             result = self._translate(expr, named=named, context=context)
#             assert result == expected

#     def _translate(self, expr, named=False, context=None):
#         translator = ClickhouseExprTranslator(expr, context=context,
#                                               named=named)
#         return translator.get_result()


# class TestValueExprs(unittest.TestCase, ExprSQLTest):

#     def setUp(self):
#         self.con = MockConnection()
#         self.table = self.con.table('alltypes')

#         self.int_cols = ['a', 'b', 'c', 'd']
#         self.bool_cols = ['h']
#         self.float_cols = ['e', 'f']

#     def _check_literals(self, cases):
#         for value, expected in cases:
#             lit_expr = L(value)
#             result = self._translate(lit_expr)
#             assert result == expected

#     # def test_decimal_builtins(self):
#     #     t = self.con.table('tpch_lineitem')
#     #     col = t.l_extendedprice
#     #     cases = [
#     #         (col.precision(), 'precision(`l_extendedprice`)'),
#     #         (col.scale(), 'scale(`l_extendedprice`)'),
#     #     ]
#     #     self._check_expr_cases(cases)

#     def test_column_ref_table_aliases(self):
#         context = ClickhouseQueryContext()

#         table1 = ibis.table([
#             ('key1', 'string'),
#             ('value1', 'double')
#         ])

#         table2 = ibis.table([
#             ('key2', 'string'),
#             ('value and2', 'double')
#         ])

#         context.set_ref(table1, 't0')
#         context.set_ref(table2, 't1')

#         expr = table1['value1'] - table2['value and2']

#         result = self._translate(expr, context=context)
#         expected = 't0.`value1` - t1.`value and2`'
#         assert result == expected

#     def test_column_ref_quoting(self):
#         schema = [('has a space', 'double')]
#         table = ibis.table(schema)
#         self._translate(table['has a space'], '`has a space`')

#     def test_identifier_quoting(self):
#         schema = [('date', 'double'), ('table', 'string')]
#         table = ibis.table(schema)
#         self._translate(table['date'], '`date`')
#         self._translate(table['table'], '`table`')

#     def test_named_expressions(self):
#         a, b, g = self.table.get_columns(['a', 'b', 'g'])

#         cases = [
#             (g.cast('double').name('g_dub'),
#              'CAST(`g` AS Float64) AS `g_dub`'),
#             (g.name('has a space'), '`g` AS `has a space`'),
#             (((a - b) * a).name('expr'), '(`a` - `b`) * `a` AS `expr`')
#         ]

#         return self._check_expr_cases(cases, named=True)

#     def test_timestamp_deltas(self):
#         units = ['year', 'month', 'week', 'day',
#                  'hour', 'minute', 'second',
#                  'millisecond', 'microsecond']

#         t = self.table.i
#         f = '`i`'

#         cases = []
#         for unit in units:
#             K = 5
#             offset = getattr(ibis, unit)(K)
#             template = '{0}s_add({1}, {2})'

#             cases.append((t + offset, template.format(unit, f, K)))
#             cases.append((t - offset, template.format(unit, f, -K)))

#         self._check_expr_cases(cases)

#     def test_correlated_predicate_subquery(self):
#         t0 = self.table
#         t1 = t0.view()

#         expr = t0.g == t1.g

#         ctx = ClickhouseQueryContext()
#         ctx.make_alias(t0)

#         # Grab alias from parent context
#         subctx = ctx.subcontext()
#         subctx.make_alias(t1)
#         subctx.make_alias(t0)

#         result = self._translate(expr, context=subctx)
#         expected = "t0.`g` = t1.`g`"
#         assert result == expected

#     def test_any_all(self):
#         t = self.table

#         bool_expr = t.f == 0

#         cases = [
#             (bool_expr.any(), 'sum(`f` = 0) > 0'),
#             (-bool_expr.any(), 'sum(`f` = 0) = 0'),
#             (bool_expr.all(), 'sum(`f` = 0) = count(*)'),
#             (-bool_expr.all(), 'sum(`f` = 0) < count(*)'),
#         ]
#         self._check_expr_cases(cases)


# class TestCaseExprs(unittest.TestCase, ExprSQLTest, ExprTestCases):

#     def setUp(self):
#         self.con = MockConnection()
#         self.table = self.con.table('alltypes')


# #     def test_search_case(self):
# #         expr = self._case_search_case()
# #         print(expr)
# #         result = self._translate(expr)
# #         expected = """CASE
# #   WHEN `f` > 0 THEN `d` * 2
# #   WHEN `c` < 0 THEN `a` * 2
# #   ELSE NULL
# # END"""
# #         assert result == expected

# # TODO: Clickhouse doesn't support null literal
# # class TestBucketHistogram(unittest.TestCase, ExprSQLTest):

# #     def setUp(self):
# #         self.con = MockConnection()
# #         self.table = self.con.table('alltypes')

# #     def test_bucket_to_case(self):
# #         buckets = [0, 10, 25, 50]

# #         expr1 = self.table.f.bucket(buckets)
# #         expected1 = """\
# # CASE
# #   WHEN (`f` >= 0) AND (`f` < 10) THEN 0
# #   WHEN (`f` >= 10) AND (`f` < 25) THEN 1
# #   WHEN (`f` >= 25) AND (`f` <= 50) THEN 2
# #   ELSE NULL
# # END"""

# #         expr2 = self.table.f.bucket(buckets, close_extreme=False)
# #         expected2 = """\
# # CASE
# #   WHEN (`f` >= 0) AND (`f` < 10) THEN 0
# #   WHEN (`f` >= 10) AND (`f` < 25) THEN 1
# #   WHEN (`f` >= 25) AND (`f` < 50) THEN 2
# #   ELSE NULL
# # END"""

# #         expr3 = self.table.f.bucket(buckets, closed='right')
# #         expected3 = """\
# # CASE
# #   WHEN (`f` >= 0) AND (`f` <= 10) THEN 0
# #   WHEN (`f` > 10) AND (`f` <= 25) THEN 1
# #   WHEN (`f` > 25) AND (`f` <= 50) THEN 2
# #   ELSE NULL
# # END"""

# #         expr4 = self.table.f.bucket(buckets, closed='right',
# #                                     close_extreme=False)
# #         expected4 = """\
# # CASE
# #   WHEN (`f` > 0) AND (`f` <= 10) THEN 0
# #   WHEN (`f` > 10) AND (`f` <= 25) THEN 1
# #   WHEN (`f` > 25) AND (`f` <= 50) THEN 2
# #   ELSE NULL
# # END"""

# #         expr5 = self.table.f.bucket(buckets, include_under=True)
# #         expected5 = """\
# # CASE
# #   WHEN `f` < 0 THEN 0
# #   WHEN (`f` >= 0) AND (`f` < 10) THEN 1
# #   WHEN (`f` >= 10) AND (`f` < 25) THEN 2
# #   WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
# #   ELSE NULL
# # END"""

# #         expr6 = self.table.f.bucket(buckets,
# #                                     include_under=True,
# #                                     include_over=True)
# #         expected6 = """\
# # CASE
# #   WHEN `f` < 0 THEN 0
# #   WHEN (`f` >= 0) AND (`f` < 10) THEN 1
# #   WHEN (`f` >= 10) AND (`f` < 25) THEN 2
# #   WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
# #   WHEN `f` > 50 THEN 4
# #   ELSE NULL
# # END"""

# #         expr7 = self.table.f.bucket(buckets,
# #                                     close_extreme=False,
# #                                     include_under=True,
# #                                     include_over=True)
# #         expected7 = """\
# # CASE
# #   WHEN `f` < 0 THEN 0
# #   WHEN (`f` >= 0) AND (`f` < 10) THEN 1
# #   WHEN (`f` >= 10) AND (`f` < 25) THEN 2
# #   WHEN (`f` >= 25) AND (`f` < 50) THEN 3
# #   WHEN `f` >= 50 THEN 4
# #   ELSE NULL
# # END"""

# #         expr8 = self.table.f.bucket(buckets, closed='right',
# #                                     close_extreme=False,
# #                                     include_under=True)
# #         expected8 = """\
# # CASE
# #   WHEN `f` <= 0 THEN 0
# #   WHEN (`f` > 0) AND (`f` <= 10) THEN 1
# #   WHEN (`f` > 10) AND (`f` <= 25) THEN 2
# #   WHEN (`f` > 25) AND (`f` <= 50) THEN 3
# #   ELSE NULL
# # END"""

# #         expr9 = self.table.f.bucket([10], closed='right',
# #                                     include_over=True,
# #                                     include_under=True)
# #         expected9 = """\
# # CASE
# #   WHEN `f` <= 10 THEN 0
# #   WHEN `f` > 10 THEN 1
# #   ELSE NULL
# # END"""

# #         expr10 = self.table.f.bucket([10], include_over=True,
# #                                      include_under=True)
# #         expected10 = """\
# # CASE
# #   WHEN `f` < 10 THEN 0
# #   WHEN `f` >= 10 THEN 1
# #   ELSE NULL
# # END"""

# #         cases = [
# #             (expr1, expected1),
# #             (expr2, expected2),
# #             (expr3, expected3),
# #             (expr4, expected4),
# #             (expr5, expected5),
# #             (expr6, expected6),
# #             (expr7, expected7),
# #             (expr8, expected8),
# #             (expr9, expected9),
# #             (expr10, expected10),
# #         ]
# #         self._check_expr_cases(cases)

# #     def test_cast_category_to_int_noop(self):
# #         # Because the bucket result is an integer, no explicit cast is
# #         # necessary
# #         expr = (self.table.f.bucket([10], include_over=True,
# #                                     include_under=True)
# #                 .cast('int32'))

# #         expected = """\
# # CASE
# #   WHEN `f` < 10 THEN 0
# #   WHEN `f` >= 10 THEN 1
# #   ELSE NULL
# # END"""

# #         expr2 = (self.table.f.bucket([10], include_over=True,
# #                                      include_under=True)
# #                  .cast('double'))

# #         expected2 = """\
# # CAST(CASE
# #   WHEN `f` < 10 THEN 0
# #   WHEN `f` >= 10 THEN 1
# #   ELSE NULL
# # END AS Float64)"""

# #         self._check_expr_cases([(expr, expected),
# #                                 (expr2, expected2)])

# #     def test_bucket_assign_labels(self):
# #         buckets = [0, 10, 25, 50]
# #         bucket = self.table.f.bucket(buckets, include_under=True)

# #         size = self.table.group_by(bucket.name('tier')).size()
# #         labelled = size.tier.label(['Under 0', '0 to 10',
# #                                     '10 to 25', '25 to 50'],
# #                                    nulls='error').name('tier2')
# #         expr = size[labelled, size['count']]

# #         expected = """\
# # SELECT
# #   CASE `tier`
# #     WHEN 0 THEN 'Under 0'
# #     WHEN 1 THEN '0 to 10'
# #     WHEN 2 THEN '10 to 25'
# #     WHEN 3 THEN '25 to 50'
# #     ELSE 'error'
# #   END AS `tier2`, `count`
# # FROM (
# #   SELECT
# #     CASE
# #       WHEN `f` < 0 THEN 0
# #       WHEN (`f` >= 0) AND (`f` < 10) THEN 1
# #       WHEN (`f` >= 10) AND (`f` < 25) THEN 2
# #       WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
# #       ELSE NULL
# #     END AS `tier`, count(*) AS `count`
# #   FROM alltypes
# #   GROUP BY 1
# # ) t0"""

# #         result = to_sql(expr)

# #         assert result == expected

# #         self.assertRaises(ValueError, size.tier.label, ['a', 'b', 'c'])
# #         self.assertRaises(ValueError, size.tier.label,
# #                           ['a', 'b', 'c', 'd', 'e'])


# class TestClickhouseExprs(ClickhouseE2E, unittest.TestCase, ExprTestCases):


#     # def test_summary_execute(self):
#     #     table = self.alltypes

#     #     # also test set_column while we're at it
#     #     table = table.set_column('double_col',
#     #                              table.double_col * 2)

#     #     expr = table.double_col.summary()
#     #     repr(expr)

#     #     result = expr.execute()
#     #     assert isinstance(result, pd.DataFrame)

#     #     expr = (table.group_by('string_col')
#     #             .aggregate([table.double_col.summary().prefix('double_'),
#     #                         table.float_col.summary().prefix('float_'),
#     #                         table.string_col.summary().suffix('_string')]))
#     #     result = expr.execute()
#     #     assert isinstance(result, pd.DataFrame)


# #     def test_parse_url(self):
# #         cases = [
# #             (L("https://www.cloudera.com").parse_url('HOST'),
# #              "www.cloudera.com"),

# #             (L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10')
# #              .parse_url('QUERY', 'v'),
# #              'kEuEcWfewf8'),
# #         ]
# #         self.assert_cases_equality(cases)

#     # def test_histogram_value_counts(self):
#     #     t = self.alltypes
#     #     expr = t.double_col.histogram(10).value_counts()
#     #     expr.execute()

# #     def test_decimal_timestamp_builtins(self):
# #         table = self.con.table('tpch_lineitem')

# #         dc = table.l_quantity
# #         ts = table.l_receiptdate.cast('timestamp')

# #         exprs = [
# #             dc % 10,
# #             dc + 5,
# #             dc + dc,
# #             dc / 2,
# #             dc * 2,
# #             dc ** 2,
# #             dc.cast('double'),

# #             api.where(table.l_discount > 0,
# #                       dc * table.l_discount, api.NA),

# #             dc.fillna(0),

# #             ts < (ibis.now() + ibis.month(3)),
# #             ts < (ibis.timestamp('2005-01-01') + ibis.month(3)),

# #             # hashing
# #             dc.hash(),
# #             ts.hash(),

# #             # truncate
# #             ts.truncate('y'),
# #             ts.truncate('q'),
# #             ts.truncate('month'),
# #             ts.truncate('d'),
# #             ts.truncate('w'),
# #             ts.truncate('h'),
# #             ts.truncate('minute'),
# #         ]

# #         timestamp_fields = ['year', 'month', 'day', 'hour', 'minute',
# #                             'second', 'millisecond', 'microsecond',
# #                             'week']
# #         for field in timestamp_fields:
# #             if hasattr(ts, field):
# #                 exprs.append(getattr(ts, field)())

# #             offset = getattr(ibis, field)(2)
# #             exprs.append(ts + offset)
# #             exprs.append(ts - offset)

# #         proj_exprs = [expr.name('e%d' % i)
# #                       for i, expr in enumerate(exprs)]

# #         projection = table[proj_exprs].limit(10)
# #         projection.execute()


#     # def test_analytic_functions(self):
#     #     t = self.alltypes.limit(1000)

#     #     g = t.group_by('string_col').order_by('double_col')
#     #     f = t.float_col

#     #     exprs = [
#     #         # f.lag(),
#     #         # f.lead(),
#     #         # f.rank(),
#     #         # f.dense_rank(),
#     #         # f.percent_rank(),
#     #         # f.ntile(buckets=7),

#     #         # f.first(),
#     #         # f.last(),

#     #         # f.first().over(ibis.window(preceding=10)),
#     #         # f.first().over(ibis.window(following=10)),

#     #         # ibis.row_number(),
#     #         # f.cumsum(),
#     #         # f.cummean(),
#     #         # f.cummin(),
#     #         # f.cummax(),

#     #         # # boolean cumulative reductions
#     #         # (f == 0).cumany(),
#     #         # (f == 0).cumall(),

#     #         f.sum(),
#     #         # f.mean(),
#     #         # f.min(),
#     #         # f.max()
#     #     ]

#     #     proj_exprs = [expr.name('e%d' % i)
#     #                   for i, expr in enumerate(exprs)]

#     #     proj_table = g.mutate(proj_exprs)
#     #     proj_table.execute()
