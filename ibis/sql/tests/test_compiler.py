# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ibis

from ibis.impala.compiler import build_ast, to_sql

from ibis import impala

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest
import ibis.common as com

import ibis.expr.api as api
import ibis.expr.operations as ops


class TestASTBuilder(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_ast_with_projection_join_filter(self):
        table = self.con.table('test1')
        table2 = self.con.table('test2')

        filter_pred = table['f'] > 0

        table3 = table[filter_pred]

        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])
        result = joined[[table3, table2['value']]]

        ast = build_ast(result)
        stmt = ast.queries[0]

        def foo():
            table3 = table[filter_pred]
            joined = table2.inner_join(table3, [join_pred])
            result = joined[[table3, table2['value']]]
            return result

        assert len(stmt.select_set) == 2

        # #790, make sure the filter stays put
        assert len(stmt.where) == 0

        # Check that the joined tables are not altered
        tbl = stmt.table_set
        tbl_node = tbl.op()
        assert isinstance(tbl_node, ops.InnerJoin)
        assert tbl_node.left is table2
        assert tbl_node.right is table3

    def test_ast_with_aggregation_join_filter(self):
        table = self.con.table('test1')
        table2 = self.con.table('test2')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]
        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])

        met1 = (table3['f'] - table2['value']).mean().name('foo')
        result = joined.aggregate([met1, table3['f'].sum().name('bar')],
                                  by=[table3['g'], table2['key']])

        ast = build_ast(result)
        stmt = ast.queries[0]

        # #790, this behavior was different before
        ex_pred = [table3['g'] == table2['key']]
        expected_table_set = \
            table2.inner_join(table3, ex_pred)
        assert stmt.table_set.equals(expected_table_set)

        # Check various exprs
        ex_metrics = [(table3['f'] - table2['value']).mean().name('foo'),
                      table3['f'].sum().name('bar')]
        ex_by = [table3['g'], table2['key']]
        for res, ex in zip(stmt.select_set, ex_by + ex_metrics):
            assert res.equals(ex)

        for res, ex in zip(stmt.group_by, ex_by):
            assert stmt.select_set[res].equals(ex)

        # The filter is in the joined subtable
        assert len(stmt.where) == 0


class TestNonTabularResults(unittest.TestCase):

    """

    """

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_simple_scalar_aggregates(self):
        from pandas import DataFrame

        # Things like table.column.{sum, mean, ...}()
        table = self.con.table('alltypes')

        expr = table[table.c > 0].f.sum()

        ast = build_ast(expr)
        query = ast.queries[0]

        sql_query = query.compile()
        expected = """SELECT sum(`f`) AS `sum`
FROM alltypes
WHERE `c` > 0"""

        assert sql_query == expected

        # Maybe the result handler should act on the cursor. Not sure.
        handler = query.result_handler
        output = DataFrame({'sum': [5]})
        assert handler(output) == 5

    def test_scalar_aggregates_multiple_tables(self):
        # #740
        table = ibis.table([('flag', 'string'),
                            ('value', 'double')],
                           'tbl')

        flagged = table[table.flag == '1']
        unflagged = table[table.flag == '0']

        expr = flagged.value.mean() / unflagged.value.mean() - 1

        result = to_sql(expr)
        expected = """\
SELECT (t0.`mean` / t1.`mean`) - 1 AS `tmp`
FROM (
  SELECT avg(`value`) AS `mean`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) AS `mean`
    FROM tbl
    WHERE `flag` = '0'
  ) t1"""
        assert result == expected

        fv = flagged.value
        uv = unflagged.value

        expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())
        result = to_sql(expr)
        expected = """\
SELECT t0.`tmp` - t1.`tmp` AS `tmp`
FROM (
  SELECT avg(`value`) / sum(`value`) AS `tmp`
  FROM tbl
  WHERE `flag` = '1'
) t0
  CROSS JOIN (
    SELECT avg(`value`) / sum(`value`) AS `tmp`
    FROM tbl
    WHERE `flag` = '0'
  ) t1"""
        assert result == expected

    def test_table_column_unbox(self):
        from pandas import DataFrame

        table = self.table
        m = table.f.sum().name('total')
        agged = table[table.c > 0].group_by('g').aggregate([m])
        expr = agged.g

        ast = build_ast(expr)
        query = ast.queries[0]

        sql_query = query.compile()
        expected = """\
SELECT `g`
FROM (
  SELECT `g`, sum(`f`) AS `total`
  FROM alltypes
  WHERE `c` > 0
  GROUP BY 1
) t0"""

        assert sql_query == expected

        # Maybe the result handler should act on the cursor. Not sure.
        handler = query.result_handler
        output = DataFrame({'g': ['foo', 'bar', 'baz']})
        assert (handler(output) == output['g']).all()

    def test_complex_array_expr_projection(self):
        # May require finding the base table and forming a projection.
        expr = (self.table.group_by('g')
                .aggregate([self.table.count().name('count')]))
        expr2 = expr.g.cast('double')

        query = impala.compile(expr2)
        expected = """SELECT CAST(`g` AS double) AS `tmp`
FROM (
  SELECT `g`, count(*) AS `count`
  FROM alltypes
  GROUP BY 1
) t0"""
        assert query == expected

    def test_scalar_exprs_no_table_refs(self):
        expr1 = ibis.now()
        expected1 = """\
SELECT now() AS `tmp`"""

        expr2 = ibis.literal(1) + ibis.literal(2)
        expected2 = """\
SELECT 1 + 2 AS `tmp`"""

        cases = [
            (expr1, expected1),
            (expr2, expected2)
        ]

        for expr, expected in cases:
            result = impala.compile(expr)
            assert result == expected

    def test_expr_list_no_table_refs(self):
        exlist = ibis.api.expr_list([ibis.literal(1).name('a'),
                                     ibis.now().name('b'),
                                     ibis.literal(2).log().name('c')])
        result = impala.compile(exlist)
        expected = """\
SELECT 1 AS `a`, now() AS `b`, ln(2) AS `c`"""
        assert result == expected

    def test_isnull_case_expr_rewrite_failure(self):
        # #172, case expression that was not being properly converted into an
        # aggregation
        reduction = self.table.g.isnull().ifelse(1, 0).sum()

        result = impala.compile(reduction)
        expected = """\
SELECT sum(CASE WHEN `g` IS NULL THEN 1 ELSE 0 END) AS `sum`
FROM alltypes"""
        assert result == expected


def _get_query(expr):
    ast = build_ast(expr)
    return ast.queries[0]

nation = api.table([
    ('n_regionkey', 'int32'),
    ('n_nationkey', 'int32'),
    ('n_name', 'string')
], 'nation')

region = api.table([
    ('r_regionkey', 'int32'),
    ('r_name', 'string')
], 'region')

customer = api.table([
    ('c_nationkey', 'int32'),
    ('c_name', 'string'),
    ('c_acctbal', 'double')
], 'customer')


def _table_wrapper(name, tname=None):
    @property
    def f(self):
        return self._table_from_schema(name, tname)
    return f


class ExprTestCases(object):

    _schemas = {
        'foo': [
            ('job', 'string'),
            ('dept_id', 'string'),
            ('year', 'int32'),
            ('y', 'double')
        ],
        'bar': [
            ('x', 'double'),
            ('job', 'string')
        ],
        't1': [
            ('key1', 'string'),
            ('key2', 'string'),
            ('value1', 'double')
        ],
        't2': [
            ('key1', 'string'),
            ('key2', 'string')
        ]
    }

    def _table_from_schema(self, name, tname=None):
        tname = tname or name
        return api.table(self._schemas[name], tname)

    def _case_multiple_joins(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')
        t3 = self.con.table('star3')

        predA = t1['foo_id'] == t2['foo_id']
        predB = t1['bar_id'] == t3['bar_id']

        what = (t1.left_join(t2, [predA])
                .inner_join(t3, [predB])
                .projection([t1, t2['value1'], t3['value2']]))
        return what

    def _case_join_between_joins(self):
        t1 = api.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value1', 'double'),
        ], 'first')

        t2 = api.table([
            ('key1', 'string'),
            ('value2', 'double'),
        ], 'second')

        t3 = api.table([
            ('key2', 'string'),
            ('key3', 'string'),
            ('value3', 'double'),
        ], 'third')

        t4 = api.table([
            ('key3', 'string'),
            ('value4', 'double')
        ], 'fourth')

        left = t1.inner_join(t2, [('key1', 'key1')])[t1, t2.value2]
        right = t3.inner_join(t4, [('key3', 'key3')])[t3, t4.value4]

        joined = left.inner_join(right, [('key2', 'key2')])

        # At one point, the expression simplification was resulting in bad refs
        # here (right.value3 referencing the table inside the right join)
        exprs = [left, right.value3, right.value4]
        projected = joined.projection(exprs)

        return projected

    def _case_join_just_materialized(self):
        t1 = self.con.table('tpch_nation')
        t2 = self.con.table('tpch_region')
        t3 = self.con.table('tpch_customer')

        # GH #491
        return (t1.inner_join(t2, t1.n_regionkey == t2.r_regionkey)
                .inner_join(t3, t1.n_nationkey == t3.c_nationkey))

    def _case_semi_anti_joins(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        sj = t1.semi_join(t2, [t1.foo_id == t2.foo_id])[[t1]]
        aj = t1.anti_join(t2, [t1.foo_id == t2.foo_id])[[t1]]

        return sj, aj

    def _case_self_reference_simple(self):
        t1 = self.con.table('star1')
        return t1.view()

    def _case_self_reference_join(self):
        t1 = self.con.table('star1')
        t2 = t1.view()
        return t1.inner_join(t2, [t1.foo_id == t2.bar_id])[[t1]]

    def _case_join_projection_subquery_bug(self):
        # From an observed bug, derived from tpch tables
        geo = (nation.inner_join(region, [('n_regionkey', 'r_regionkey')])
               [nation.n_nationkey,
                nation.n_name.name('nation'),
                region.r_name.name('region')])

        expr = (geo.inner_join(customer, [('n_nationkey', 'c_nationkey')])
                [customer, geo])

        return expr

    def _case_where_simple_comparisons(self):
        t1 = self.con.table('star1')

        what = t1.filter([t1.f > 0, t1.c < t1.f * 2])

        return what

    def _case_where_with_join(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        # This also tests some cases of predicate pushdown
        e1 = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
              .projection([t1, t2.value1, t2.value3])
              .filter([t1.f > 0, t2.value3 < 1000]))

        # e2 = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
        #       .filter([t1.f > 0, t2.value3 < 1000])
        #       .projection([t1, t2.value1, t2.value3]))

        # return e1, e2

        return e1

    def _case_subquery_used_for_self_join(self):
        # There could be cases that should look in SQL like
        # WITH t0 as (some subquery)
        # select ...
        # from t0 t1
        #   join t0 t2
        #     on t1.kind = t2.subkind
        # ...
        # However, the Ibis code will simply have an expression (projection or
        # aggregation, say) built on top of the subquery expression, so we need
        # to extract the subquery unit (we see that it appears multiple times
        # in the tree).
        t = self.con.table('alltypes')

        agged = t.aggregate([t.f.sum().name('total')], by=['g', 'a', 'b'])
        view = agged.view()
        metrics = [(agged.total - view.total).max().name('metric')]
        expr = (agged.inner_join(view, [agged.a == view.b])
                .aggregate(metrics, by=[agged.g]))

        return expr

    def _case_subquery_factor_correlated_subquery(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

        fields_of_interest = [customer,
                              region.r_name.name('region'),
                              orders.o_totalprice.name('amount'),
                              orders.o_orderdate
                              .cast('timestamp').name('odate')]

        tpch = (region.join(nation, region.r_regionkey == nation.n_regionkey)
                .join(customer, customer.c_nationkey == nation.n_nationkey)
                .join(orders, orders.o_custkey == customer.c_custkey)
                [fields_of_interest])

        # Self-reference + correlated subquery complicates things
        t2 = tpch.view()
        conditional_avg = t2[t2.region == tpch.region].amount.mean()
        amount_filter = tpch.amount > conditional_avg

        return tpch[amount_filter].limit(10)

    def _case_self_join_subquery_distinct_equal(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        j1 = (region.join(nation, region.r_regionkey == nation.n_regionkey)
              [region, nation])

        j2 = (region.join(nation, region.r_regionkey == nation.n_regionkey)
              [region, nation].view())

        expr = (j1.join(j2, j1.r_regionkey == j2.r_regionkey)
                [j1.r_name, j2.n_name])

        return expr

    def _case_cte_factor_distinct_but_equal(self):
        t = self.con.table('alltypes')
        tt = self.con.table('alltypes')

        expr1 = t.group_by('g').aggregate(t.f.sum().name('metric'))
        expr2 = tt.group_by('g').aggregate(tt.f.sum().name('metric')).view()

        expr = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]

        return expr

    def _case_tpch_self_join_failure(self):
        # duplicating the integration test here

        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

        fields_of_interest = [
            region.r_name.name('region'),
            nation.n_name.name('nation'),
            orders.o_totalprice.name('amount'),
            orders.o_orderdate.cast('timestamp').name('odate')]

        joined_all = (
            region.join(nation, region.r_regionkey == nation.n_regionkey)
            .join(customer, customer.c_nationkey == nation.n_nationkey)
            .join(orders, orders.o_custkey == customer.c_custkey)
            [fields_of_interest])

        year = joined_all.odate.year().name('year')
        total = joined_all.amount.sum().cast('double').name('total')
        annual_amounts = (joined_all
                          .group_by(['region', year])
                          .aggregate(total))

        current = annual_amounts
        prior = annual_amounts.view()

        yoy_change = (current.total - prior.total).name('yoy_change')
        yoy = (current.join(prior, current.year == (prior.year - 1))
               [current.region, current.year, yoy_change])
        return yoy

    def _case_subquery_in_filter_predicate(self):
        # E.g. comparing against some scalar aggregate value. See Ibis #43
        t1 = self.con.table('star1')

        pred = t1.f > t1.f.mean()
        expr = t1[pred]

        # This brought out another expression rewriting bug, since the filtered
        # table isn't found elsewhere in the expression.
        pred2 = t1.f > t1[t1.foo_id == 'foo'].f.mean()
        expr2 = t1[pred2]

        return expr, expr2

    def _case_filter_subquery_derived_reduction(self):
        t1 = self.con.table('star1')

        # Reduction can be nested inside some scalar expression
        pred3 = t1.f > t1[t1.foo_id == 'foo'].f.mean().log()
        pred4 = t1.f > (t1[t1.foo_id == 'foo'].f.mean().log() + 1)

        expr3 = t1[pred3]
        expr4 = t1[pred4]

        return expr3, expr4

    def _case_topk_operation(self):
        # TODO: top K with filter in place

        table = api.table([
            ('foo', 'string'),
            ('bar', 'string'),
            ('city', 'string'),
            ('v1', 'double'),
            ('v2', 'double'),
        ], 'tbl')

        what = table.city.topk(10, by=table.v2.mean())
        e1 = table[what]

        # Test the default metric (count)
        what = table.city.topk(10)
        e2 = table[what]

        return e1, e2

    def _case_simple_aggregate_query(self):
        t1 = self.con.table('star1')
        cases = [
            t1.aggregate([t1['f'].sum().name('total')],
                         [t1['foo_id']]),
            t1.aggregate([t1['f'].sum().name('total')],
                         ['foo_id', 'bar_id'])
        ]

        return cases

    def _case_aggregate_having(self):
        # Filtering post-aggregation predicate
        t1 = self.con.table('star1')

        total = t1.f.sum().name('total')
        metrics = [total]

        e1 = t1.aggregate(metrics, by=['foo_id'], having=[total > 10])
        e2 = t1.aggregate(metrics, by=['foo_id'], having=[t1.count() > 100])

        return e1, e2

    def _case_aggregate_count_joined(self):
        # count on more complicated table
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        join_expr = region.r_regionkey == nation.n_regionkey
        joined = region.inner_join(nation, join_expr)
        table_ref = joined[nation, region.r_name.name('region')]

        return table_ref.count()

    def _case_sort_by(self):
        table = self.con.table('star1')

        return [
            table.sort_by('f'),
            table.sort_by(('f', 0)),
            table.sort_by(['c', ('f', 0)])
        ]

    def _case_limit(self):
        star1 = self.con.table('star1')

        cases = [
            star1.limit(10),
            star1.limit(10, offset=5),
            star1[star1.f > 0].limit(10),

            # Semantically, this should produce a subquery
            star1.limit(10)[lambda x: x.f > 0]
        ]

        return cases

    foo = _table_wrapper('foo')
    bar = _table_wrapper('bar')
    t1 = _table_wrapper('t1', 'foo')
    t2 = _table_wrapper('t2', 'bar')

    def _case_where_uncorrelated_subquery(self):
        return self.foo[self.foo.job.isin(self.bar.job)]

    def _case_where_correlated_subquery(self):
        t1 = self.foo
        t2 = t1.view()

        stat = t2[t1.dept_id == t2.dept_id].y.mean()
        return t1[t1.y > stat]

    def _case_exists(self):
        t1, t2 = self.t1, self.t2

        cond = (t1.key1 == t2.key1).any()
        expr = t1[cond]

        cond2 = ((t1.key1 == t2.key1) & (t2.key2 == 'foo')).any()
        expr2 = t1[cond2]

        return expr, expr2

    def _case_not_exists(self):
        t1, t2 = self.t1, self.t2

        cond = (t1.key1 == t2.key1).any()
        return t1[-cond]

    def _case_join_with_limited_table(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        limited = t1.limit(100)
        joined = (limited.inner_join(t2, [limited.foo_id == t2.foo_id])
                  [[limited]])
        return joined

    def _case_union(self, distinct=False):
        table = self.con.table('functional_alltypes')

        t1 = (table[table.int_col > 0]
              [table.string_col.name('key'),
               table.float_col.cast('double').name('value')])
        t2 = (table[table.int_col <= 0]
                   [table.string_col.name('key'),
                    table.double_col.name('value')])

        expr = t1.union(t2, distinct=distinct)

        return expr

    def _case_simple_case(self):
        t = self.con.table('alltypes')
        return (t.g.case()
                .when('foo', 'bar')
                .when('baz', 'qux')
                .else_('default')
                .end())

    def _case_search_case(self):
        t = self.con.table('alltypes')
        return (ibis.case()
                .when(t.f > 0, t.d * 2)
                .when(t.c < 0, t.a * 2)
                .end())

    def _case_self_reference_in_exists(self):
        t = self.con.table('functional_alltypes')
        t2 = t.view()

        cond = (t.string_col == t2.string_col).any()
        semi = t[cond]
        anti = t[-cond]

        return semi, anti

    def _case_self_reference_limit_exists(self):
        alltypes = self.con.table('functional_alltypes')
        t = alltypes.limit(100)
        t2 = t.view()
        return t[-(t.string_col == t2.string_col).any()]

    def _case_limit_cte_extract(self):
        alltypes = self.con.table('functional_alltypes')
        t = alltypes.limit(100)
        t2 = t.view()
        return t.join(t2).projection(t)

    def _case_subquery_aliased(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        agged = t1.aggregate([t1.f.sum().name('total')], by=['foo_id'])
        what = (agged.inner_join(t2, [agged.foo_id == t2.foo_id])
                [agged, t2.value1])

        return what

    def _case_filter_self_join_analysis_bug(self):
        purchases = ibis.table([('region', 'string'),
                                ('kind', 'string'),
                                ('user', 'int64'),
                                ('amount', 'double')], 'purchases')

        metric = purchases.amount.sum().name('total')
        agged = (purchases.group_by(['region', 'kind'])
                 .aggregate(metric))

        left = agged[agged.kind == 'foo']
        right = agged[agged.kind == 'bar']

        joined = left.join(right, left.region == right.region)
        result = joined[left.region,
                        (left.total - right.total).name('diff')]

        return result, purchases

    def _case_projection_fuse_filter(self):
        # Probably test this during the evaluation phase. In SQL, "fusable"
        # table operations will be combined together into a single select
        # statement
        #
        # see ibis #71 for more on this

        t = ibis.table([
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean')
        ], 'foo')

        proj = t['a', 'b', 'c']

        # Rewrite a little more aggressively here
        expr1 = proj[t.a > 0]

        # at one point these yielded different results
        filtered = t[t.a > 0]

        expr2 = filtered[t.a, t.b, t.c]
        expr3 = filtered.projection(['a', 'b', 'c'])

        return expr1, expr2, expr3


class TestSelectSQL(unittest.TestCase, ExprTestCases):

    @classmethod
    def setUpClass(cls):
        cls.con = MockConnection()

    def _compare_sql(self, expr, expected):
        result = to_sql(expr)
        assert result == expected

    def test_nameless_table(self):
        # Ensure that user gets some kind of sensible error
        nameless = api.table([('key', 'string')])
        self.assertRaises(com.RelationError, to_sql, nameless)

        with_name = api.table([('key', 'string')], name='baz')
        result = to_sql(with_name)
        assert result == 'SELECT *\nFROM baz'

    def test_physical_table_reference_translate(self):
        # If an expression's table leaves all reference database tables, verify
        # we translate correctly
        table = self.con.table('alltypes')

        query = _get_query(table)
        sql_string = query.compile()
        expected = "SELECT *\nFROM alltypes"
        assert sql_string == expected

    def test_simple_joins(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        pred = t1['foo_id'] == t2['foo_id']
        pred2 = t1['bar_id'] == t2['foo_id']
        cases = [
            (t1.inner_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""),
            (t1.left_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  LEFT OUTER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""),
            (t1.outer_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  FULL OUTER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""),
            # multiple predicates
            (t1.inner_join(t2, [pred, pred2])[[t1]],
             """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id` AND
       t0.`bar_id` = t1.`foo_id`"""),
        ]

        for expr, expected_sql in cases:
            result_sql = to_sql(expr)
            assert result_sql == expected_sql

    def test_multiple_joins(self):
        what = self._case_multiple_joins()

        result_sql = to_sql(what)
        expected_sql = """SELECT t0.*, t1.`value1`, t2.`value2`
FROM star1 t0
  LEFT OUTER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
  INNER JOIN star3 t2
    ON t0.`bar_id` = t2.`bar_id`"""
        assert result_sql == expected_sql

    def test_join_between_joins(self):
        projected = self._case_join_between_joins()

        result = to_sql(projected)
        expected = """SELECT t0.*, t1.`value3`, t1.`value4`
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

    def test_join_just_materialized(self):
        joined = self._case_join_just_materialized()
        result = to_sql(joined)
        expected = """SELECT *
FROM tpch_nation t0
  INNER JOIN tpch_region t1
    ON t0.`n_regionkey` = t1.`r_regionkey`
  INNER JOIN tpch_customer t2
    ON t0.`n_nationkey` = t2.`c_nationkey`"""
        assert result == expected

        result = to_sql(joined.materialize())
        assert result == expected

    def test_join_no_predicates_for_impala(self):
        # Impala requires that joins without predicates be written explicitly
        # as CROSS JOIN, since result sets can accidentally get too large if a
        # query is executed before predicates are written
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        joined2 = t1.cross_join(t2)[[t1]]

        expected = """SELECT t0.*
FROM star1 t0
  CROSS JOIN star2 t1"""
        result2 = to_sql(joined2)
        assert result2 == expected

        for jtype in ['inner_join', 'left_join', 'outer_join']:
            joined = getattr(t1, jtype)(t2)[[t1]]

            result = to_sql(joined)
            assert result == expected

    def test_semi_anti_joins(self):
        sj, aj = self._case_semi_anti_joins()

        result = to_sql(sj)
        expected = """SELECT t0.*
FROM star1 t0
  LEFT SEMI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
        assert result == expected

        result = to_sql(aj)
        expected = """SELECT t0.*
FROM star1 t0
  LEFT ANTI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
        assert result == expected

    def test_self_reference_simple(self):
        expr = self._case_self_reference_simple()

        result_sql = to_sql(expr)
        expected_sql = "SELECT *\nFROM star1"
        assert result_sql == expected_sql

    def test_join_self_reference(self):
        result = self._case_self_reference_join()

        result_sql = to_sql(result)
        expected_sql = """SELECT t0.*
FROM star1 t0
  INNER JOIN star1 t1
    ON t0.`foo_id` = t1.`bar_id`"""
        assert result_sql == expected_sql

    def test_join_projection_subquery_broken_alias(self):
        expr = self._case_join_projection_subquery_bug()

        result = to_sql(expr)
        expected = """SELECT t1.*, t0.*
FROM (
  SELECT t2.`n_nationkey`, t2.`n_name` AS `nation`, t3.`r_name` AS `region`
  FROM nation t2
    INNER JOIN region t3
      ON t2.`n_regionkey` = t3.`r_regionkey`
) t0
  INNER JOIN customer t1
    ON t0.`n_nationkey` = t1.`c_nationkey`"""
        assert result == expected

    def test_where_simple_comparisons(self):
        what = self._case_where_simple_comparisons()
        result = to_sql(what)
        expected = """SELECT *
FROM star1
WHERE `f` > 0 AND
      `c` < (`f` * 2)"""
        assert result == expected

    def test_where_in_array_literal(self):
        # e.g.
        # where string_col in (v1, v2, v3)
        raise unittest.SkipTest

    def test_where_with_join(self):
        e1 = self._case_where_with_join()

        expected_sql = """SELECT t0.*, t1.`value1`, t1.`value3`
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`
WHERE t0.`f` > 0 AND
      t1.`value3` < 1000"""

        result_sql = to_sql(e1)
        assert result_sql == expected_sql

        # result2_sql = to_sql(e2)
        # assert result2_sql == expected_sql

    def test_where_no_pushdown_possible(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        joined = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                  [t1, (t1.f - t2.value1).name('diff')])

        filtered = joined[joined.diff > 1]

        # TODO: I'm not sure if this is exactly what we want
        expected_sql = """SELECT *
FROM (
  SELECT t0.*, t0.`f` - t1.`value1` AS `diff`
  FROM star1 t0
    INNER JOIN star2 t1
      ON t0.`foo_id` = t1.`foo_id`
  WHERE t0.`f` > 0 AND
        t1.`value3` < 1000
)
WHERE `diff` > 1"""

        raise unittest.SkipTest

        result_sql = to_sql(filtered)
        assert result_sql == expected_sql

    def test_where_with_between(self):
        t = self.con.table('alltypes')

        what = t.filter([t.a > 0, t.f.between(0, 1)])
        result = to_sql(what)
        expected = """SELECT *
FROM alltypes
WHERE `a` > 0 AND
      `f` BETWEEN 0 AND 1"""
        assert result == expected

    def test_where_analyze_scalar_op(self):
        # root cause of #310

        table = self.con.table('functional_alltypes')

        expr = (table.filter([table.timestamp_col <
                             (ibis.timestamp('2010-01-01') + ibis.month(3)),
                             table.timestamp_col < (ibis.now() +
                                                    ibis.day(10))])
                .count())

        result = to_sql(expr)
        expected = """\
SELECT count(*) AS `count`
FROM functional_alltypes
WHERE `timestamp_col` < months_add('2010-01-01 00:00:00', 3) AND
      `timestamp_col` < days_add(now(), 10)"""
        assert result == expected

    def test_bug_duplicated_where(self):
        # GH #539
        table = self.con.table('airlines')

        t = table['arrdelay', 'dest']
        expr = (t.group_by('dest')
                .mutate(dest_avg=t.arrdelay.mean(),
                        dev=t.arrdelay - t.arrdelay.mean()))

        tmp1 = expr[expr.dev.notnull()]
        tmp2 = tmp1.sort_by(ibis.desc('dev'))
        worst = tmp2.limit(10)

        result = to_sql(worst)
        expected = """\
SELECT *
FROM (
  SELECT `arrdelay`, `dest`,
         avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dest_avg`,
         `arrdelay` - avg(`arrdelay`) OVER (PARTITION BY `dest`) AS `dev`
  FROM airlines
) t0
WHERE `dev` IS NOT NULL
ORDER BY `dev` DESC
LIMIT 10"""
        assert result == expected

    def test_simple_aggregate_query(self):
        expected = [
            """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1""",
            """SELECT `foo_id`, `bar_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1, 2"""
        ]

        cases = self._case_simple_aggregate_query()
        for expr, expected_sql in zip(cases, expected):
            result_sql = to_sql(expr)
            assert result_sql == expected_sql

    def test_aggregate_having(self):
        e1, e2 = self._case_aggregate_having()

        result = to_sql(e1)
        expected = """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1
HAVING sum(`f`) > 10"""
        assert result == expected

        result = to_sql(e2)
        expected = """SELECT `foo_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1
HAVING count(*) > 100"""
        assert result == expected

    def test_aggregate_table_count_metric(self):
        expr = self.con.table('star1').count()

        result = to_sql(expr)
        expected = """SELECT count(*) AS `count`
FROM star1"""
        assert result == expected

    def test_aggregate_count_joined(self):
        expr = self._case_aggregate_count_joined()

        result = to_sql(expr)
        expected = """SELECT count(*) AS `count`
FROM (
  SELECT t2.*, t1.`r_name` AS `region`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
) t0"""
        assert result == expected

    def test_expr_template_field_name_binding(self):
        # Given an expression with no concrete links to actual database tables,
        # indicate a mapping between the distinct unbound table leaves of the
        # expression and some database tables with compatible schemas but
        # potentially different column names
        pass

    def test_no_aliases_needed(self):
        table = api.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value', 'double')
        ])

        expr = table.aggregate([table['value'].sum().name('total')],
                               by=['key1', 'key2'])

        query = _get_query(expr)
        context = query.context
        assert not context.need_aliases()

    def test_table_names_overlap_default_aliases(self):
        # see discussion in #104; this actually is not needed for query
        # correctness, and only makes the generated SQL nicer
        raise unittest.SkipTest

        t0 = api.table([
            ('key', 'string'),
            ('v1', 'double')
        ], 't1')

        t1 = api.table([
            ('key', 'string'),
            ('v2', 'double')
        ], 't0')

        expr = t0.join(t1, t0.key == t1.key)[t0.key, t0.v1, t1.v2]

        result = to_sql(expr)
        expected = """\
SELECT t2.`key`, t2.`v1`, t3.`v2`
FROM t0 t2
  INNER JOIN t1 t3
    ON t2.`key` = t3.`key`"""

        assert result == expected

    def test_context_aliases_multiple_join(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')
        t3 = self.con.table('star3')

        expr = (t1.left_join(t2, [t1['foo_id'] == t2['foo_id']])
                .inner_join(t3, [t1['bar_id'] == t3['bar_id']])
                [[t1, t2['value1'], t3['value2']]])

        query = _get_query(expr)
        context = query.context

        assert context.get_ref(t1) == 't0'
        assert context.get_ref(t2) == 't1'
        assert context.get_ref(t3) == 't2'

    def test_fuse_projections(self):
        table = api.table([
            ('foo', 'int32'),
            ('bar', 'int64'),
            ('value', 'double')
        ], name='tbl')

        # Cases where we project in both cases using the base table reference
        f1 = (table['foo'] + table['bar']).name('baz')
        pred = table['value'] > 0

        table2 = table[table, f1]
        table2_filtered = table2[pred]

        f2 = (table2['foo'] * 2).name('qux')
        f3 = (table['foo'] * 2).name('qux')

        table3 = table2.projection([table2, f2])

        # fusion works even if there's a filter
        table3_filtered = table2_filtered.projection([table2, f2])

        expected = table[table, f1, f3]
        expected2 = table[pred][table, f1, f3]

        assert table3.equals(expected)
        assert table3_filtered.equals(expected2)

        ex_sql = """SELECT *, `foo` + `bar` AS `baz`, `foo` * 2 AS `qux`
FROM tbl"""

        ex_sql2 = """SELECT *, `foo` + `bar` AS `baz`, `foo` * 2 AS `qux`
FROM tbl
WHERE `value` > 0"""

        table3_sql = to_sql(table3)
        table3_filt_sql = to_sql(table3_filtered)

        assert table3_sql == ex_sql
        assert table3_filt_sql == ex_sql2

        # Use the intermediate table refs
        table3 = table2.projection([table2, f2])

        # fusion works even if there's a filter
        table3_filtered = table2_filtered.projection([table2, f2])

        expected = table[table, f1, f3]
        expected2 = table[pred][table, f1, f3]

        assert table3.equals(expected)
        assert table3_filtered.equals(expected2)

    def test_projection_filter_fuse(self):
        expr1, expr2, expr3 = self._case_projection_fuse_filter()

        sql1 = to_sql(expr1)
        sql2 = to_sql(expr2)
        sql3 = to_sql(expr3)

        assert sql1 == sql2
        assert sql1 == sql3

    def test_bug_project_multiple_times(self):
        # 108
        customer = self.con.table('tpch_customer')
        nation = self.con.table('tpch_nation')
        region = self.con.table('tpch_region')

        joined = (
            customer.inner_join(nation,
                                [customer.c_nationkey == nation.n_nationkey])
            .inner_join(region,
                        [nation.n_regionkey == region.r_regionkey])
        )
        proj1 = [customer, nation.n_name, region.r_name]
        step1 = joined[proj1]

        topk_by = step1.c_acctbal.cast('double').sum()
        pred = step1.n_name.topk(10, by=topk_by)

        proj_exprs = [step1.c_name, step1.r_name, step1.n_name]
        step2 = step1[pred]
        expr = step2.projection(proj_exprs)

        # it works!
        result = to_sql(expr)
        expected = """\
SELECT t0.`c_name`, t2.`r_name`, t1.`n_name`
FROM tpch_customer t0
  INNER JOIN tpch_nation t1
    ON t0.`c_nationkey` = t1.`n_nationkey`
  INNER JOIN tpch_region t2
    ON t1.`n_regionkey` = t2.`r_regionkey`
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT t1.`n_name`, sum(CAST(t0.`c_acctbal` AS double)) AS `sum`
      FROM tpch_customer t0
        INNER JOIN tpch_nation t1
          ON t0.`c_nationkey` = t1.`n_nationkey`
        INNER JOIN tpch_region t2
          ON t1.`n_regionkey` = t2.`r_regionkey`
      GROUP BY 1
    ) t4
    ORDER BY `sum` DESC
    LIMIT 10
  ) t3
    ON t1.`n_name` = t3.`n_name`"""
        assert result == expected

    def test_aggregate_projection_subquery(self):
        t = self.con.table('alltypes')

        proj = t[t.f > 0][t, (t.a + t.b).name('foo')]

        result = to_sql(proj)
        expected = """SELECT *, `a` + `b` AS `foo`
FROM alltypes
WHERE `f` > 0"""
        assert result == expected

        def agg(x):
            return x.aggregate([x.foo.sum().name('foo total')], by=['g'])

        # predicate gets pushed down
        filtered = proj[proj.g == 'bar']

        result = to_sql(filtered)
        expected = """SELECT *, `a` + `b` AS `foo`
FROM alltypes
WHERE `f` > 0 AND
      `g` = 'bar'"""
        assert result == expected

        agged = agg(filtered)
        result = to_sql(agged)
        expected = """SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE `f` > 0 AND
        `g` = 'bar'
) t0
GROUP BY 1"""
        assert result == expected

        # Pushdown is not possible (in Impala, Postgres, others)
        agged2 = agg(proj[proj.foo < 10])

        result = to_sql(agged2)
        expected = """SELECT `g`, sum(`foo`) AS `foo total`
FROM (
  SELECT *, `a` + `b` AS `foo`
  FROM alltypes
  WHERE `f` > 0
) t0
WHERE `foo` < 10
GROUP BY 1"""
        assert result == expected

    def test_subquery_aliased(self):
        case = self._case_subquery_aliased()

        expected = """SELECT t0.*, t1.`value1`
FROM (
  SELECT `foo_id`, sum(`f`) AS `total`
  FROM star1
  GROUP BY 1
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""
        self._compare_sql(case, expected)

    def test_double_nested_subquery_no_aliases(self):
        # We don't require any table aliasing anywhere
        t = api.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value', 'double')
        ], 'foo_table')

        agg1 = t.aggregate([t.value.sum().name('total')],
                           by=['key1', 'key2', 'key3'])
        agg2 = agg1.aggregate([agg1.total.sum().name('total')],
                              by=['key1', 'key2'])
        agg3 = agg2.aggregate([agg2.total.sum().name('total')],
                              by=['key1'])

        result = to_sql(agg3)
        expected = """SELECT `key1`, sum(`total`) AS `total`
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

    def test_aggregate_projection_alias_bug(self):
        # Observed in use
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        what = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                [[t1, t2.value1]])

        what = what.aggregate([what.value1.sum().name('total')],
                              by=[what.foo_id])

        # TODO: Not fusing the aggregation with the projection yet
        result = to_sql(what)
        expected = """SELECT `foo_id`, sum(`value1`) AS `total`
FROM (
  SELECT t1.*, t2.`value1`
  FROM star1 t1
    INNER JOIN star2 t2
      ON t1.`foo_id` = t2.`foo_id`
) t0
GROUP BY 1"""
        assert result == expected

    def test_aggregate_fuse_with_projection(self):
        # see above test case
        pass

    def test_subquery_used_for_self_join(self):
        expr = self._case_subquery_used_for_self_join()

        result = to_sql(expr)
        expected = """WITH t0 AS (
  SELECT `g`, `a`, `b`, sum(`f`) AS `total`
  FROM alltypes
  GROUP BY 1, 2, 3
)
SELECT t0.`g`, max(t0.`total` - t1.`total`) AS `metric`
FROM t0
  INNER JOIN t0 t1
    ON t0.`a` = t1.`b`
GROUP BY 1"""
        assert result == expected

    def test_subquery_in_union(self):
        t = self.con.table('alltypes')

        expr1 = t.group_by(['a', 'g']).aggregate(t.f.sum().name('metric'))
        expr2 = expr1.view()

        join1 = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]
        join2 = join1.view()

        expr = join1.union(join2)
        result = to_sql(expr)
        expected = """\
(WITH t0 AS (
  SELECT `a`, `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1, 2
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1
    ON t0.`g` = t1.`g`)
UNION ALL
(WITH t0 AS (
  SELECT `a`, `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1, 2
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1
    ON t0.`g` = t1.`g`)"""
        assert result == expected

    def test_subquery_factor_correlated_subquery(self):
        # #173, #183 and other issues

        expr = self._case_subquery_factor_correlated_subquery()

        result = to_sql(expr)
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

    def test_self_join_subquery_distinct_equal(self):
        expr = self._case_self_join_subquery_distinct_equal()

        result = to_sql(expr)
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

    def test_limit_with_self_join(self):
        t = self.con.table('functional_alltypes')
        t2 = t.view()

        expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

        # it works
        result = to_sql(expr)
        expected = """\
SELECT count(*) AS `count`
FROM functional_alltypes t0
  INNER JOIN functional_alltypes t1
    ON t0.`tinyint_col` < extract(t1.`timestamp_col`, 'minute')"""
        assert result == expected

    def test_cte_factor_distinct_but_equal(self):
        expr = self._case_cte_factor_distinct_but_equal()

        result = to_sql(expr)
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

    def test_tpch_self_join_failure(self):
        yoy = self._case_tpch_self_join_failure()
        to_sql(yoy)

    def test_extract_subquery_nested_lower(self):
        # We may have a join between two tables requiring subqueries, and
        # buried inside these there may be a common subquery. Let's test that
        # we find it and pull it out to the top level to avoid repeating
        # ourselves.
        pass

    def test_subquery_in_filter_predicate(self):
        expr, expr2 = self._case_subquery_in_filter_predicate()

        result = to_sql(expr)
        expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`) AS `mean`
  FROM star1
)"""
        assert result == expected

        result = to_sql(expr2)
        expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT avg(`f`) AS `mean`
  FROM star1
  WHERE `foo_id` = 'foo'
)"""
        assert result == expected

    def test_filter_subquery_derived_reduction(self):
        expr3, expr4 = self._case_filter_subquery_derived_reduction()

        result = to_sql(expr3)
        expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT ln(avg(`f`)) AS `tmp`
  FROM star1
  WHERE `foo_id` = 'foo'
)"""
        assert result == expected

        result = to_sql(expr4)
        expected = """SELECT *
FROM star1
WHERE `f` > (
  SELECT ln(avg(`f`)) + 1 AS `tmp`
  FROM star1
  WHERE `foo_id` = 'foo'
)"""
        assert result == expected

    def test_topk_operation(self):
        filtered, filtered2 = self._case_topk_operation()

        query = to_sql(filtered)
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

        query = to_sql(filtered2)
        expected = """SELECT t0.*
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

    def test_topk_predicate_pushdown_bug(self):
        # Observed on TPCH data
        cplusgeo = (
            customer.inner_join(nation, [customer.c_nationkey ==
                                         nation.n_nationkey])
                    .inner_join(region, [nation.n_regionkey ==
                                         region.r_regionkey])
            [customer, nation.n_name, region.r_name])

        pred = cplusgeo.n_name.topk(10, by=cplusgeo.c_acctbal.sum())
        expr = cplusgeo.filter([pred])

        result = to_sql(expr)
        expected = """\
SELECT t0.*, t1.`n_name`, t2.`r_name`
FROM customer t0
  INNER JOIN nation t1
    ON t0.`c_nationkey` = t1.`n_nationkey`
  INNER JOIN region t2
    ON t1.`n_regionkey` = t2.`r_regionkey`
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT t1.`n_name`, sum(t0.`c_acctbal`) AS `sum`
      FROM customer t0
        INNER JOIN nation t1
          ON t0.`c_nationkey` = t1.`n_nationkey`
        INNER JOIN region t2
          ON t1.`n_regionkey` = t2.`r_regionkey`
      GROUP BY 1
    ) t4
    ORDER BY `sum` DESC
    LIMIT 10
  ) t3
    ON t1.`n_name` = t3.`n_name`"""

        assert result == expected

    def test_topk_analysis_bug(self):
        # GH #398
        airlines = ibis.table([('dest', 'string'),
                               ('origin', 'string'),
                               ('arrdelay', 'int32')], 'airlines')

        dests = ['ORD', 'JFK', 'SFO']
        delay_filter = airlines.dest.topk(10, by=airlines.arrdelay.mean())
        t = airlines[airlines.dest.isin(dests)]
        expr = t[delay_filter].group_by('origin').size()

        result = to_sql(expr)
        expected = """\
SELECT t0.`origin`, count(*) AS `count`
FROM airlines t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `dest`, avg(`arrdelay`) AS `mean`
      FROM airlines
      GROUP BY 1
    ) t2
    ORDER BY `mean` DESC
    LIMIT 10
  ) t1
    ON t0.`dest` = t1.`dest`
WHERE t0.`dest` IN ('ORD', 'JFK', 'SFO')
GROUP BY 1"""

        assert result == expected

    def test_topk_to_aggregate(self):
        t = ibis.table([('dest', 'string'),
                        ('origin', 'string'),
                        ('arrdelay', 'int32')], 'airlines')

        top = t.dest.topk(10, by=t.arrdelay.mean())

        result = to_sql(top)
        expected = to_sql(top.to_aggregation())
        assert result == expected

    def test_bottomk(self):
        pass

    def test_topk_antijoin(self):
        # Get the "other" category somehow
        pass

    def test_case_in_projection(self):
        t = self.con.table('alltypes')

        expr = (t.g.case()
                .when('foo', 'bar')
                .when('baz', 'qux')
                .else_('default').end())

        expr2 = (api.case()
                 .when(t.g == 'foo', 'bar')
                 .when(t.g == 'baz', t.g)
                 .end())

        proj = t[expr.name('col1'), expr2.name('col2'), t]

        result = to_sql(proj)
        expected = """SELECT
  CASE `g`
    WHEN 'foo' THEN 'bar'
    WHEN 'baz' THEN 'qux'
    ELSE 'default'
  END AS `col1`,
  CASE
    WHEN `g` = 'foo' THEN 'bar'
    WHEN `g` = 'baz' THEN `g`
    ELSE NULL
  END AS `col2`, *
FROM alltypes"""
        assert result == expected

    def test_identifier_quoting(self):
        data = api.table([
            ('date', 'int32'),
            ('explain', 'string')
        ], 'table')

        expr = data[data.date.name('else'), data.explain.name('join')]

        result = to_sql(expr)
        expected = """SELECT `date` AS `else`, `explain` AS `join`
FROM `table`"""
        assert result == expected

    def test_scalar_subquery_different_table(self):
        t1, t2 = self.foo, self.bar
        expr = t1[t1.y > t2.x.max()]

        result = to_sql(expr)
        expected = """SELECT *
FROM foo
WHERE `y` > (
  SELECT max(`x`) AS `max`
  FROM bar
)"""
        assert result == expected

    def test_where_uncorrelated_subquery(self):
        expr = self._case_where_uncorrelated_subquery()

        result = to_sql(expr)
        expected = """SELECT *
FROM foo
WHERE `job` IN (
  SELECT `job`
  FROM bar
)"""
        assert result == expected

    def test_where_correlated_subquery(self):
        expr = self._case_where_correlated_subquery()
        result = to_sql(expr)
        expected = """SELECT t0.*
FROM foo t0
WHERE t0.`y` > (
  SELECT avg(t1.`y`) AS `mean`
  FROM foo t1
  WHERE t0.`dept_id` = t1.`dept_id`
)"""
        assert result == expected

    def test_where_array_correlated(self):
        # Test membership in some record-dependent values, if this is supported
        pass

    def test_exists(self):
        e1, e2 = self._case_exists()

        result = to_sql(e1)
        expected = """SELECT t0.*
FROM foo t0
WHERE EXISTS (
  SELECT 1
  FROM bar t1
  WHERE t0.`key1` = t1.`key1`
)"""
        assert result == expected

        result = to_sql(e2)
        expected = """SELECT t0.*
FROM foo t0
WHERE EXISTS (
  SELECT 1
  FROM bar t1
  WHERE t0.`key1` = t1.`key1` AND
        t1.`key2` = 'foo'
)"""
        assert result == expected

    def test_exists_subquery_repr(self):
        # GH #660
        t1, t2 = self.t1, self.t2

        cond = t1.key1 == t2.key1
        expr = t1[cond.any()]
        stmt = build_ast(expr).queries[0]

        repr(stmt.where[0])

    def test_not_exists(self):
        expr = self._case_not_exists()
        result = to_sql(expr)
        expected = """SELECT t0.*
FROM foo t0
WHERE NOT EXISTS (
  SELECT 1
  FROM bar t1
  WHERE t0.`key1` = t1.`key1`
)"""
        assert result == expected

    def test_filter_inside_exists(self):
        events = ibis.table([('session_id', 'int64'),
                             ('user_id', 'int64'),
                             ('event_type', 'int32'),
                             ('ts', 'timestamp')], 'events')

        purchases = ibis.table([('item_id', 'int64'),
                                ('user_id', 'int64'),
                                ('price', 'double'),
                                ('ts', 'timestamp')], 'purchases')
        filt = purchases.ts > '2015-08-15'
        cond = (events.user_id == purchases[filt].user_id).any()
        expr = events[cond]

        result = to_sql(expr)
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

    def test_self_reference_in_exists(self):
        semi, anti = self._case_self_reference_in_exists()

        result = to_sql(semi)
        expected = """\
SELECT t0.*
FROM functional_alltypes t0
WHERE EXISTS (
  SELECT 1
  FROM functional_alltypes t1
  WHERE t0.`string_col` = t1.`string_col`
)"""
        assert result == expected

        result = to_sql(anti)
        expected = """\
SELECT t0.*
FROM functional_alltypes t0
WHERE NOT EXISTS (
  SELECT 1
  FROM functional_alltypes t1
  WHERE t0.`string_col` = t1.`string_col`
)"""
        assert result == expected

    def test_self_reference_limit_exists(self):
        case = self._case_self_reference_limit_exists()

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
        self._compare_sql(case, expected)

    def test_limit_cte_extract(self):
        case = self._case_limit_cte_extract()

        expected = """\
WITH t0 AS (
  SELECT *
  FROM functional_alltypes
  LIMIT 100
)
SELECT t0.*
FROM t0
  CROSS JOIN t0 t1"""

        self._compare_sql(case, expected)

    def test_sort_by(self):
        cases = self._case_sort_by()

        expected = [
            """SELECT *
FROM star1
ORDER BY `f`""",
            """SELECT *
FROM star1
ORDER BY `f` DESC""",
            """SELECT *
FROM star1
ORDER BY `c`, `f` DESC"""
        ]

        for case, ex in zip(cases, expected):
            result = to_sql(case)
            assert result == ex

    def test_limit(self):
        cases = self._case_limit()

        expected = [
            """SELECT *
FROM star1
LIMIT 10""",
            """SELECT *
FROM star1
LIMIT 10 OFFSET 5""",
            """SELECT *
FROM star1
WHERE `f` > 0
LIMIT 10""",
            """SELECT *
FROM (
  SELECT *
  FROM star1
  LIMIT 10
) t0
WHERE `f` > 0"""
        ]

        for case, ex in zip(cases, expected):
            result = to_sql(case)
            assert result == ex

    def test_join_with_limited_table(self):
        joined = self._case_join_with_limited_table()

        result = to_sql(joined)
        expected = """SELECT t0.*
FROM (
  SELECT *
  FROM star1
  LIMIT 100
) t0
  INNER JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`"""

        assert result == expected

    def test_sort_by_on_limit_yield_subquery(self):
        # x.limit(...).sort_by(...)
        #   is semantically different from
        # x.sort_by(...).limit(...)
        #   and will often yield different results
        t = self.con.table('functional_alltypes')
        expr = (t.group_by('string_col')
                .aggregate([t.count().name('nrows')])
                .limit(5)
                .sort_by('string_col'))

        result = to_sql(expr)
        expected = """SELECT *
FROM (
  SELECT `string_col`, count(*) AS `nrows`
  FROM functional_alltypes
  GROUP BY 1
  LIMIT 5
) t0
ORDER BY `string_col`"""
        assert result == expected

    def test_multiple_limits(self):
        t = self.con.table('functional_alltypes')

        expr = t.limit(20).limit(10)
        stmt = build_ast(expr).queries[0]

        assert stmt.limit['n'] == 10

    def test_top_convenience(self):
        # x.top(10, by=field)
        # x.top(10, by=[field1, field2])
        pass

    def test_self_aggregate_in_predicate(self):
        # Per ibis #43
        pass

    def test_self_join_filter_analysis_bug(self):
        expr, _ = self._case_filter_self_join_analysis_bug()

        expected = """\
SELECT t0.`region`, t0.`total` - t1.`total` AS `diff`
FROM (
  SELECT `region`, `kind`, sum(`amount`) AS `total`
  FROM purchases
  WHERE `kind` = 'foo'
  GROUP BY 1, 2
) t0
  INNER JOIN (
    SELECT `region`, `kind`, sum(`amount`) AS `total`
    FROM purchases
    WHERE `kind` = 'bar'
    GROUP BY 1, 2
  ) t1
    ON t0.`region` = t1.`region`"""
        self._compare_sql(expr, expected)

    def test_join_filtered_tables_no_pushdown(self):
        # #790, #781
        tbl_a = ibis.table([('year', 'int32'),
                            ('month', 'int32'),
                            ('day', 'int32'),
                            ('value_a', 'double')], 'a')

        tbl_b = ibis.table([('year', 'int32'),
                            ('month', 'int32'),
                            ('day', 'int32'),
                            ('value_b', 'double')], 'b')

        tbl_a_filter = tbl_a.filter([
            tbl_a.year == 2016,
            tbl_a.month == 2,
            tbl_a.day == 29
        ])

        tbl_b_filter = tbl_b.filter([
            tbl_b.year == 2016,
            tbl_b.month == 2,
            tbl_b.day == 29
        ])

        joined = tbl_a_filter.left_join(tbl_b_filter, ['year', 'month', 'day'])
        result = joined[tbl_a_filter.value_a, tbl_b_filter.value_b]

        join_op = result.op().table.op()
        assert join_op.left.equals(tbl_a_filter)
        assert join_op.right.equals(tbl_b_filter)

        result_sql = ibis.impala.compile(result)
        expected_sql = """\
SELECT t0.`value_a`, t1.`value_b`
FROM (
  SELECT *
  FROM a
  WHERE `year` = 2016 AND
        `month` = 2 AND
        `day` = 29
) t0
  LEFT OUTER JOIN (
    SELECT *
    FROM b
    WHERE `year` = 2016 AND
          `month` = 2 AND
          `day` = 29
  ) t1
    ON t0.`year` = t1.`year` AND
       t0.`month` = t1.`month` AND
       t0.`day` = t1.`day`"""

        assert result_sql == expected_sql

    def test_loj_subquery_filter_handling(self):
        # #781
        left = ibis.table([('id', 'int32'), ('desc', 'string')], 'foo')

        right = ibis.table([('id', 'int32'), ('desc', 'string')], 'bar')
        left = left[left.id < 2]
        right = right[right.id < 3]

        joined = left.left_join(right, ['id', 'desc'])
        joined = joined[
            [left[name].name('left_' + name) for name in left.columns] +
            [right[name].name('right_' + name) for name in right.columns]
        ]

        result = to_sql(joined)
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
    ON t0.`id` = t1.`id` AND
       t0.`desc` = t1.`desc`"""

        assert result == expected


class TestUnions(unittest.TestCase, ExprTestCases):

    def setUp(self):
        self.con = MockConnection()

    def test_union(self):
        union1 = self._case_union()

        result = to_sql(union1)
        expected = """\
SELECT `string_col` AS `key`, CAST(`float_col` AS double) AS `value`
FROM functional_alltypes
WHERE `int_col` > 0
UNION ALL
SELECT `string_col` AS `key`, `double_col` AS `value`
FROM functional_alltypes
WHERE `int_col` <= 0"""
        assert result == expected

    def test_union_distinct(self):
        union = self._case_union(distinct=True)
        result = to_sql(union)
        expected = """\
SELECT `string_col` AS `key`, CAST(`float_col` AS double) AS `value`
FROM functional_alltypes
WHERE `int_col` > 0
UNION
SELECT `string_col` AS `key`, `double_col` AS `value`
FROM functional_alltypes
WHERE `int_col` <= 0"""
        assert result == expected

    def test_union_project_column(self):
        # select a column, get a subquery
        union1 = self._case_union()
        expr = union1[[union1.key]]
        result = to_sql(expr)
        expected = """SELECT `key`
FROM (
  SELECT `string_col` AS `key`, CAST(`float_col` AS double) AS `value`
  FROM functional_alltypes
  WHERE `int_col` > 0
  UNION ALL
  SELECT `string_col` AS `key`, `double_col` AS `value`
  FROM functional_alltypes
  WHERE `int_col` <= 0
) t0"""
        assert result == expected


class TestDistinct(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_table_distinct(self):
        t = self.con.table('functional_alltypes')

        expr = t[t.string_col, t.int_col].distinct()

        result = to_sql(expr)
        expected = """SELECT DISTINCT `string_col`, `int_col`
FROM functional_alltypes"""
        assert result == expected

    def test_array_distinct(self):
        t = self.con.table('functional_alltypes')
        expr = t.string_col.distinct()

        result = to_sql(expr)
        expected = """SELECT DISTINCT `string_col`
FROM functional_alltypes"""
        assert result == expected

    def test_count_distinct(self):
        t = self.con.table('functional_alltypes')

        metric = t.int_col.nunique().name('nunique')
        expr = t[t.bigint_col > 0].group_by('string_col').aggregate([metric])

        result = to_sql(expr)
        expected = """\
SELECT `string_col`, COUNT(DISTINCT `int_col`) AS `nunique`
FROM functional_alltypes
WHERE `bigint_col` > 0
GROUP BY 1"""
        assert result == expected

    def test_multiple_count_distinct(self):
        # Impala and some other databases will not execute multiple
        # count-distincts in a single aggregation query. This error reporting
        # will be left to the database itself, for now.
        t = self.con.table('functional_alltypes')
        metrics = [t.int_col.nunique().name('int_card'),
                   t.smallint_col.nunique().name('smallint_card')]

        expr = t.group_by('string_col').aggregate(metrics)

        result = to_sql(expr)
        expected = """\
SELECT `string_col`, COUNT(DISTINCT `int_col`) AS `int_card`,
       COUNT(DISTINCT `smallint_col`) AS `smallint_card`
FROM functional_alltypes
GROUP BY 1"""
        assert result == expected


def test_pushdown_with_or():
    t = ibis.table(
        [('double_col', 'double'),
         ('string_col', 'string'),
         ('int_col', 'int32'),
         ('float_col', 'float')],
        'functional_alltypes',
    )
    subset = t[(t.double_col > 3.14) & t.string_col.contains('foo')]
    filt = subset[(subset.int_col - 1 == 0) | (subset.float_col <= 1.34)]
    result = to_sql(filt)
    expected = """\
SELECT *
FROM functional_alltypes
WHERE (`double_col` > 3.14) AND (locate('foo', `string_col`) - 1 >= 0) AND
      (((`int_col` - 1) = 0) OR (`float_col` <= 1.34))"""
    assert result == expected


def test_having_size():
    t = ibis.table(
        [('double_col', 'double'),
         ('string_col', 'string'),
         ('int_col', 'int32'),
         ('float_col', 'float')],
        'functional_alltypes',
    )
    expr = t.group_by(t.string_col).having(t.double_col.max() == 1).size()
    result = to_sql(expr)
    assert result == """\
SELECT `string_col`, count(*) AS `count`
FROM functional_alltypes
GROUP BY 1
HAVING max(`double_col`) = 1"""


def test_having_from_filter():
    t = ibis.table([('a', 'int64'), ('b', 'string')], 't')
    filt = t[t.b == 'm']
    gb = filt.group_by(filt.b)
    having = gb.having(filt.a.max() == 2)
    agg = having.aggregate(filt.a.sum().name('sum'))
    result = to_sql(agg)
    expected = """\
SELECT `b`, sum(`a`) AS `sum`
FROM t
WHERE `b` = 'm'
GROUP BY 1
HAVING max(`a`) = 2"""
    assert result == expected
