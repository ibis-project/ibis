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

from ibis.compat import unittest
from ibis.expr.tests.mocks import BasicTestCase
import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.common as com

from ibis.tests.util import assert_equal


# Place to collect esoteric expression analysis bugs and tests


class TestTableExprBasics(BasicTestCase, unittest.TestCase):

    def test_rewrite_substitute_distinct_tables(self):
        t = self.con.table('test1')
        tt = self.con.table('test1')

        expr = t[t.c > 0]
        expr2 = tt[tt.c > 0]

        metric = t.f.sum().name('metric')
        expr3 = expr.aggregate(metric)

        result = L.sub_for(expr3, [(expr2, t)])
        expected = t.aggregate(metric)

        assert_equal(result, expected)

    def test_rewrite_join_projection_without_other_ops(self):
        # See #790, predicate pushdown in joins not supported

        # Star schema with fact table
        table = self.con.table('star1')
        table2 = self.con.table('star2')
        table3 = self.con.table('star3')

        filtered = table[table['f'] > 0]

        pred1 = table['foo_id'] == table2['foo_id']
        pred2 = filtered['bar_id'] == table3['bar_id']

        j1 = filtered.left_join(table2, [pred1])
        j2 = j1.inner_join(table3, [pred2])

        # Project out the desired fields
        view = j2[[filtered, table2['value1'], table3['value2']]]

        # Construct the thing we expect to obtain
        ex_pred2 = table['bar_id'] == table3['bar_id']
        ex_expr = (table.left_join(table2, [pred1])
                   .inner_join(table3, [ex_pred2]))

        rewritten_proj = L.substitute_parents(view)
        op = rewritten_proj.op()

        assert not op.table.equals(ex_expr)

    def test_rewrite_past_projection(self):
        table = self.con.table('test1')

        # Rewrite past a projection
        table3 = table[['c', 'f']]
        expr = table3['c'] == 2

        result = L.substitute_parents(expr)
        expected = table['c'] == 2
        assert_equal(result, expected)

        # Unsafe to rewrite past projection
        table5 = table[(table.f * 2).name('c'), table.f]
        expr = table5['c'] == 2
        result = L.substitute_parents(expr)
        assert result is expr

    def test_multiple_join_deeper_reference(self):
        # Join predicates down the chain might reference one or more root
        # tables in the hierarchy.
        table1 = ibis.table({'key1': 'string', 'key2': 'string',
                            'value1': 'double'})
        table2 = ibis.table({'key3': 'string', 'value2': 'double'})
        table3 = ibis.table({'key4': 'string', 'value3': 'double'})

        joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
        joined2 = joined.inner_join(table3, [table1['key2'] == table3['key4']])

        # it works, what more should we test here?
        materialized = joined2.materialize()
        repr(materialized)

    def test_filter_on_projected_field(self):
        # See #173. Impala and other SQL engines do not allow filtering on a
        # just-created alias in a projection
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

        fields_of_interest = [customer,
                              region.r_name.name('region'),
                              orders.o_totalprice.name('amount'),
                              orders.o_orderdate
                              .cast('timestamp').name('odate')]

        all_join = (
            region.join(nation, region.r_regionkey == nation.n_regionkey)
            .join(customer, customer.c_nationkey == nation.n_nationkey)
            .join(orders, orders.o_custkey == customer.c_custkey))

        tpch = all_join[fields_of_interest]

        # Correlated subquery, yikes!
        t2 = tpch.view()
        conditional_avg = t2[(t2.region == tpch.region)].amount.mean()

        # `amount` is part of the projection above as an aliased field
        amount_filter = tpch.amount > conditional_avg

        result = tpch.filter([amount_filter])

        # Now then! Predicate pushdown here is inappropriate, so we check that
        # it didn't occur.
        assert isinstance(result.op(), ops.Selection)
        assert result.op().table is tpch

    def test_bad_join_predicate_raises(self):
        # Join predicate references a derived table, but we can salvage and
        # rewrite it to get the join semantics out
        # see ibis #74
        table = ibis.table([
            ('c', 'int32'),
            ('f', 'double'),
            ('g', 'string')
        ], 'foo_table')

        table2 = ibis.table([
            ('key', 'string'),
            ('value', 'double')
        ], 'bar_table')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]

        with self.assertRaises(com.ExpressionError):
            table.inner_join(table2, [table3['g'] == table2['key']])

        # expected = table.inner_join(table2, [table['g'] == table2['key']])
        # assert_equal(result, expected)

    def test_filter_self_join(self):
        # GH #667
        purchases = ibis.table([('region', 'string'),
                                ('kind', 'string'),
                                ('user', 'int64'),
                                ('amount', 'double')], 'purchases')

        metric = purchases.amount.sum().name('total')
        agged = (purchases.group_by(['region', 'kind'])
                 .aggregate(metric))

        left = agged[agged.kind == 'foo']
        right = agged[agged.kind == 'bar']

        cond = left.region == right.region
        joined = left.join(right, cond)

        # unmodified by analysis
        assert_equal(joined.op().predicates[0], cond)

        metric = (left.total - right.total).name('diff')
        what = [left.region, metric]
        projected = joined.projection(what)

        proj_exprs = projected.op().selections

        # proj exprs unaffected by analysis
        assert_equal(proj_exprs[0], left.region)
        assert_equal(proj_exprs[1], metric)

    # def test_fuse_filter_projection(self):
    #     data = ibis.table([('kind', 'string'),
    #                        ('year', 'int64')], 'data')

    #     pred = data.year == 2010

    #     result = data.projection(['kind'])[pred]
    #     expected = data.filter(pred).kind

    #     assert isinstance(result, ops.Selection)
    #     assert result.equals(expected)

    def test_fuse_projection_sort_by(self):
        pass

    def test_fuse_filter_sort_by(self):
        pass

    # Refactoring deadpool

    def test_no_rewrite(self):
        table = self.con.table('test1')

        # Substitution not fully possible if we depend on a new expr in a
        # projection
        table4 = table[['c', (table['c'] * 2).name('foo')]]
        expr = table4['c'] == table4['foo']
        result = L.substitute_parents(expr)
        expected = table['c'] == table4['foo']
        assert_equal(result, expected)

    # def test_projection_with_join_pushdown_rewrite_refs(self):
    #     # Observed this expression IR issue in a TopK-rewrite context
    #     table1 = ibis.table([
    #         ('a_key1', 'string'),
    #         ('a_key2', 'string'),
    #         ('a_value', 'double')
    #     ], 'foo')

    #     table2 = ibis.table([
    #         ('b_key1', 'string'),
    #         ('b_name', 'string'),
    #         ('b_value', 'double')
    #     ], 'bar')

    #     table3 = ibis.table([
    #         ('c_key2', 'string'),
    #         ('c_name', 'string')
    #     ], 'baz')

    #     proj = (table1.inner_join(table2, [('a_key1', 'b_key1')])
    #             .inner_join(table3, [(table1.a_key2, table3.c_key2)])
    #             [table1, table2.b_name.name('b'), table3.c_name.name('c'),
    #              table2.b_value])

    #     cases = [
    #         (proj.a_value > 0, table1.a_value > 0),
    #         (proj.b_value > 0, table2.b_value > 0)
    #     ]

    #     for higher_pred, lower_pred in cases:
    #         result = proj.filter([higher_pred])
    #         op = result.op()
    #         assert isinstance(op, ops.Selection)
    #         new_pred = op.predicates[0]
    #         assert_equal(new_pred, lower_pred)

    # def test_rewrite_expr_with_parent(self):
    #     table = self.con.table('test1')

    #     table2 = table[table['f'] > 0]

    #     expr = table2['c'] == 2

    #     result = L.substitute_parents(expr)
    #     expected = table['c'] == 2
    #     assert_equal(result, expected)

    # def test_rewrite_distinct_but_equal_objects(self):
    #     t = self.con.table('test1')
    #     t_copy = self.con.table('test1')

    #     table2 = t[t_copy['f'] > 0]

    #     expr = table2['c'] == 2

    #     result = L.substitute_parents(expr)
    #     expected = t['c'] == 2
    #     assert_equal(result, expected)


def test_join_table_choice():
    # GH807
    x = ibis.table(ibis.schema([('n', 'int64')]), 'x')
    t = x.aggregate(cnt=x.n.count())
    predicate = t.cnt > 0
    assert L.sub_for(predicate, [(t, t.op().table)]).equals(predicate)
