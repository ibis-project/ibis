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

from ibis.expr.types import ArrayExpr, TableExpr, RelationError
from ibis.common import ExpressionError
from ibis.expr.datatypes import array_type
import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis

from ibis.compat import unittest
from ibis.expr.tests.mocks import MockConnection, BasicTestCase

import ibis.common as com
import ibis.config as config


from ibis.tests.util import assert_equal


class TestTableExprBasics(BasicTestCase, unittest.TestCase):

    def test_empty_schema(self):
        table = api.table([], 'foo')
        assert len(table.schema()) == 0

    def test_columns(self):
        t = self.con.table('alltypes')
        result = t.columns
        expected = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        assert result == expected

    def test_view_new_relation(self):
        # For assisting with self-joins and other self-referential operations
        # where we need to be able to treat instances of the same TableExpr as
        # semantically distinct
        #
        # This thing is not exactly a projection, since it has no semantic
        # meaning when it comes to execution
        tview = self.table.view()

        roots = tview._root_tables()
        assert len(roots) == 1
        assert roots[0] is tview.op()

    def test_get_type(self):
        for k, v in self.schema_dict.items():
            assert self.table._get_type(k) == v

    def test_getitem_column_select(self):
        for k, v in self.schema_dict.items():
            col = self.table[k]

            # Make sure it's the right type
            assert isinstance(col, ArrayExpr)
            assert isinstance(col, array_type(v))

            # Ensure we have a field selection with back-reference to the table
            parent = col.parent()
            assert isinstance(parent, ops.TableColumn)
            assert parent.parent() is self.table

    def test_getitem_attribute(self):
        result = self.table.a
        assert_equal(result, self.table['a'])

        assert 'a' in dir(self.table)

        # Project and add a name that conflicts with a TableExpr built-in
        # attribute
        view = self.table[[self.table, self.table['a'].name('schema')]]
        assert not isinstance(view.schema, ArrayExpr)

    def test_projection(self):
        cols = ['f', 'a', 'h']

        proj = self.table[cols]
        assert isinstance(proj, TableExpr)
        assert isinstance(proj.op(), ops.Selection)

        assert proj.schema().names == cols
        for c in cols:
            expr = proj[c]
            assert isinstance(expr, type(self.table[c]))

    def test_projection_no_list(self):
        expr = (self.table.f * 2).name('bar')
        result = self.table.select(expr)
        expected = self.table.projection([expr])
        assert_equal(result, expected)

    def test_projection_with_exprs(self):
        # unnamed expr to test
        mean_diff = (self.table['a'] - self.table['c']).mean()

        col_exprs = [self.table['b'].log().name('log_b'),
                     mean_diff.name('mean_diff')]

        proj = self.table[col_exprs + ['g']]
        schema = proj.schema()
        assert schema.names == ['log_b', 'mean_diff', 'g']
        assert schema.types == ['double', 'double', 'string']

        # Test with unnamed expr
        self.assertRaises(ExpressionError, self.table.projection,
                          ['g', self.table['a'] - self.table['c']])

    def test_projection_duplicate_names(self):
        self.assertRaises(com.IntegrityError, self.table.projection,
                          [self.table.c, self.table.c])

    def test_projection_invalid_root(self):
        schema1 = {
            'foo': 'double',
            'bar': 'int32'
        }

        left = api.table(schema1, name='foo')
        right = api.table(schema1, name='bar')

        exprs = [right['foo'], right['bar']]
        self.assertRaises(RelationError, left.projection, exprs)

    def test_projection_unnamed_literal_interactive_blowup(self):
        # #147 and #153 alike
        table = self.con.table('functional_alltypes')

        with config.option_context('interactive', True):
            try:
                table.select([table.bigint_col, ibis.literal(5)])
            except Exception as e:
                assert 'named' in e.args[0]

    def test_projection_of_aggregated(self):
        # Fully-formed aggregations "block"; in a projection, column
        # expressions referencing table expressions below the aggregation are
        # invalid.
        pass

    def test_projection_with_star_expr(self):
        new_expr = (self.table['a'] * 5).name('bigger_a')

        t = self.table

        # it lives!
        proj = t[t, new_expr]
        repr(proj)

        ex_names = self.table.schema().names + ['bigger_a']
        assert proj.schema().names == ex_names

        # cannot pass an invalid table expression
        t2 = t.aggregate([t['a'].sum().name('sum(a)')], by=['g'])
        self.assertRaises(RelationError, t.__getitem__, [t2])

        # TODO: there may be some ways this can be invalid

    def test_projection_convenient_syntax(self):
        proj = self.table[self.table, self.table['a'].name('foo')]
        proj2 = self.table[[self.table, self.table['a'].name('foo')]]
        assert_equal(proj, proj2)

    def test_projection_mutate_analysis_bug(self):
        # GH #549

        t = self.con.table('airlines')

        filtered = t[t.depdelay.notnull()]
        leg = ibis.literal('-').join([t.origin, t.dest])
        mutated = filtered.mutate(leg=leg)

        # it works!
        mutated['year', 'month', 'day', 'depdelay', 'leg']

    def test_projection_self(self):
        result = self.table[self.table]
        expected = self.table.projection(self.table)

        assert_equal(result, expected)

    def test_projection_array_expr(self):
        result = self.table[self.table.a]
        expected = self.table[[self.table.a]]
        assert_equal(result, expected)

    def test_add_column(self):
        # Creates a projection with a select-all on top of a non-projection
        # TableExpr
        new_expr = (self.table['a'] * 5).name('bigger_a')

        t = self.table

        result = t.add_column(new_expr)
        expected = t[[t, new_expr]]
        assert_equal(result, expected)

        result = t.add_column(new_expr, 'wat')
        expected = t[[t, new_expr.name('wat')]]
        assert_equal(result, expected)

    def test_add_column_scalar_expr(self):
        # Check literals, at least
        pass

    def test_add_column_aggregate_crossjoin(self):
        # A new column that depends on a scalar value produced by this or some
        # other table.
        #
        # For example:
        # SELECT *, b - VAL
        # FROM table1
        #
        # Here, VAL could be something produced by aggregating table1 or any
        # other table for that matter.
        pass

    def test_add_column_existing_projection(self):
        # The "blocking" predecessor table is a projection; we can simply add
        # the column to the existing projection
        foo = (self.table.f * 2).name('foo')
        bar = (self.table.f * 4).name('bar')
        t2 = self.table.add_column(foo)
        t3 = t2.add_column(bar)

        expected = self.table[self.table, foo, bar]
        assert_equal(t3, expected)

    def test_mutate(self):
        one = self.table.f * 2
        foo = (self.table.a + self.table.b).name('foo')

        expr = self.table.mutate(foo, one=one, two=2)
        expected = self.table[self.table, foo, one.name('one'),
                              ibis.literal(2).name('two')]
        assert_equal(expr, expected)

    def test_mutate_alter_existing_columns(self):
        new_f = self.table.f * 2
        foo = self.table.d * 2
        expr = self.table.mutate(f=new_f, foo=foo)

        expected = self.table['a', 'b', 'c', 'd', 'e',
                              new_f.name('f'), 'g', 'h',
                              foo.name('foo')]

        assert_equal(expr, expected)

    def test_replace_column(self):
        tb = api.table([
            ('a', 'int32'),
            ('b', 'double'),
            ('c', 'string')
        ])

        expr = tb.b.cast('int32')
        tb2 = tb.set_column('b', expr)
        expected = tb[tb.a, expr.name('b'), tb.c]

        assert_equal(tb2, expected)

    def test_filter_no_list(self):
        pred = self.table.a > 5

        result = self.table.filter(pred)
        expected = self.table[pred]
        assert_equal(result, expected)

    def test_add_predicate(self):
        pred = self.table['a'] > 5
        result = self.table[pred]
        assert isinstance(result.op(), ops.Selection)

    def test_invalid_predicate(self):
        # a lookalike
        table2 = api.table(self.schema, name='bar')
        self.assertRaises(RelationError, self.table.__getitem__,
                          table2['a'] > 5)

    def test_add_predicate_coalesce(self):
        # Successive predicates get combined into one rather than nesting. This
        # is mainly to enhance readability since we could handle this during
        # expression evaluation anyway.
        pred1 = self.table['a'] > 5
        pred2 = self.table['b'] > 0

        result = self.table[pred1][pred2]
        expected = self.table.filter([pred1, pred2])
        assert_equal(result, expected)

        # 59, if we are not careful, we can obtain broken refs
        interm = self.table[pred1]
        result = interm.filter([interm['b'] > 0])
        assert_equal(result, expected)

    def test_repr_same_but_distinct_objects(self):
        t = self.con.table('test1')
        t_copy = self.con.table('test1')
        table2 = t[t_copy['f'] > 0]

        result = repr(table2)
        assert result.count('DatabaseTable') == 1

    def test_filter_fusion_distinct_table_objects(self):
        t = self.con.table('test1')
        tt = self.con.table('test1')

        expr = t[t.f > 0][t.c > 0]
        expr2 = t[t.f > 0][tt.c > 0]
        expr3 = t[tt.f > 0][tt.c > 0]
        expr4 = t[tt.f > 0][t.c > 0]

        assert_equal(expr, expr2)
        assert repr(expr) == repr(expr2)
        assert_equal(expr, expr3)
        assert_equal(expr, expr4)

    def test_column_relabel(self):
        # GH #551. Keeping the test case very high level to not presume that
        # the relabel is necessarily implemented using a projection
        types = ['int32', 'string', 'double']
        table = api.table(zip(['foo', 'bar', 'baz'], types))
        result = table.relabel({'foo': 'one', 'baz': 'three'})

        schema = result.schema()
        ex_schema = api.schema(zip(['one', 'bar', 'three'], types))
        assert_equal(schema, ex_schema)

    def test_limit(self):
        limited = self.table.limit(10, offset=5)
        assert limited.op().n == 10
        assert limited.op().offset == 5

    def test_sort_by(self):
        # Commit to some API for ascending and descending
        #
        # table.sort_by(['g', expr1, desc(expr2), desc(expr3)])
        #
        # Default is ascending for anything coercable to an expression,
        # and we'll have ascending/descending wrappers to help.
        result = self.table.sort_by(['f'])

        sort_key = result.op().sort_keys[0].op()

        assert_equal(sort_key.expr, self.table.f)
        assert sort_key.ascending

        # non-list input. per #150
        result2 = self.table.sort_by('f')
        assert_equal(result, result2)

        result2 = self.table.sort_by([('f', False)])
        result3 = self.table.sort_by([('f', 'descending')])
        result4 = self.table.sort_by([('f', 0)])

        key2 = result2.op().sort_keys[0].op()
        key3 = result3.op().sort_keys[0].op()
        key4 = result4.op().sort_keys[0].op()

        assert not key2.ascending
        assert not key3.ascending
        assert not key4.ascending
        assert_equal(result2, result3)

    def test_sort_by_desc_deferred_sort_key(self):
        result = (self.table.group_by('g')
                  .size()
                  .sort_by(ibis.desc('count')))

        tmp = self.table.group_by('g').size()
        expected = tmp.sort_by((tmp['count'], False))
        expected2 = tmp.sort_by(ibis.desc(tmp['count']))

        assert_equal(result, expected)
        assert_equal(result, expected2)

    def test_slice_convenience(self):
        expr = self.table[:5]
        expr2 = self.table[:5:1]
        assert_equal(expr, self.table.limit(5))
        assert_equal(expr, expr2)

        expr = self.table[2:7]
        expr2 = self.table[2:7:1]
        assert_equal(expr, self.table.limit(5, offset=2))
        assert_equal(expr, expr2)

        self.assertRaises(ValueError, self.table.__getitem__, slice(2, 15, 2))
        self.assertRaises(ValueError, self.table.__getitem__, slice(5, None))
        self.assertRaises(ValueError, self.table.__getitem__, slice(None, -5))
        self.assertRaises(ValueError, self.table.__getitem__, slice(-10, -5))


class TestAggregation(BasicTestCase, unittest.TestCase):

    def test_count(self):
        result = self.table['a'].count()
        assert isinstance(result, api.Int64Scalar)
        assert isinstance(result.op(), ops.Count)

    def test_table_count(self):
        result = self.table.count()
        assert isinstance(result, api.Int64Scalar)
        assert isinstance(result.op(), ops.Count)
        assert result.get_name() == 'count'

    def test_len_raises_expression_error(self):
        with self.assertRaises(com.ExpressionError):
            len(self.table)

    def test_sum_expr_basics(self):
        # Impala gives bigint for all integer types
        ex_class = api.Int64Scalar
        for c in self.int_cols + self.bool_cols:
            result = self.table[c].sum()
            assert isinstance(result, ex_class)
            assert isinstance(result.op(), ops.Sum)

        # Impala gives double for all floating point types
        ex_class = api.DoubleScalar
        for c in self.float_cols:
            result = self.table[c].sum()
            assert isinstance(result, ex_class)
            assert isinstance(result.op(), ops.Sum)

    def test_mean_expr_basics(self):
        cols = self.int_cols + self.float_cols + self.bool_cols
        for c in cols:
            result = self.table[c].mean()
            assert isinstance(result, api.DoubleScalar)
            assert isinstance(result.op(), ops.Mean)

    def test_aggregate_no_keys(self):
        agg_exprs = [self.table['a'].sum().name('sum(a)'),
                     self.table['c'].mean().name('mean(c)')]

        # A TableExpr, which in SQL at least will yield a table with a single
        # row
        result = self.table.aggregate(agg_exprs)
        assert isinstance(result, TableExpr)

    def test_aggregate_keys_basic(self):
        agg_exprs = [self.table['a'].sum().name('sum(a)'),
                     self.table['c'].mean().name('mean(c)')]

        # A TableExpr, which in SQL at least will yield a table with a single
        # row
        result = self.table.aggregate(agg_exprs, by=['g'])
        assert isinstance(result, TableExpr)

        # it works!
        repr(result)

    def test_aggregate_non_list_inputs(self):
        # per #150
        metric = self.table.f.sum().name('total')
        by = 'g'
        having = self.table.c.sum() > 10

        result = self.table.aggregate(metric, by=by, having=having)
        expected = self.table.aggregate([metric], by=[by], having=[having])
        assert_equal(result, expected)

    def test_aggregate_keywords(self):
        t = self.table

        expr = t.aggregate(foo=t.f.sum(), bar=lambda x: x.f.mean(),
                           by='g')
        expr2 = t.group_by('g').aggregate(foo=t.f.sum(),
                                          bar=lambda x: x.f.mean())
        expected = t.aggregate([t.f.mean().name('bar'),
                                t.f.sum().name('foo')], by='g')

        assert_equal(expr, expected)
        assert_equal(expr2, expected)

    def test_groupby_alias(self):
        t = self.table

        result = t.groupby('g').size()
        expected = t.group_by('g').size()
        assert_equal(result, expected)

    def test_summary_expand_list(self):
        summ = self.table.f.summary()

        metric = self.table.g.group_concat().name('bar')
        result = self.table.aggregate([metric, summ])
        expected = self.table.aggregate([metric] + summ.exprs())
        assert_equal(result, expected)

    def test_aggregate_invalid(self):
        # Pass a non-aggregation or non-scalar expr
        pass

    def test_filter_aggregate_pushdown_predicate(self):
        # In the case where we want to add a predicate to an aggregate
        # expression after the fact, rather than having to backpedal and add it
        # before calling aggregate.
        #
        # TODO (design decision): This could happen automatically when adding a
        # predicate originating from the same root table; if an expression is
        # created from field references from the aggregated table then it
        # becomes a filter predicate applied on top of a view

        pred = self.table.f > 0
        metrics = [self.table.a.sum().name('total')]
        agged = self.table.aggregate(metrics, by=['g'])
        filtered = agged.filter([pred])
        expected = self.table[pred].aggregate(metrics, by=['g'])
        assert_equal(filtered, expected)

    def test_filter_aggregate_partial_pushdown(self):
        pass

    def test_aggregate_post_predicate(self):
        # Test invalid having clause
        metrics = [self.table.f.sum().name('total')]
        by = ['g']

        invalid_having_cases = [
            self.table.f.sum(),
            self.table.f > 2
        ]
        for case in invalid_having_cases:
            self.assertRaises(com.ExpressionError, self.table.aggregate,
                              metrics, by=by, having=[case])

    def test_group_by_having_api(self):
        # #154, add a HAVING post-predicate in a composable way
        metric = self.table.f.sum().name('foo')
        postp = self.table.d.mean() > 1

        expr = (self.table
                .group_by('g')
                .having(postp)
                .aggregate(metric))

        expected = self.table.aggregate(metric, by='g', having=postp)
        assert_equal(expr, expected)

    def test_group_by_kwargs(self):
        t = self.table
        expr = (t.group_by(['f', t.h], z='g', z2=t.d)
                 .aggregate(t.d.mean().name('foo')))
        expected = (t.group_by(['f', t.h, t.g.name('z'), t.d.name('z2')])
                    .aggregate(t.d.mean().name('foo')))
        assert_equal(expr, expected)

    def test_aggregate_root_table_internal(self):
        pass

    def test_compound_aggregate_expr(self):
        # See ibis #24
        compound_expr = (self.table['a'].sum() /
                         self.table['a'].mean()).name('foo')
        assert ops.is_reduction(compound_expr)

        # Validates internally
        self.table.aggregate([compound_expr])

    def test_groupby_convenience(self):
        metrics = [self.table.f.sum().name('total')]

        expr = self.table.group_by('g').aggregate(metrics)
        expected = self.table.aggregate(metrics, by=['g'])
        assert_equal(expr, expected)

        group_expr = self.table.g.cast('double').name('g')
        expr = self.table.group_by(group_expr).aggregate(metrics)
        expected = self.table.aggregate(metrics, by=[group_expr])
        assert_equal(expr, expected)

    def test_group_by_count_size(self):
        # #148, convenience for interactive use, and so forth
        result1 = self.table.group_by('g').size()
        result2 = self.table.group_by('g').count()

        expected = (self.table.group_by('g')
                    .aggregate([self.table.count().name('count')]))

        assert_equal(result1, expected)
        assert_equal(result2, expected)

        result = self.table.group_by('g').count('foo')
        expected = (self.table.group_by('g')
                    .aggregate([self.table.count().name('foo')]))
        assert_equal(result, expected)

    def test_group_by_column_select_api(self):
        grouped = self.table.group_by('g')

        result = grouped.f.sum()
        expected = grouped.aggregate(self.table.f.sum().name('sum(f)'))
        assert_equal(result, expected)

        supported_functions = ['sum', 'mean', 'count', 'size', 'max', 'min']

        # make sure they all work
        for fn in supported_functions:
            getattr(grouped.f, fn)()

    def test_value_counts_convenience(self):
        # #152
        result = self.table.g.value_counts()
        expected = (self.table.group_by('g')
                    .aggregate(self.table.count().name('count')))

        assert_equal(result, expected)

    def test_isin_value_counts(self):
        # #157, this code path was untested before
        bool_clause = self.table.g.notin(['1', '4', '7'])
        # it works!
        bool_clause.name('notin').value_counts()

    def test_value_counts_unnamed_expr(self):
        nation = self.con.table('tpch_nation')

        expr = nation.n_name.lower().value_counts()
        expected = nation.n_name.lower().name('unnamed').value_counts()
        assert_equal(expr, expected)

    def test_aggregate_unnamed_expr(self):
        nation = self.con.table('tpch_nation')
        expr = nation.n_name.lower().left(1)
        self.assertRaises(com.ExpressionError, nation.group_by(expr).aggregate,
                          nation.count().name('metric'))

    def test_default_reduction_names(self):
        d = self.table.f
        cases = [
            (d.count(), 'count'),
            (d.sum(), 'sum'),
            (d.mean(), 'mean'),
            (d.approx_nunique(), 'approx_nunique'),
            (d.approx_median(), 'approx_median'),
            (d.min(), 'min'),
            (d.max(), 'max')
        ]

        for expr, ex_name in cases:
            assert expr.get_name() == ex_name


class TestJoinsUnions(BasicTestCase, unittest.TestCase):

    def test_join_no_predicate_list(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        pred = region.r_regionkey == nation.n_regionkey
        joined = region.inner_join(nation, pred)
        expected = region.inner_join(nation, [pred])
        assert_equal(joined, expected)

    def test_equijoin_schema_merge(self):
        table1 = ibis.table([('key1',  'string'), ('value1', 'double')])
        table2 = ibis.table([('key2',  'string'), ('stuff', 'int32')])

        pred = table1['key1'] == table2['key2']
        join_types = ['inner_join', 'left_join', 'outer_join']

        ex_schema = api.Schema(['key1', 'value1', 'key2', 'stuff'],
                               ['string', 'double', 'string', 'int32'])

        for fname in join_types:
            f = getattr(table1, fname)
            joined = f(table2, [pred]).materialize()
            assert_equal(joined.schema(), ex_schema)

    def test_join_combo_with_projection(self):
        # Test a case where there is column name overlap, but the projection
        # passed makes it a non-issue. Highly relevant with self-joins
        #
        # For example, where left/right have some field names in common:
        # SELECT left.*, right.a, right.b
        # FROM left join right on left.key = right.key
        t = self.table
        t2 = t.add_column(t['f'] * 2, 'foo')
        t2 = t2.add_column(t['f'] * 4, 'bar')

        # this works
        joined = t.left_join(t2, [t['g'] == t2['g']])
        proj = joined.projection([t, t2['foo'], t2['bar']])
        repr(proj)

    def test_join_getitem_projection(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        pred = region.r_regionkey == nation.n_regionkey
        joined = region.inner_join(nation, pred)

        result = joined[nation]
        expected = joined.projection(nation)
        assert_equal(result, expected)

    def test_self_join(self):
        # Self-joins are problematic with this design because column
        # expressions may reference either the left or right self. For example:
        #
        # SELECT left.key, sum(left.value - right.value) as total_deltas
        # FROM table left
        #  INNER JOIN table right
        #    ON left.current_period = right.previous_period + 1
        # GROUP BY 1
        #
        # One way around the self-join issue is to force the user to add
        # prefixes to the joined fields, then project using those. Not that
        # satisfying, though.
        left = self.table
        right = self.table.view()
        metric = (left['a'] - right['b']).mean().name('metric')

        joined = left.inner_join(right, [right['g'] == left['g']])
        # basic check there's no referential problems
        result_repr = repr(joined)
        assert 'ref_0' in result_repr
        assert 'ref_1' in result_repr

        # Cannot be immediately materialized because of the schema overlap
        self.assertRaises(RelationError, joined.materialize)

        # Project out left table schema
        proj = joined[[left]]
        assert_equal(proj.schema(), left.schema())

        # Try aggregating on top of joined
        aggregated = joined.aggregate([metric], by=[left['g']])
        ex_schema = api.Schema(['g', 'metric'], ['string', 'double'])
        assert_equal(aggregated.schema(), ex_schema)

    def test_self_join_no_view_convenience(self):
        # #165, self joins ought to be possible when the user specifies the
        # column names to join on rather than referentially-valid expressions

        result = self.table.join(self.table, [('g', 'g')])

        t2 = self.table.view()
        expected = self.table.join(t2, self.table.g == t2.g)
        assert_equal(result, expected)

    def test_materialized_join_reference_bug(self):
        # GH#403
        orders = self.con.table('tpch_orders')
        customer = self.con.table('tpch_customer')
        lineitem = self.con.table('tpch_lineitem')

        items = (orders
                 .join(lineitem, orders.o_orderkey == lineitem.l_orderkey)
                 [lineitem, orders.o_custkey, orders.o_orderpriority]
                 .join(customer, [('o_custkey', 'c_custkey')])
                 .materialize())
        items['o_orderpriority'].value_counts()

    def test_join_project_after(self):
        # e.g.
        #
        # SELECT L.foo, L.bar, R.baz, R.qux
        # FROM table1 L
        #   INNER JOIN table2 R
        #     ON L.key = R.key
        #
        # or
        #
        # SELECT L.*, R.baz
        # ...
        #
        # The default for a join is selecting all fields if possible
        table1 = ibis.table([('key1',  'string'), ('value1', 'double')])
        table2 = ibis.table([('key2',  'string'), ('stuff', 'int32')])

        pred = table1['key1'] == table2['key2']

        joined = table1.left_join(table2, [pred])
        projected = joined.projection([table1, table2['stuff']])
        assert projected.schema().names == ['key1', 'value1', 'stuff']

        projected = joined.projection([table2, table1['key1']])
        assert projected.schema().names == ['key2', 'stuff', 'key1']

    def test_semi_join_schema(self):
        # A left semi join discards the schema of the right table
        table1 = ibis.table([('key1',  'string'), ('value1', 'double')])
        table2 = ibis.table([('key2',  'string'), ('stuff', 'double')])

        pred = table1['key1'] == table2['key2']
        semi_joined = table1.semi_join(table2, [pred]).materialize()

        result_schema = semi_joined.schema()
        assert_equal(result_schema, table1.schema())

    def test_cross_join(self):
        agg_exprs = [self.table['a'].sum().name('sum_a'),
                     self.table['b'].mean().name('mean_b')]
        scalar_aggs = self.table.aggregate(agg_exprs)

        joined = self.table.cross_join(scalar_aggs).materialize()
        agg_schema = api.Schema(['sum_a', 'mean_b'], ['int64', 'double'])
        ex_schema = self.table.schema().append(agg_schema)
        assert_equal(joined.schema(), ex_schema)

    def test_cross_join_multiple(self):
        a = self.table['a', 'b', 'c']
        b = self.table['d', 'e']
        c = self.table['f', 'h']

        joined = ibis.cross_join(a, b, c)
        expected = a.cross_join(b.cross_join(c))
        assert joined.equals(expected)

    def test_join_compound_boolean_predicate(self):
        # The user might have composed predicates through logical operations
        pass

    def test_filter_join_unmaterialized(self):
        table1 = ibis.table({'key1': 'string', 'key2': 'string',
                            'value1': 'double'})
        table2 = ibis.table({'key3': 'string', 'value2': 'double'})

        # It works!
        joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
        filtered = joined.filter([table1.value1 > 0])
        repr(filtered)

    def test_join_overlapping_column_names(self):
        t1 = ibis.table([('foo', 'string'),
                         ('bar', 'string'),
                         ('value1', 'double')])
        t2 = ibis.table([('foo', 'string'),
                         ('bar', 'string'),
                         ('value2', 'double')])

        joined = t1.join(t2, 'foo')
        expected = t1.join(t2, t1.foo == t2.foo)
        assert_equal(joined, expected)

        joined = t1.join(t2, ['foo', 'bar'])
        expected = t1.join(t2, [t1.foo == t2.foo,
                                t1.bar == t2.bar])
        assert_equal(joined, expected)

    def test_join_key_alternatives(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        # Join with tuples
        joined = t1.inner_join(t2, [('foo_id', 'foo_id')])
        joined2 = t1.inner_join(t2, [(t1.foo_id, t2.foo_id)])

        # Join with single expr
        joined3 = t1.inner_join(t2, t1.foo_id == t2.foo_id)

        expected = t1.inner_join(t2, [t1.foo_id == t2.foo_id])

        assert_equal(joined, expected)
        assert_equal(joined2, expected)
        assert_equal(joined3, expected)

        self.assertRaises(com.ExpressionError, t1.inner_join, t2,
                          [('foo_id', 'foo_id', 'foo_id')])

    def test_join_invalid_refs(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')
        t3 = self.con.table('star3')

        predicate = t1.bar_id == t3.bar_id
        self.assertRaises(com.RelationError, t1.inner_join, t2, [predicate])

    def test_join_non_boolean_expr(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        # oops
        predicate = t1.f * t2.value1
        self.assertRaises(com.ExpressionError, t1.inner_join, t2, [predicate])

    def test_unravel_compound_equijoin(self):
        t1 = ibis.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value1', 'double')
        ], 'foo_table')

        t2 = ibis.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value2', 'double')
        ], 'bar_table')

        p1 = t1.key1 == t2.key1
        p2 = t1.key2 == t2.key2
        p3 = t1.key3 == t2.key3

        joined = t1.inner_join(t2, [p1 & p2 & p3])
        expected = t1.inner_join(t2, [p1, p2, p3])
        assert_equal(joined, expected)

    def test_join_add_prefixes(self):
        pass

    def test_join_nontrivial_exprs(self):
        pass

    def test_union(self):
        schema1 = [
            ('key', 'string'),
            ('value', 'double')
        ]
        schema2 = [
            ('key', 'string'),
            ('key2', 'string'),
            ('value', 'double')
        ]
        t1 = ibis.table(schema1, 'foo')
        t2 = ibis.table(schema1, 'bar')
        t3 = ibis.table(schema2, 'baz')

        result = t1.union(t2)
        assert isinstance(result.op(), ops.Union)
        assert not result.op().distinct

        result = t1.union(t2, distinct=True)
        assert isinstance(result.op(), ops.Union)
        assert result.op().distinct

        self.assertRaises(ir.RelationError, t1.union, t3)

    def test_column_ref_on_projection_rename(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')

        joined = (region.inner_join(
            nation, [region.r_regionkey == nation.n_regionkey])
            .inner_join(
                customer, [customer.c_nationkey == nation.n_nationkey]))

        proj_exprs = [customer, nation.n_name.name('nation'),
                      region.r_name.name('region')]
        joined = joined.projection(proj_exprs)

        metrics = [joined.c_acctbal.sum().name('metric')]

        # it works!
        joined.aggregate(metrics, by=['region'])


class TestSemiAntiJoinPredicates(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

        self.t1 = ibis.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value1', 'double')
        ], 'foo')

        self.t2 = ibis.table([
            ('key1', 'string'),
            ('key2', 'string')
        ], 'bar')

    def test_simple_existence_predicate(self):
        cond = (self.t1.key1 == self.t2.key1).any()

        assert isinstance(cond, ir.BooleanArray)
        op = cond.op()
        assert isinstance(op, ops.Any)

        # it works!
        expr = self.t1[cond]
        assert isinstance(expr.op(), ops.Selection)

    def test_cannot_use_existence_expression_in_join(self):
        # Join predicates must consist only of comparisons
        pass

    def test_not_exists_predicate(self):
        cond = -(self.t1.key1 == self.t2.key1).any()
        assert isinstance(cond.op(), ops.NotAny)


class TestLateBindingFunctions(BasicTestCase, unittest.TestCase):

    def test_aggregate_metrics(self):
        functions = [lambda x: x.e.sum().name('esum'),
                     lambda x: x.f.sum().name('fsum')]
        exprs = [self.table.e.sum().name('esum'),
                 self.table.f.sum().name('fsum')]

        result = self.table.aggregate(functions[0])
        expected = self.table.aggregate(exprs[0])
        assert_equal(result, expected)

        result = self.table.aggregate(functions)
        expected = self.table.aggregate(exprs)
        assert_equal(result, expected)

    def test_group_by_keys(self):
        m = self.table.mutate(foo=self.table.f * 2,
                              bar=self.table.e / 2)

        expr = m.group_by(lambda x: x.foo).size()
        expected = m.group_by('foo').size()
        assert_equal(expr, expected)

        expr = m.group_by([lambda x: x.foo, lambda x: x.bar]).size()
        expected = m.group_by(['foo', 'bar']).size()
        assert_equal(expr, expected)

    def test_having(self):
        m = self.table.mutate(foo=self.table.f * 2,
                              bar=self.table.e / 2)

        expr = (m.group_by('foo')
                .having(lambda x: x.foo.sum() > 10)
                .size())
        expected = (m.group_by('foo')
                    .having(m.foo.sum() > 10)
                    .size())

        assert_equal(expr, expected)

    def test_filter(self):
        m = self.table.mutate(foo=self.table.f * 2,
                              bar=self.table.e / 2)

        result = m.filter(lambda x: x.foo > 10)
        result2 = m[lambda x: x.foo > 10]
        expected = m[m.foo > 10]

        assert_equal(result, expected)
        assert_equal(result2, expected)

        result = m.filter([lambda x: x.foo > 10,
                           lambda x: x.bar < 0])
        expected = m.filter([m.foo > 10, m.bar < 0])
        assert_equal(result, expected)

    def test_sort_by(self):
        m = self.table.mutate(foo=self.table.e + self.table.f)

        result = m.sort_by(lambda x: -x.foo)
        expected = m.sort_by(-m.foo)
        assert_equal(result, expected)

        result = m.sort_by(lambda x: ibis.desc(x.foo))
        expected = m.sort_by(ibis.desc('foo'))
        assert_equal(result, expected)

        result = m.sort_by(ibis.desc(lambda x: x.foo))
        expected = m.sort_by(ibis.desc('foo'))
        assert_equal(result, expected)

    def test_projection(self):
        m = self.table.mutate(foo=self.table.f * 2)

        def f(x):
            return (x.foo * 2).name('bar')

        result = m.projection([f, 'f'])
        result2 = m[f, 'f']
        expected = m.projection([f(m), 'f'])
        assert_equal(result, expected)
        assert_equal(result2, expected)

    def test_mutate(self):
        m = self.table.mutate(foo=self.table.f * 2)

        def g(x):
            return x.foo * 2

        def h(x):
            return x.bar * 2

        result = m.mutate(bar=g).mutate(baz=h)

        m2 = m.mutate(bar=g(m))
        expected = m2.mutate(baz=h(m2))

        assert_equal(result, expected)

    def test_add_column(self):
        def g(x):
            return x.f * 2

        result = self.table.add_column(g, name='foo')
        expected = self.table.mutate(foo=g)
        assert_equal(result, expected)

    def test_groupby_mutate(self):
        t = self.table

        g = t.group_by('g').order_by('f')
        expr = g.mutate(foo=lambda x: x.f.lag(),
                        bar=lambda x: x.f.rank())
        expected = g.mutate(foo=t.f.lag(),
                            bar=t.f.rank())

        assert_equal(expr, expected)

    def test_groupby_projection(self):
        t = self.table

        g = t.group_by('g').order_by('f')
        expr = g.projection([lambda x: x.f.lag().name('foo'),
                             lambda x: x.f.rank().name('bar')])
        expected = g.projection([t.f.lag().name('foo'),
                                 t.f.rank().name('bar')])

        assert_equal(expr, expected)

    def test_set_column(self):
        def g(x):
            return x.f * 2

        result = self.table.set_column('f', g)
        expected = self.table.set_column('f', self.table.f * 2)
        assert_equal(result, expected)
