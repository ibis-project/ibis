# copyright 2014 Cloudera Inc.
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

import itertools
import operator
import unittest

from ibis.expr.types import ArrayExpr, TableExpr, RelationError
from ibis.common import ExpressionError
import ibis.expr.analysis as L
import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis

from ibis.expr.format import ExprFormatter
from ibis.expr.tests.mocks import MockConnection

import ibis.common as com
import ibis.config as config


class TestParameters(unittest.TestCase):
    pass


class TestSchema(unittest.TestCase):
    pass


class TestLiterals(unittest.TestCase):

    def test_null(self):
        expr = api.literal(None)
        assert isinstance(expr, api.NullScalar)
        assert isinstance(expr.op(), ops.NullLiteral)

        expr2 = api.null()
        assert expr.equals(expr2)

    def test_boolean(self):
        val = True
        expr = api.literal(val)
        self._check_literal(expr, api.BooleanScalar, val)

        val = False
        expr = api.literal(val)
        self._check_literal(expr, api.BooleanScalar, val)

    def test_float(self):
        val = 1.5
        expr = api.literal(val)
        self._check_literal(expr, api.DoubleScalar, val)

    def test_string(self):
        val = 'foo'
        expr = api.literal(val)
        self._check_literal(expr, api.StringScalar, val)

    def _check_literal(self, expr, ex_klass, val):
        assert isinstance(expr, ex_klass)

        arg = expr.op()
        assert isinstance(arg, ops.Literal)
        assert arg.value == val

        # Console formatting works
        repr(expr)

    def test_unicode(self):
        # UTF-8 support in Impala non-existent at the moment?
        pass

    def test_int_literal_cases(self):
        cases = [
            (5, 'int8'),
            (127, 'int8'),
            (128, 'int16'),
            (32767, 'int16'),
            (32768, 'int32'),
            (2147483647, 'int32'),
            (2147483648, 'int64'),
            (-5, 'int8'),
            (-128, 'int8'),
            (-129, 'int16'),
            (-32769, 'int32'),
            (-2147483649, 'int64'),
            (1.5, 'double'),
            ('foo', 'string')
        ]

        for value, ex_type in cases:
            expr = api.literal(value)
            klass = ir.scalar_type(ex_type)
            assert isinstance(expr, klass)
            assert isinstance(expr.op(), ops.Literal)
            assert expr.op().value is value

    def test_literal_list(self):
        what = [1, 2, 1000]
        expr = api.as_value_expr(what)

        assert isinstance(expr, ir.ArrayExpr)
        assert isinstance(expr.op(), ops.ValueList)
        assert isinstance(expr.op().values[2], ir.Int16Scalar)

        # it works!
        repr(expr)

    def test_mixed_arity(self):
        table = api.table(_all_types_schema)
        what = ["bar", table.g, "foo"]
        expr = api.as_value_expr(what)

        values = expr.op().values
        assert isinstance(values[1], ir.StringArray)

        # it works!
        repr(expr)


_all_types_schema = [
    ('a', 'int8'),
    ('b', 'int16'),
    ('c', 'int32'),
    ('d', 'int64'),
    ('e', 'float'),
    ('f', 'double'),
    ('g', 'string'),
    ('h', 'boolean')
]


class BasicTestCase(object):

    def setUp(self):
        self.schema = _all_types_schema
        self.schema_dict = dict(self.schema)
        self.table = api.table(self.schema)

        self.int_cols = ['a', 'b', 'c', 'd']
        self.bool_cols = ['h']
        self.float_cols = ['e', 'f']

        self.con = MockConnection()


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
        for k, v in self.schema_dict.iteritems():
            assert self.table._get_type(k) == v

    def test_getitem_column_select(self):
        for k, v in self.schema_dict.iteritems():
            col = self.table[k]

            # Make sure it's the right type
            assert isinstance(col, ArrayExpr)
            assert isinstance(col, ir.array_type(v))

            # Ensure we have a field selection with back-reference to the table
            parent = col.parent()
            assert isinstance(parent, ops.TableColumn)
            assert parent.parent() is self.table

    def test_getitem_attribute(self):
        result = self.table.a
        assert result.equals(self.table['a'])

        assert 'a' in dir(self.table)

        # Project and add a name that conflicts with a TableExpr built-in
        # attribute
        view = self.table[[self.table, self.table['a'].name('schema')]]
        assert not isinstance(view.schema, ArrayExpr)

    def test_projection(self):
        cols = ['f', 'a', 'h']

        proj = self.table[cols]
        assert isinstance(proj, TableExpr)
        assert isinstance(proj.op(), ops.Projection)

        assert proj.schema().names == cols
        for c in cols:
            expr = proj[c]
            assert type(expr) == type(self.table[c])

    def test_projection_no_list(self):
        expr = (self.table.f * 2).name('bar')
        result = self.table.select(expr)
        expected = self.table.projection([expr])
        assert result.equals(expected)

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
                          ['g', mean_diff])

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
                assert 'named' in e.message

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
        assert proj.equals(proj2)

    def test_add_column(self):
        # Creates a projection with a select-all on top of a non-projection
        # TableExpr
        new_expr = (self.table['a'] * 5).name('bigger_a')

        t = self.table

        result = t.add_column(new_expr)
        expected = t[[t, new_expr]]
        assert result.equals(expected)

        result = t.add_column(new_expr, 'wat')
        expected = t[[t, new_expr.name('wat')]]
        assert result.equals(expected)

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
        assert t3.equals(expected)

    def test_filter_no_list(self):
        pred = self.table.a > 5

        result = self.table.filter(pred)
        expected = self.table[pred]
        assert result.equals(expected)

    def test_add_predicate(self):
        pred = self.table['a'] > 5
        result = self.table[pred]
        assert isinstance(result.op(), ops.Filter)

    def test_filter_root_table_preserved(self):
        result = self.table[self.table['a'] > 5]
        roots = result.op().root_tables()
        assert roots[0] is self.table.op()

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
        assert result.equals(expected)

        # 59, if we are not careful, we can obtain broken refs
        interm = self.table[pred1]
        result = interm.filter([interm['b'] > 0])
        assert result.equals(expected)

    def test_rewrite_expr_with_parent(self):
        table = self.con.table('test1')

        table2 = table[table['f'] > 0]

        expr = table2['c'] == 2

        result = L.substitute_parents(expr)
        expected = table['c'] == 2
        assert result.equals(expected)

        # Substitution not fully possible if we depend on a new expr in a
        # projection

        table4 = table[['c', (table['c'] * 2).name('foo')]]
        expr = table4['c'] == table4['foo']
        result = L.substitute_parents(expr)
        expected = table['c'] == table4['foo']
        assert result.equals(expected)

    def test_rewrite_distinct_but_equal_objects(self):
        t = self.con.table('test1')
        t_copy = self.con.table('test1')

        table2 = t[t_copy['f'] > 0]

        expr = table2['c'] == 2

        result = L.substitute_parents(expr)
        expected = t['c'] == 2
        assert result.equals(expected)

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

        assert expr.equals(expr2)
        assert repr(expr) == repr(expr2)
        assert expr.equals(expr3)
        assert expr.equals(expr4)

    def test_rewrite_substitute_distinct_tables(self):
        t = self.con.table('test1')
        tt = self.con.table('test1')

        expr = t[t.c > 0]
        expr2 = tt[tt.c > 0]

        metric = t.f.sum().name('metric')
        expr3 = expr.aggregate(metric)

        result = L.sub_for(expr3, [(expr2, t)])
        expected = t.aggregate(metric)

        assert result.equals(expected)

    def test_rewrite_join_projection_without_other_ops(self):
        # Drop out filters and other commutative table operations. Join
        # predicates are "lifted" to reference the base, unmodified join roots

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
        assert op.table.equals(ex_expr)

        # Ensure that filtered table has been substituted with the base table
        assert op.selections[0] is table

    def test_rewrite_past_projection(self):
        table = self.con.table('test1')

        # Rewrite past a projection
        table3 = table[['c', 'f']]
        expr = table3['c'] == 2

        result = L.substitute_parents(expr)
        expected = table['c'] == 2
        assert result.equals(expected)

        # Unsafe to rewrite past projection
        table5 = table[(table.f * 2).name('c'), table.f]
        expr = table5['c'] == 2
        result = L.substitute_parents(expr)
        assert result is expr

    def test_projection_predicate_pushdown(self):
        # Probably test this during the evaluation phase. In SQL, "fusable"
        # table operations will be combined together into a single select
        # statement
        #
        # see ibis #71 for more on this
        t = self.table
        proj = t['a', 'b', 'c']

        # Rewrite a little more aggressively here
        result = proj[t.a > 0]

        # at one point these yielded different results
        filtered = t[t.a > 0]
        expected = filtered[t.a, t.b, t.c]
        expected2 = filtered.projection(['a', 'b', 'c'])

        assert result.equals(expected)
        assert result.equals(expected2)

    def test_projection_with_join_pushdown_rewrite_refs(self):
        # Observed this expression IR issue in a TopK-rewrite context
        table1 = api.table([
            ('a_key1', 'string'),
            ('a_key2', 'string'),
            ('a_value', 'double')
        ], 'foo')

        table2 = api.table([
            ('b_key1', 'string'),
            ('b_name', 'string'),
            ('b_value', 'double')
        ], 'bar')

        table3 = api.table([
            ('c_key2', 'string'),
            ('c_name', 'string')
        ], 'baz')

        proj = (table1.inner_join(table2, [('a_key1', 'b_key1')])
                .inner_join(table3, [(table1.a_key2, table3.c_key2)])
                [table1, table2.b_name.name('b'), table3.c_name.name('c'),
                 table2.b_value])

        cases = [
            (proj.a_value > 0, table1.a_value > 0),
            (proj.b_value > 0, table2.b_value > 0)
        ]

        for higher_pred, lower_pred in cases:
            result = proj.filter([higher_pred])
            op = result.op()
            assert isinstance(op, ops.Projection)
            filter_op = op.table.op()
            assert isinstance(filter_op, ops.Filter)
            new_pred = filter_op.predicates[0]
            assert new_pred.equals(lower_pred)

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
        sort_key = result.op().keys[0]
        assert sort_key.expr.equals(self.table.f)
        assert sort_key.ascending

        # non-list input. per #150
        result2 = self.table.sort_by('f')
        assert result.equals(result2)

        result2 = self.table.sort_by([('f', False)])
        result3 = self.table.sort_by([('f', 'descending')])
        result4 = self.table.sort_by([('f', 0)])

        key2 = result2.op().keys[0]
        key3 = result3.op().keys[0]
        key4 = result4.op().keys[0]

        assert not key2.ascending
        assert not key3.ascending
        assert not key4.ascending
        assert result2.equals(result3)

    def test_sort_by_desc_deferred_sort_key(self):
        result = (self.table.group_by('g')
                  .size()
                  .sort_by(ibis.desc('count')))

        tmp = self.table.group_by('g').size()
        expected = tmp.sort_by((tmp['count'], False))
        expected2 = tmp.sort_by(ibis.desc(tmp['count']))

        assert result.equals(expected)
        assert result.equals(expected2)

    def test_slice_convenience(self):
        expr = self.table[:5]
        expr2 = self.table[:5:1]
        assert expr.equals(self.table.limit(5))
        assert expr.equals(expr2)

        expr = self.table[2:7]
        expr2 = self.table[2:7:1]
        assert expr.equals(self.table.limit(5, offset=2))
        assert expr.equals(expr2)

        self.assertRaises(ValueError, self.table.__getitem__, slice(2, 15, 2))
        self.assertRaises(ValueError, self.table.__getitem__, slice(5, None))
        self.assertRaises(ValueError, self.table.__getitem__, slice(None, -5))
        self.assertRaises(ValueError, self.table.__getitem__, slice(-10, -5))


class TestContains(BasicTestCase, unittest.TestCase):

    def test_isin_notin_list(self):
        vals = [1, 2, 3]

        expr = self.table.a.isin(vals)
        not_expr = self.table.a.notin(vals)

        assert isinstance(expr, ir.BooleanArray)
        assert isinstance(expr.op(), ops.Contains)

        assert isinstance(not_expr, ir.BooleanArray)
        assert isinstance(not_expr.op(), ops.NotContains)

    def test_isin_not_comparable(self):
        pass

    def test_isin_array_expr(self):
        #
        pass

    def test_isin_invalid_cases(self):
        # For example, array expression in a list of values, where the inner
        # array values originate from some other table
        pass

    def test_isin_notin_scalars(self):
        a, b, c = [api.literal(x) for x in [1, 1, 2]]

        result = a.isin([1, 2])
        assert isinstance(result, ir.BooleanScalar)

        result = a.notin([b, c])
        assert isinstance(result, ir.BooleanScalar)

    def test_isin_null(self):
        pass

    def test_negate_isin(self):
        # Should yield a NotContains
        pass

    def test_scalar_isin_list_with_array(self):
        val = api.literal(2)

        options = [self.table.a, self.table.b, self.table.c]

        expr = val.isin(options)
        assert isinstance(expr, ir.BooleanArray)

        not_expr = val.notin(options)
        assert isinstance(not_expr, ir.BooleanArray)


class TestDistinct(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_distinct_basic(self):
        expr = self.table.distinct()
        assert isinstance(expr.op(), ops.Distinct)
        assert expr.op().table is self.table

        expr = self.table.string_col.distinct()
        assert isinstance(expr.op(), ops.DistinctArray)

        ex_projection = self.table[[self.table.string_col]]
        assert expr.op().table.op().table.equals(ex_projection)

    # def test_distinct_array_interactions(self):
    # TODO

    # array cardinalities / shapes are likely to be different.
    #     a = self.table.int_col.distinct()
    #     b = self.table.bigint_col

    #     self.assertRaises(ir.RelationError, a.__add__, b)

    def test_distinct_count(self):
        result = self.table.string_col.distinct().count()
        expected = self.table.string_col.nunique()
        assert result.equals(expected)
        assert isinstance(result.op(), ops.CountDistinct)

    def test_distinct_count_numeric_types(self):
        table = self.table
        metric = (table.bigint_col.distinct().count()
                  .name('unique_bigints'))

        table.group_by('string_col').aggregate(metric)

    def test_nunique(self):
        expr = self.table.string_col.nunique()
        assert isinstance(expr.op(), ops.CountDistinct)

    def test_project_with_distinct(self):
        pass


class TestExprFormatting(unittest.TestCase):
    # Uncertain about how much we want to commit to unit tests around the
    # particulars of the output at the moment.

    def setUp(self):
        self.schema = [
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean')
        ]
        self.schema_dict = dict(self.schema)
        self.table = api.table(self.schema)

    def test_format_projection(self):
        # This should produce a ref to the projection
        proj = self.table[['c', 'a', 'f']]
        repr(proj['a'])

    def test_table_type_output(self):
        foo = api.table(
            [
                ('job', 'string'),
                ('dept_id', 'string'),
                ('year', 'int32'),
                ('y', 'double')
            ], 'foo')

        expr = foo.dept_id == foo.view().dept_id
        result = repr(expr)
        assert 'SelfReference[table]' in result
        assert 'UnboundTable[table]' in result

    def test_memoize_aggregate_correctly(self):
        table = self.table

        agg_expr = (table['c'].sum() / table['c'].mean() - 1).name('analysis')
        agg_exprs = [table['a'].sum().name('sum(a)'),
                     table['b'].mean().name('mean(b)'), agg_expr]

        result = table.aggregate(agg_exprs, by=['g'])

        formatter = ExprFormatter(result)
        formatted = formatter.get_result()

        alias = formatter.memo.get_alias(table.op())
        assert formatted.count(alias) == 7

    def test_aggregate_arg_names(self):
        # Not sure how to test this *well*

        t = self.table

        by_exprs = [t.g.name('key1'), t.f.round().name('key2')]
        agg_exprs = [t.c.sum().name('c'), t.d.mean().name('d')]

        expr = self.table.group_by(by_exprs).aggregate(agg_exprs)
        result = repr(expr)
        assert 'metrics' in result
        assert 'by' in result

    def test_format_multiple_join_with_projection(self):
        # Star schema with fact table
        table = api.table([
            ('c', 'int32'),
            ('f', 'double'),
            ('foo_id', 'string'),
            ('bar_id', 'string'),
        ])

        table2 = api.table([
            ('foo_id', 'string'),
            ('value1', 'double')
        ])

        table3 = api.table([
            ('bar_id', 'string'),
            ('value2', 'double')
        ])

        filtered = table[table['f'] > 0]

        pred1 = table['foo_id'] == table2['foo_id']
        pred2 = filtered['bar_id'] == table3['bar_id']

        j1 = filtered.left_join(table2, [pred1])
        j2 = j1.inner_join(table3, [pred2])

        # Project out the desired fields
        view = j2[[table, table2['value1'], table3['value2']]]

        # it works!
        repr(view)

    def test_memoize_database_table(self):
        con = MockConnection()
        table = con.table('test1')
        table2 = con.table('test2')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]
        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])

        met1 = (table3['f'] - table2['value']).mean().name('foo')
        result = joined.aggregate([met1, table3['f'].sum().name('bar')],
                                  by=[table3['g'], table2['key']])

        formatted = repr(result)
        assert formatted.count('test1') == 1
        assert formatted.count('test2') == 1


class TestNullOps(BasicTestCase, unittest.TestCase):

    def test_isnull(self):
        expr = self.table['g'].isnull()
        assert isinstance(expr, api.BooleanArray)
        assert isinstance(expr.op(), ops.IsNull)

        expr = api.literal('foo').isnull()
        assert isinstance(expr, api.BooleanScalar)
        assert isinstance(expr.op(), ops.IsNull)

    def test_notnull(self):
        expr = self.table['g'].notnull()
        assert isinstance(expr, api.BooleanArray)
        assert isinstance(expr.op(), ops.NotNull)

        expr = api.literal('foo').notnull()
        assert isinstance(expr, api.BooleanScalar)
        assert isinstance(expr.op(), ops.NotNull)

    def test_null_literal(self):
        pass


class TestMathUnaryOps(BasicTestCase, unittest.TestCase):

    def test_log_variants(self):
        opnames = ['ln', 'log', 'log2', 'log10']

        columns = ['a', 'b', 'c', 'd', 'e', 'f']

        for opname in opnames:
            f = lambda x: getattr(x, opname)()

            for c in columns:
                result = f(self.table[c])
                assert isinstance(result, api.DoubleArray)

                # is this what we want?
                # assert result.get_name() == c

            assert isinstance(f(api.literal(5)), api.DoubleScalar)
            assert isinstance(f(api.literal(5.5)), api.DoubleScalar)

            klass = getattr(ops, opname.capitalize())
            self.assertRaises(TypeError, klass(self.table['g']).to_expr)

            # boolean not implemented for these
            self.assertRaises(TypeError, f, self.table['h'])

    def test_exp(self):
        pass

    def test_sqrt(self):
        pass

    def test_trig_functions(self):
        pass

    def test_round(self):
        pass


class TestTypeCasting(BasicTestCase, unittest.TestCase):

    def test_cast_same_type_noop(self):
        col = self.table['g']
        result = col.cast('string')
        assert result is col

    def test_string_to_number(self):
        types = ['int8', 'int32', 'double', 'float']

        for t in types:
            c = 'g'
            casted = self.table[c].cast(t)
            assert isinstance(casted, ir.array_type(t))
            assert casted.get_name() == c

            casted_literal = api.literal('5').name('bar').cast(t)
            assert isinstance(casted_literal, ir.scalar_type(t))
            assert casted_literal.get_name() == 'bar'

    def test_number_to_string(self):
        cols = ['a', 'b', 'c', 'd', 'e', 'f', 'h']
        for c in cols:
            casted = self.table[c].cast('string')
            assert isinstance(casted, api.StringArray)
            assert casted.get_name() == c

        casted_literal = api.literal(5).name('bar').cast('string')
        assert isinstance(casted_literal, api.StringScalar)
        assert casted_literal.get_name() == 'bar'


class TestBooleanUnaryOps(BasicTestCase, unittest.TestCase):

    def test_nonzero(self):
        pass

    def test_negate(self):
        for name in self.int_cols + self.float_cols + self.bool_cols:
            col = self.table[name]
            result = -col
            assert isinstance(result, type(col))
            assert isinstance(result.op(), ops.Negate)

        result = -api.literal(False)
        assert isinstance(result, api.BooleanScalar)
        assert isinstance(result.op(), ops.Negate)

    def test_isnull_notnull(self):
        pass


class TestComparisons(BasicTestCase, unittest.TestCase):

    def test_numbers_compare_numeric_literal(self):
        opnames = ['lt', 'gt', 'ge', 'le', 'eq', 'ne']

        ex_op_class = {
            'eq': ops.Equals,
            'ne': ops.NotEquals,
            'le': ops.LessEqual,
            'lt': ops.Less,
            'ge': ops.GreaterEqual,
            'gt': ops.Greater,
        }

        columns = ['a', 'b', 'c', 'd', 'e', 'f']

        cases = [2, 2 ** 9, 2 ** 17, 2 ** 33, 1.5]
        for opname, cname, val in itertools.product(opnames, columns, cases):
            f = getattr(operator, opname)
            col = self.table[cname]

            result = f(col, val)
            assert isinstance(result, api.BooleanArray)

            assert isinstance(result.op(), ex_op_class[opname])

    def test_boolean_comparisons(self):
        bool_col = self.table['h']

        result = bool_col == True
        assert isinstance(result, api.BooleanArray)

        result = bool_col == False
        assert isinstance(result, api.BooleanArray)

    def test_string_comparisons(self):
        string_col = self.table['g']

        ops = ['lt', 'gt', 'ge', 'le', 'eq', 'ne']

        for opname in ops:
            f = getattr(operator, opname)
            result = f(string_col, 'foo')
            assert isinstance(result, api.BooleanArray)

    def test_boolean_logical_ops(self):
        expr = self.table['a'] > 0
        ops = ['xor', 'or_', 'and_']

        for opname in ops:
            f = getattr(operator, opname)
            result = f(expr, self.table['h'])
            assert isinstance(result, api.BooleanArray)

            result = f(expr, True)
            refl_result = f(True, expr)
            assert isinstance(result, api.BooleanArray)
            assert isinstance(refl_result, api.BooleanArray)

            true = api.literal(True)
            false = api.literal(False)

            result = f(true, false)
            assert isinstance(result, api.BooleanScalar)

    def test_string_compare_numeric_array(self):
        self.assertRaises(TypeError, self.table.g.__eq__, self.table.f)
        self.assertRaises(TypeError, self.table.g.__eq__, self.table.c)

    def test_string_compare_numeric_literal(self):
        self.assertRaises(TypeError, self.table.g.__eq__, api.literal(1.5))
        self.assertRaises(TypeError, self.table.g.__eq__, api.literal(5))

    def test_between(self):
        result = self.table.f.between(0, 1)

        assert isinstance(result, ir.BooleanArray)
        assert isinstance(result.op(), ops.Between)

        # it works!
        result = self.table.g.between('a', 'f')
        assert isinstance(result, ir.BooleanArray)

        result = api.literal(1).between(self.table.a, self.table.c)
        assert isinstance(result, ir.BooleanArray)

        result = api.literal(7).between(5, 10)
        assert isinstance(result, ir.BooleanScalar)

        # Cases where between should immediately fail, e.g. incomparables
        self.assertRaises(TypeError, self.table.f.between, '0', '1')
        self.assertRaises(TypeError, self.table.f.between, 0, '1')
        self.assertRaises(TypeError, self.table.f.between, '0', 1)


class TestBinaryArithOps(BasicTestCase, unittest.TestCase):

    def test_binop_string_type_error(self):
        # Strings are not valid for any numeric arithmetic
        ints = self.table['a']
        strs = self.table['g']

        ops = ['add', 'mul', 'div', 'sub']
        for name in ops:
            f = getattr(operator, name)
            self.assertRaises(TypeError, f, ints, strs)
            self.assertRaises(TypeError, f, strs, ints)

    def test_add_literal_promotions(self):
        cases = [
            ('a', 0, 'int8'),
            ('a', 5, 'int16'),
            ('a', 100000, 'int32'),
            ('a', -100000, 'int32'),

            ('a', 1.5, 'double'),

            ('b', 0, 'int16'),
            ('b', 5, 'int32'),
            ('b', -5, 'int32'),

            ('c', 0, 'int32'),
            ('c', 5, 'int64'),
            ('c', -5, 'int64'),

            # technically this can overflow, but we allow it
            ('d', 5, 'int64')
        ]
        self._check_literal_promote_cases(operator.add, cases)

    def test_multiply_literal_promotions(self):
        cases = [
            ('a', 0, 'int8'),
            ('a', 5, 'int16'),
            ('a', 2 ** 24, 'int32'),
            ('a', -2 ** 24 + 1, 'int32'),

            ('a', 1.5, 'double'),

            ('b', 0, 'int16'),
            ('b', 5, 'int32'),
            ('b', -5, 'int32'),
            ('c', 0, 'int32'),
            ('c', 5, 'int64'),
            ('c', -5, 'int64'),

            # technically this can overflow, but we allow it
            ('d', 5, 'int64')
        ]
        self._check_literal_promote_cases(operator.mul, cases)

    def test_subtract_literal_promotions(self):
        cases = [
            ('a', 0, 'int8'),
            ('a', 5, 'int16'),
            ('a', 100000, 'int32'),
            ('a', -100000, 'int32'),

            ('a', 1.5, 'double'),

            ('b', 0, 'int16'),
            ('b', 5, 'int32'),
            ('b', -5, 'int32'),
            ('c', 0, 'int32'),
            ('c', 5, 'int64'),
            ('c', -5, 'int64'),

            # technically this can overflow, but we allow it
            ('d', 5, 'int64')
        ]
        self._check_literal_promote_cases(operator.sub, cases)

    def test_divide_literal_promotions(self):
        cases = [
            ('a', 5, 'double'),
            ('a', 1.5, 'double'),
            ('b', 5, 'double'),
            ('b', -5, 'double'),
            ('c', 5, 'double'),
        ]
        self._check_literal_promote_cases(operator.div, cases)

    def test_pow_literal_promotions(self):
        cases = [
            ('a', 0, 'int8'),
            ('b', 0, 'int16'),
            ('c', 0, 'int32'),
            ('d', 0, 'int64'),
            ('e', 0, 'float'),
            ('f', 0, 'double'),

            ('a', 2, 'int16'),
            ('b', 2, 'int32'),
            ('c', 2, 'int64'),
            ('d', 2, 'int64'),

            ('a', 1.5, 'double'),
            ('b', 1.5, 'double'),
            ('c', 1.5, 'double'),
            ('d', 1.5, 'double'),

            ('a', -2, 'double'),
            ('b', -2, 'double'),
            ('c', -2, 'double'),
            ('d', -2, 'double'),

            ('e', 2, 'float'),
            ('f', 2, 'double')
        ]
        self._check_literal_promote_cases(operator.pow, cases)

    def _check_literal_promote_cases(self, op, cases):
        for name, val, ex_type in cases:
            col = self.table[name]

            result = op(col, val)
            ex_class = ir.array_type(ex_type)
            assert isinstance(result, ex_class)

            result = op(val, col)
            ex_class = ir.array_type(ex_type)
            assert isinstance(result, ex_class)

    def test_add_array_promotions(self):
        pass

    def test_subtract_array_promotions(self):
        pass

    def test_multiply_array_promotions(self):
        pass

    def test_divide_array_promotions(self):
        pass

    def test_string_add_concat(self):
        pass


class TestAggregation(BasicTestCase, unittest.TestCase):

    def test_count(self):
        result = self.table['a'].count()
        assert isinstance(result, api.Int64Scalar)
        assert isinstance(result.op(), ops.Count)

        result = self.table.count()
        assert isinstance(result, api.Int64Scalar)
        assert isinstance(result.op(), ops.Count)

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
        assert result.equals(expected)

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
        assert filtered.equals(expected)

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
        assert expr.equals(expected)

    def test_aggregate_root_table_internal(self):
        pass

    def test_compound_aggregate_expr(self):
        # See ibis #24
        compound_expr = (self.table['a'].sum() /
                         self.table['a'].mean()).name('foo')
        assert compound_expr.is_reduction()

        # Validates internally
        self.table.aggregate([compound_expr])

    def test_groupby_convenience(self):
        metrics = [self.table.f.sum().name('total')]

        expr = self.table.group_by('g').aggregate(metrics)
        expected = self.table.aggregate(metrics, by=['g'])
        assert expr.equals(expected)

        group_expr = self.table.g.cast('double')
        expr = self.table.group_by(group_expr).aggregate(metrics)
        expected = self.table.aggregate(metrics, by=[group_expr])
        assert expr.equals(expected)

    def test_group_by_count_size(self):
        # #148, convenience for interactive use, and so forth
        result1 = self.table.group_by('g').size()
        result2 = self.table.group_by('g').count()

        expected = (self.table.group_by('g')
                    .aggregate([self.table.count().name('count')]))

        assert result1.equals(expected)
        assert result2.equals(expected)

        result = self.table.group_by('g').count('foo')
        expected = (self.table.group_by('g')
                    .aggregate([self.table.count().name('foo')]))
        assert result.equals(expected)

    def test_group_by_column_select_api(self):
        grouped = self.table.group_by('g')

        result = grouped.f.sum()
        expected = grouped.aggregate(self.table.f.sum().name('sum(f)'))
        assert result.equals(expected)

        supported_functions = ['sum', 'mean', 'count', 'size', 'max', 'min']

        # make sure they all work
        for fn in supported_functions:
            getattr(grouped.f, fn)()

    def test_value_counts_convenience(self):
        # #152
        result = self.table.g.value_counts()
        expected = (self.table.group_by('g')
                    .aggregate(self.table.count().name('count')))

        assert result.equals(expected)

    def test_isin_value_counts(self):
        # #157, this code path was untested before
        bool_clause = self.table.g.notin(['1', '4', '7'])
        # it works!
        bool_clause.name('notin').value_counts()

    def test_value_counts_unnamed_expr(self):
        nation = self.con.table('tpch_nation')

        expr = nation.n_name.lower().value_counts()
        expected = nation.n_name.lower().name('unnamed').value_counts()
        assert expr.equals(expected)

    def test_aggregate_unnamed_expr(self):
        nation = self.con.table('tpch_nation')
        expr = nation.n_name.lower().left(1)
        self.assertRaises(com.ExpressionError, nation.group_by(expr).aggregate,
                          nation.count().name('metric'))


class TestJoinsUnions(BasicTestCase, unittest.TestCase):

    def test_join_no_predicate_list(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')

        pred = region.r_regionkey == nation.n_regionkey
        joined = region.inner_join(nation, pred)
        expected = region.inner_join(nation, [pred])
        assert joined.equals(expected)

    def test_equijoin_schema_merge(self):
        table1 = api.table([('key1',  'string'), ('value1', 'double')])
        table2 = api.table([('key2',  'string'), ('stuff', 'int32')])

        pred = table1['key1'] == table2['key2']
        join_types = ['inner_join', 'left_join', 'outer_join']

        ex_schema = api.Schema(['key1', 'value1', 'key2', 'stuff'],
                               ['string', 'double', 'string', 'int32'])

        for fname in join_types:
            f = getattr(table1, fname)
            joined = f(table2, [pred]).materialize()
            assert joined.schema().equals(ex_schema)

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
        assert proj.schema().equals(left.schema())

        # Try aggregating on top of joined
        aggregated = joined.aggregate([metric], by=[left['g']])
        ex_schema = api.Schema(['g', 'metric'], ['string', 'double'])
        assert aggregated.schema().equals(ex_schema)

    def test_self_join_no_view_convenience(self):
        # #165, self joins ought to be possible when the user specifies the
        # column names to join on rather than referentially-valid expressions

        result = self.table.join(self.table, [('g', 'g')])

        t2 = self.table.view()
        expected = self.table.join(t2, self.table.g == t2.g)
        assert result.equals(expected)

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
        table1 = api.table([('key1',  'string'), ('value1', 'double')])
        table2 = api.table([('key2',  'string'), ('stuff', 'int32')])

        pred = table1['key1'] == table2['key2']

        joined = table1.left_join(table2, [pred])
        projected = joined.projection([table1, table2['stuff']])
        assert projected.schema().names == ['key1', 'value1', 'stuff']

        projected = joined.projection([table2, table1['key1']])
        assert projected.schema().names == ['key2', 'stuff', 'key1']

    def test_semi_join_schema(self):
        # A left semi join discards the schema of the right table
        table1 = api.table([('key1',  'string'), ('value1', 'double')])
        table2 = api.table([('key2',  'string'), ('stuff', 'double')])

        pred = table1['key1'] == table2['key2']
        semi_joined = table1.semi_join(table2, [pred]).materialize()

        result_schema = semi_joined.schema()
        assert result_schema.equals(table1.schema())

    def test_cross_join(self):
        agg_exprs = [self.table['a'].sum().name('sum_a'),
                     self.table['b'].mean().name('mean_b')]
        scalar_aggs = self.table.aggregate(agg_exprs)

        joined = self.table.cross_join(scalar_aggs).materialize()
        agg_schema = api.Schema(['sum_a', 'mean_b'], ['int64', 'double'])
        ex_schema = self.table.schema().append(agg_schema)
        assert joined.schema().equals(ex_schema)

    def test_join_compound_boolean_predicate(self):
        # The user might have composed predicates through logical operations
        pass

    def test_multiple_join_deeper_reference(self):
        # Join predicates down the chain might reference one or more root
        # tables in the hierarchy.
        table1 = api.table({'key1': 'string', 'key2': 'string',
                            'value1': 'double'})
        table2 = api.table({'key3': 'string', 'value2': 'double'})
        table3 = api.table({'key4': 'string', 'value3': 'double'})

        joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
        joined2 = joined.inner_join(table3, [table1['key2'] == table3['key4']])

        # it works, what more should we test here?
        materialized = joined2.materialize()
        repr(materialized)

    def test_filter_join_unmaterialized(self):
        table1 = api.table({'key1': 'string', 'key2': 'string',
                            'value1': 'double'})
        table2 = api.table({'key3': 'string', 'value2': 'double'})

        # It works!
        joined = table1.inner_join(table2, [table1['key1'] == table2['key3']])
        filtered = joined.filter([table1.value1 > 0])
        repr(filtered)

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
                              orders.o_orderdate.cast('timestamp').name('odate')]

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

        # If filter were pushed below projection, the top-level operator type
        # would be Projection instead.
        assert type(result.op()) == ops.Filter

    def test_join_can_rewrite_errant_predicate(self):
        # Join predicate references a derived table, but we can salvage and
        # rewrite it to get the join semantics out
        # see ibis #74
        table = api.table([
            ('c', 'int32'),
            ('f', 'double'),
            ('g', 'string')
        ], 'foo_table')

        table2 = api.table([
            ('key', 'string'),
            ('value', 'double')
        ], 'bar_table')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]

        result = table.inner_join(table2, [table3['g'] == table2['key']])
        expected = table.inner_join(table2, [table['g'] == table2['key']])
        assert result.equals(expected)

    def test_non_equijoins(self):
        # Move non-equijoin predicates to WHERE during SQL translation if
        # possible, per #107
        pass

    def test_join_overlapping_column_names(self):
        pass

    def test_join_key_alternatives(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        # Join with tuples
        joined = t1.inner_join(t2, [('foo_id', 'foo_id')])
        joined2 = t1.inner_join(t2, [(t1.foo_id, t2.foo_id)])

        # Join with single expr
        joined3 = t1.inner_join(t2, t1.foo_id == t2.foo_id)

        expected = t1.inner_join(t2, [t1.foo_id == t2.foo_id])

        assert joined.equals(expected)
        assert joined2.equals(expected)
        assert joined3.equals(expected)

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
        t1 = api.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value1', 'double')
        ], 'foo_table')

        t2 = api.table([
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
        assert joined.equals(expected)

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
        t1 = api.table(schema1, 'foo')
        t2 = api.table(schema1, 'bar')
        t3 = api.table(schema2, 'baz')

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

        self.t1 = api.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value1', 'double')
        ], 'foo')

        self.t2 = api.table([
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
        assert isinstance(expr.op(), ops.Filter)

    def test_cannot_use_existence_expression_in_join(self):
        # Join predicates must consist only of comparisons
        pass

    def test_not_exists_predicate(self):
        cond = -(self.t1.key1 == self.t2.key1).any()
        assert isinstance(cond.op(), ops.NotAny)


class TestCaseExpressions(BasicTestCase, unittest.TestCase):

    def test_ifelse(self):
        bools = self.table.g.isnull()
        result = bools.ifelse("foo", "bar")
        assert isinstance(result, ir.StringArray)

    def test_ifelse_literal(self):
        pass

    def test_simple_case_expr(self):
        case1, result1 = "foo", self.table.a
        case2, result2 = "bar", self.table.c
        default_result = self.table.b

        expr1 = self.table.g.lower().cases(
            [(case1, result1),
             (case2, result2)],
            default=default_result
        )

        expr2 = (self.table.g.lower().case()
                 .when(case1, result1)
                 .when(case2, result2)
                 .else_(default_result)
                 .end())

        assert expr1.equals(expr2)
        assert isinstance(expr1, ir.Int32Array)

    def test_multiple_case_expr(self):
        case1 = self.table.a == 5
        case2 = self.table.b == 128
        case3 = self.table.c == 1000

        result1 = self.table.f
        result2 = self.table.b * 2
        result3 = self.table.e

        default = self.table.d

        expr = (api.case()
                .when(case1, result1)
                .when(case2, result2)
                .when(case3, result3)
                .else_(default)
                .end())

        op = expr.op()
        assert isinstance(expr, ir.DoubleArray)
        assert isinstance(op, ops.SearchedCase)
        assert op.default is default

    def test_simple_case_no_default(self):
        # TODO: this conflicts with the null else cases below. Make a decision
        # about what to do, what to make the default behavior based on what the
        # user provides. SQL behavior is to use NULL when nothing else
        # provided. The .replace convenience API could use the field values as
        # the default, getting us around this issue.
        pass

    def test_simple_case_null_else(self):
        expr = self.table.g.case().when("foo", "bar").end()
        op = expr.op()

        assert isinstance(expr, ir.StringArray)
        assert isinstance(op.default, ir.ValueExpr)
        assert isinstance(op.default.op(), ops.NullLiteral)

    def test_multiple_case_null_else(self):
        expr = api.case().when(self.table.g == "foo", "bar").end()
        op = expr.op()

        assert isinstance(expr, ir.StringArray)
        assert isinstance(op.default, ir.ValueExpr)
        assert isinstance(op.default.op(), ops.NullLiteral)

    def test_case_type_precedence(self):
        pass

    def test_no_implicit_cast_possible(self):
        pass


class TestInteractiveUse(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_interactive_execute_on_repr(self):
        table = self.con.table('functional_alltypes')
        expr = table.bigint_col.sum()
        with config.option_context('interactive', True):
            repr(expr)

        assert self.con.last_executed_expr is expr

    def test_default_limit(self):
        table = self.con.table('functional_alltypes')

        with config.option_context('interactive', True):
            repr(table)

        result = self.con.last_executed_expr
        limit = config.options.sql.default_limit
        assert result.equals(table.limit(limit))

    def test_interactive_non_compilable_repr_not_fail(self):
        # #170
        table = self.con.table('functional_alltypes')

        expr = table.string_col.topk(3)

        # it works!
        with config.option_context('interactive', True):
            repr(expr)

    def test_histogram_repr_no_query_execute(self):
        t = self.con.table('functional_alltypes')
        tier = t.double_col.histogram(10).name('bucket')
        expr = t.group_by(tier).size()
        with config.option_context('interactive', True):
            expr._repr()
        assert self.con.last_executed_expr is None
