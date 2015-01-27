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

import itertools
import operator
import unittest

from ibis.expr.base import ArrayExpr, TableExpr, RelationError
import ibis.expr.base as api
import ibis.expr.base as operations

from ibis.expr.format import ExprFormatter
from ibis.expr.tests.mocks import MockConnection


import ibis.common as com


class TestParameters(unittest.TestCase):
    pass


class TestSchema(unittest.TestCase):
    pass


class TestLiterals(unittest.TestCase):

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
        assert isinstance(arg, api.Literal)
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
            klass = api.scalar_class(ex_type)
            assert isinstance(expr, klass)
            assert isinstance(expr.op(), api.Literal)
            assert expr.op().value is value


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
            assert isinstance(col, api.array_class(v))

            # Ensure we have a field selection with back-reference to the table
            parent = col.parent()
            assert isinstance(parent, api.TableColumn)
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
        assert isinstance(proj.op(), api.Projection)

        assert proj.schema().names == cols
        for c in cols:
            expr = proj[c]
            assert type(expr) == type(self.table[c])

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
        self.assertRaises(ValueError, self.table.projection,
                          ['g', mean_diff])

    def test_projection_duplicate_names(self):
        self.assertRaises(com.IntegrityError, self.table.projection,
                          [self.table.c, self.table.c])

    def test_projection_unary_name_passthrough(self):
        # Can fix this later if we add different default names
        proj = self.table[[self.table['a'].log()]]
        assert proj.schema().names == ['a']

    def test_projection_invalid_root(self):
        schema1 = {
            'foo': 'double',
            'bar': 'int32'
        }

        left = api.table(schema1)
        right = api.table(schema1)

        exprs = [right['foo'], right['bar']]
        self.assertRaises(RelationError, left.projection, exprs)

    def test_projection_of_aggregated(self):
        # Fully-formed aggregations "block"; in a projection, column
        # expressions referencing table expressions below the aggregation are
        # invalid.
        pass

    def test_projection_with_star_expr(self):
        new_expr = (self.table['a'] * 5).name('bigger_a')

        t = self.table

        # it lives!
        proj = t[[t, new_expr]]
        repr(proj)

        ex_names = self.table.schema().names + ['bigger_a']
        assert proj.schema().names == ex_names

        # cannot pass an invalid table expression
        t2 = t.aggregate([t['a'].sum()], by=['g'])
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

    def test_add_predicate(self):
        pred = self.table['a'] > 5
        result = self.table[pred]
        assert isinstance(result.op(), api.Filter)

    def test_filter_root_table_preserved(self):
        result = self.table[self.table['a'] > 5]
        roots = result.op().root_tables()
        assert roots[0] is self.table.op()

    def test_invalid_predicate(self):
        # a lookalike
        table2 = api.table(self.schema)
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

        # #59, if we are not careful, we can obtain broken refs
        interm = self.table[pred1]
        result = interm[interm['b'] > 0]
        assert result.equals(expected)

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

    def test_filter_projection_partial_pushdown(self):
        pass

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

    def test_sort_by_aggregate_or_projection_field(self):
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

    def test_memoize_aggregate_correctly(self):
        table = self.table

        agg_expr = (table['c'].sum() / table['c'].mean() - 1).name('analysis')
        agg_exprs = [table['a'].sum(), table['b'].mean(), agg_expr]

        result = table.aggregate(agg_exprs, by=['g'])

        formatter = ExprFormatter(result)
        formatted = formatter.get_result()

        alias = formatter.memo.get_alias(table.op())
        assert formatted.count(alias) == 7

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
        assert isinstance(expr.op(), operations.IsNull)

        expr = api.literal('foo').isnull()
        assert isinstance(expr, api.BooleanScalar)
        assert isinstance(expr.op(), operations.IsNull)

    def test_notnull(self):
        expr = self.table['g'].notnull()
        assert isinstance(expr, api.BooleanArray)
        assert isinstance(expr.op(), operations.NotNull)

        expr = api.literal('foo').notnull()
        assert isinstance(expr, api.BooleanScalar)
        assert isinstance(expr.op(), operations.NotNull)

    def test_null_literal(self):
        pass


class TestMathUnaryOps(BasicTestCase, unittest.TestCase):

    def test_log_variants(self):
        ops = ['log', 'log2', 'log10']

        columns = ['a', 'b', 'c', 'd', 'e', 'f']

        for opname in ops:
            f = lambda x: getattr(x, opname)()

            for c in columns:
                result = f(self.table[c])
                assert isinstance(result, api.DoubleArray)

                # is this what we want?
                # assert result.get_name() == c

            assert isinstance(f(api.literal(5)), api.DoubleScalar)
            assert isinstance(f(api.literal(5.5)), api.DoubleScalar)

            klass = getattr(operations, opname.capitalize())
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
            assert isinstance(casted, api.array_class(t))
            assert casted.get_name() == c

            casted_literal = api.literal('5').name('bar').cast(t)
            assert isinstance(casted_literal, api.scalar_class(t))
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
            assert isinstance(result.op(), operations.Negate)

        result = -api.literal(False)
        assert isinstance(result, api.BooleanScalar)
        assert isinstance(result.op(), operations.Negate)

    def test_isnull_notnull(self):
        pass


class TestBooleanBinaryOps(BasicTestCase, unittest.TestCase):

    def test_numbers_compare_numeric_literal(self):
        ops = ['lt', 'gt', 'ge', 'le', 'eq', 'ne']

        ex_op_class = {
            'eq': api.Equals,
            'ne': api.NotEquals,
            'le': api.LessEqual,
            'lt': api.Less,
            'ge': api.GreaterEqual,
            'gt': api.Greater,
        }

        columns = ['a', 'b', 'c', 'd', 'e', 'f']

        cases = [2, 2 ** 9, 2 ** 17, 2 ** 33, 1.5]
        for opname, cname, val in itertools.product(ops, columns, cases):
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
            ex_class = api.array_class(ex_type)
            assert isinstance(result, ex_class)

            result = op(val, col)
            ex_class = api.array_class(ex_type)
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
        assert isinstance(result.op(), operations.Count)

        result = self.table.count()
        assert isinstance(result, api.Int64Scalar)
        assert isinstance(result.op(), operations.Count)

    def test_sum_expr_basics(self):
        # Impala gives bigint for all integer types
        ex_class = api.Int64Scalar
        for c in self.int_cols + self.bool_cols:
            result = self.table[c].sum()
            assert isinstance(result, ex_class)
            assert isinstance(result.op(), api.Sum)

            assert result.get_name() == c

        # Impala gives double for all floating point types
        ex_class = api.DoubleScalar
        for c in self.float_cols:
            result = self.table[c].sum()
            assert isinstance(result, ex_class)
            assert isinstance(result.op(), api.Sum)

    def test_mean_expr_basics(self):
        cols = self.int_cols + self.float_cols + self.bool_cols
        for c in cols:
            result = self.table[c].mean()
            assert isinstance(result, api.DoubleScalar)
            assert isinstance(result.op(), api.Mean)

    def test_aggregate_no_keys(self):
        agg_exprs = [self.table['a'].sum(),
                     self.table['c'].mean()]

        # A TableExpr, which in SQL at least will yield a table with a single
        # row
        result = self.table.aggregate(agg_exprs)
        assert isinstance(result, TableExpr)

    def test_aggregate_keys_basic(self):
        agg_exprs = [self.table['a'].sum(),
                     self.table['c'].mean()]

        # A TableExpr, which in SQL at least will yield a table with a single
        # row
        result = self.table.aggregate(agg_exprs, by=['g'])
        assert isinstance(result, TableExpr)

        # it works!
        repr(result)

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
        filtered = agged[pred]
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

    def test_aggregate_root_table_internal(self):
        pass

    def test_group_by_expr(self):
        # Should not be an issue, as long as expr originates from table and has
        # a name
        pass

    def test_compound_aggregate_expr(self):
        # See ibis #24
        compound_expr = (self.table['a'].sum() /
                         self.table['a'].mean()).name('foo')
        assert compound_expr.is_reduction()

        # Validates internally
        self.table.aggregate([compound_expr])


class TestJoins(BasicTestCase, unittest.TestCase):

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
