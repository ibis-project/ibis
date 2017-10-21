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

import operator
from collections import OrderedDict
from operator import methodcaller
from datetime import date, datetime, time

import pytest

from ibis.common import IbisTypeError
from ibis.expr import (
    rules, api, datatypes as dt, types as ir, operations as ops
)
import ibis
from ibis.compat import PY2

from ibis import literal
from ibis.tests.util import assert_equal

import toolz


def test_null():
    expr = ibis.literal(None)
    assert isinstance(expr, ir.NullScalar)
    assert isinstance(expr.op(), ir.NullLiteral)
    assert expr._arg.value is None

    expr2 = ibis.null()
    assert_equal(expr, expr2)


@pytest.mark.xfail(
    raises=AssertionError,
    reason='UTF-8 support in Impala non-existent at the moment?'
)
def test_unicode():
    assert False


@pytest.mark.parametrize(
    ['value', 'expected_type'],
    [
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
        ('foo', 'string'),
    ]
)
def test_literal_cases(value, expected_type):
    expr = ibis.literal(value)
    klass = dt.scalar_type(expected_type)
    assert isinstance(expr, klass)
    assert isinstance(expr.op(), ir.Literal)
    assert expr.op().value is value


@pytest.mark.parametrize(
    ['value', 'expected_type'],
    [
        (5, 'int16'),
        (127, 'double'),
        (128, 'int64'),
        (32767, 'double'),
        (32768, 'float'),
        (2147483647, 'int64'),
        (-5, 'int16'),
        (-128, 'int32'),
        (-129, 'int64'),
        (-32769, 'float'),
        (-2147483649, 'double'),
        (1.5, 'double'),
        ('foo', 'string'),
    ]
)
def test_literal_with_different_type(value, expected_type):
    expr = ibis.literal(value, type=expected_type)
    assert expr.type().equals(dt.validate_type(expected_type))


@pytest.mark.parametrize(
    ['value', 'expected_type', 'expected_class'],
    [
        (list('abc'), 'array<string>', ir.ArrayScalar),
        ([1, 2, 3], 'array<int8>', ir.ArrayScalar),
        ({'a': 1, 'b': 2, 'c': 3}, 'map<string, int8>', ir.MapScalar),
        ({1: 2, 3: 4, 5: 6}, 'map<int8, int8>', ir.MapScalar),
        (
            {'a': [1.0, 2.0], 'b': [], 'c': [3.0]},
            'map<string, array<double>>',
            ir.MapScalar
        ),
        (
            OrderedDict([
                ('a', 1),
                ('b', list('abc')),
                ('c', OrderedDict([('foo', [1.0, 2.0])]))
            ]),
            'struct<a: int8, b: array<string>, c: struct<foo: array<double>>>',
            ir.StructScalar
        )
    ]
)
def test_literal_complex_types(value, expected_type, expected_class):
    expr = ibis.literal(value)
    expr_type = expr.type()
    assert expr_type.equals(dt.validate_type(expected_type))
    assert isinstance(expr, expected_class)
    assert isinstance(expr.op(), ir.Literal)
    assert expr.op().value is value


def test_struct_operations():
    value = OrderedDict([
        ('a', 1),
        ('b', list('abc')),
        ('c', OrderedDict([('foo', [1.0, 2.0])]))
    ])
    expr = ibis.literal(value)
    assert isinstance(expr, ir.StructValue)
    assert isinstance(expr.b, ir.ArrayValue)
    assert isinstance(expr.a.op(), ops.StructField)


def test_simple_map_operations():
    value = {'a': [1.0, 2.0], 'b': [], 'c': [3.0]}
    value2 = {'a': [1.0, 2.0], 'c': [3.0], 'd': [4.0, 5.0]}
    expr = ibis.literal(value)
    expr2 = ibis.literal(value2)
    assert isinstance(expr, ir.MapValue)
    assert isinstance(expr['b'].op(), ops.MapValueForKey)
    assert isinstance(expr.length().op(), ops.MapLength)
    assert isinstance(expr.keys().op(), ops.MapKeys)
    assert isinstance(expr.values().op(), ops.MapValues)
    assert isinstance((expr + expr2).op(), ops.MapConcat)
    assert isinstance((expr2 + expr).op(), ops.MapConcat)


@pytest.mark.parametrize(
    ['value', 'expected_type'],
    [
        (32767, 'int8'),
        (32768, 'int16'),
        (2147483647, 'int16'),
        (2147483648, 'int32'),
        ('foo', 'double'),
    ]
)
def test_literal_with_different_type_failure(value, expected_type):
    with pytest.raises(TypeError):
        ibis.literal(value, type=expected_type)


def test_literal_list():
    what = [1, 2, 1000]
    expr = api.as_value_expr(what)

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr.op(), ir.ValueList)
    assert isinstance(expr.op().values[2], ir.Int16Scalar)

    # it works!
    repr(expr)


def test_literal_array():
    what = []
    expr = api.literal(what)
    assert isinstance(expr, ir.ArrayValue)
    assert expr.type().equals(dt.Array(dt.null))


def test_mixed_arity(table):
    what = ["bar", table.g, "foo"]
    expr = api.as_value_expr(what)

    values = expr.op().values
    assert isinstance(values[1], ir.StringColumn)

    # it works!
    repr(expr)


def test_isin_notin_list(table):
    vals = [1, 2, 3]

    expr = table.a.isin(vals)
    not_expr = table.a.notin(vals)

    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.Contains)

    assert isinstance(not_expr, ir.BooleanColumn)
    assert isinstance(not_expr.op(), ops.NotContains)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isin_not_comparable():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isin_array_expr():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isin_invalid_cases():
    # For example, array expression in a list of values, where the inner
    # array values originate from some other table
    assert False


def test_isin_notin_scalars():
    a, b, c = [ibis.literal(x) for x in [1, 1, 2]]

    result = a.isin([1, 2])
    assert isinstance(result, ir.BooleanScalar)

    result = a.notin([b, c])
    assert isinstance(result, ir.BooleanScalar)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isin_null():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_negate_isin():
    # Should yield a NotContains
    assert False


def test_scalar_isin_list_with_array(table):
    val = ibis.literal(2)

    options = [table.a, table.b, table.c]

    expr = val.isin(options)
    assert isinstance(expr, ir.BooleanColumn)

    not_expr = val.notin(options)
    assert isinstance(not_expr, ir.BooleanColumn)


def test_distinct_basic(functional_alltypes):
    expr = functional_alltypes.distinct()
    assert isinstance(expr.op(), ops.Distinct)
    assert isinstance(expr, ir.TableExpr)
    assert expr.op().table is functional_alltypes

    expr = functional_alltypes.string_col.distinct()
    assert isinstance(expr.op(), ops.DistinctColumn)

    assert isinstance(expr, ir.StringColumn)


@pytest.mark.xfail(reason='NYT')
def test_distinct_array_interactions(functional_alltypes):
    # array cardinalities / shapes are likely to be different.
    a = functional_alltypes.int_col.distinct()
    b = functional_alltypes.bigint_col

    with pytest.raises(ir.RelationError):
        a + b


def test_distinct_count(functional_alltypes):
    result = functional_alltypes.string_col.distinct().count()
    expected = functional_alltypes.string_col.nunique().name('count')
    assert_equal(result, expected)
    assert isinstance(result.op(), ops.CountDistinct)


def test_distinct_unnamed_array_expr():
    table = ibis.table([('year', 'int32'),
                        ('month', 'int32'),
                        ('day', 'int32')], 'foo')

    # it works!
    expr = (ibis.literal('-')
            .join([table.year.cast('string'),
                   table.month.cast('string'),
                   table.day.cast('string')])
            .distinct())
    repr(expr)


def test_distinct_count_numeric_types(functional_alltypes):
    metric = functional_alltypes.bigint_col.distinct().count().name(
        'unique_bigints'
    )
    functional_alltypes.group_by('string_col').aggregate(metric)


def test_nunique(functional_alltypes):
    expr = functional_alltypes.string_col.nunique()
    assert isinstance(expr.op(), ops.CountDistinct)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_project_with_distinct():
    assert False


def test_isnull(table):
    expr = table['g'].isnull()
    assert isinstance(expr, api.BooleanColumn)
    assert isinstance(expr.op(), ops.IsNull)

    expr = ibis.literal('foo').isnull()
    assert isinstance(expr, api.BooleanScalar)
    assert isinstance(expr.op(), ops.IsNull)


def test_notnull(table):
    expr = table['g'].notnull()
    assert isinstance(expr, api.BooleanColumn)
    assert isinstance(expr.op(), ops.NotNull)

    expr = ibis.literal('foo').notnull()
    assert isinstance(expr, api.BooleanScalar)
    assert isinstance(expr.op(), ops.NotNull)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_null_literal():
    assert False


@pytest.mark.parametrize(
    ['column', 'operation'],
    [
        ('d', 'cumsum'),
        ('d', 'cummean'),
        ('d', 'cummin'),
        ('d', 'cummax'),
        ('h', 'cumany'),
        ('h', 'cumall'),
    ]
)
def test_cumulative_yield_array_types(table, column, operation):
    expr = getattr(getattr(table, column), operation)()
    assert isinstance(expr, ir.ColumnExpr)


@pytest.fixture(params=['ln', 'log', 'log2', 'log10'])
def log(request):
    return operator.methodcaller(request.param)


@pytest.mark.parametrize('column', list('abcdef'))
def test_log(table, log, column):
    result = log(table[column])
    assert isinstance(result, api.DoubleColumn)

    # is this what we want?
    # assert result.get_name() == c


def test_log_string(table):
    g = table.g

    with pytest.raises(IbisTypeError):
        ops.Log(g, None).to_expr()


@pytest.mark.parametrize('klass', [ops.Ln, ops.Log2, ops.Log10])
def test_log_variants_string(table, klass):
    g = table.g

    with pytest.raises(IbisTypeError):
        klass(g).to_expr()


def test_log_boolean(table, log):
    # boolean not implemented for these
    h = table['h']
    with pytest.raises(IbisTypeError):
        log(h)


def test_log_literal(log):
    assert isinstance(log(ibis.literal(5)), api.DoubleScalar)
    assert isinstance(log(ibis.literal(5.5)), api.DoubleScalar)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_exp():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_sqrt():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_trig_functions():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_round():
    assert False


def test_cast_same_type_noop(table):
    c = table.g
    assert c.cast('string') is c


@pytest.mark.parametrize('type', ['int8', 'int32', 'double', 'float'])
def test_string_to_number(table, type):
    casted = table.g.cast(type)
    assert isinstance(casted, dt.array_type(type))

    casted_literal = ibis.literal('5').cast(type).name('bar')
    assert isinstance(casted_literal, dt.scalar_type(type))
    assert casted_literal.get_name() == 'bar'


@pytest.mark.parametrize('col', list('abcdefh'))
def test_number_to_string_column(table, col):
    casted = table[col].cast('string')
    assert isinstance(casted, api.StringColumn)


def test_number_to_string_scalar():
    casted_literal = ibis.literal(5).cast('string').name('bar')
    assert isinstance(casted_literal, api.StringScalar)
    assert casted_literal.get_name() == 'bar'


def test_casted_exprs_are_named(table):
    expr = table.f.cast('string')
    assert expr.get_name() == 'cast(f, string)'

    # it works! per GH #396
    expr.value_counts()


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_nonzero():
    assert False


@pytest.mark.parametrize('col', list('abcdefh'))
def test_negate(table, col):
    c = table[col]
    result = -c
    assert isinstance(result, type(c))
    assert isinstance(result.op(), ops.Negate)


def test_negate_boolean_scalar():
    result = -ibis.literal(False)
    assert isinstance(result, api.BooleanScalar)
    assert isinstance(result.op(), ops.Negate)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isnull_notnull():
    assert False


@pytest.mark.parametrize(
    ['column', 'operation'],
    [
        ('h', lambda column: column.any()),
        ('h', lambda column: column.notany()),
        ('h', lambda column: column.all()),
        ('c', lambda column: (column == 0).any()),
        ('c', lambda column: (column == 0).all()),
    ]
)
def test_any_all_notany(table, column, operation):
    expr = operation(table[column])
    assert isinstance(expr, api.BooleanScalar)
    assert ops.is_reduction(expr)


@pytest.mark.parametrize(
    'operation',
    [
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ]
)
@pytest.mark.parametrize('column', list('abcdef'))
@pytest.mark.parametrize('case', [2, 2 ** 9, 2 ** 17, 2 ** 33, 1.5])
def test_numbers_compare_numeric_literal(table, operation, column, case):
    ex_op_class = {
        operator.eq: ops.Equals,
        operator.ne: ops.NotEquals,
        operator.le: ops.LessEqual,
        operator.lt: ops.Less,
        operator.ge: ops.GreaterEqual,
        operator.gt: ops.Greater,
    }

    col = table[column]

    result = operation(col, case)
    assert isinstance(result, api.BooleanColumn)
    assert isinstance(result.op(), ex_op_class[operation])


def test_boolean_comparisons(table):
    bool_col = table.h

    result = bool_col == True  # noqa
    assert isinstance(result, api.BooleanColumn)

    result = bool_col == False  # noqa
    assert isinstance(result, api.BooleanColumn)


@pytest.mark.parametrize(
    'operation',
    [operator.lt, operator.gt, operator.ge, operator.le,
     operator.eq, operator.ne]
)
def test_string_comparisons(table, operation):
    string_col = table.g
    result = operation(string_col, 'foo')
    assert isinstance(result, api.BooleanColumn)


@pytest.mark.parametrize(
    'operation',
    [operator.xor, operator.or_, operator.and_]
)
def test_boolean_logical_ops(table, operation):
    expr = table.a > 0

    result = operation(expr, table.h)
    assert isinstance(result, api.BooleanColumn)

    result = operation(expr, True)
    refl_result = operation(True, expr)
    assert isinstance(result, api.BooleanColumn)
    assert isinstance(refl_result, api.BooleanColumn)

    true = ibis.literal(True)
    false = ibis.literal(False)

    result = operation(true, false)
    assert isinstance(result, api.BooleanScalar)


def test_null_column():
    t = ibis.table([('a', 'string')], name='t')
    s = t.mutate(b=ibis.NA)
    assert s.b.type() == dt.null
    assert isinstance(s.b, ir.NullColumn)


def test_null_column_union():
    s = ibis.table([('a', 'string'), ('b', 'double')])
    t = ibis.table([('a', 'string')])
    with pytest.raises(ibis.common.RelationError):
        s.union(t.mutate(b=ibis.NA))  # needs a type
    assert (
        s.union(t.mutate(b=ibis.NA.cast('double'))).schema() == s.schema()
    )


def test_string_compare_numeric_array(table):
    with pytest.raises(TypeError):
        table.g == table.f

    with pytest.raises(TypeError):
        table.g == table.c


def test_string_compare_numeric_literal(table):
    with pytest.raises(TypeError):
        table.g == ibis.literal(1.5)

    with pytest.raises(TypeError):
        table.g == ibis.literal(5)


def test_between(table):
    result = table.f.between(0, 1)

    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(result.op(), ops.Between)

    # it works!
    result = table.g.between('a', 'f')
    assert isinstance(result, ir.BooleanColumn)

    result = ibis.literal(1).between(table.a, table.c)
    assert isinstance(result, ir.BooleanColumn)

    result = ibis.literal(7).between(5, 10)
    assert isinstance(result, ir.BooleanScalar)

    # Cases where between should immediately fail, e.g. incomparables
    with pytest.raises(TypeError):
        table.f.between('0', '1')

    with pytest.raises(TypeError):
        table.f.between(0, '1')

    with pytest.raises(TypeError):
        table.f.between('0', 1)


def test_chained_comparisons_not_allowed(table):
    with pytest.raises(ValueError):
        0 < table.f < 1


@pytest.mark.parametrize(
    'operation',
    [
        operator.add,
        operator.mul,
        operator.truediv,
        operator.sub
    ]
)
def test_binop_string_type_error(table, operation):
    # Strings are not valid for any numeric arithmetic
    ints = table.d
    strs = table.g

    with pytest.raises(TypeError):
        operation(ints, strs)

    with pytest.raises(TypeError):
        operation(strs, ints)


@pytest.mark.parametrize(
    ['op', 'name', 'case', 'ex_type'],
    [
        (operator.add, 'a', 0, 'int8'),
        (operator.add, 'a', 5, 'int16'),
        (operator.add, 'a', 100000, 'int32'),
        (operator.add, 'a', -100000, 'int32'),

        (operator.add, 'a', 1.5, 'double'),

        (operator.add, 'b', 0, 'int16'),
        (operator.add, 'b', 5, 'int32'),
        (operator.add, 'b', -5, 'int32'),

        (operator.add, 'c', 0, 'int32'),
        (operator.add, 'c', 5, 'int64'),
        (operator.add, 'c', -5, 'int64'),

        # technically this can overflow, but we allow it
        (operator.add, 'd', 5, 'int64'),

        (operator.mul, 'a', 0, 'int8'),
        (operator.mul, 'a', 5, 'int16'),
        (operator.mul, 'a', 2 ** 24, 'int32'),
        (operator.mul, 'a', -2 ** 24 + 1, 'int32'),

        (operator.mul, 'a', 1.5, 'double'),

        (operator.mul, 'b', 0, 'int16'),
        (operator.mul, 'b', 5, 'int32'),
        (operator.mul, 'b', -5, 'int32'),
        (operator.mul, 'c', 0, 'int32'),
        (operator.mul, 'c', 5, 'int64'),
        (operator.mul, 'c', -5, 'int64'),

        # technically this can overflow, but we allow it
        (operator.mul, 'd', 5, 'int64'),

        (operator.sub, 'a', 0, 'int8'),
        (operator.sub, 'a', 5, 'int16'),
        (operator.sub, 'a', 100000, 'int32'),
        (operator.sub, 'a', -100000, 'int32'),


        (operator.sub, 'a', 1.5, 'double'),
        (operator.sub, 'b', 0, 'int16'),
        (operator.sub, 'b', 5, 'int32'),
        (operator.sub, 'b', -5, 'int32'),
        (operator.sub, 'c', 0, 'int32'),
        (operator.sub, 'c', 5, 'int64'),
        (operator.sub, 'c', -5, 'int64'),

        # technically this can overflow, but we allow it
        (operator.sub, 'd', 5, 'int64'),

        (operator.truediv, 'a', 5, 'double'),
        (operator.truediv, 'a', 1.5, 'double'),
        (operator.truediv, 'b', 5, 'double'),
        (operator.truediv, 'b', -5, 'double'),
        (operator.truediv, 'c', 5, 'double'),

        (operator.pow, 'a', 0, 'int8'),
        (operator.pow, 'b', 0, 'int16'),
        (operator.pow, 'c', 0, 'int32'),
        (operator.pow, 'd', 0, 'int64'),
        (operator.pow, 'e', 0, 'float'),
        (operator.pow, 'f', 0, 'double'),

        (operator.pow, 'a', 2, 'int16'),
        (operator.pow, 'b', 2, 'int32'),
        (operator.pow, 'c', 2, 'int64'),
        (operator.pow, 'd', 2, 'int64'),

        (operator.pow, 'a', 1.5, 'double'),
        (operator.pow, 'b', 1.5, 'double'),
        (operator.pow, 'c', 1.5, 'double'),
        (operator.pow, 'd', 1.5, 'double'),

        (operator.pow, 'a', -2, 'double'),
        (operator.pow, 'b', -2, 'double'),
        (operator.pow, 'c', -2, 'double'),
        (operator.pow, 'd', -2, 'double'),

        (operator.pow, 'e', 2, 'float'),
        (operator.pow, 'f', 2, 'double'),
    ]
)
def test_literal_promotions(table, op, name, case, ex_type):
    col = table[name]

    result = op(col, case)
    ex_class = dt.array_type(ex_type)
    assert isinstance(result, ex_class)

    result = op(case, col)
    ex_class = dt.array_type(ex_type)
    assert isinstance(result, ex_class)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_add_array_promotions():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_subtract_array_promotions():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_multiply_array_promotions():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_divide_array_promotions():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_string_add_concat():
    assert False


@pytest.fixture
def expr():
    exprs = [ibis.literal(1).name('a'),
             ibis.literal(2).name('b')]

    return ibis.expr_list(exprs)


def test_names(expr):
    assert expr.names() == ['a', 'b']


def test_prefix(expr):
    prefixed = expr.prefix('foo_')
    result = prefixed.names()
    assert result == ['foo_a', 'foo_b']


def test_rename(expr):
    renamed = expr.rename(lambda x: 'foo({0})'.format(x))
    result = renamed.names()
    assert result == ['foo(a)', 'foo(b)']


def test_suffix(expr):
    suffixed = expr.suffix('.x')
    result = suffixed.names()
    assert result == ['a.x', 'b.x']


def test_concat():
    exprs = [ibis.literal(1).name('a'), ibis.literal(2).name('b')]
    exprs2 = [ibis.literal(3).name('c'), ibis.literal(4).name('d')]

    list1 = ibis.expr_list(exprs)
    list2 = ibis.expr_list(exprs2)

    result = list1.concat(list2)
    expected = ibis.expr_list(exprs + exprs2)
    assert_equal(result, expected)


def test_substitute_dict():
    table = ibis.table([('foo', 'string'), ('bar', 'string')], 't1')
    subs = {'a': 'one', 'b': table.bar}

    result = table.foo.substitute(subs)
    expected = (table.foo.case()
                .when('a', 'one')
                .when('b', table.bar)
                .else_(table.foo).end())
    assert_equal(result, expected)

    result = table.foo.substitute(subs, else_=ibis.NA)
    expected = (table.foo.case()
                .when('a', 'one')
                .when('b', table.bar)
                .else_(ibis.NA).end())
    assert_equal(result, expected)


@pytest.mark.parametrize(
    'typ',
    [
        'array<map<string, array<array<double>>>>',
        'string',
        'double',
        'float',
        'int64',
    ]
)
def test_not_without_boolean(typ):
    t = ibis.table([('a', typ)], name='t')
    c = t.a
    with pytest.raises(TypeError):
        ~c


@pytest.mark.parametrize(
    ('position', 'names'),
    [
        (0, 'foo'),
        (1, 'bar'),
        ([0], ['foo']),
        ([1], ['bar']),
        ([0, 1], ['foo', 'bar']),
        ([1, 0], ['bar', 'foo']),
    ]
)
@pytest.mark.parametrize(
    'expr_func',
    [
        lambda t, args: t[args],
        lambda t, args: t.sort_by(args),
        lambda t, args: t.group_by(args).aggregate(bar_avg=t.bar.mean())
    ]
)
def test_table_operations_with_integer_column(position, names, expr_func):
    t = ibis.table([('foo', 'string'), ('bar', 'double')])
    result = expr_func(t, position)
    expected = expr_func(t, names)
    assert result.equals(expected)


@pytest.mark.parametrize(
    'value',
    [
        'abcdefg',
        ['a', 'b', 'c'],
        [1, 2, 3],
    ]
)
@pytest.mark.parametrize(
    'operation',
    [
        'pow',
        'sub',
        'truediv',
        'floordiv',
        'mod',
    ]
)
def test_generic_value_api_no_arithmetic(value, operation):
    func = getattr(operator, operation)
    expr = ibis.literal(value)
    with pytest.raises(TypeError):
        func(expr, expr)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (5, dt.int8),
        (5.4, dt.double),
        ('abc', dt.string),
    ]
)
def test_fillna_null(value, expected):
    assert ibis.NA.fillna(value).type().equals(expected)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (literal('2017-04-01'), date(2017, 4, 2)),
        (date(2017, 4, 2), literal('2017-04-01')),
        (literal('2017-04-01 01:02:33'), datetime(2017, 4, 1, 1, 3, 34)),
        (datetime(2017, 4, 1, 1, 3, 34), literal('2017-04-01 01:02:33')),
    ]
)
@pytest.mark.parametrize(
    'op',
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        lambda left, right: ibis.timestamp(
            '2017-04-01 00:02:34'
        ).between(left, right),
        lambda left, right: ibis.timestamp(
            '2017-04-01'
        ).cast(dt.date).between(left, right)
    ]
)
def test_string_temporal_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ('value', 'type', 'expected_type_class'),
    [
        (2.21, 'decimal', dt.Decimal),
        (3.14, 'double', dt.Double),
        (4.2, 'int64', dt.Double),
        (4, 'int64', dt.Int64),
    ]
)
def test_decimal_modulo_output_type(value, type, expected_type_class):
    t = ibis.table([('a', type)])
    expr = t.a % value
    assert isinstance(expr.type(), expected_type_class)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (literal('10:00'), time(10, 0)),
        (time(10, 0), literal('10:00')),
    ]
)
@pytest.mark.parametrize(
    'op',
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ]
)
@pytest.mark.skipif(PY2, reason="time comparsions not available on PY2")
def test_time_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (literal('10:00'), date(2017, 4, 2)),
        (literal('10:00'), datetime(2017, 4, 2, 1, 1)),
        (literal('10:00'), literal('2017-04-01')),
    ]
)
@pytest.mark.parametrize(
    'op',
    [
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        ]
)
def test_time_timestamp_invalid_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.skipif(not PY2, reason="invalid compare of time on PY2")
def test_time_invalid_compare_on_py2():

    # we cannot actually compare datetime.time objects and literals
    # in a deferred way in python 2, they short circuit in the CPython
    result = operator.eq(time(10, 0), literal('10:00'))
    assert not result


def test_scalar_parameter_repr():
    value = ibis.param(dt.timestamp, name='value')
    assert repr(value) == 'value = ScalarParameter[timestamp]'

    value_op = value.op()
    assert repr(value_op) == "ScalarParameter(name='value', type=timestamp)"


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (
            # same value type, same name
            ibis.param(dt.timestamp, name='value1'),
            ibis.param(dt.timestamp, name='value1'),
            True,
        ),
        (
            # different value type, same name
            ibis.param(dt.date, name='value1'),
            ibis.param(dt.timestamp, name='value1'),
            False,
        ),
        (
            # same value type, different name
            ibis.param(dt.timestamp, name='value1'),
            ibis.param(dt.timestamp, name='value2'),
            False,
        ),
        (
            # different value type, different name
            ibis.param(dt.date, name='value1'),
            ibis.param(dt.timestamp, name='value2'),
            False,
        ),
        (
            # different Python class, left side is param
            ibis.param(dt.timestamp, 'value'),
            dt.date,
            False
        ),
        (
            # different Python class, right side is param
            dt.date,
            ibis.param(dt.timestamp, 'value'),
            False
        ),
    ]
)
def test_scalar_parameter_compare(left, right, expected):
    assert left.equals(right) == expected


@pytest.mark.parametrize(
    ('case', 'creator'),
    [
        (datetime.now(), toolz.compose(methodcaller('time'), ibis.timestamp)),
        ('now', toolz.compose(methodcaller('time'), ibis.timestamp)),
        (datetime.now().time(), ibis.time),
        ('10:37', ibis.time),
    ]
)
@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (1, 'a'),
        ('a', 1),
        (1.0, 2.0),
        (['a'], [1]),
    ]
)
@pytest.mark.xfail(PY2, reason='Not supported on Python 2')
def test_between_time_failure_time(case, creator, left, right):
    value = creator(case)
    with pytest.raises(TypeError):
        value.between(left, right)


def test_custom_type_binary_operations():
    class Foo(ir.ValueExpr):

        def __add__(self, other):
            op = self.op()
            return type(op)(op.value + other).to_expr()

        __radd__ = __add__

    class FooNode(ops.ValueOp):

        input_type = [ibis.expr.rules.integer(name='value')]

        def output_type(self):
            return Foo

    left = ibis.literal(2)
    right = FooNode(3).to_expr()
    result = left + right
    assert isinstance(result, Foo)
    assert isinstance(result.op(), FooNode)

    left = FooNode(3).to_expr()
    right = ibis.literal(2)
    result = left + right
    assert isinstance(result, Foo)
    assert isinstance(result.op(), FooNode)


def test_empty_array_as_argument():
    class Foo(ir.Expr):
        pass

    class FooNode(ops.ValueOp):

        input_type = [rules.array(dt.int64, name='value')]

        def output_type(self):
            return Foo

    node = FooNode([])
    value = node.value
    expected = literal([]).cast(dt.Array(dt.int64))
    assert not value.type().equals(dt.Array(dt.null))
    assert value.type().equals(dt.Array(dt.int64))
    assert value.equals(expected)


def test_struct_field_dir():
    t = ibis.table([('struct_col', 'struct<my_field: string>')])
    assert 'struct_col' in dir(t)
    assert 'my_field' in dir(t.struct_col)
