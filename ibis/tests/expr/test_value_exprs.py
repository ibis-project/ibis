import functools
import operator
import os
from collections import OrderedDict
from datetime import date, datetime, time
from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
import toolz

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as L
import ibis.expr.api as api
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis import literal
from ibis.common.exceptions import IbisTypeError
from ibis.expr.signature import Argument as Arg
from ibis.tests.util import assert_equal


def test_null():
    expr = ibis.literal(None)
    assert isinstance(expr, ir.NullScalar)
    assert isinstance(expr.op(), ops.NullLiteral)
    assert expr._arg.value is None

    expr2 = ibis.null()
    assert_equal(expr, expr2)

    assert expr is expr2
    assert expr.type() is dt.null
    assert expr2.type() is dt.null


@pytest.mark.xfail(
    raises=AssertionError,
    reason='UTF-8 support in Impala non-existent at the moment?',
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
        ([1, 2, 3], 'array<int8>'),
    ],
)
def test_literal_with_implicit_type(value, expected_type):
    expr = ibis.literal(value)

    assert isinstance(expr, ir.ScalarExpr)
    assert expr.type() == dt.dtype(expected_type)

    assert isinstance(expr.op(), ops.Literal)
    assert expr.op().value is value


pointA = (1, 2)
pointB = (-3, 4)
pointC = (5, 19)
lineAB = [pointA, pointB]
lineBC = [pointB, pointC]
lineCA = [pointC, pointA]
polygon1 = [lineAB, lineBC, lineCA]
polygon2 = [lineAB, lineBC, lineCA]
multilinestring = [lineAB, lineBC, lineCA]
multipoint = [pointA, pointB, pointC]
multipolygon1 = [polygon1, polygon2]


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
        (list(pointA), 'point'),
        (tuple(pointA), 'point'),
        (list(lineAB), 'linestring'),
        (tuple(lineAB), 'linestring'),
        (list(polygon1), 'polygon'),
        (tuple(polygon1), 'polygon'),
        (list(multilinestring), 'multilinestring'),
        (tuple(multilinestring), 'multilinestring'),
        (list(multipoint), 'multipoint'),
        (tuple(multipoint), 'multipoint'),
        (list(multipolygon1), 'multipolygon'),
        (tuple(multipolygon1), 'multipolygon'),
    ],
)
def test_literal_with_explicit_type(value, expected_type):
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
            ir.MapScalar,
        ),
        (
            OrderedDict(
                [
                    ('a', 1),
                    ('b', list('abc')),
                    ('c', OrderedDict([('foo', [1.0, 2.0])])),
                ]
            ),
            'struct<a: int8, b: array<string>, c: struct<foo: array<double>>>',
            ir.StructScalar,
        ),
    ],
)
def test_literal_complex_types(value, expected_type, expected_class):
    expr = ibis.literal(value)
    expr_type = expr.type()
    assert expr_type.equals(dt.validate_type(expected_type))
    assert isinstance(expr, expected_class)
    assert isinstance(expr.op(), ops.Literal)
    assert expr.op().value is value


def test_struct_operations():
    value = OrderedDict(
        [
            ('a', 1),
            ('b', list('abc')),
            ('c', OrderedDict([('foo', [1.0, 2.0])])),
        ]
    )
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
    assert isinstance(expr.length().op(), ops.MapLength)
    assert isinstance((expr + expr2).op(), ops.MapConcat)
    assert isinstance((expr2 + expr).op(), ops.MapConcat)

    default = ibis.literal([0.0])
    assert isinstance(expr.get('d', default).op(), ops.MapValueOrDefaultForKey)

    # test for an invalid default type, nulls are ok
    with pytest.raises(IbisTypeError):
        expr.get('d', ibis.literal('foo'))

    assert isinstance(
        expr.get('d', ibis.literal(None)).op(), ops.MapValueOrDefaultForKey
    )

    assert isinstance(expr['b'].op(), ops.MapValueForKey)
    assert isinstance(expr.keys().op(), ops.MapKeys)
    assert isinstance(expr.values().op(), ops.MapValues)


@pytest.mark.parametrize(
    ['value', 'expected_type'],
    [
        (32767, 'int8'),
        (32768, 'int16'),
        (2147483647, 'int16'),
        (2147483648, 'int32'),
        ('foo', 'double'),
    ],
)
def test_literal_with_non_coercible_type(value, expected_type):
    expected_msg = 'Value .* cannot be safely coerced to .*'
    with pytest.raises(TypeError, match=expected_msg):
        ibis.literal(value, type=expected_type)


def test_non_inferrable_literal():
    expected_msg = (
        'The datatype of value .* cannot be inferred, try '
        'passing it explicitly with the `type` keyword.'
    )

    value = tuple(pointA)

    with pytest.raises(TypeError, match=expected_msg):
        ibis.literal(value)

    point = ibis.literal(value, type='point')
    assert point.type() == dt.point


def test_literal_list():
    what = [1, 2, 1000]
    expr = api.literal(what)

    assert isinstance(expr, ir.ArrayScalar)

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


@pytest.mark.parametrize('container', [list, tuple, set, frozenset])
def test_isin_notin_list(table, container):
    values = container([1, 2, 3, 4])

    expr = table.a.isin(values)
    not_expr = table.a.notin(values)

    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.Contains)

    assert isinstance(not_expr, ir.BooleanColumn)
    assert isinstance(not_expr.op(), ops.NotContains)


def test_value_counts(table, string_col):
    bool_clause = table[string_col].notin(['1', '4', '7'])
    expr = table[bool_clause][string_col].value_counts()
    assert isinstance(expr, ir.TableExpr)


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

    result = a.notin([b, c, 3])
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


@pytest.mark.parametrize('where', [lambda t: None, lambda t: t.int_col != 0])
def test_distinct_count(functional_alltypes, where):
    result = functional_alltypes.string_col.distinct().count(
        where=where(functional_alltypes)
    )
    assert isinstance(result.op(), ops.CountDistinct)

    expected = functional_alltypes.string_col.nunique(
        where=where(functional_alltypes)
    ).name('count')
    assert result.equals(expected)


def test_distinct_unnamed_array_expr():
    table = ibis.table(
        [('year', 'int32'), ('month', 'int32'), ('day', 'int32')], 'foo'
    )

    # it works!
    expr = (
        ibis.literal('-')
        .join(
            [
                table.year.cast('string'),
                table.month.cast('string'),
                table.day.cast('string'),
            ]
        )
        .distinct()
    )
    repr(expr)


def test_distinct_count_numeric_types(functional_alltypes):
    metric = (
        functional_alltypes.bigint_col.distinct()
        .count()
        .name('unique_bigints')
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
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsNull)

    expr = ibis.literal('foo').isnull()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsNull)


def test_notnull(table):
    expr = table['g'].notnull()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.NotNull)

    expr = ibis.literal('foo').notnull()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.NotNull)


@pytest.mark.parametrize('column', ['e', 'f'], ids=['float', 'double'])
def test_isnan_isinf_column(table, column):
    expr = table[column].isnan()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsNan)

    expr = table[column].isinf()
    assert isinstance(expr, ir.BooleanColumn)
    assert isinstance(expr.op(), ops.IsInf)


@pytest.mark.parametrize('value', [1.3, np.nan, np.inf, -np.inf])
def test_isnan_isinf_scalar(value):
    expr = ibis.literal(value).isnan()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsNan)

    expr = ibis.literal(value).isinf()
    assert isinstance(expr, ir.BooleanScalar)
    assert isinstance(expr.op(), ops.IsInf)


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
    ],
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
    assert isinstance(result, ir.FloatingColumn)

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
    assert isinstance(log(ibis.literal(5)), ir.FloatingScalar)
    assert isinstance(log(ibis.literal(5.5)), ir.FloatingScalar)


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

    i = ibis.literal(5)
    assert i.cast('int8') is i


@pytest.mark.parametrize('type', ['int8', 'int32', 'double', 'float'])
def test_string_to_number(table, type):
    casted = table.g.cast(type)
    casted_literal = ibis.literal('5').cast(type).name('bar')

    assert isinstance(casted, ir.ColumnExpr)
    assert casted.type() == dt.dtype(type)

    assert isinstance(casted_literal, ir.ScalarExpr)
    assert casted_literal.type() == dt.dtype(type)
    assert casted_literal.get_name() == 'bar'


@pytest.mark.parametrize('col', list('abcdefh'))
def test_number_to_string_column(table, col):
    casted = table[col].cast('string')
    assert isinstance(casted, ir.StringColumn)


def test_number_to_string_scalar():
    casted_literal = ibis.literal(5).cast('string').name('bar')
    assert isinstance(casted_literal, ir.StringScalar)
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
    result = -(ibis.literal(False))
    assert isinstance(result, ir.BooleanScalar)
    assert isinstance(result.op(), ops.Negate)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_isnull_notnull():
    assert False


@pytest.mark.parametrize('column', ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
@pytest.mark.parametrize('how', [None, 'first', 'last', 'heavy'])
@pytest.mark.parametrize('condition_fn', [lambda t: None, lambda t: t.a > 8])
def test_arbitrary(table, column, how, condition_fn):
    col = table[column]
    where = condition_fn(table)
    expr = col.arbitrary(how=how, where=where)
    assert expr.type() == col.type()
    assert isinstance(expr, ir.ScalarExpr)
    assert L.is_reduction(expr)


@pytest.mark.parametrize(
    ['column', 'operation'],
    [
        ('h', lambda column: column.any()),
        ('h', lambda column: column.notany()),
        ('h', lambda column: column.all()),
        ('c', lambda column: (column == 0).any()),
        ('c', lambda column: (column == 0).all()),
    ],
)
def test_any_all_notany(table, column, operation):
    expr = operation(table[column])
    assert isinstance(expr, ir.BooleanScalar)
    assert L.is_reduction(expr)


@pytest.mark.parametrize(
    'operation',
    [
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ],
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
    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(result.op(), ex_op_class[operation])


def test_boolean_comparisons(table):
    bool_col = table.h

    result = bool_col == True  # noqa
    assert isinstance(result, ir.BooleanColumn)

    result = bool_col == False  # noqa
    assert isinstance(result, ir.BooleanColumn)


@pytest.mark.parametrize(
    'operation',
    [
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ],
)
def test_string_comparisons(table, operation):
    string_col = table.g
    result = operation(string_col, 'foo')
    assert isinstance(result, ir.BooleanColumn)


@pytest.mark.parametrize(
    'operation', [operator.xor, operator.or_, operator.and_]
)
def test_boolean_logical_ops(table, operation):
    expr = table.a > 0

    result = operation(expr, table.h)
    assert isinstance(result, ir.BooleanColumn)

    result = operation(expr, True)
    refl_result = operation(True, expr)
    assert isinstance(result, ir.BooleanColumn)
    assert isinstance(refl_result, ir.BooleanColumn)

    true = ibis.literal(True)
    false = ibis.literal(False)

    result = operation(true, false)
    assert isinstance(result, ir.BooleanScalar)


def test_null_column():
    t = ibis.table([('a', 'string')], name='t')
    s = t.mutate(b=ibis.NA)
    assert s.b.type() == dt.null
    assert isinstance(s.b, ir.NullColumn)


def test_null_column_union():
    s = ibis.table([('a', 'string'), ('b', 'double')])
    t = ibis.table([('a', 'string')])
    with pytest.raises(ibis.common.exceptions.RelationError):
        s.union(t.mutate(b=ibis.NA))  # needs a type
    assert s.union(t.mutate(b=ibis.NA.cast('double'))).schema() == s.schema()


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
    'operation', [operator.add, operator.mul, operator.truediv, operator.sub]
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
        (operator.mul, 'a', -(2 ** 24) + 1, 'int32'),
        (operator.mul, 'a', 1.5, 'double'),
        (operator.mul, 'b', 0, 'int16'),
        (operator.mul, 'b', 5, 'int32'),
        (operator.mul, 'b', -5, 'int32'),
        (operator.mul, 'c', 0, 'int32'),
        (operator.mul, 'c', 5, 'int64'),
        (operator.mul, 'c', -5, 'int64'),
        # technically this can overflow, but we allow it
        (operator.mul, 'd', 5, 'int64'),
        (operator.sub, 'a', 5, 'int16'),
        (operator.sub, 'a', 100000, 'int32'),
        (operator.sub, 'a', -100000, 'int32'),
        (operator.sub, 'a', 1.5, 'double'),
        (operator.sub, 'b', 5, 'int32'),
        (operator.sub, 'b', -5, 'int32'),
        (operator.sub, 'c', 5, 'int64'),
        (operator.sub, 'c', -5, 'int64'),
        # technically this can overflow, but we allow it
        (operator.sub, 'd', 5, 'int64'),
        (operator.truediv, 'a', 5, 'double'),
        (operator.truediv, 'a', 1.5, 'double'),
        (operator.truediv, 'b', 5, 'double'),
        (operator.truediv, 'b', -5, 'double'),
        (operator.truediv, 'c', 5, 'double'),
        (operator.pow, 'a', 0, 'double'),
        (operator.pow, 'b', 0, 'double'),
        (operator.pow, 'c', 0, 'double'),
        (operator.pow, 'd', 0, 'double'),
        (operator.pow, 'e', 0, 'float'),
        (operator.pow, 'f', 0, 'double'),
        (operator.pow, 'a', 2, 'double'),
        (operator.pow, 'b', 2, 'double'),
        (operator.pow, 'c', 2, 'double'),
        (operator.pow, 'd', 2, 'double'),
        (operator.pow, 'a', 1.5, 'double'),
        (operator.pow, 'b', 1.5, 'double'),
        (operator.pow, 'c', 1.5, 'double'),
        (operator.pow, 'd', 1.5, 'double'),
        (operator.pow, 'e', 2, 'float'),
        (operator.pow, 'f', 2, 'double'),
        (operator.pow, 'a', -2, 'double'),
        (operator.pow, 'b', -2, 'double'),
        (operator.pow, 'c', -2, 'double'),
        (operator.pow, 'd', -2, 'double'),
    ],
    ids=lambda arg: str(getattr(arg, '__name__', arg)),
)
def test_literal_promotions(table, op, name, case, ex_type):
    col = table[name]

    result = op(col, case)
    assert result.type() == dt.dtype(ex_type)

    result = op(case, col)
    assert result.type() == dt.dtype(ex_type)


@pytest.mark.parametrize(
    ('op', 'left_fn', 'right_fn', 'ex_type'),
    [
        (operator.sub, lambda t: t['a'], lambda t: 0, 'int8'),
        (operator.sub, lambda t: 0, lambda t: t['a'], 'int16'),
        (operator.sub, lambda t: t['b'], lambda t: 0, 'int16'),
        (operator.sub, lambda t: 0, lambda t: t['b'], 'int32'),
        (operator.sub, lambda t: t['c'], lambda t: 0, 'int32'),
        (operator.sub, lambda t: 0, lambda t: t['c'], 'int64'),
    ],
    ids=lambda arg: str(getattr(arg, '__name__', arg)),
)
def test_zero_subtract_literal_promotions(
    table, op, left_fn, right_fn, ex_type
):
    # in case of zero subtract the order of operands matters
    left, right = left_fn(table), right_fn(table)
    result = op(left, right)

    assert result.type() == dt.dtype(ex_type)


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
    exprs = [ibis.literal(1).name('a'), ibis.literal(2).name('b')]

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
    expected = (
        table.foo.case()
        .when('a', 'one')
        .when('b', table.bar)
        .else_(table.foo)
        .end()
    )
    assert_equal(result, expected)

    result = table.foo.substitute(subs, else_=ibis.NA)
    expected = (
        table.foo.case()
        .when('a', 'one')
        .when('b', table.bar)
        .else_(ibis.NA)
        .end()
    )
    assert_equal(result, expected)


@pytest.mark.parametrize(
    'typ',
    [
        'array<map<string, array<array<double>>>>',
        'string',
        'double',
        'float',
        'int64',
    ],
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
    ],
)
@pytest.mark.parametrize(
    'expr_func',
    [
        lambda t, args: t[args],
        lambda t, args: t.sort_by(args),
        lambda t, args: t.group_by(args).aggregate(bar_avg=t.bar.mean()),
    ],
)
def test_table_operations_with_integer_column(position, names, expr_func):
    t = ibis.table([('foo', 'string'), ('bar', 'double')])
    result = expr_func(t, position)
    expected = expr_func(t, names)
    assert result.equals(expected)


@pytest.mark.parametrize('value', ['abcdefg', ['a', 'b', 'c'], [1, 2, 3]])
@pytest.mark.parametrize(
    'operation', ['pow', 'sub', 'truediv', 'floordiv', 'mod']
)
def test_generic_value_api_no_arithmetic(value, operation):
    func = getattr(operator, operation)
    expr = ibis.literal(value)
    with pytest.raises(TypeError):
        func(expr, expr)


@pytest.mark.parametrize(
    ('value', 'expected'), [(5, dt.int8), (5.4, dt.double), ('abc', dt.string)]
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
    ],
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
        lambda left, right: ibis.timestamp('2017-04-01 00:02:34').between(
            left, right
        ),
        lambda left, right: ibis.timestamp('2017-04-01')
        .cast(dt.date)
        .between(left, right),
    ],
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
    ],
)
def test_decimal_modulo_output_type(value, type, expected_type_class):
    t = ibis.table([('a', type)])
    expr = t.a % value
    assert isinstance(expr.type(), expected_type_class)


@pytest.mark.parametrize(
    ('left', 'right'),
    [(literal('10:00'), time(10, 0)), (time(10, 0), literal('10:00'))],
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
    ],
)
def test_time_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (literal('10:00'), date(2017, 4, 2)),
        (literal('10:00'), datetime(2017, 4, 2, 1, 1)),
        (literal('10:00'), literal('2017-04-01')),
    ],
)
@pytest.mark.parametrize(
    'op', [operator.eq, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_time_timestamp_invalid_compare(op, left, right):
    result = op(left, right)
    assert result.type().equals(dt.boolean)


def test_scalar_parameter_set():
    value = ibis.param({dt.int64})

    assert isinstance(value.op(), ops.ScalarParameter)
    assert value.type().equals(dt.Set(dt.int64))


def test_scalar_parameter_repr():
    value = ibis.param(dt.timestamp).name('value')
    assert repr(value) == 'value = ScalarParameter[timestamp]'

    value_op = value.op()
    assert repr(value_op) == "ScalarParameter(type=timestamp)"


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (
            # same value type, same name
            ibis.param(dt.timestamp),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # different value type, same name
            ibis.param(dt.date),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # same value type, different name
            ibis.param(dt.timestamp),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # different value type, different name
            ibis.param(dt.date),
            ibis.param(dt.timestamp),
            False,
        ),
        (
            # different Python class, left side is param
            ibis.param(dt.timestamp),
            dt.date,
            False,
        ),
        (
            # different Python class, right side is param
            dt.date,
            ibis.param(dt.timestamp),
            False,
        ),
    ],
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
    ],
)
@pytest.mark.parametrize(
    ('left', 'right'), [(1, 'a'), ('a', 1), (1.0, 2.0), (['a'], [1])]
)
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
        value = Arg(rlz.integer)

        def output_type(self):
            return functools.partial(Foo, dtype=dt.int64)

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
        value = Arg(rlz.value(dt.Array(dt.int64)))

        def output_type(self):
            return Foo

    node = FooNode([])
    value = node.value
    expected = literal([]).cast(dt.Array(dt.int64))

    assert value.type().equals(dt.Array(dt.null))
    assert value.cast(dt.Array(dt.int64)).equals(expected)


def test_struct_field_dir():
    t = ibis.table([('struct_col', 'struct<my_field: string>')])
    assert 'struct_col' in dir(t)
    assert 'my_field' in dir(t.struct_col)


def test_nullable_column_propagated():
    t = ibis.table(
        [
            ('a', dt.Int32(nullable=True)),
            ('b', dt.Int32(nullable=False)),
            ('c', dt.String(nullable=False)),
            ('d', dt.double),  # nullable by default
            ('f', dt.Double(nullable=False)),
        ]
    )

    assert t.a.type().nullable is True
    assert t.b.type().nullable is False
    assert t.c.type().nullable is False
    assert t.d.type().nullable is True
    assert t.f.type().nullable is False

    s = t.a + t.d
    assert s.type().nullable is True

    s = t.b + t.d
    assert s.type().nullable is True

    s = t.b + t.f
    assert s.type().nullable is False


@pytest.mark.parametrize(
    'base_expr',
    [
        ibis.table([('interval_col', dt.Interval(unit='D'))]).interval_col,
        ibis.interval(seconds=42),
    ],
)
def test_interval_negate(base_expr):
    expr = -base_expr
    expr2 = base_expr.negate()
    expr3 = ibis.negate(base_expr)
    assert isinstance(expr.op(), ops.Negate)
    assert expr.equals(expr2)
    assert expr.equals(expr3)


def test_large_timestamp():
    expr = ibis.timestamp('4567-02-03')
    expected = datetime(year=4567, month=2, day=3)
    result = expr.op().value
    assert result == expected


@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_timestamp_with_timezone(tz):
    expr = ibis.timestamp('2017-01-01', timezone=tz)
    expected = pd.Timestamp('2017-01-01', tz=tz)
    result = expr.op().value
    assert expected == result


@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_timestamp_timezone_type(tz):
    expr = ibis.timestamp('2017-01-01', timezone=tz)
    expected = dt.Timestamp(timezone=tz)
    assert expected == expr.op().dtype


def test_map_get_broadcast():
    t = ibis.table([('a', 'string')], name='t')
    lookup_table = ibis.literal({'a': 1, 'b': 2})
    expr = lookup_table.get(t.a)
    assert isinstance(expr, ir.IntegerColumn)


def test_map_getitem_broadcast():
    t = ibis.table([('a', 'string')], name='t')
    lookup_table = ibis.literal({'a': 1, 'b': 2})
    expr = lookup_table[t.a]
    assert isinstance(expr, ir.IntegerColumn)


def test_map_keys_output_type():
    mapping = ibis.literal({'a': 1, 'b': 2})
    assert mapping.keys().type() == dt.Array(dt.string)


def test_map_values_output_type():
    mapping = ibis.literal({'a': 1, 'b': 2})
    assert mapping.values().type() == dt.Array(dt.int8)


def test_scalar_isin_map_keys():
    mapping = ibis.literal({'a': 1, 'b': 2})
    key = ibis.literal('a')
    expr = key.isin(mapping.keys())
    assert isinstance(expr, ir.BooleanScalar)


def test_column_isin_map_keys():
    t = ibis.table([('a', 'string')], name='t')
    mapping = ibis.literal({'a': 1, 'b': 2})
    expr = t.a.isin(mapping.keys())
    assert isinstance(expr, ir.BooleanColumn)


def test_map_get_with_compatible_value_smaller():
    value = ibis.literal({'A': 1000, 'B': 2000})
    expr = value.get('C', 3)
    assert value.type() == dt.Map(dt.string, dt.int16)
    assert expr.type() == dt.int16


def test_map_get_with_compatible_value_bigger():
    value = ibis.literal({'A': 1, 'B': 2})
    expr = value.get('C', 3000)
    assert value.type() == dt.Map(dt.string, dt.int8)
    assert expr.type() == dt.int16


def test_map_get_with_incompatible_value_different_kind():
    value = ibis.literal({'A': 1000, 'B': 2000})
    with pytest.raises(IbisTypeError):
        value.get('C', 3.0)


@pytest.mark.parametrize('null_value', [None, ibis.NA])
def test_map_get_with_null_on_not_nullable(null_value):
    map_type = dt.Map(dt.string, dt.Int16(nullable=False))
    value = ibis.literal({'A': 1000, 'B': 2000}).cast(map_type)
    assert value.type() == map_type
    with pytest.raises(IbisTypeError):
        assert value.get('C', null_value)


@pytest.mark.parametrize('null_value', [None, ibis.NA])
def test_map_get_with_null_on_nullable(null_value):
    value = ibis.literal({'A': 1000, 'B': None})
    result = value.get('C', null_value)
    assert result.type().nullable


@pytest.mark.parametrize('null_value', [None, ibis.NA])
def test_map_get_with_null_on_null_type_with_null(null_value):
    value = ibis.literal({'A': None, 'B': None})
    result = value.get('C', null_value)
    assert result.type().nullable


def test_map_get_with_null_on_null_type_with_non_null():
    value = ibis.literal({'A': None, 'B': None})
    assert value.get('C', 1).type() == dt.int8


def test_map_get_with_incompatible_value():
    value = ibis.literal({'A': 1000, 'B': 2000})
    with pytest.raises(IbisTypeError):
        value.get('C', ['A'])


@pytest.mark.parametrize(
    ('value', 'expected_type'),
    [
        (datetime.now(), dt.timestamp),
        (datetime.now().date(), dt.date),
        (datetime.now().time(), dt.time),
    ],
)
def test_invalid_negate(value, expected_type):
    expr = ibis.literal(value)
    assert expr.type() == expected_type
    with pytest.raises(TypeError):
        -expr


@pytest.mark.parametrize(
    'type',
    [
        np.float16,
        np.float32,
        np.float64,
        np.int16,
        np.int32,
        np.int64,
        np.int64,
        np.int8,
        np.timedelta64,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uint64,
        np.uint8,
        float,
        int,
    ],
)
def test_valid_negate(type):
    value = type(1)
    expr = ibis.literal(value)
    assert -expr is not None


@pytest.mark.xfail(
    reason='Type not supported in most backends', raises=TypeError
)
@pytest.mark.skipif(
    os.name == 'nt', reason='np.float128 not appear to exist on windows'
)
def test_valid_negate_float128():
    value = np.float128(1)
    expr = ibis.literal(value)
    assert -expr is not None


@pytest.mark.parametrize(
    ('kind', 'begin', 'end'),
    [
        ('preceding', None, None),
        ('preceding', 1, None),
        ('preceding', -1, 1),
        ('preceding', 1, -1),
        ('preceding', -1, -1),
        ('following', None, None),
        ('following', None, 1),
        ('following', -1, 1),
        ('following', 1, -1),
        ('following', -1, -1),
    ],
)
def test_window_unbounded_invalid(kind, begin, end):
    kwargs = {kind: (begin, end)}
    with pytest.raises(com.IbisInputError):
        ibis.window(**kwargs)


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (ibis.literal(1), ibis.literal(1.0), dt.float64),
        (ibis.literal('a'), ibis.literal('b'), dt.string),
        (ibis.literal(1.0), ibis.literal(1), dt.float64),
        (ibis.literal(1), ibis.literal(1), dt.int8),
        (ibis.literal(1), ibis.literal(1000), dt.int16),
        (ibis.literal(2 ** 16), ibis.literal(2 ** 17), dt.int32),
        (ibis.literal(2 ** 50), ibis.literal(1000), dt.int64),
        (ibis.literal([1, 2]), ibis.literal([1, 2]), dt.Array(dt.int8)),
        (ibis.literal(['a']), ibis.literal([]), dt.Array(dt.string)),
        (ibis.literal([]), ibis.literal(['a']), dt.Array(dt.string)),
        (ibis.literal([]), ibis.literal([]), dt.Array(dt.null)),
    ],
)
def test_nullif_type(left, right, expected):
    assert left.nullif(right).type() == expected


@pytest.mark.parametrize(
    ('left', 'right'), [(ibis.literal(1), ibis.literal('a'))]
)
def test_nullif_fail(left, right):
    with pytest.raises(com.IbisTypeError):
        left.nullif(right)
    with pytest.raises(com.IbisTypeError):
        right.nullif(left)


@pytest.mark.parametrize(
    "join_method",
    [
        "left_join",
        pytest.param(
            "right_join",
            marks=pytest.mark.xfail(
                raises=AttributeError, reason="right_join is not an ibis API"
            ),
        ),
        "inner_join",
        "outer_join",
        "asof_join",
        pytest.param(
            "semi_join",
            marks=pytest.mark.xfail(
                raises=com.IbisTypeError,
                reason=(
                    "semi_join only gives access to the left table's "
                    "columns"
                ),
            ),
        ),
    ],
)
@pytest.mark.xfail(
    raises=(com.IbisError, AttributeError),
    reason="Select from unambiguous joins not implemented",
)
def test_select_on_unambiguous_join(join_method):
    t = ibis.table([("a0", dt.int64), ("b1", dt.string)], name="t")
    s = ibis.table([("a1", dt.int64), ("b2", dt.string)], name="s")
    method = getattr(t, join_method)
    join = method(s, t.b1 == s.b2)
    expr1 = join["a0", "a1"]
    expr2 = join[["a0", "a1"]]
    expr3 = join.select(["a0", "a1"])
    assert expr1.equals(expr2)
    assert expr1.equals(expr3)


def test_chained_select_on_join():
    t = ibis.table([("a", dt.int64)], name="t")
    s = ibis.table([("a", dt.int64), ("b", dt.string)], name="s")
    join = t.join(s)[t.a, s.b]
    expr1 = join["a", "b"]
    expr2 = join.select(["a", "b"])
    assert expr1.equals(expr2)


def test_repr_list_of_lists():
    lit = ibis.literal([[1]])
    result = repr(lit)
    expected = """\
Literal[array<array<int8>>]
  [[1]]"""
    assert result == expected


def test_repr_list_of_lists_in_table():
    t = ibis.table([('a', 'int64')], name='t')
    lit = ibis.literal([[1]])
    expr = t[t, lit.name('array_of_array')]
    result = repr(expr)
    expected = """\
ref_0
UnboundTable[table]
  name: t
  schema:
    a : int64

Selection[table]
  table:
    Table: ref_0
  selections:
    Table: ref_0
    array_of_array = Literal[array<array<int8>>]
      [[1]]"""
    assert result == expected
