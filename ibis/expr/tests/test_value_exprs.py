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

import pytest

from ibis.common import IbisTypeError
import ibis.expr.api as api
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis

from ibis.tests.util import assert_equal


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


def test_literal_list():
    what = [1, 2, 1000]
    expr = api.as_value_expr(what)

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr.op(), ir.ValueList)
    assert isinstance(expr.op().values[2], ir.Int16Scalar)

    # it works!
    repr(expr)


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


@pytest.fixture
def dtable(con):
    return con.table('functional_alltypes')


def test_distinct_basic(dtable):
    expr = dtable.distinct()
    assert isinstance(expr.op(), ops.Distinct)
    assert isinstance(expr, ir.TableExpr)
    assert expr.op().table is dtable

    expr = dtable.string_col.distinct()
    assert isinstance(expr.op(), ops.DistinctColumn)

    assert isinstance(expr, ir.StringColumn)


@pytest.mark.xfail(reason='NYT')
def test_distinct_array_interactions(dtable):
    # array cardinalities / shapes are likely to be different.
    a = dtable.int_col.distinct()
    b = dtable.bigint_col

    with pytest.raises(ir.RelationError):
        a + b


def test_distinct_count(dtable):
    result = dtable.string_col.distinct().count()
    expected = dtable.string_col.nunique().name('count')
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


def test_distinct_count_numeric_types(dtable):
    metric = dtable.bigint_col.distinct().count().name('unique_bigints')
    dtable.group_by('string_col').aggregate(metric)


def test_nunique(dtable):
    expr = dtable.string_col.nunique()
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
