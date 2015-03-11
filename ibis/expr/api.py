# Copyright 2015 Cloudera Inc.
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


from ibis.expr.types import (Schema,
                             ValueExpr,
                             ScalarExpr,
                             ArrayExpr,
                             TableExpr,
                             NumericValue, NumericArray,
                             Int8Value, Int8Scalar, Int8Array,
                             Int16Value, Int16Scalar, Int16Array,
                             Int32Value, Int32Scalar, Int32Array,
                             Int64Value, Int64Scalar, Int64Array,
                             NullScalar,
                             BooleanValue, BooleanScalar, BooleanArray,
                             FloatValue, FloatScalar, FloatArray,
                             DoubleValue, DoubleScalar, DoubleArray,
                             StringValue, StringScalar, StringArray,
                             DecimalValue, DecimalScalar, DecimalArray,
                             TimestampValue, TimestampScalar, TimestampArray,
                             unnamed)
import ibis.expr.types as ir

from ibis.expr.operations import (as_value_expr, table, literal, null,
                                  value_list, desc)
import ibis.expr.operations as ops


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)
    """
    return ops.SearchedCaseBuilder()


def now():
    """
    Compute the current timestamp

    Returns
    -------
    now : Timestamp scalar
    """
    return ops.TimestampNow().to_expr()


def _add_methods(klass, method_table):
    for k, v in method_table.items():
        setattr(klass, k, v)


def _unary_op(name, klass, doc=None):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    if doc is not None:
        f.__doc__ = doc
    else:
        f.__doc__ = klass.__doc__
    return f


def _negate(expr):
    op = expr.op()
    if hasattr(op, 'negate'):
        return op.negate()
    else:
        return ops.Negate(expr)


def _count(expr):
    """
    Compute cardinality / sequence size of expression

    Returns
    -------
    counts : int64 type
    """
    op = expr.op()
    if isinstance(op, ops.DistinctArray):
        return op.count()
    else:
        return ops.Count(expr)


def _binop_expr(name, klass):
    def f(self, other):
        other = as_value_expr(other)
        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary ops, like radd, etc.
    def f(self, other):
        other = as_value_expr(other)
        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _boolean_binary_op(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _boolean_binary_rop(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _agg_function(name, klass):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    return f


def _extract_field(name, klass):
    def f(self):
        op = klass(self)
        return op.to_expr()
    f.__name__ = name
    return f


#----------------------------------------------------------------------
# Generic value API


def cast(arg, target_type):
    """
    Cast value(s) to indicated data type. Values that cannot be
    successfully casted

    Parameters
    ----------
    target_type : data type name

    Returns
    -------
    cast_expr : ValueExpr
    """
    # validate
    op = ops.Cast(arg, target_type)

    if op.target_type == arg.type():
        # noop case if passed type is the same
        return arg
    else:
        return op.to_expr()


_generic_value_methods = dict(
    cast=cast,
    isnull=_unary_op('isnull', ops.IsNull),
    notnull=_unary_op('notnull', ops.NotNull),
    __add__=_binop_expr('__add__', ops.Add),
    __sub__=_binop_expr('__sub__', ops.Subtract),
    __mul__=_binop_expr('__mul__', ops.Multiply),
    __div__=_binop_expr('__div__', ops.Divide),
    __pow__=_binop_expr('__pow__', ops.Power),

    __radd__=_rbinop_expr('__radd__', ops.Add),
    __rsub__=_rbinop_expr('__rsub__', ops.Subtract),
    __rmul__=_rbinop_expr('__rmul__', ops.Multiply),
    __rdiv__=_rbinop_expr('__rdiv__', ops.Divide),
    __rpow__=_binop_expr('__rpow__', ops.Power),

    __eq__=_binop_expr('__eq__', ops.Equals),
    __ne__=_binop_expr('__ne__', ops.NotEquals),
    __ge__=_binop_expr('__ge__', ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', ops.Greater),
    __le__=_binop_expr('__le__', ops.LessEqual),
    __lt__=_binop_expr('__lt__', ops.Less)
)

_generic_array_methods = dict(
    count=_unary_op('count', _count)
)


_add_methods(ValueExpr, _generic_value_methods)
_add_methods(ArrayExpr, _generic_array_methods)


#----------------------------------------------------------------------
# Numeric API


_numeric_value_methods = dict(
    __neg__=_unary_op('__neg__', _negate),
    abs=_unary_op('abs', ops.Abs),
    ceil=_unary_op('ceil', ops.Ceil),
    floor=_unary_op('floor', ops.Floor),
    sign=_unary_op('sign', ops.Sign),
    exp=_unary_op('exp', ops.Exp),
    sqrt=_unary_op('sqrt', ops.Sqrt),
    log=_unary_op('log', ops.Log),
    ln=_unary_op('log', ops.Log),
    log2=_unary_op('log2', ops.Log2),
    log10=_unary_op('log10', ops.Log10),
)


_numeric_array_methods = dict(
    sum=_agg_function('sum', ops.Sum),
    mean=_agg_function('mean', ops.Mean),
    min=_agg_function('min', ops.Min),
    max=_agg_function('max', ops.Max)
)

_add_methods(NumericValue, _numeric_value_methods)
_add_methods(NumericArray, _numeric_array_methods)


#----------------------------------------------------------------------
# Boolean API


# TODO: logical binary operators for BooleanValue


_boolean_value_methods = dict(
    __and__=_boolean_binary_op('__and__', ops.And),
    __or__=_boolean_binary_op('__or__', ops.Or),
    __xor__=_boolean_binary_op('__xor__', ops.Xor),
    __rand__=_boolean_binary_rop('__rand__', ops.And),
    __ror__=_boolean_binary_rop('__ror__', ops.Or),
    __rxor__=_boolean_binary_rop('__rxor__', ops.Xor)
)


_boolean_array_methods = dict(
    any=_unary_op('any', ops.Any)
)


_add_methods(BooleanValue, _boolean_value_methods)
_add_methods(BooleanArray, _boolean_array_methods)


#----------------------------------------------------------------------
# String API

def _string_substr(self, start, length=None):
    """
    Pull substrings out of each string value by position and maximum
    length.

    Parameters
    ----------
    start : int
      First character to start splitting, indices starting at 0 (like
      Python)
    length : int, optional
      Maximum length of each substring. If not supplied, splits each string
      to the end

    Returns
    -------
    substrings : type of caller
    """
    op = ops.Substring(self, start, length)
    return op.to_expr()


def _string_left(self, nchars):
    """
    Return left-most up to N characters from each string. Convenience
    use of substr.

    Returns
    -------
    substrings : type of caller
    """
    return self.substr(0, length=nchars)


def _string_right(self, nchars):
    """
    Split up to nchars starting from end of each string.

    Returns
    -------
    substrings : type of caller
    """
    return ops.StrRight(self, nchars).to_expr()


_string_value_methods = dict(
    length=_unary_op('length', ops.StringLength),
    lower=_unary_op('lower', ops.Lowercase),
    upper=_unary_op('upper', ops.Uppercase),

    substr=_string_substr,
    left=_string_left,
    right=_string_right
)


_add_methods(StringValue, _string_value_methods)


#----------------------------------------------------------------------
# Timestamp API

_timestamp_value_methods = dict(
    year=_extract_field('year', ops.ExtractYear),
    month=_extract_field('month', ops.ExtractMonth),
    day=_extract_field('day', ops.ExtractDay),
    hour=_extract_field('hour', ops.ExtractHour),
    minute=_extract_field('minute', ops.ExtractMinute),
    second=_extract_field('second', ops.ExtractSecond),
    millisecond=_extract_field('millisecond', ops.ExtractMillisecond)
)

_add_methods(TimestampValue, _timestamp_value_methods)


#----------------------------------------------------------------------
# Decimal API


#----------------------------------------------------------------------
# Table API

def join(left, right, predicates=(), prefixes=None, how='inner'):
    pass


def cross_join(left, right, prefixes=None):
    """

    """
    op = ops.CrossJoin(left, right)
    return TableExpr(op)


def _regular_join_method(name, klass, doc=None):
    def f(self, other, predicates=(), prefixes=None):
        op = klass(self, other, predicates)
        return TableExpr(op)
    if doc:
        f.__doc__ = doc
    f.__name__ = name

    return f


_table_methods = dict(
    cross_join=cross_join,
    inner_join=_regular_join_method('inner_join', ops.InnerJoin),
    left_join=_regular_join_method('left_join', ops.LeftJoin),
    outer_join=_regular_join_method('outer_join', ops.OuterJoin),
    semi_join=_regular_join_method('semi_join', ops.LeftSemiJoin),
    anti_join=_regular_join_method('anti_join', ops.LeftAntiJoin),
)


_add_methods(TableExpr, _table_methods)
