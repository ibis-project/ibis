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


from ibis.expr.types import (Schema, Expr,
                             ValueExpr, ScalarExpr, ArrayExpr,
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

from ibis.expr.operations import (as_value_expr, table, literal, null,
                                  value_list, desc)

import ibis.common as _com
import ibis.expr.analysis as _L
import ibis.expr.operations as _ops


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)
    """
    return _ops.SearchedCaseBuilder()


def now():
    """
    Compute the current timestamp

    Returns
    -------
    now : Timestamp scalar
    """
    return _ops.TimestampNow().to_expr()


e = _ops.E().to_expr()


def _add_methods(klass, method_table):
    for k, v in method_table.items():
        setattr(klass, k, v)


def _unary_op(name, klass, doc=None):
    def f(arg):
        return klass(arg).to_expr()
    f.__name__ = name
    if doc is not None:
        f.__doc__ = doc
    else:
        f.__doc__ = klass.__doc__
    return f


def negate(arg):
    """
    Negate a numeric expression

    Parameters
    ----------
    arg : numeric value expression

    Returns
    -------
    negated : type of caller
    """
    op = arg.op()
    if hasattr(op, 'negate'):
        result = op.negate()
    else:
        result = _ops.Negate(arg)

    return result.to_expr()


def count(expr):
    """
    Compute cardinality / sequence size of expression. For array expressions,
    the count is excluding nulls. For tables, it's the size of the entire
    table.

    Returns
    -------
    counts : int64 type
    """
    op = expr.op()
    if isinstance(op, _ops.DistinctArray):
        return op.count().to_expr()
    else:
        return _ops.Count(expr).to_expr()


def group_concat(arg, sep=','):
    """
    Concatenate values using the indicated separator (comma by default) to
    produce a string

    Parameters
    ----------
    arg : array expression
    sep : string, default ','

    Returns
    -------
    concatenated : string scalar
    """
    return _ops.GroupConcat(arg, sep=sep).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        other = as_value_expr(other)
        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary _ops, like radd, etc.
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
        return klass(self).to_expr()
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
    op = _ops.Cast(arg, target_type)

    if op.target_type == arg.type():
        # noop case if passed type is the same
        return arg
    else:
        return op.to_expr()


def fillna(arg, fill_value):
    """
    Replace any null values with the indicated fill value

    Parameters
    ----------
    fill_value : scalar / array value or expression

    Examples
    --------
    result = table.col.fillna(5)
    result2 = table.col.fillna(table.other_col * 3)

    Returns
    -------
    filled : type of caller
    """
    return arg.isnull().ifelse(fill_value, arg)


def coalesce(*args):
    """
    Compute the first non-null value(s) from the passed arguments in
    left-to-right order. This is also known as "combine_first" in pandas.

    Parameters
    ----------
    *args : variable-length value list

    Examples
    --------
    result = coalesce(expr1, expr2, 5)

    Returns
    -------
    coalesced : type of first provided argument
    """
    return _ops.Coalesce(args).to_expr()


def greatest(*args):
    """
    Compute the largest value (row-wise, if any arrays are present) among the
    supplied arguments.
    """
    return _ops.Greatest(args).to_expr()


def least(*args):
    """
    Compute the smallest value (row-wise, if any arrays are present) among the
    supplied arguments.
    """
    return _ops.Least(args).to_expr()


def where(boolean_expr, true_expr, false_null_expr):
    """

    Parameters
    ----------
    boolean_expr : BooleanValue (array or scalar)
    true_expr : value
      Values for each True value
    false_null_expr : value
      Values for False or NULL values

    Returns
    -------
    result : arity depending on inputs
      Type of true_expr used to determine output type
    """
    pass


def value_counts(arg, metric_name='count'):
    """
    Compute a frequency table for this value expression

    Parameters
    ----------

    Returns
    -------
    counts : TableExpr
      Aggregated table
    """
    base = _L.find_base_table(arg)
    metric = base.count().name(metric_name)

    try:
        arg.get_name()
    except _com.ExpressionError:
        arg = arg.name('unnamed')

    return base.group_by(arg).aggregate(metric)


def nullif(value, null_if_expr):
    """
    Set values to null if they match/equal a particular expression (scalar or
    array-valued).

    Common use to avoid divide-by-zero problems (get NULL instead of INF on
    divide-by-zero):
      5 / expr.nullif(0)

    Parameters
    ----------
    value : value expression
      Value to modify
    null_if_expr : value expression (array or scalar)

    Returns
    -------
    null_if : type of caller
    """
    return _ops.NullIf(value, null_if_expr).to_expr()


_generic_value_methods = dict(
    cast=cast,
    fillna=fillna,
    nullif=nullif,
    isnull=_unary_op('isnull', _ops.IsNull),
    notnull=_unary_op('notnull', _ops.NotNull),
    __add__=_binop_expr('__add__', _ops.Add),
    __sub__=_binop_expr('__sub__', _ops.Subtract),
    __mul__=_binop_expr('__mul__', _ops.Multiply),
    __div__=_binop_expr('__div__', _ops.Divide),
    __pow__=_binop_expr('__pow__', _ops.Power),

    __radd__=_rbinop_expr('__radd__', _ops.Add),
    __rsub__=_rbinop_expr('__rsub__', _ops.Subtract),
    __rmul__=_rbinop_expr('__rmul__', _ops.Multiply),
    __rdiv__=_rbinop_expr('__rdiv__', _ops.Divide),
    __rpow__=_binop_expr('__rpow__', _ops.Power),

    __eq__=_binop_expr('__eq__', _ops.Equals),
    __ne__=_binop_expr('__ne__', _ops.NotEquals),
    __ge__=_binop_expr('__ge__', _ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', _ops.Greater),
    __le__=_binop_expr('__le__', _ops.LessEqual),
    __lt__=_binop_expr('__lt__', _ops.Less)
)


approx_nunique = _agg_function('approx_nunique', _ops.HLLCardinality)
approx_median = _agg_function('approx_median', _ops.CMSMedian)


_generic_array_methods = dict(
    count=count,
    approx_median=approx_median,
    approx_nunique=approx_nunique,
    group_concat=group_concat,
    value_counts=value_counts
)


_add_methods(ValueExpr, _generic_value_methods)
_add_methods(ArrayExpr, _generic_array_methods)


#----------------------------------------------------------------------
# Numeric API

def round(arg, digits=None):
    """
    Round values either to integer or indicated number of decimal places.

    Returns
    -------
    rounded : type depending on digits argument
      digits None or 0
        decimal types: decimal
        other numeric types: bigint
      digits nonzero
        decimal types: decimal
        other numeric types: double
    """
    op = _ops.Round(arg, digits)
    return op.to_expr()


def log(arg, base=None):
    """
    Perform the logarithm using a specified base

    Parameters
    ----------
    base : number, default None
      If None, base e is used

    Returns
    -------
    logarithm : double type
    """
    op = _ops.Log(arg, base)
    return op.to_expr()


abs = _unary_op('abs', _ops.Abs)
ceil = _unary_op('ceil', _ops.Ceil)
exp = _unary_op('exp', _ops.Exp)
floor = _unary_op('floor', _ops.Floor)
log2 = _unary_op('log2', _ops.Log2)
log10 = _unary_op('log10', _ops.Log10)
ln = _unary_op('ln', _ops.Ln)
sign = _unary_op('sign', _ops.Sign)
sqrt = _unary_op('sqrt', _ops.Sqrt)


_numeric_value_methods = dict(
    __neg__=negate,
    abs=abs,
    ceil=ceil,
    floor=floor,
    sign=sign,
    exp=exp,
    sqrt=sqrt,
    log=log,
    ln=ln,
    log2=log2,
    log10=log10,
    round=round,
    zeroifnull=_unary_op('zeroifnull', _ops.ZeroIfNull),
)

mean = _agg_function('mean', _ops.Mean)
max = _agg_function('max', _ops.Max)
min = _agg_function('min', _ops.Min)
sum = _agg_function('sum', _ops.Sum)


_numeric_array_methods = dict(
    mean=mean,
    min=min,
    max=max,
    sum=sum,
)

_add_methods(NumericValue, _numeric_value_methods)
_add_methods(NumericArray, _numeric_array_methods)


#----------------------------------------------------------------------
# Boolean API


# TODO: logical binary operators for BooleanValue


_boolean_value_methods = dict(
    __and__=_boolean_binary_op('__and__', _ops.And),
    __or__=_boolean_binary_op('__or__', _ops.Or),
    __xor__=_boolean_binary_op('__xor__', _ops.Xor),
    __rand__=_boolean_binary_rop('__rand__', _ops.And),
    __ror__=_boolean_binary_rop('__ror__', _ops.Or),
    __rxor__=_boolean_binary_rop('__rxor__', _ops.Xor)
)


_boolean_array_methods = dict(
    any=_unary_op('any', _ops.Any)
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
    op = _ops.Substring(self, start, length)
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
    return _ops.StrRight(self, nchars).to_expr()


_string_value_methods = dict(
    length=_unary_op('length', _ops.StringLength),
    lower=_unary_op('lower', _ops.Lowercase),
    upper=_unary_op('upper', _ops.Uppercase),

    substr=_string_substr,
    left=_string_left,
    right=_string_right
)


_add_methods(StringValue, _string_value_methods)


#----------------------------------------------------------------------
# Timestamp API

_timestamp_value_methods = dict(
    year=_extract_field('year', _ops.ExtractYear),
    month=_extract_field('month', _ops.ExtractMonth),
    day=_extract_field('day', _ops.ExtractDay),
    hour=_extract_field('hour', _ops.ExtractHour),
    minute=_extract_field('minute', _ops.ExtractMinute),
    second=_extract_field('second', _ops.ExtractSecond),
    millisecond=_extract_field('millisecond', _ops.ExtractMillisecond)
)

_add_methods(TimestampValue, _timestamp_value_methods)


#----------------------------------------------------------------------
# Decimal API


#----------------------------------------------------------------------
# Table API

_join_classes = {
    'inner': _ops.InnerJoin,
    'left': _ops.LeftJoin,
    'outer': _ops.OuterJoin,
    'left_semi': _ops.LeftSemiJoin,
    'semi': _ops.LeftSemiJoin,
    'anti': _ops.LeftAntiJoin,
    'cross': _ops.CrossJoin
}


def join(left, right, predicates=(), how='inner'):
    """
    Perform a relational join between two tables. Does not resolve resulting
    table schema.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr
    predicates : join expression(s)
    how : string, default 'inner'
      - 'inner': inner join
      - 'left': left join
      - 'outer': full outer join
      - 'semi' or 'left_semi': left semi join
      - 'anti': anti join

    Returns
    -------
    joined : TableExpr
      Note, schema is not materialized yet
    """
    klass = _join_classes[how.lower()]
    if isinstance(predicates, Expr):
        predicates = _L.unwrap_ands(predicates)

    op = klass(left, right, predicates)
    return TableExpr(op)


def cross_join(left, right, prefixes=None):
    """

    """
    op = _ops.CrossJoin(left, right)
    return TableExpr(op)


def _regular_join_method(name, how, doc=None):
    def f(self, other, predicates=()):
        return self.join(other, predicates, how=how)
    if doc:
        f.__doc__ = doc
    else:
        # XXX
        f.__doc__ = join.__doc__
    f.__name__ = name
    return f


_table_methods = dict(
    join=join,
    cross_join=cross_join,
    inner_join=_regular_join_method('inner_join', 'inner'),
    left_join=_regular_join_method('left_join', 'left'),
    outer_join=_regular_join_method('outer_join', 'outer'),
    semi_join=_regular_join_method('semi_join', 'semi'),
    anti_join=_regular_join_method('anti_join', 'anti')
)


_add_methods(TableExpr, _table_methods)
