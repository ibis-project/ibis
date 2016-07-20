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

import six

from ibis.expr.datatypes import Schema  # noqa
from ibis.expr.types import (Expr,  # noqa
                             ValueExpr, ScalarExpr, ArrayExpr,
                             TableExpr,
                             NumericValue, NumericArray,
                             IntegerValue,
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
                             CategoryValue, unnamed, as_value_expr, literal,
                             null, sequence)

# __all__ is defined
from ibis.expr.temporal import *  # noqa

import ibis.common as _com

from ibis.compat import py_string
from ibis.expr.analytics import bucket, histogram
from ibis.expr.groupby import GroupedTableExpr  # noqa
from ibis.expr.window import window, trailing_window, cumulative_window
import ibis.expr.analytics as _analytics
import ibis.expr.analysis as _L
import ibis.expr.types as ir
import ibis.expr.operations as _ops
import ibis.expr.temporal as _T
import ibis.util as util


__all__ = [
    'schema', 'table', 'literal', 'expr_list', 'timestamp',
    'case', 'where', 'sequence',
    'now', 'desc', 'null', 'NA',
    'cast', 'coalesce', 'greatest', 'least',
    'cross_join', 'join',
    'aggregate',
    'row_number',
    'negate', 'ifelse',
    'Expr', 'Schema',
    'window', 'trailing_window', 'cumulative_window'
]
__all__ += _T.__all__


NA = null()


_data_type_docs = """\
Ibis uses its own type aliases that map onto database types. See, for
example, the correspondence between Ibis type names and Impala type names:

Ibis type      Impala Type
~~~~~~~~~      ~~~~~~~~~~~
int8           TINYINT
int16          SMALLINT
int32          INT
int64          BIGINT
float          FLOAT
double         DOUBLE
boolean        BOOLEAN
string         STRING
timestamp      TIMESTAMP
decimal(p, s)  DECIMAL(p,s)"""


def schema(pairs=None, names=None, types=None):
    if pairs is not None:
        return Schema.from_tuples(pairs)
    else:
        return Schema(names, types)


def table(schema, name=None):
    """
    Create an unbound Ibis table for creating expressions. Cannot be executed
    without being bound to some physical table.

    Useful for testing

    Parameters
    ----------
    schema : ibis Schema
    name : string, default None
      Name for table

    Returns
    -------
    table : TableExpr
    """
    if not isinstance(schema, Schema):
        if isinstance(schema, dict):
            schema = Schema.from_dict(schema)
        else:
            schema = Schema.from_tuples(schema)

    node = _ops.UnboundTable(schema, name=name)
    return TableExpr(node)


def desc(expr):
    """
    Create a sort key (when used in sort_by) by the passed array expression or
    column name.

    Parameters
    ----------
    expr : array expression or string
      Can be a column name in the table being sorted

    Examples
    --------
    result = (self.table.group_by('g')
              .size('count')
              .sort_by(ibis.desc('count')))
    """
    if not isinstance(expr, Expr):
        return _ops.DeferredSortKey(expr, ascending=False)
    else:
        return _ops.SortKey(expr, ascending=False).to_expr()


def timestamp(value):
    """
    Returns a timestamp literal if value is likely coercible to a timestamp
    """
    if isinstance(value, py_string):
        from pandas import Timestamp
        value = Timestamp(value)
    op = ir.Literal(value)
    return ir.TimestampScalar(op)


schema.__doc__ = """\
Validate and return an Ibis Schema object

{0}

Parameters
----------
pairs : list of (name, type) tuples
  Mutually exclusive with names/types
names : list of string
  Field names
types : list of string
  Field types

Examples
--------
sc = schema([('foo', 'string'),
             ('bar', 'int64'),
             ('baz', 'boolean')])

sc2 = schema(names=['foo', 'bar', 'baz'],
             types=['string', 'int64', 'boolean'])

Returns
-------
schema : Schema
""".format(_data_type_docs)


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)

    Use the .when method on the resulting object followed by .end to create a
    complete case.

    Examples
    --------
    expr = (ibis.case()
            .when(cond1, result1)
            .when(cond2, result2).end())

    Returns
    -------
    case : CaseBuilder
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


def row_number():
    """
    Analytic function for the current row number, starting at 0

    Returns
    -------
    row_number : IntArray
    """
    return _ops.RowNumber().to_expr()


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


def count(expr, where=None):
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
        if where is not None:
            raise NotImplementedError
        result = op.count().to_expr()
    else:
        result = _ops.Count(expr, where).to_expr()

    return result.name('count')


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
    return _ops.GroupConcat(arg, sep).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        try:
            other = as_value_expr(other)
            op = klass(self, other)
            return op.to_expr()
        except _com.InputTypeError:
            return NotImplemented

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


def _agg_function(name, klass, assign_default_name=True):
    def f(self, where=None):
        expr = klass(self, where).to_expr()
        if assign_default_name:
            expr = expr.name(name)
        return expr
    f.__name__ = name
    return f


def _extract_field(name, klass):
    def f(self):
        expr = klass(self).to_expr()
        return expr.name(name)
    f.__name__ = name
    return f


# ---------------------------------------------------------------------
# Generic value API


def cast(arg, target_type):
    # validate
    op = _ops.Cast(arg, target_type)

    if op.args[1] == arg.type():
        # noop case if passed type is the same
        return arg
    else:
        result = op.to_expr()
        try:
            expr_name = ('cast({0}, {1!s})'
                         .format(arg.get_name(),
                                 op.args[1]))
            result = result.name(expr_name)
        except:
            pass
        return result

cast.__doc__ = """
Cast value(s) to indicated data type. Values that cannot be
successfully casted

Parameters
----------
target_type : data type name

Notes
-----
{0}

Returns
-------
cast_expr : ValueExpr
""".format(_data_type_docs)


def typeof(arg):
    """
    Return the data type of the argument according to the current backend

    Returns
    -------
    typeof_arg : string
    """
    return _ops.TypeOf(arg).to_expr()


def hash(arg, how='fnv'):
    """
    Compute an integer hash value for the indicated value expression.

    Parameters
    ----------
    arg : value expression
    how : {'fnv'}, default 'fnv'
      Hash algorithm to use

    Returns
    -------
    hash_value : int64 expression
    """
    return _ops.Hash(arg, how).to_expr()


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
    return _ops.IfNull(arg, fill_value).to_expr()


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
    return _ops.Coalesce(*args).to_expr()


def greatest(*args):
    """
    Compute the largest value (row-wise, if any arrays are present) among the
    supplied arguments.

    Returns
    -------
    greatest : type depending on arguments
    """
    return _ops.Greatest(*args).to_expr()


def least(*args):
    """
    Compute the smallest value (row-wise, if any arrays are present) among the
    supplied arguments.

    Returns
    -------
    least : type depending on arguments
    """
    return _ops.Least(*args).to_expr()


def where(boolean_expr, true_expr, false_null_expr):
    """
    Equivalent to the ternary expression: if X then Y else Z

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
    op = _ops.Where(boolean_expr, true_expr, false_null_expr)
    return op.to_expr()


def over(expr, window):
    """
    Turn an aggregation or full-sample analytic operation into a windowed
    operation. See ibis.window for more details on window configuration

    Parameters
    ----------
    expr : value expression
    window : ibis.Window

    Returns
    -------
    expr : type of input
    """
    prior_op = expr.op()

    if isinstance(prior_op, _ops.WindowOp):
        op = prior_op.over(window)
    else:
        op = _ops.WindowOp(expr, window)

    result = op.to_expr()

    try:
        result = result.name(expr.get_name())
    except:
        pass

    return result


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
    base = ir.find_base_table(arg)
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
    divide-by-zero): 5 / expr.nullif(0)

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


def between(arg, lower, upper):
    """
    Check if the input expr falls between the lower/upper bounds
    passed. Bounds are inclusive. All arguments must be comparable.

    Returns
    -------
    is_between : BooleanValue
    """
    lower = _ops.as_value_expr(lower)
    upper = _ops.as_value_expr(upper)
    op = _ops.Between(arg, lower, upper)
    return op.to_expr()


def isin(arg, values):
    """
    Check whether the value expression is contained within the indicated
    list of values.

    Parameters
    ----------
    values : list, tuple, or array expression
      The values can be scalar or array-like. Each of them must be
      comparable with the calling expression, or None (NULL).

    Examples
    --------
    expr = table.strings.isin(['foo', 'bar', 'baz'])

    expr2 = table.strings.isin(table2.other_string_col)

    Returns
    -------
    contains : BooleanValue
    """
    op = _ops.Contains(arg, values)
    return op.to_expr()


def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = _ops.NotContains(arg, values)
    return op.to_expr()


add = _binop_expr('__add__', _ops.Add)
sub = _binop_expr('__sub__', _ops.Subtract)
mul = _binop_expr('__mul__', _ops.Multiply)
div = _binop_expr('__div__', _ops.Divide)
floordiv = _binop_expr('__floordiv__', _ops.FloorDivide)
pow = _binop_expr('__pow__', _ops.Power)
mod = _binop_expr('__mod__', _ops.Modulus)

rsub = _rbinop_expr('__rsub__', _ops.Subtract)
rdiv = _rbinop_expr('__rdiv__', _ops.Divide)
rfloordiv = _rbinop_expr('__rfloordiv__', _ops.FloorDivide)


def substitute(arg, value, replacement=None, else_=None):
    """
    Substitute (replace) one or more values in a value expression

    Parameters
    ----------
    value : expr-like or dict
    replacement : expr-like, optional
      If an expression is passed to value, this must be passed
    else_ : expr, optional

    Returns
    -------
    replaced : case statement (for now!)

    """
    expr = arg.case()
    if isinstance(value, dict):
        for k, v in sorted(value.items()):
            expr = expr.when(k, v)
    else:
        expr = expr.when(value, replacement)

    if else_ is not None:
        expr = expr.else_(else_)
    else:
        expr = expr.else_(arg)

    return expr.end()


def _case(arg):
    """
    Create a new SimpleCaseBuilder to chain multiple if-else
    statements. Add new search expressions with the .when method. These
    must be comparable with this array expression. Conclude by calling
    .end()

    Examples
    --------
    case_expr = (expr.case()
                 .when(case1, output1)
                 .when(case2, output2)
                 .default(default_output)
                 .end())

    Returns
    -------
    builder : CaseBuilder
    """
    return _ops.SimpleCaseBuilder(arg)


def cases(arg, case_result_pairs, default=None):
    """
    Create a case expression in one shot.

    Returns
    -------
    case_expr : SimpleCase
    """
    builder = arg.case()
    for case, result in case_result_pairs:
        builder = builder.when(case, result)
    if default is not None:
        builder = builder.else_(default)
    return builder.end()


_generic_value_methods = dict(
    hash=hash,
    cast=cast,
    coalesce=coalesce,
    typeof=typeof,
    fillna=fillna,
    nullif=nullif,
    between=between,
    isin=isin,
    notin=notin,
    isnull=_unary_op('isnull', _ops.IsNull),
    notnull=_unary_op('notnull', _ops.NotNull),

    over=over,

    case=_case,
    cases=cases,
    substitute=substitute,

    __add__=add,
    add=add,

    __sub__=sub,
    sub=sub,

    __mul__=mul,
    mul=mul,

    __div__=div,
    __truediv__=div,
    __floordiv__=floordiv,
    div=div,
    floordiv=floordiv,

    __rdiv__=rdiv,
    __rtruediv__=rdiv,
    __rfloordiv__=rfloordiv,
    rdiv=rdiv,
    rfloordiv=rfloordiv,

    __pow__=pow,
    pow=pow,

    __radd__=add,

    __rsub__=rsub,
    rsub=rsub,

    __rmul__=_rbinop_expr('__rmul__', _ops.Multiply),
    __rpow__=_binop_expr('__rpow__', _ops.Power),

    __mod__=mod,
    __rmod__=_rbinop_expr('__rmod__', _ops.Modulus),

    __eq__=_binop_expr('__eq__', _ops.Equals),
    __ne__=_binop_expr('__ne__', _ops.NotEquals),
    __ge__=_binop_expr('__ge__', _ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', _ops.Greater),
    __le__=_binop_expr('__le__', _ops.LessEqual),
    __lt__=_binop_expr('__lt__', _ops.Less)
)


approx_nunique = _agg_function('approx_nunique', _ops.HLLCardinality, True)
approx_median = _agg_function('approx_median', _ops.CMSMedian, True)
max = _agg_function('max', _ops.Max, True)
min = _agg_function('min', _ops.Min, True)


def lag(arg, offset=None, default=None):
    return _ops.Lag(arg, offset, default).to_expr()


def lead(arg, offset=None, default=None):
    return _ops.Lead(arg, offset, default).to_expr()


first = _unary_op('first', _ops.FirstValue)
last = _unary_op('last', _ops.LastValue)
rank = _unary_op('rank', _ops.MinRank)
dense_rank = _unary_op('dense_rank', _ops.DenseRank)
cummin = _unary_op('cummin', _ops.CumulativeMin)
cummax = _unary_op('cummax', _ops.CumulativeMax)


def nth(arg, k):
    """
    Analytic operation computing nth value from start of sequence

    Parameters
    ----------
    arg : array expression
    k : int
        Desired rank value

    Returns
    -------
    nth : type of argument
    """
    return _ops.NthValue(arg, k).to_expr()


def distinct(arg):
    """
    Compute set of unique values occurring in this array. Can not be used
    in conjunction with other array expressions from the same context
    (because it's a cardinality-modifying pseudo-reduction).
    """
    op = _ops.DistinctArray(arg)
    return op.to_expr()


def nunique(arg):
    """
    Shorthand for foo.distinct().count(); computing the number of unique
    values in an array.
    """
    return _ops.CountDistinct(arg).to_expr()


def topk(arg, k, by=None):
    """
    Produces

    Returns
    -------
    topk : TopK filter expression
    """
    op = _ops.TopK(arg, k, by=by)
    return op.to_expr()


def bottomk(arg, k, by=None):
    raise NotImplementedError


def _generic_summary(arg, exact_nunique=False, prefix=None):
    """
    Compute a set of summary metrics from the input value expression

    Parameters
    ----------
    arg : value expression
    exact_nunique : boolean, default False
      Compute the exact number of distinct values (slower)
    prefix : string, default None
      String prefix for metric names

    Returns
    -------
    summary : (count, # nulls, nunique)
    """
    metrics = [
        arg.count(),
        arg.isnull().sum().name('nulls')
    ]

    if exact_nunique:
        unique_metric = arg.nunique().name('uniques')
    else:
        unique_metric = arg.approx_nunique().name('uniques')

    metrics.append(unique_metric)
    return _wrap_summary_metrics(metrics, prefix)


def _numeric_summary(arg, exact_nunique=False, prefix=None):
    """
    Compute a set of summary metrics from the input numeric value expression

    Parameters
    ----------
    arg : numeric value expression
    exact_nunique : boolean, default False
    prefix : string, default None
      String prefix for metric names

    Returns
    -------
    summary : (count, # nulls, min, max, sum, mean, nunique)
    """
    metrics = [
        arg.count(),
        arg.isnull().sum().name('nulls'),
        arg.min(),
        arg.max(),
        arg.sum(),
        arg.mean()
    ]

    if exact_nunique:
        unique_metric = arg.nunique().name('nunique')
    else:
        unique_metric = arg.approx_nunique().name('approx_nunique')

    metrics.append(unique_metric)
    return _wrap_summary_metrics(metrics, prefix)


def _wrap_summary_metrics(metrics, prefix):
    result = expr_list(metrics)
    if prefix is not None:
        result = result.prefix(prefix)
    return result


def expr_list(exprs):
    for e in exprs:
        e.get_name()
    return ir.ExpressionList(exprs).to_expr()


_generic_array_methods = dict(
    bottomk=bottomk,
    distinct=distinct,
    nunique=nunique,
    topk=topk,
    summary=_generic_summary,
    count=count,
    min=min,
    max=max,
    approx_median=approx_median,
    approx_nunique=approx_nunique,
    group_concat=group_concat,
    value_counts=value_counts,

    first=first,
    last=last,
    dense_rank=dense_rank,
    rank=rank,
    # nth=nth,
    lag=lag,
    lead=lead,
    cummin=cummin,
    cummax=cummax,
)


_add_methods(ValueExpr, _generic_value_methods)
_add_methods(ArrayExpr, _generic_array_methods)


# ---------------------------------------------------------------------
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


def _integer_to_timestamp(arg, unit='s'):
    """
    Convert integer UNIX timestamp (at some resolution) to a timestamp type

    Parameters
    ----------
    unit : {'s', 'ms', 'us'}
      Second (s), millisecond (ms), or microsecond (us) resolution

    Returns
    -------
    timestamp : timestamp value expression
    """
    op = _ops.TimestampFromUNIX(arg, unit)
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
    nullifzero=_unary_op('nullifzero', _ops.NullIfZero),
    zeroifnull=_unary_op('zeroifnull', _ops.ZeroIfNull),
)


def convert_base(arg, from_base, to_base):
    """
    Convert number (as integer or string) from one base to another

    Parameters
    ----------
    arg : string or integer
    from_base : integer
    to_base : integer

    Returns
    -------
    converted : string
    """
    return _ops.BaseConvert(arg, from_base, to_base).to_expr()


_integer_value_methods = dict(
    to_timestamp=_integer_to_timestamp,
    convert_base=convert_base
)


mean = _agg_function('mean', _ops.Mean, True)
cummean = _unary_op('cummean', _ops.CumulativeMean)

sum = _agg_function('sum', _ops.Sum, True)
cumsum = _unary_op('cumsum', _ops.CumulativeSum)


def std(arg, where=None, how='sample'):
    """
    Compute standard deviation of numeric array

    Parameters
    ----------
    how : {'sample', 'pop'}, default 'sample'

    Returns
    -------
    stdev : double scalar
    """
    expr = _ops.StandardDev(arg, where, how).to_expr()
    expr = expr.name('std')
    return expr


def variance(arg, where=None, how='sample'):
    """
    Compute standard deviation of numeric array

    Parameters
    ----------
    how : {'sample', 'pop'}, default 'sample'

    Returns
    -------
    stdev : double scalar
    """
    expr = _ops.Variance(arg, where, how).to_expr()
    expr = expr.name('var')
    return expr


_numeric_array_methods = dict(
    mean=mean,
    cummean=cummean,

    sum=sum,
    cumsum=cumsum,

    std=std,
    var=variance,

    bucket=bucket,
    histogram=histogram,
    summary=_numeric_summary,
)

_add_methods(NumericValue, _numeric_value_methods)
_add_methods(IntegerValue, _integer_value_methods)

_add_methods(NumericArray, _numeric_array_methods)


# ----------------------------------------------------------------------
# Boolean API


# TODO: logical binary operators for BooleanValue


def ifelse(arg, true_expr, false_expr):
    """
    Shorthand for implementing ternary expressions

    bool_expr.ifelse(0, 1)
    e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END
    """
    # Result will be the result of promotion of true/false exprs. These
    # might be conflicting types; same type resolution as case expressions
    # must be used.
    case = _ops.SearchedCaseBuilder()
    return case.when(arg, true_expr).else_(false_expr).end()


_boolean_value_methods = dict(
    ifelse=ifelse,
    __and__=_boolean_binary_op('__and__', _ops.And),
    __or__=_boolean_binary_op('__or__', _ops.Or),
    __xor__=_boolean_binary_op('__xor__', _ops.Xor),
    __rand__=_boolean_binary_rop('__rand__', _ops.And),
    __ror__=_boolean_binary_rop('__ror__', _ops.Or),
    __rxor__=_boolean_binary_rop('__rxor__', _ops.Xor)
)


_boolean_array_methods = dict(
    any=_unary_op('any', _ops.Any),
    notany=_unary_op('notany', _ops.NotAny),
    all=_unary_op('all', _ops.All),
    notall=_unary_op('notany', _ops.NotAll),
    cumany=_unary_op('cumany', _ops.CumulativeAny),
    cumall=_unary_op('cumall', _ops.CumulativeAll)
)


_add_methods(BooleanValue, _boolean_value_methods)
_add_methods(BooleanArray, _boolean_array_methods)


# ---------------------------------------------------------------------
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


def repeat(self, n):
    """
    Returns the argument string repeated n times

    Parameters
    ----------
    n : int

    Returns
    -------
    result : string
    """
    return _ops.Repeat(self, n).to_expr()


def _translate(self, from_str, to_str):
    """
    Returns string with set of 'from' characters replaced
    by set of 'to' characters.
    from_str[x] is replaced by to_str[x].
    To avoid unexpected behavior, from_str should be
    shorter than to_string.

    Parameters
    ----------
    from_str : string
    to_str : string

    Examples
    --------
    expr = table.strings.translate('a', 'b')
    expr = table.string.translate('a', 'bc')
    Returns
    -------
    translated : string
    """
    return _ops.Translate(self, from_str, to_str).to_expr()


def _string_find(self, substr, start=None, end=None):
    """
    Returns position (0 indexed) of first occurence of substring,
    optionally after a particular position (0 indexed)

    Parameters
    ----------
    substr : string
    start : int, default None
    end : int, default None
        Not currently implemented

    Returns
    -------
    position : int, 0 indexed
    """
    if end is not None:
        raise NotImplementedError
    return _ops.StringFind(self, substr, start, end).to_expr()


def _lpad(self, length, pad=' '):
    """
    Returns string of given length by truncating (on right)
    or padding (on left) original string

    Parameters
    ----------
    length : int
    pad : string, default is ' '

    Examples
    --------
    table.strings.lpad(5, '-')
    'a' becomes '----a'
    'abcdefg' becomes 'abcde'

    Returns
    -------
    padded : string
    """
    return _ops.LPad(self, length, pad).to_expr()


def _rpad(self, length, pad=' '):
    """
    Returns string of given length by truncating (on right)
    or padding (on right) original string

    Parameters
    ----------
    length : int
    pad : string, default is ' '

    Examples
    --------
    table.strings.rpad(5, '-')
    'a' becomes 'a----'
    'abcdefg' becomes 'abcde'

    Returns
    -------
    padded : string
    """
    return _ops.RPad(self, length, pad).to_expr()


def _find_in_set(self, str_list):
    """
    Returns postion (0 indexed) of first occurence of argument within
    a list of strings. No string in list can have a comma
    Returns -1 if search string isn't found or if search string contains ','


    Parameters
    ----------
    str_list : list of strings

    Examples
    --------
    table.strings.find_in_set(['a', 'b'])

    Returns
    -------
    position : int
    """
    return _ops.FindInSet(self, str_list).to_expr()


def _string_join(self, strings):
    """
    Joins a list of strings together using the calling string as a separator

    Parameters
    ----------
    strings : list of strings

    Examples
    --------
    sep = ibis.literal(',')
    sep.join(['a','b','c'])

    Returns
    -------
    joined : string
    """
    return _ops.StringJoin(self, strings).to_expr()


def _string_like(self, pattern):
    """
    Wildcard fuzzy matching function equivalent to the SQL LIKE directive. Use
    % as a multiple-character wildcard or _ (underscore) as a single-character
    wildcard.

    Use re_search or rlike for regex-based matching.

    Parameters
    ----------
    pattern : string

    Returns
    -------
    matched : boolean value
    """
    return _ops.StringSQLLike(self, pattern).to_expr()


def re_search(arg, pattern):
    """
    Search string values using a regular expression. Returns True if the regex
    matches a string and False otherwise.

    Parameters
    ----------
    pattern : string (regular expression string)

    Returns
    -------
    searched : boolean value
    """
    return _ops.RegexSearch(arg, pattern).to_expr()


def regex_extract(arg, pattern, index):
    """
    Returns specified index, 0 indexed, from string based on regex pattern
    given

    Parameters
    ----------
    pattern : string (regular expression string)
    index : int, 0 indexed

    Returns
    -------
    extracted : string
    """
    return _ops.RegexExtract(arg, pattern, index).to_expr()


def regex_replace(arg, pattern, replacement):
    """
    Replaces match found by regex with replacement string.
    Replacement string can also be a regex

    Parameters
    ----------
    pattern : string (regular expression string)
    replacement : string (can be regular expression string)

    Examples
    --------
    table.strings.replace('(b+)', r'<\1>')
    'aaabbbaa' becomes 'aaa<bbb>aaa'

    Returns
    -------
    modified : string
    """
    return _ops.RegexReplace(arg, pattern, replacement).to_expr()


def _string_replace(arg, pattern, replacement):
    """
    Replaces each exactly occurrence of pattern with given replacement
    string. Like Python built-in str.replace

    Parameters
    ----------
    pattern : string
    replacement : string

    Examples
    --------
    table.strings.replace('aaa', 'foo')
    'aaabbbaaa' becomes 'foobbbfoo'

    Returns
    -------
    replaced : string
    """
    return _ops.StringReplace(arg, pattern, replacement).to_expr()


def parse_url(arg, extract, key=None):
    """
    Returns the portion of a URL corresponding to a part specified
    by 'extract'
    Can optionally specify a key to retrieve an associated value
    if extract parameter is 'QUERY'

    Parameters
    ----------
    extract : one of {'PROTOCOL', 'HOST', 'PATH', 'REF',
                'AUTHORITY', 'FILE', 'USERINFO', 'QUERY'}
    key : string (optional)

    Examples
    --------
    parse_url("https://www.youtube.com/watch?v=kEuEcWfewf8&t=10", 'QUERY', 'v')
    yields 'kEuEcWfewf8'

    Returns
    -------
    extracted : string
    """
    return _ops.ParseURL(arg, extract, key).to_expr()


def _string_contains(arg, substr):
    """
    Determine if indicated string is exactly contained in the calling string.

    Parameters
    ----------
    substr

    Returns
    -------
    contains : boolean
    """
    return arg.find(substr) >= 0


def _string_dunder_contains(arg, substr):
    raise TypeError('Use val.contains(arg)')


def _string_getitem(self, key):
    if isinstance(key, slice):
        start, stop, step = key.start, key.stop, key.step
        if step and step != 1:
            raise ValueError('Step can only be 1')

        start = start or 0

        if start < 0 or stop < 0:
            raise ValueError('negative slicing not yet supported')

        return self.substr(start, stop - start)
    else:
        raise NotImplementedError


_string_value_methods = dict(
    __getitem__=_string_getitem,

    length=_unary_op('length', _ops.StringLength),
    lower=_unary_op('lower', _ops.Lowercase),
    upper=_unary_op('upper', _ops.Uppercase),
    reverse=_unary_op('reverse', _ops.Reverse),
    ascii_str=_unary_op('ascii', _ops.StringAscii),
    strip=_unary_op('strip', _ops.Strip),
    lstrip=_unary_op('lstrip', _ops.LStrip),
    rstrip=_unary_op('rstrip', _ops.RStrip),
    capitalize=_unary_op('initcap', _ops.Capitalize),

    convert_base=convert_base,

    __contains__=_string_dunder_contains,
    contains=_string_contains,
    like=_string_like,
    rlike=re_search,
    replace=_string_replace,
    re_search=re_search,
    re_extract=regex_extract,
    re_replace=regex_replace,
    parse_url=parse_url,

    substr=_string_substr,
    left=_string_left,
    right=_string_right,
    repeat=repeat,
    find=_string_find,
    translate=_translate,
    find_in_set=_find_in_set,
    join=_string_join,
    lpad=_lpad,
    rpad=_rpad,
)


_add_methods(StringValue, _string_value_methods)


# ---------------------------------------------------------------------
# Timestamp API

def _timestamp_truncate(arg, unit):
    """
    Zero out smaller-size units beyond indicated unit. Commonly used for time
    series resampling.

    Parameters
    ----------
    unit : string, one of below table
      'Y': year
      'Q': quarter
      'M': month
      'D': day
      'W': week
      'H': hour
      'MI': minute

    Returns
    -------
    truncated : timestamp
    """
    return _ops.Truncate(arg, unit).to_expr()


def _timestamp_strftime(arg, format_str):
    """
    Format timestamp according to the passed format string. Format string may
    depend on backend, but we try to conform to ANSI strftime (e.g. Python
    built-in datetime.strftime)

    Parameters
    ----------
    format_str : string

    Returns
    -------
    formatted : string
    """
    return _ops.Strftime(arg, format_str).to_expr()


_timestamp_value_methods = dict(
    strftime=_timestamp_strftime,
    year=_extract_field('year', _ops.ExtractYear),
    month=_extract_field('month', _ops.ExtractMonth),
    day=_extract_field('day', _ops.ExtractDay),
    hour=_extract_field('hour', _ops.ExtractHour),
    minute=_extract_field('minute', _ops.ExtractMinute),
    second=_extract_field('second', _ops.ExtractSecond),
    millisecond=_extract_field('millisecond', _ops.ExtractMillisecond),
    truncate=_timestamp_truncate
)


_add_methods(TimestampValue, _timestamp_value_methods)


# ---------------------------------------------------------------------
# Decimal API

_decimal_value_methods = dict(
    precision=_unary_op('precision', _ops.DecimalPrecision),
    scale=_unary_op('scale', _ops.DecimalScale),
)


_add_methods(DecimalValue, _decimal_value_methods)


# ----------------------------------------------------------------------
# Category API


_category_value_methods = dict(
    label=_analytics.category_label
)

_add_methods(CategoryValue, _category_value_methods)


# ---------------------------------------------------------------------
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


def cross_join(*args, **kwargs):
    """
    Perform a cross join (cartesian product) amongst a list of tables, with
    optional set of prefixes to apply to overlapping column names

    Parameters
    ----------
    positional args: tables to join
    prefixes keyword : prefixes for each table
      Not yet implemented

    Examples
    --------
    >>> joined1 = ibis.cross_join(a, b, c, d, e)
    >>> joined2 = ibis.cross_join(a, b, c, prefixes=['a_', 'b_', 'c_']))

    Returns
    -------
    joined : TableExpr
      If prefixes not provided, the result schema is not yet materialized
    """
    op = _ops.CrossJoin(*args, **kwargs)
    return TableExpr(op)


def _table_count(self):
    """
    Returns the computed number of rows in the table expression

    Returns
    -------
    count : Int64Scalar
    """
    return _ops.Count(self, None).to_expr().name('count')


def _table_info(self, buf=None):
    """
    Similar to pandas DataFrame.info. Show column names, types, and null
    counts. Output to stdout by default
    """
    metrics = [self.count().name('nrows')]
    for col in self.columns:
        metrics.append(self[col].count().name(col))

    metrics = self.aggregate(metrics).execute().loc[0]

    names = ['Column', '------'] + self.columns
    types = ['Type', '----'] + [repr(x) for x in self.schema().types]
    counts = ['Non-null #', '----------'] + [str(x) for x in metrics[1:]]
    col_metrics = util.adjoin(2, names, types, counts)

    result = ('Table rows: {0}\n\n'
              '{1}'
              .format(metrics[0], col_metrics))

    if buf is None:
        import sys
        sys.stdout.write(result)
        sys.stdout.write('\n')
    else:
        buf.write(result)


def _table_set_column(table, name, expr):
    """
    Replace an existing column with a new expression

    Parameters
    ----------
    name : string
      Column name to replace
    expr : value expression
      New data for column

    Returns
    -------
    set_table : TableExpr
      New table expression
    """
    expr = table._ensure_expr(expr)

    if expr._name != name:
        expr = expr.name(name)

    if name not in table:
        raise KeyError('{0} is not in the table'.format(name))

    # TODO: This assumes that projection is required; may be backend-dependent
    proj_exprs = []
    for key in table.columns:
        if key == name:
            proj_exprs.append(expr)
        else:
            proj_exprs.append(table[key])

    return table.projection(proj_exprs)


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


def filter(table, predicates):
    """
    Select rows from table based on boolean expressions

    Parameters
    ----------
    predicates : boolean array expressions, or list thereof

    Returns
    -------
    filtered_expr : TableExpr
    """
    resolved_predicates = _resolve_predicates(table, predicates)
    return _L.apply_filter(table, resolved_predicates)


def _resolve_predicates(table, predicates):
    if isinstance(predicates, Expr):
        predicates = _L.unwrap_ands(predicates)
    predicates = util.promote_list(predicates)
    predicates = [ir.bind_expr(table, x) for x in predicates]
    resolved_predicates = []
    for pred in predicates:
        if isinstance(pred, ir.AnalyticExpr):
            pred = pred.to_filter()
        resolved_predicates.append(pred)

    return resolved_predicates


def aggregate(table, metrics=None, by=None, having=None, **kwds):
    """
    Aggregate a table with a given set of reductions, with grouping
    expressions, and post-aggregation filters.

    Parameters
    ----------
    table : table expression
    metrics : expression or expression list
    by : optional, default None
      Grouping expressions
    having : optional, default None
      Post-aggregation filters

    Returns
    -------
    agg_expr : TableExpr
    """
    if metrics is None:
        metrics = []

    for k, v in sorted(kwds.items()):
        v = table._ensure_expr(v)
        metrics.append(v.name(k))

    op = table.op().aggregate(table, metrics, by=by, having=having)
    return TableExpr(op)


def _table_distinct(self):
    """
    Compute set of unique rows/tuples occurring in this table
    """
    op = _ops.Distinct(self)
    return op.to_expr()


def _table_limit(table, n, offset=0):
    """
    Select the first n rows at beginning of table (may not be deterministic
    depending on implementatino and presence of a sorting).

    Parameters
    ----------
    n : int
      Rows to include
    offset : int, default 0
      Number of rows to skip first

    Returns
    -------
    limited : TableExpr
    """
    op = _ops.Limit(table, n, offset=offset)
    return TableExpr(op)


def _table_sort_by(table, sort_exprs):
    """
    Sort table by the indicated column expressions and sort orders
    (ascending/descending)

    Parameters
    ----------
    sort_exprs : sorting expressions
      Must be one of:
        - Column name or expression
        - Sort key, e.g. desc(col)
        - (column name, True (ascending) / False (descending))

    Examples
    --------
    sorted = table.sort_by([('a', True), ('b', False)])

    Returns
    -------
    sorted : TableExpr
    """
    op = table.op()
    result = op.sort_by(table, sort_exprs)

    return TableExpr(result)


def _table_union(left, right, distinct=False):
    """
    Form the table set union of two table expressions having identical
    schemas.

    Parameters
    ----------
    right : TableExpr
    distinct : boolean, default False
        Only union distinct rows not occurring in the calling table (this
        can be very expensive, be careful)

    Returns
    -------
    union : TableExpr
    """
    op = _ops.Union(left, right, distinct=distinct)
    return TableExpr(op)


def _table_to_array(self):
    """
    Single column tables can be viewed as arrays.
    """
    op = _ops.TableArrayView(self)
    return op.to_expr()


def _table_materialize(table):
    """
    Force schema resolution for a joined table, selecting all fields from
    all tables.
    """
    if table._is_materialized():
        return table
    else:
        op = _ops.MaterializedJoin(table)
        return TableExpr(op)


def mutate(table, exprs=None, **kwds):
    """
    Convenience function for table projections involving adding columns

    Parameters
    ----------
    exprs : list, default None
      List of named expressions to add as columns
    kwds : keywords for new columns

    Examples
    --------
    expr = table.mutate(qux=table.foo + table.bar, baz=5)

    Returns
    -------
    mutated : TableExpr
    """
    if exprs is None:
        exprs = []
    else:
        exprs = util.promote_list(exprs)

    for k, v in sorted(kwds.items()):
        if util.is_function(v):
            v = v(table)
        else:
            v = as_value_expr(v)
        exprs.append(v.name(k))

    has_replacement = False
    for expr in exprs:
        if expr.get_name() in table:
            has_replacement = True

    if has_replacement:
        by_name = dict((x.get_name(), x) for x in exprs)
        used = set()
        proj_exprs = []
        for c in table.columns:
            if c in by_name:
                proj_exprs.append(by_name[c])
                used.add(c)
            else:
                proj_exprs.append(c)

        for x in exprs:
            if x.get_name() not in used:
                proj_exprs.append(x)

        return table.projection(proj_exprs)
    else:
        return table.projection([table] + exprs)


def projection(table, exprs):
    """
    Compute new table expression with the indicated column expressions from
    this table.

    Parameters
    ----------
    exprs : column expression, or string, or list of column expressions and
      strings. If strings passed, must be columns in the table already

    Returns
    -------
    projection : TableExpr
    """
    import ibis.expr.analysis as L

    if isinstance(exprs, (Expr,) + six.string_types):
        exprs = [exprs]

    projector = L.Projector(table, exprs)

    op = projector.get_result()
    return TableExpr(op)


def _table_relabel(table, substitutions, replacements=None):
    """
    Change table column names, otherwise leaving table unaltered

    Parameters
    ----------
    substitutions

    Returns
    -------
    relabeled : TableExpr
    """
    if replacements is not None:
        raise NotImplementedError

    observed = set()

    exprs = []
    for c in table.columns:
        expr = table[c]
        if c in substitutions:
            expr = expr.name(substitutions[c])
            observed.add(c)
        exprs.append(expr)

    for c in substitutions:
        if c not in observed:
            raise KeyError('{0!r} is not an existing column'.format(c))

    return table.projection(exprs)


def _table_view(self):
    """
    Create a new table expression that is semantically equivalent to the
    current one, but is considered a distinct relation for evaluation
    purposes (e.g. in SQL).

    For doing any self-referencing operations, like a self-join, you will
    use this operation to create a reference to the current table
    expression.

    Returns
    -------
    expr : TableExpr
    """
    return TableExpr(_ops.SelfReference(self))


def _table_drop(self, fields):
    if len(fields) == 0:
        # noop
        return self

    fields = set(fields)
    to_project = []
    for name in self.schema():
        if name in fields:
            fields.remove(name)
        else:
            to_project.append(name)

    if len(fields) > 0:
        raise KeyError('Fields not in table: {0!s}'.format(fields))

    return self.projection(to_project)


_table_methods = dict(
    aggregate=aggregate,
    count=_table_count,
    distinct=_table_distinct,
    drop=_table_drop,
    info=_table_info,
    limit=_table_limit,
    set_column=_table_set_column,
    filter=filter,
    materialize=_table_materialize,
    mutate=mutate,
    projection=projection,
    select=projection,
    relabel=_table_relabel,
    join=join,
    cross_join=cross_join,
    inner_join=_regular_join_method('inner_join', 'inner'),
    left_join=_regular_join_method('left_join', 'left'),
    outer_join=_regular_join_method('outer_join', 'outer'),
    semi_join=_regular_join_method('semi_join', 'semi'),
    anti_join=_regular_join_method('anti_join', 'anti'),
    sort_by=_table_sort_by,
    to_array=_table_to_array,
    union=_table_union,
    view=_table_view
)


_add_methods(TableExpr, _table_methods)
