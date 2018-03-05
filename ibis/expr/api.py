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

from __future__ import print_function

import warnings
import operator
import datetime
import functools
import collections

import six
import toolz

from ibis.expr.schema import Schema
from ibis.expr import datatypes as dt
from ibis.expr import schema as sch
from ibis.expr.types import (Expr,  # noqa
                             ValueExpr, ScalarExpr, ColumnExpr,
                             TableExpr,
                             NumericValue, NumericColumn,
                             IntegerValue,
                             Int8Value, Int8Scalar, Int8Column,
                             Int16Value, Int16Scalar, Int16Column,
                             Int32Value, Int32Scalar, Int32Column,
                             Int64Value, Int64Scalar, Int64Column,
                             NullScalar,
                             BooleanValue, BooleanScalar, BooleanColumn,
                             FloatingValue,
                             FloatValue, FloatScalar, FloatColumn,
                             DoubleValue, DoubleScalar, DoubleColumn,
                             StringValue, StringScalar, StringColumn,
                             DecimalValue, DecimalScalar, DecimalColumn,
                             TimestampValue, TimestampScalar, TimestampColumn,
                             IntervalValue, IntervalScalar, IntervalColumn,
                             DateValue, TimeValue,
                             ArrayValue, ArrayScalar, ArrayColumn,
                             MapValue, MapScalar, MapColumn,
                             StructValue, StructScalar, StructColumn,
                             CategoryValue, unnamed, as_value_expr, literal,
                             param, null, sequence)

import ibis.common as _com
from ibis.compat import PY2, to_time, to_date
from ibis.expr.analytics import bucket, histogram
from ibis.expr.groupby import GroupedTableExpr  # noqa
from ibis.expr.window import window, trailing_window, cumulative_window
import ibis.expr.analytics as _analytics
import ibis.expr.analysis as _L
import ibis.expr.types as ir
import ibis.expr.operations as _ops
import ibis.util as util


__all__ = [
    'infer_dtype', 'infer_schema',
    'schema', 'table', 'literal', 'expr_list',
    'timestamp', 'time', 'date', 'interval', 'param',
    'nanosecond', 'microsecond', 'millisecond', 'second',
    'minute', 'hour', 'day', 'week', 'month', 'year',
    'case', 'where', 'sequence',
    'now', 'desc', 'null', 'NA',
    'cast', 'coalesce', 'greatest', 'least',
    'cross_join', 'join',
    'aggregate',
    'row_number',
    'negate', 'ifelse',
    'Expr', 'Schema',
    'window', 'trailing_window', 'cumulative_window',
]


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
decimal(p, s)  DECIMAL(p,s)
interval(u)    INTERVAL(u)"""


infer_dtype = dt.infer
infer_schema = sch.infer


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
    >>> import ibis
    >>> t = ibis.table([('g', 'string')])
    >>> result = t.group_by('g').size('count').sort_by(ibis.desc('count'))
    """
    if not isinstance(expr, Expr):
        return _ops.DeferredSortKey(expr, ascending=False)
    else:
        return _ops.SortKey(expr, ascending=False).to_expr()


def timestamp(value):
    """
    Returns a timestamp literal if value is likely coercible to a timestamp

    Parameters
    ----------
    value : timestamp value as string

    Returns
    --------
    result : TimestampScalar
    """
    if isinstance(value, six.string_types):
        from pandas import Timestamp
        value = Timestamp(value)
    if isinstance(value, six.integer_types):
        warnings.warn(
            'Integer values for timestamp literals are deprecated in 0.11.0 '
            'and will be removed in 0.12.0. To pass integers as timestamp '
            'literals, use pd.Timestamp({:d}, unit=...)'.format(value)
        )
    return ir.TimestampScalar(ir.literal(value).op())


def date(value):
    """
    Returns a date literal if value is likely coercible to a date

    Parameters
    ----------
    value : date value as string

    Returns
    --------
    result : TimeScalar
    """
    if isinstance(value, six.string_types):
        value = to_date(value)
    return ir.DateScalar(ir.literal(value).op())


def time(value):
    """
    Returns a time literal if value is likely coercible to a time

    Parameters
    ----------
    value : time value as string

    Returns
    --------
    result : TimeScalar
    """
    if isinstance(value, six.string_types):
        value = to_time(value)
    return ir.TimeScalar(ir.literal(value).op())


def interval(value=None, unit='s', years=None, quarters=None, months=None,
             weeks=None, days=None, hours=None, minutes=None, seconds=None,
             milliseconds=None, microseconds=None, nanoseconds=None):
    """
    Returns an interval literal

    Parameters
    ----------
    value : int or datetime.timedelta, default None
    years : int, default None
    quarters : int, default None
    months : int, default None
    days : int, default None
    weeks : int, default None
    hours : int, default None
    minutes : int, default None
    seconds : int, default None
    milliseconds : int, default None
    microseconds : int, default None
    nanoseconds : int, default None

    Returns
    --------
    result : IntervalScalar
    """
    if value is not None:
        if isinstance(value, datetime.timedelta):
            unit = 's'
            value = int(value.total_seconds())
        elif not isinstance(value, six.integer_types):
            raise ValueError('Interval value must be an integer')
    else:
        kwds = [
            ('Y', years),
            ('Q', quarters),
            ('M', months),
            ('W', weeks),
            ('D', days),
            ('h', hours),
            ('m', minutes),
            ('s', seconds),
            ('ms', milliseconds),
            ('us', microseconds),
            ('ns', nanoseconds)
        ]
        defined_units = [(k, v) for k, v in kwds if v is not None]

        if len(defined_units) != 1:
            raise ValueError('Exactly one argument is required')

        unit, value = defined_units[0]

    value_type = ir.literal(value).type()
    type = dt.Interval(unit, value_type)

    return ir.literal(value, type=type).op().to_expr()


@functools.wraps(interval)
def timedelta(*args, **kwargs):
    warnings.warn('ibis.timedelta is deprecated, use ibis.interval instead',
                  DeprecationWarning)
    return interval(*args, **kwargs)


def _timedelta(name, unit):
    def f(value=1):
        msg = 'ibis.{0} is deprecated, use ibis.interval({0}s=n) instead'
        warnings.warn(msg.format(name), DeprecationWarning)
        return interval(value, unit=unit)
    f.__name__ = name
    return f


year = _timedelta('year', 'Y')
quarter = _timedelta('quarter', 'Q')
month = _timedelta('month', 'M')
week = _timedelta('week', 'W')
day = _timedelta('day', 'D')
hour = _timedelta('hour', 'h')
minute = _timedelta('minute', 'm')
second = _timedelta('second', 's')
millisecond = _timedelta('millisecond', 'ms')
microsecond = _timedelta('microsecond', 'us')
nanosecond = _timedelta('nanosecond', 'ns')


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
>>> from ibis import schema
>>> sc = schema([('foo', 'string'),
...              ('bar', 'int64'),
...              ('baz', 'boolean')])
>>> sc2 = schema(names=['foo', 'bar', 'baz'],
...              types=['string', 'int64', 'boolean'])

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
    >>> import ibis
    >>> cond1 = ibis.literal(1) == 1
    >>> cond2 = ibis.literal(2) == 1
    >>> result1 = 3
    >>> result2 = 4
    >>> expr = (ibis.case()
    ...         .when(cond1, result1)
    ...         .when(cond2, result2).end())

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
    """Analytic function for the current row number, starting at 0.

    This function does not require an ORDER BY clause, however, without an
    ORDER BY clause the order of the result is nondeterministic.

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
    if isinstance(op, _ops.DistinctColumn):
        result = _ops.CountDistinct(op.args[0], where).to_expr()
    else:
        result = _ops.Count(expr, where).to_expr()

    return result.name('count')


def group_concat(arg, sep=',', where=None):
    """
    Concatenate values using the indicated separator (comma by default) to
    produce a string

    Parameters
    ----------
    arg : array expression
    sep : string, default ','
    where : bool, default None

    Returns
    -------
    concatenated : string scalar
    """
    return _ops.GroupConcat(arg, sep, where).to_expr()


def arbitrary(arg, where=None, how='first'):
    """
    Selects the first / last non-null value in a column

    Parameters
    ----------
    arg : array expression
    where: bool, default None
    how : {'first', 'last', 'heavy'}, default 'first'
      Heavy selects a frequently occurring value using the heavy hitters
      algorithm. Heavy is only supported by Clickhouse backend.

    Returns
    -------
    arbitrary element : scalar type of caller
    """
    return _ops.Arbitrary(arg, how, where).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        try:
            other = as_value_expr(other)
            op = klass(self, other)
            return op.to_expr()
        except (_com.IbisTypeError, NotImplementedError):
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


def _boolean_unary_op(name, klass):
    def f(self):
        return klass(self).to_expr()
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
    to = op.args[1]

    if to.equals(arg.type()):
        # noop case if passed type is the same
        return arg
    else:
        result = op.to_expr()
        if not arg.has_name():
            return result
        expr_name = 'cast({}, {})'.format(arg.get_name(), op.args[1])
        return result.name(expr_name)


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
    >>> import ibis
    >>> table = ibis.table([('col', 'int64'), ('other_col', 'int64')])
    >>> result = table.col.fillna(5)
    >>> result2 = table.col.fillna(table.other_col * 3)

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
    >>> import ibis
    >>> expr1 = None
    >>> expr2 = 4
    >>> result = ibis.coalesce(expr1, expr2, 5)

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
        name = expr.get_name()
    except _com.ExpressionError:
        pass
    else:
        result = result.name(name)

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
    >>> import ibis
    >>> table = ibis.table([('string_col', 'string')])
    >>> table2 = ibis.table([('other_string_col', 'string')])
    >>> expr = table.string_col.isin(['foo', 'bar', 'baz'])
    >>> expr2 = table.string_col.isin(table2.other_string_col)

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

radd = _rbinop_expr('__radd__', _ops.Add)
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
    """Create a new SimpleCaseBuilder to chain multiple if-else statements. Add
    new search expressions with the .when method. These must be comparable with
    this array expression. Conclude by calling .end()

    Returns
    -------
    builder : CaseBuilder

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('string_col', 'string')], name='t')
    >>> expr = t.string_col
    >>> case_expr = (expr.case()
    ...              .when('a', 'an a')
    ...              .when('b', 'a b')
    ...              .else_('null or (not a and not b)')
    ...              .end())
    >>> case_expr  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        string_col : string
    <BLANKLINE>
    SimpleCase[string*]
      base:
        string_col = Column[string*] 'string_col' from table
          ref_0
      cases:
        ValueList[string*]
          Literal[string]
            a
          Literal[string]
            b
      results:
        ValueList[string*]
          Literal[string]
            an a
          Literal[string]
            a b
      default:
        Literal[string]
          null or (not a and not b)
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

    __eq__=_binop_expr('__eq__', _ops.Equals),
    __ne__=_binop_expr('__ne__', _ops.NotEquals),
    __ge__=_binop_expr('__ge__', _ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', _ops.Greater),
    __le__=_binop_expr('__le__', _ops.LessEqual),
    __lt__=_binop_expr('__lt__', _ops.Less),
    collect=_unary_op('collect', _ops.ArrayCollect),
    identical_to=_binop_expr('identical_to', _ops.IdenticalTo),
)


approx_nunique = _agg_function('approx_nunique', _ops.HLLCardinality, True)
approx_median = _agg_function('approx_median', _ops.CMSMedian, True)
max = _agg_function('max', _ops.Max, True)
min = _agg_function('min', _ops.Min, True)
nunique = _agg_function('nunique', _ops.CountDistinct, True)


def lag(arg, offset=None, default=None):
    return _ops.Lag(arg, offset, default).to_expr()


def lead(arg, offset=None, default=None):
    return _ops.Lead(arg, offset, default).to_expr()


first = _unary_op('first', _ops.FirstValue)
last = _unary_op('last', _ops.LastValue)
rank = _unary_op('rank', _ops.MinRank)
dense_rank = _unary_op('dense_rank', _ops.DenseRank)
percent_rank = _unary_op('percent_rank', _ops.PercentRank)
cummin = _unary_op('cummin', _ops.CumulativeMin)
cummax = _unary_op('cummax', _ops.CumulativeMax)


def ntile(arg, buckets):
    return _ops.NTile(arg, buckets).to_expr()


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
    op = _ops.DistinctColumn(arg)
    return op.to_expr()


def topk(arg, k, by=None):
    """
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


_generic_column_methods = dict(
    bottomk=bottomk,
    distinct=distinct,
    nunique=nunique,
    topk=topk,
    summary=_generic_summary,
    count=count,
    arbitrary=arbitrary,
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
    percent_rank=percent_rank,
    # nth=nth,
    ntile=ntile,
    lag=lag,
    lead=lead,
    cummin=cummin,
    cummax=cummax,
)


_add_methods(ValueExpr, _generic_value_methods)
_add_methods(ColumnExpr, _generic_column_methods)


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


def clip(arg, lower=None, upper=None):
    """
    Trim values at input threshold(s).

    Parameters
    ----------
    lower : float
    upper : float

    Returns
    -------
    clipped : same as type of the input
    """
    if lower is None and upper is None:
        raise ValueError("at least one of lower and "
                         "upper must be provided")

    op = _ops.Clip(arg, lower, upper)
    return op.to_expr()


def quantile(arg, quantile, interpolation='linear'):
    """
    Return value at the given quantile, a la numpy.percentile.

    Parameters
    ----------
    quantile : float/int or array-like
        0 <= quantile <= 1, the quantile(s) to compute
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        * linear: `i + (j - i) * fraction`, where `fraction` is the
          fractional part of the index surrounded by `i` and `j`.
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j` whichever is nearest.
        * midpoint: (`i` + `j`) / 2.

    Returns
    -------
    quantile
        if scalar input, scalar type, same as input
        if array input, list of scalar type
    """
    if isinstance(quantile, collections.Sequence):
        op = _ops.MultiQuantile(arg, quantile, interpolation)
    else:
        op = _ops.Quantile(arg, quantile, interpolation)
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


def _integer_to_interval(arg, unit='s'):
    """
    Convert integer interval with the same inner type

    Parameters
    ----------
    unit : {'Y', 'M', 'W', 'D', 'h', 'm', s', 'ms', 'us', 'ns'}

    Returns
    -------
    interval : interval value expression
    """
    op = _ops.IntervalFromInteger(arg, unit)
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
    clip=clip,

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
    radd=add,

    __rsub__=rsub,
    rsub=rsub,

    __rmul__=_rbinop_expr('__rmul__', _ops.Multiply),
    __rpow__=_rbinop_expr('__rpow__', _ops.Power),

    __mod__=mod,
    __rmod__=_rbinop_expr('__rmod__', _ops.Modulus),
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
    to_interval=_integer_to_interval,
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
    expr = _ops.StandardDev(arg, how, where).to_expr()
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
    expr = _ops.Variance(arg, how, where).to_expr()
    expr = expr.name('var')
    return expr


_numeric_column_methods = dict(
    mean=mean,
    cummean=cummean,

    sum=sum,
    cumsum=cumsum,

    quantile=quantile,

    std=std,
    var=variance,

    bucket=bucket,
    histogram=histogram,
    summary=_numeric_summary,
)

_floating_value_methods = dict(
    isnan=_unary_op('isnull', _ops.IsNan),
    isinf=_unary_op('isinf', _ops.IsInf),
)

_add_methods(NumericValue, _numeric_value_methods)
_add_methods(IntegerValue, _integer_value_methods)
_add_methods(FloatingValue, _floating_value_methods)

_add_methods(NumericColumn, _numeric_column_methods)


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
    __rxor__=_boolean_binary_rop('__rxor__', _ops.Xor),
    __invert__=_boolean_unary_op('__invert__', _ops.Not),
)


_boolean_column_methods = dict(
    any=_unary_op('any', _ops.Any),
    notany=_unary_op('notany', _ops.NotAny),
    all=_unary_op('all', _ops.All),
    notall=_unary_op('notany', _ops.NotAll),
    cumany=_unary_op('cumany', _ops.CumulativeAny),
    cumall=_unary_op('cumall', _ops.CumulativeAll)
)


_add_methods(BooleanValue, _boolean_value_methods)
_add_methods(BooleanColumn, _boolean_column_methods)


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
    Return up to nchars starting from end of each string.

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
    >>> import ibis
    >>> table = ibis.table([('string_col', 'string')])
    >>> expr = table.string_col.translate('a', 'b')
    >>> expr = table.string_col.translate('a', 'bc')

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
    >>> import ibis
    >>> table = ibis.table([('strings', 'string')])
    >>> expr = table.strings.lpad(5, '-')
    >>> expr = ibis.literal('a').lpad(5, '-')  # 'a' becomes '----a'
    >>> expr = ibis.literal('abcdefg').lpad(5, '-')  # 'abcdefg' becomes 'abcde'  # noqa: E501

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
    >>> import ibis
    >>> table = ibis.table([('string_col', 'string')])
    >>> expr = table.string_col.rpad(5, '-')
    >>> expr = ibis.literal('a').rpad(5, '-')  # 'a' becomes 'a----'
    >>> expr = ibis.literal('abcdefg').rpad(5, '-')  # 'abcdefg' becomes 'abcde'  # noqa: E501

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
    >>> import ibis
    >>> table = ibis.table([('strings', 'string')])
    >>> result = table.strings.find_in_set(['a', 'b'])

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
    >>> import ibis
    >>> sep = ibis.literal(',')
    >>> result = sep.join(['a', 'b', 'c'])

    Returns
    -------
    joined : string
    """
    return _ops.StringJoin(self, strings).to_expr()


def _string_like(self, patterns):
    """
    Wildcard fuzzy matching function equivalent to the SQL LIKE directive. Use
    % as a multiple-character wildcard or _ (underscore) as a single-character
    wildcard.

    Use re_search or rlike for regex-based matching.

    Parameters
    ----------
    pattern : str or List[str]
        A pattern or list of patterns to match. If `pattern` is a list, then if
        **any** pattern matches the input then the corresponding row in the
        output is ``True``.

    Returns
    -------
    matched : ir.BooleanColumn
    """
    return functools.reduce(
        operator.or_,
        (
            _ops.StringSQLLike(self, pattern).to_expr()
            for pattern in util.promote_list(patterns)
        )
    )


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
    >>> import ibis
    >>> table = ibis.table([('strings', 'string')])
    >>> result = table.strings.replace('(b+)', r'<\1>')  # 'aaabbbaa' becomes 'aaa<bbb>aaa'  # noqa: E501

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
    >>> import ibis
    >>> table = ibis.table([('strings', 'string')])
    >>> result = table.strings.replace('aaa', 'foo')  # 'aaabbbaaa' becomes 'foobbbfoo'  # noqa: E501

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
    >>> url = "https://www.youtube.com/watch?v=kEuEcWfewf8&t=10"
    >>> parse_url(url, 'QUERY', 'v')  # doctest: +SKIP
    'kEuEcWfewf8'

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
    substr : str or ibis.expr.types.StringValue

    Returns
    -------
    contains : ibis.expr.types.BooleanValue
    """
    return arg.find(substr) >= 0


def _string_split(arg, delimiter):
    """Split `arg` on `delimiter`.

    Parameters
    ----------
    arg : str or ibis.expr.types.StringValue
    delimiter : str or ibis.expr.types.StringValue

    Returns
    -------
    splitsville : Array[String]
    """
    return _ops.StringSplit(arg, delimiter).to_expr()


def _string_concat(*args):
    return _ops.StringConcat(*args).to_expr()


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
    elif isinstance(key, six.integer_types):
        return self.substr(key, 1)
    else:
        raise NotImplementedError(
            'string __getitem__[{}]'.format(type(key).__name__)
        )


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
    split=_string_split,
    join=_string_join,
    lpad=_lpad,
    rpad=_rpad,
    __add__=_string_concat,
    __radd__=lambda *args: _string_concat(*args[::-1]),
    __mul__=mul,
    __rmul__=mul,
)


_add_methods(StringValue, _string_value_methods)


# ---------------------------------------------------------------------
# Array API


def _array_slice(array, index):
    """Slice or index `array` at `index`.

    Parameters
    ----------
    index : int or ibis.expr.types.IntegerValue or slice

    Returns
    -------
    sliced_array : ibis.expr.types.ValueExpr
        If `index` is an ``int`` or :class:`~ibis.expr.types.IntegerValue` then
        the return type is the element type of `array`. If `index` is a
        ``slice`` then the return type is the same type as the input.
    """
    if isinstance(index, slice):
        start = index.start
        stop = index.stop
        if ((start is not None and start < 0) or
                (stop is not None and stop < 0)):
            raise ValueError('negative slicing not yet supported')

        step = index.step

        if step is not None and step != 1:
            raise NotImplementedError('step can only be 1')

        op = _ops.ArraySlice(
            array,
            start if start is not None else 0,
            stop,
        )
    else:
        op = _ops.ArrayIndex(array, index)
    return op.to_expr()


_array_column_methods = dict(
    length=_unary_op('length', _ops.ArrayLength),
    __getitem__=_array_slice,
    __add__=_binop_expr('__add__', _ops.ArrayConcat),
    __radd__=toolz.flip(_binop_expr('__radd__', _ops.ArrayConcat)),
    __mul__=_binop_expr('__mul__', _ops.ArrayRepeat),
    __rmul__=_binop_expr('__rmul__', _ops.ArrayRepeat),
)

_add_methods(ArrayValue, _array_column_methods)


# ---------------------------------------------------------------------
# Map API

def get(expr, key, default):
    """
    Return the mapped value for this key, or the default
    if the key does not exist

    Parameters
    ----------
    key : any
    default : any
    """
    return _ops.MapValueOrDefaultForKey(expr, key, default).to_expr()


_map_column_methods = dict(
    length=_unary_op('length', _ops.MapLength),
    __getitem__=_binop_expr('__getitem__', _ops.MapValueForKey),
    get=get,
    keys=_unary_op('keys', _ops.MapKeys),
    values=_unary_op('values', _ops.MapValues),
    __add__=_binop_expr('__add__', _ops.MapConcat),
    __radd__=toolz.flip(_binop_expr('__radd__', _ops.MapConcat)),
)

_add_methods(MapValue, _map_column_methods)

# ---------------------------------------------------------------------
# Struct API


def _struct_get_field(expr, field_name):
    """Get the `field_name` field from the ``Struct`` expression `expr`.

    Parameters
    ----------
    field_name : str
        The name of the field to access from the ``Struct`` typed expression
        `expr`. Must be a Python ``str`` type; programmatic struct field
        access is not yet supported.

    Returns
    -------
    value_expr : ibis.expr.types.ValueExpr
        An expression with the type of the field being accessed.
    """
    return _ops.StructField(expr, field_name).to_expr().name(field_name)


_struct_column_methods = dict(
    __getattr__=_struct_get_field,
    __getitem__=_struct_get_field,
)

_add_methods(StructValue, _struct_column_methods)


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
      'W': week
      'D': day
      'h': hour
      'm': minute
      's': second
      'ms': millisecond
      'us': microsecond
      'ns': nanosecond

    Returns
    -------
    truncated : timestamp
    """
    return _ops.TimestampTruncate(arg, unit).to_expr()


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


def _timestamp_time(arg):
    """
    Return a Time node for a Timestamp
    We can then perform certain operations on this node
    w/o actually instantiating the underlying structure
    (which is inefficient in pandas/numpy)

    Returns
    -------
    Time node
    """
    if PY2:
        raise ValueError("time support is not enabled on python 2")
    return _ops.Time(arg).to_expr()


def _timestamp_date(arg):
    """
    Return a Date node for a Timestamp
    We can then perform certain operations on this node
    w/o actually instantiating the underlying structure
    (which is inefficient in pandas/numpy)

    Returns
    -------
    Date node
    """
    return _ops.Date(arg).to_expr()


def _timestamp_sub(left, right):
    right = as_value_expr(right)

    if isinstance(right, ir.TimestampValue):
        op = _ops.TimestampDiff(left, right)
    else:
        op = _ops.TimestampSub(left, right)  # let the operation validate

    return op.to_expr()


_timestamp_add = _binop_expr('__add__', _ops.TimestampAdd)
_timestamp_radd = _binop_expr('__radd__', _ops.TimestampAdd)


_timestamp_value_methods = dict(
    strftime=_timestamp_strftime,
    year=_extract_field('year', _ops.ExtractYear),
    month=_extract_field('month', _ops.ExtractMonth),
    day=_extract_field('day', _ops.ExtractDay),
    hour=_extract_field('hour', _ops.ExtractHour),
    minute=_extract_field('minute', _ops.ExtractMinute),
    second=_extract_field('second', _ops.ExtractSecond),
    millisecond=_extract_field('millisecond', _ops.ExtractMillisecond),
    truncate=_timestamp_truncate,
    time=_timestamp_time,
    date=_timestamp_date,

    __sub__=_timestamp_sub,
    sub=_timestamp_sub,

    __add__=_timestamp_add,
    add=_timestamp_add,

    __radd__=_timestamp_radd,
    radd=_timestamp_radd,
    day_of_week=property(lambda self: _ops.DayOfWeekNode([self]).to_expr()),
)


# ---------------------------------------------------------------------
# Date API


def _date_truncate(arg, unit):
    """
    Zero out smaller-size units beyond indicated unit. Commonly used for time
    series resampling.

    Parameters
    ----------
    unit : string, one of below table
      'Y': year
      'Q': quarter
      'M': month
      'W': week
      'D': day

    Returns
    -------
    truncated : date
    """
    return _ops.DateTruncate(arg, unit).to_expr()


def _date_sub(left, right):
    right = as_value_expr(right)

    if isinstance(right, ir.DateValue):
        op = _ops.DateDiff(left, right)
    else:
        op = _ops.DateSub(left, right)  # let the operation validate

    return op.to_expr()


_date_add = _binop_expr('__add__', _ops.DateAdd)

_date_value_methods = dict(
    strftime=_timestamp_strftime,
    year=_extract_field('year', _ops.ExtractYear),
    month=_extract_field('month', _ops.ExtractMonth),
    day=_extract_field('day', _ops.ExtractDay),
    day_of_week=property(lambda self: _ops.DayOfWeekNode([self]).to_expr()),

    truncate=_date_truncate,

    __sub__=_date_sub,
    sub=_date_sub,

    __add__=_date_add,
    add=_date_add,

    __radd__=_date_add,
    radd=_date_add
)

_add_methods(TimestampValue, _timestamp_value_methods)
_add_methods(DateValue, _date_value_methods)


def _convert_unit(value, unit, to):
    units = ('W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns')
    factors = (7, 24, 60, 60, 1000, 1000, 1000)

    monthly_units = ('Y', 'Q', 'M')
    monthly_factors = (4, 3)

    try:
        i, j = units.index(unit), units.index(to)
    except ValueError:
        try:
            i, j = monthly_units.index(unit), monthly_units.index(to)
            factors = monthly_factors
        except ValueError:
            raise ValueError('Cannot convert to or from '
                             'non-fixed-length interval')

    factor = functools.reduce(operator.mul, factors[i:j], 1)

    if i < j:
        return value * factor
    elif i > j:
        return value // factor
    else:
        return value


def _to_unit(arg, target_unit):
    if arg.meta.unit != target_unit:
        arg = _convert_unit(arg, arg.meta.unit, target_unit)
        arg.unit = target_unit
    return arg


def _interval_property(target_unit):
    return property(functools.partial(_to_unit, target_unit=target_unit))


_interval_add = _binop_expr('__add__', _ops.IntervalAdd)
_interval_radd = _binop_expr('__radd__', _ops.IntervalAdd)
_interval_mul = _binop_expr('__mul__', _ops.IntervalMultiply)
_interval_rmul = _binop_expr('__rmul__', _ops.IntervalMultiply)
_interval_floordiv = _binop_expr('__floordiv__', _ops.IntervalFloorDivide)

_interval_value_methods = dict(
    to_unit=_to_unit,
    years=_interval_property('Y'),
    quarters=_interval_property('Q'),
    months=_interval_property('M'),
    weeks=_interval_property('W'),
    days=_interval_property('D'),
    hours=_interval_property('h'),
    minutes=_interval_property('m'),
    seconds=_interval_property('s'),
    milliseconds=_interval_property('ms'),
    microseconds=_interval_property('us'),
    nanoseconds=_interval_property('ns'),

    __add__=_interval_add,
    add=_interval_add,

    __radd__=_interval_radd,
    radd=_interval_radd,

    __mul__=_interval_mul,
    mul=_interval_mul,

    __rmul__=_interval_rmul,
    rmul=_interval_rmul,

    __floordiv__=_interval_floordiv,
    floordiv=_interval_floordiv
)

_add_methods(IntervalValue, _interval_value_methods)


# ---------------------------------------------------------------------
# Time API

def between_time(arg, lower, upper, timezone=None):
    """
    Check if the input expr falls between the lower/upper bounds
    passed. Bounds are inclusive. All arguments must be comparable.

    Parameters
    ----------
    lower : str, datetime.time
    upper : str, datetime.time
    timezone : str, timezone, default None

    Returns
    -------
    is_between : BooleanValue
    """

    if isinstance(arg.op(), _ops.Time):
        # Here we pull out the first argument to the underlying Time operation
        # which is by definition (in _timestamp_value_methods) a
        # TimestampValue. We do this so that we can potentially specialize the
        # "between time" operation for timestamp_value_expr.time().between().
        # A similar mechanism is triggered when creating expressions like
        # t.column.distinct().count(), which is turned into t.column.nunique().
        arg = arg.op().args[0]
        if timezone is not None:
            arg = arg.cast(dt.Timestamp(timezone=timezone))
        op = _ops.BetweenTime(arg, lower, upper)
    else:
        op = _ops.Between(arg, lower, upper)

    return op.to_expr()


def _time_truncate(arg, unit):
    """
    Zero out smaller-size units beyond indicated unit. Commonly used for time
    series resampling.

    Parameters
    ----------
    unit : string, one of below table
      'h': hour
      'm': minute
      's': second
      'ms': millisecond
      'us': microsecond
      'ns': nanosecond

    Returns
    -------
    truncated : time
    """
    return _ops.TimeTruncate(arg, unit).to_expr()


def _time_sub(left, right):
    right = as_value_expr(right)

    if isinstance(right, ir.TimeValue):
        op = _ops.TimeDiff(left, right)
    else:
        op = _ops.TimeSub(left, right)  # let the operation validate

    return op.to_expr()


_time_add = _binop_expr('__add__', _ops.TimeAdd)


_time_value_methods = dict(
    between=between_time,
    truncate=_time_truncate,

    __sub__=_time_sub,
    sub=_time_sub,

    __add__=_time_add,
    add=_time_add,

    __radd__=_time_add,
    radd=_time_add
)

_add_methods(TimeValue, _time_value_methods)


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
    'any_inner': _ops.AnyInnerJoin,
    'any_left': _ops.AnyLeftJoin,
    'outer': _ops.OuterJoin,
    'right': _ops.RightJoin,
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
      - 'right': right outer join
      - 'semi' or 'left_semi': left semi join
      - 'anti': anti join

    Returns
    -------
    joined : TableExpr
      Note, schema is not materialized yet
    """
    klass = _join_classes[how.lower()]
    if isinstance(predicates, Expr):
        predicates = _L.flatten_predicate(predicates)

    op = klass(left, right, predicates)
    return TableExpr(op)


def asof_join(left, right, predicates=(), by=()):
    """
    Perform an asof join between two tables.  Similar to a left join
    except that the match is done on nearest key rather than equal keys.

    Optionally, match keys with 'by' before joining with predicates.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr
    predicates : join expression(s)
    by : string
      column to group by before joining

    Returns
    -------
    joined : TableExpr
      Note, schema is not materialized yet
    """
    return _ops.AsOfJoin(left, right, predicates, by).to_expr()


def cross_join(*tables, **kwargs):
    """
    Perform a cross join (cartesian product) amongst a list of tables, with
    optional set of prefixes to apply to overlapping column names

    Parameters
    ----------
    tables : ibis.expr.types.TableExpr

    Returns
    -------
    joined : TableExpr

    Examples
    --------
    >>> import ibis
    >>> schemas = [(name, 'int64') for name in 'abcde']
    >>> a, b, c, d, e = [
    ...     ibis.table([(name, type)], name=name) for name, type in schemas
    ... ]
    >>> joined1 = ibis.cross_join(a, b, c, d, e)
    >>> joined1  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: a
      schema:
        a : int64
    ref_1
    UnboundTable[table]
      name: b
      schema:
        b : int64
    ref_2
    UnboundTable[table]
      name: c
      schema:
        c : int64
    ref_3
    UnboundTable[table]
      name: d
      schema:
        d : int64
    ref_4
    UnboundTable[table]
      name: e
      schema:
        e : int64
    CrossJoin[table]
      left:
        Table: ref_0
      right:
        CrossJoin[table]
          left:
            CrossJoin[table]
              left:
                CrossJoin[table]
                  left:
                    Table: ref_1
                  right:
                    Table: ref_2
              right:
                Table: ref_3
          right:
            Table: ref_4
    """
    # TODO(phillipc): Implement prefix keyword argument
    return TableExpr(_ops.CrossJoin(*tables, **kwargs))


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
    result = 'Table rows: {}\n\n{}'.format(metrics[0], col_metrics)

    print(result, file=buf)


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
        predicates = _L.flatten_predicate(predicates)
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
    depending on implementation and presence of a sorting).

    Parameters
    ----------
    n : int
      Number of rows to include
    offset : int, default 0
      Number of rows to skip first

    Returns
    -------
    limited : TableExpr
    """
    op = _ops.Limit(table, n, offset=offset)
    return TableExpr(op)


def _head(table, n=5):
    """
    Select the first n rows at beginning of a table (may not be deterministic
    depending on implementation and presence of a sorting).

    Parameters
    ----------
    n : int
      Number of rows to include, defaults to 5

    Returns
    -------
    limited : TableExpr

    See Also
    --------
    ibis.expr.types.TableExpr.limit
    """
    return _table_limit(table, n=n)


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
    >>> import ibis
    >>> t = ibis.table([('a', 'int64'), ('b', 'string')])
    >>> ab_sorted = t.sort_by([('a', True), ('b', False)])

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


def add_column(table, expr, name=None):
    """
    Add indicated column expression to table, producing a new table. Note:
    this is a shortcut for performing a projection having the same effect.

    Returns
    -------
    modified_table : TableExpr
    """
    warnings.warn('add_column is deprecated, use mutate(name=expr, ...)',
                  DeprecationWarning)
    if name is not None:
        return table.mutate(**{name: expr})
    else:
        return table.mutate(expr)


def _safe_get_name(expr):
    try:
        return expr.get_name()
    except _com.ExpressionError:
        return None


def mutate(table, exprs=None, **mutations):
    """
    Convenience function for table projections involving adding columns

    Parameters
    ----------
    exprs : list, default None
      List of named expressions to add as columns
    mutations : keywords for new columns

    Returns
    -------
    mutated : TableExpr

    Examples
    --------
    Using keywords arguments to name the new columns

    >>> import ibis
    >>> table = ibis.table([('foo', 'double'), ('bar', 'double')], name='t')
    >>> expr = table.mutate(qux=table.foo + table.bar, baz=5)
    >>> expr  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        foo : double
        bar : double
    <BLANKLINE>
    Selection[table]
      table:
        Table: ref_0
      selections:
        Table: ref_0
        baz = Literal[int8]
          5
        qux = Add[double*]
          left:
            foo = Column[double*] 'foo' from table
              ref_0
          right:
            bar = Column[double*] 'bar' from table
              ref_0

    Using the :meth:`ibis.expr.types.Expr.name` method to name the new columns

    >>> new_columns = [ibis.literal(5).name('baz',),
    ...                (table.foo + table.bar).name('qux')]
    >>> expr2 = table.mutate(new_columns)
    >>> expr.equals(expr2)
    True
    """
    if exprs is None:
        exprs = []
    else:
        exprs = util.promote_list(exprs)

    for k, v in sorted(mutations.items()):
        if util.is_function(v):
            v = v(table)
        else:
            v = as_value_expr(v)

        # TODO(phillipc): Fix this by making expressions hashable
        named_v = v.name(k) if _safe_get_name(v) != k else v
        exprs.append(named_v)

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

    Notes
    -----
    Passing an aggregate function to this method will broadcast the aggregate's
    value over the number of rows in the table. See the examples section for
    more details.

    Examples
    --------
    Simple projection

    >>> import ibis
    >>> fields = [('a', 'int64'), ('b', 'double')]
    >>> t = ibis.table(fields, name='t')
    >>> proj = t.projection([t.a, (t.b + 1).name('b_plus_1')])
    >>> proj  # doctest: +NORMALIZE_WHITESPACE
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : int64
        b : double
    <BLANKLINE>
    Selection[table]
      table:
        Table: ref_0
      selections:
        a = Column[int64*] 'a' from table
          ref_0
        b_plus_1 = Add[double*]
          left:
            b = Column[double*] 'b' from table
              ref_0
          right:
            Literal[int8]
              1
    >>> proj2 = t[t.a, (t.b + 1).name('b_plus_1')]
    >>> proj.equals(proj2)
    True

    Aggregate projection

    >>> agg_proj = t[t.a.sum().name('sum_a'), t.b.mean().name('mean_b')]
    >>> agg_proj  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    ref_0
    UnboundTable[table]
      name: t
      schema:
        a : int64
        b : double
    <BLANKLINE>
    Selection[table]
      table:
        Table: ref_0
      selections:
        sum_a = WindowOp[int64*]
          sum_a = Sum[int64]
            a = Column[int64*] 'a' from table
              ref_0
            where:
              None
          <ibis.expr.window.Window object at 0x...>
        mean_b = WindowOp[double*]
          mean_b = Mean[double]
            b = Column[double*] 'b' from table
              ref_0
            where:
              None
          <ibis.expr.window.Window object at 0x...>

    Note the ``<ibis.expr.window.Window>`` objects here, their existence means
    that the result of the aggregation will be broadcast across the number of
    rows in the input column. The purpose of this expression rewrite is to make
    it easy to write column/scalar-aggregate operations like

    .. code-block:: python

       t[(t.a - t.a.mean()).name('demeaned_a')]
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
    head=_head,
    set_column=_table_set_column,
    filter=filter,
    materialize=_table_materialize,
    add_column=add_column,
    mutate=mutate,
    projection=projection,
    select=projection,
    relabel=_table_relabel,
    join=join,
    cross_join=cross_join,
    inner_join=_regular_join_method('inner_join', 'inner'),
    left_join=_regular_join_method('left_join', 'left'),
    any_inner_join=_regular_join_method('any_inner_join', 'any_inner'),
    any_left_join=_regular_join_method('any_left_join', 'any_left'),
    outer_join=_regular_join_method('outer_join', 'outer'),
    semi_join=_regular_join_method('semi_join', 'semi'),
    anti_join=_regular_join_method('anti_join', 'anti'),
    asof_join=asof_join,
    sort_by=_table_sort_by,
    to_array=_table_to_array,
    union=_table_union,
    view=_table_view
)


_add_methods(TableExpr, _table_methods)
