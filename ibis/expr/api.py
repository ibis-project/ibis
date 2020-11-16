"""Ibis expression API definitions."""

import collections
import datetime
import functools
import numbers
import operator
from typing import Union

import dateutil.parser
import pandas as pd
import toolz

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as _L
import ibis.expr.analytics as _analytics
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.expr.analytics import bucket, histogram
from ibis.expr.groupby import GroupedTableExpr  # noqa
from ibis.expr.random import random  # noqa
from ibis.expr.schema import Schema
from ibis.expr.types import (  # noqa
    ArrayColumn,
    ArrayScalar,
    ArrayValue,
    BooleanColumn,
    BooleanScalar,
    BooleanValue,
    CategoryScalar,
    CategoryValue,
    ColumnExpr,
    DateColumn,
    DateScalar,
    DateValue,
    DecimalColumn,
    DecimalScalar,
    DecimalValue,
    DestructColumn,
    DestructScalar,
    DestructValue,
    Expr,
    FloatingColumn,
    FloatingScalar,
    FloatingValue,
    GeoSpatialColumn,
    GeoSpatialScalar,
    GeoSpatialValue,
    IntegerColumn,
    IntegerScalar,
    IntegerValue,
    IntervalColumn,
    IntervalScalar,
    IntervalValue,
    LineStringColumn,
    LineStringScalar,
    LineStringValue,
    MapColumn,
    MapScalar,
    MapValue,
    MultiLineStringColumn,
    MultiLineStringScalar,
    MultiLineStringValue,
    MultiPointColumn,
    MultiPointScalar,
    MultiPointValue,
    MultiPolygonColumn,
    MultiPolygonScalar,
    MultiPolygonValue,
    NullColumn,
    NullScalar,
    NullValue,
    NumericColumn,
    NumericScalar,
    NumericValue,
    PointColumn,
    PointScalar,
    PointValue,
    PolygonColumn,
    PolygonScalar,
    PolygonValue,
    ScalarExpr,
    StringColumn,
    StringScalar,
    StringValue,
    StructColumn,
    StructScalar,
    StructValue,
    TableExpr,
    TimeColumn,
    TimeScalar,
    TimestampColumn,
    TimestampScalar,
    TimestampValue,
    TimeValue,
    ValueExpr,
    as_value_expr,
    literal,
    null,
    param,
    sequence,
)
from ibis.expr.window import (
    cumulative_window,
    range_window,
    rows_with_max_lookback,
    trailing_range_window,
    trailing_window,
    window,
)

__all__ = (
    'aggregate',
    'case',
    'cast',
    'coalesce',
    'cross_join',
    'cumulative_window',
    'date',
    'desc',
    'Expr',
    'expr_list',
    'geo_area',
    'geo_as_binary',
    'geo_as_ewkb',
    'geo_as_ewkt',
    'geo_as_text',
    'geo_azimuth',
    'geo_buffer',
    'geo_centroid',
    'geo_contains',
    'geo_contains_properly',
    'geo_covers',
    'geo_covered_by',
    'geo_crosses',
    'geo_d_fully_within',
    'geo_disjoint',
    'geo_difference',
    'geo_d_within',
    'geo_envelope',
    'geo_equals',
    'geo_geometry_n',
    'geo_geometry_type',
    'geo_intersection',
    'geo_intersects',
    'geo_is_valid',
    'geo_line_locate_point',
    'geo_line_merge',
    'geo_line_substring',
    'geo_ordering_equals',
    'geo_overlaps',
    'geo_touches',
    'geo_distance',
    'geo_end_point',
    'geo_length',
    'geo_max_distance',
    'geo_n_points',
    'geo_n_rings',
    'geo_perimeter',
    'geo_point',
    'geo_point_n',
    'geo_simplify',
    'geo_srid',
    'geo_start_point',
    'geo_transform',
    'geo_unary_union',
    'geo_union',
    'geo_within',
    'geo_x',
    'geo_x_max',
    'geo_x_min',
    'geo_y',
    'geo_y_max',
    'geo_y_min',
    'greatest',
    'ifelse',
    'infer_dtype',
    'infer_schema',
    'interval',
    'join',
    'least',
    'literal',
    'NA',
    'negate',
    'now',
    'null',
    'param',
    'pi',
    'prevent_rewrite',
    'random',
    'range_window',
    'row_number',
    'rows_with_max_lookback',
    'schema',
    'Schema',
    'sequence',
    'table',
    'time',
    'timestamp',
    'trailing_range_window',
    'trailing_window',
    'where',
    'window',
)


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


NA = null()


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

    node = ops.UnboundTable(schema, name=name)
    return node.to_expr()


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
        return ops.DeferredSortKey(expr, ascending=False)
    else:
        return ops.SortKey(expr, ascending=False).to_expr()


def timestamp(value, timezone=None):
    """
    Returns a timestamp literal if value is likely coercible to a timestamp

    Parameters
    ----------
    value : timestamp value as string
    timezone: timezone as string
        defaults to None

    Returns
    --------
    result : TimestampScalar
    """
    if isinstance(value, str):
        try:
            value = pd.Timestamp(value, tz=timezone)
        except pd.errors.OutOfBoundsDatetime:
            value = dateutil.parser.parse(value)
    if isinstance(value, numbers.Integral):
        raise TypeError(
            (
                "Passing an integer to ibis.timestamp is not supported. Use "
                "ibis.literal({value:d}).to_timestamp() to create a timestamp "
                "expression from an integer."
            ).format(value=value)
        )
    return literal(value, type=dt.Timestamp(timezone=timezone))


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
    if isinstance(value, str):
        value = pd.to_datetime(value).date()
    return literal(value, type=dt.date)


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
    if isinstance(value, str):
        value = pd.to_datetime(value).time()
    return literal(value, type=dt.time)


def interval(
    value=None,
    unit='s',
    years=None,
    quarters=None,
    months=None,
    weeks=None,
    days=None,
    hours=None,
    minutes=None,
    seconds=None,
    milliseconds=None,
    microseconds=None,
    nanoseconds=None,
):
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
        elif not isinstance(value, int):
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
            ('ns', nanoseconds),
        ]
        defined_units = [(k, v) for k, v in kwds if v is not None]

        if len(defined_units) != 1:
            raise ValueError('Exactly one argument is required')

        unit, value = defined_units[0]

    value_type = literal(value).type()
    type = dt.Interval(unit, value_type)

    return literal(value, type=type).op().to_expr()


schema.__doc__ = """\
Validate and return an Ibis Schema object

{}

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
""".format(
    _data_type_docs
)


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
    return ops.SearchedCaseBuilder()


def now():
    """
    Compute the current timestamp

    Returns
    -------
    now : Timestamp scalar
    """
    return ops.TimestampNow().to_expr()


def row_number():
    """Analytic function for the current row number, starting at 0.

    This function does not require an ORDER BY clause, however, without an
    ORDER BY clause the order of the result is nondeterministic.

    Returns
    -------
    row_number : IntArray
    """
    return ops.RowNumber().to_expr()


e = ops.E().to_expr()

pi = ops.Pi().to_expr()


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
        result = ops.Negate(arg)

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
    if isinstance(op, ops.DistinctColumn):
        result = ops.CountDistinct(op.args[0], where).to_expr()
    else:
        result = ops.Count(expr, where).to_expr()

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
    return ops.GroupConcat(arg, sep, where).to_expr()


def arbitrary(arg, where=None, how=None):
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
    return ops.Arbitrary(arg, how, where).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        try:
            other = as_value_expr(other)
            op = klass(self, other)
            return op.to_expr()
        except (com.IbisTypeError, NotImplementedError):
            return NotImplemented

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

        if not isinstance(other, ir.BooleanValue):
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

        if not isinstance(other, ir.BooleanValue):
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
    op = ops.Cast(arg, to=target_type)

    if op.to.equals(arg.type()):
        # noop case if passed type is the same
        return arg

    if isinstance(op.to, (dt.Geography, dt.Geometry)):
        from_geotype = arg.type().geotype or 'geometry'
        to_geotype = op.to.geotype
        if from_geotype == to_geotype:
            return arg

    result = op.to_expr()
    if not arg.has_name():
        return result
    expr_name = 'cast({}, {})'.format(arg.get_name(), op.to)
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
""".format(
    _data_type_docs
)


def typeof(arg):
    """
    Return the data type of the argument according to the current backend

    Returns
    -------
    typeof_arg : string
    """
    return ops.TypeOf(arg).to_expr()


def hash(arg, how='fnv'):
    """
    Compute an integer hash value for the indicated value expression.

    Parameters
    ----------
    arg : value expression
    how : {'fnv', 'farm_fingerprint'}, default 'fnv'
      Hash algorithm to use

    Returns
    -------
    hash_value : int64 expression
    """
    return ops.Hash(arg, how).to_expr()


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
    return ops.IfNull(arg, fill_value).to_expr()


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
    return ops.Coalesce(args).to_expr()


def greatest(*args):
    """
    Compute the largest value (row-wise, if any arrays are present) among the
    supplied arguments.

    Returns
    -------
    greatest : type depending on arguments
    """
    return ops.Greatest(args).to_expr()


def least(*args):
    """
    Compute the smallest value (row-wise, if any arrays are present) among the
    supplied arguments.

    Returns
    -------
    least : type depending on arguments
    """
    return ops.Least(args).to_expr()


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
    op = ops.Where(boolean_expr, true_expr, false_null_expr)
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

    if isinstance(prior_op, ops.WindowOp):
        op = prior_op.over(window)
    else:
        op = ops.WindowOp(expr, window)

    result = op.to_expr()

    try:
        name = expr.get_name()
    except com.ExpressionError:
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
    except com.ExpressionError:
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
    return ops.NullIf(value, null_if_expr).to_expr()


def between(arg, lower, upper):
    """
    Check if the input expr falls between the lower/upper bounds
    passed. Bounds are inclusive. All arguments must be comparable.

    Returns
    -------
    is_between : BooleanValue
    """
    lower = as_value_expr(lower)
    upper = as_value_expr(upper)

    op = ops.Between(arg, lower, upper)
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
    op = ops.Contains(arg, values)
    return op.to_expr()


def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = ops.NotContains(arg, values)
    return op.to_expr()


add = _binop_expr('__add__', ops.Add)
sub = _binop_expr('__sub__', ops.Subtract)
mul = _binop_expr('__mul__', ops.Multiply)
div = _binop_expr('__div__', ops.Divide)
floordiv = _binop_expr('__floordiv__', ops.FloorDivide)
pow = _binop_expr('__pow__', ops.Power)
mod = _binop_expr('__mod__', ops.Modulus)

radd = _rbinop_expr('__radd__', ops.Add)
rsub = _rbinop_expr('__rsub__', ops.Subtract)
rdiv = _rbinop_expr('__rdiv__', ops.Divide)
rfloordiv = _rbinop_expr('__rfloordiv__', ops.FloorDivide)


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
        Literal[string]
          a
        Literal[string]
          b
      results:
        Literal[string]
          an a
        Literal[string]
          a b
      default:
        Literal[string]
          null or (not a and not b)
    """
    return ops.SimpleCaseBuilder(arg)


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
    isnull=_unary_op('isnull', ops.IsNull),
    notnull=_unary_op('notnull', ops.NotNull),
    over=over,
    case=_case,
    cases=cases,
    substitute=substitute,
    __eq__=_binop_expr('__eq__', ops.Equals),
    __ne__=_binop_expr('__ne__', ops.NotEquals),
    __ge__=_binop_expr('__ge__', ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', ops.Greater),
    __le__=_binop_expr('__le__', ops.LessEqual),
    __lt__=_binop_expr('__lt__', ops.Less),
    collect=_unary_op('collect', ops.ArrayCollect),
    identical_to=_binop_expr('identical_to', ops.IdenticalTo),
)


approx_nunique = _agg_function('approx_nunique', ops.HLLCardinality, True)
approx_median = _agg_function('approx_median', ops.CMSMedian, True)
max = _agg_function('max', ops.Max, True)
min = _agg_function('min', ops.Min, True)
nunique = _agg_function('nunique', ops.CountDistinct, True)


def lag(arg, offset=None, default=None):
    return ops.Lag(arg, offset, default).to_expr()


def lead(arg, offset=None, default=None):
    return ops.Lead(arg, offset, default).to_expr()


first = _unary_op('first', ops.FirstValue)
last = _unary_op('last', ops.LastValue)
rank = _unary_op('rank', ops.MinRank)
dense_rank = _unary_op('dense_rank', ops.DenseRank)
percent_rank = _unary_op('percent_rank', ops.PercentRank)
cummin = _unary_op('cummin', ops.CumulativeMin)
cummax = _unary_op('cummax', ops.CumulativeMax)


def ntile(arg, buckets):
    return ops.NTile(arg, buckets).to_expr()


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
    return ops.NthValue(arg, k).to_expr()


def distinct(arg):
    """
    Compute set of unique values occurring in this array. Can not be used
    in conjunction with other array expressions from the same context
    (because it's a cardinality-modifying pseudo-reduction).
    """
    op = ops.DistinctColumn(arg)
    return op.to_expr()


def topk(arg, k, by=None):
    """
    Returns
    -------
    topk : TopK filter expression
    """
    op = ops.TopK(arg, k, by=by)
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
    metrics = [arg.count(), arg.isnull().sum().name('nulls')]

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
        arg.mean(),
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
    return ops.ExpressionList(exprs).to_expr()


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


# TODO: should bound to AnyValue and AnyColumn instead, but that breaks
#       doc builds, because it checks methods on ColumnExpr
_add_methods(ir.ValueExpr, _generic_value_methods)
_add_methods(ir.ColumnExpr, _generic_column_methods)


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
    op = ops.Round(arg, digits)
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
    op = ops.Log(arg, base)
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
        raise ValueError("at least one of lower and " "upper must be provided")

    op = ops.Clip(arg, lower, upper)
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
    if isinstance(quantile, collections.abc.Sequence):
        op = ops.MultiQuantile(arg, quantile, interpolation)
    else:
        op = ops.Quantile(arg, quantile, interpolation)
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
    op = ops.TimestampFromUNIX(arg, unit)
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
    op = ops.IntervalFromInteger(arg, unit)
    return op.to_expr()


abs = _unary_op('abs', ops.Abs)
ceil = _unary_op('ceil', ops.Ceil)
degrees = _unary_op('degrees', ops.Degrees)
exp = _unary_op('exp', ops.Exp)
floor = _unary_op('floor', ops.Floor)
log2 = _unary_op('log2', ops.Log2)
log10 = _unary_op('log10', ops.Log10)
ln = _unary_op('ln', ops.Ln)
radians = _unary_op('radians', ops.Radians)
sign = _unary_op('sign', ops.Sign)
sqrt = _unary_op('sqrt', ops.Sqrt)

# TRIGONOMETRIC OPERATIONS
acos = _unary_op('acos', ops.Acos)
asin = _unary_op('asin', ops.Asin)
atan = _unary_op('atan', ops.Atan)
atan2 = _binop_expr('atan2', ops.Atan2)
cos = _unary_op('cos', ops.Cos)
cot = _unary_op('cot', ops.Cot)
sin = _unary_op('sin', ops.Sin)
tan = _unary_op('tan', ops.Tan)


_numeric_value_methods = dict(
    __neg__=negate,
    abs=abs,
    ceil=ceil,
    degrees=degrees,
    deg2rad=radians,
    floor=floor,
    radians=radians,
    rad2deg=degrees,
    sign=sign,
    exp=exp,
    sqrt=sqrt,
    log=log,
    ln=ln,
    log2=log2,
    log10=log10,
    round=round,
    nullifzero=_unary_op('nullifzero', ops.NullIfZero),
    zeroifnull=_unary_op('zeroifnull', ops.ZeroIfNull),
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
    __rmul__=_rbinop_expr('__rmul__', ops.Multiply),
    __rpow__=_rbinop_expr('__rpow__', ops.Power),
    __mod__=mod,
    __rmod__=_rbinop_expr('__rmod__', ops.Modulus),
    # trigonometric operations
    acos=acos,
    asin=asin,
    atan=atan,
    atan2=atan2,
    cos=cos,
    cot=cot,
    sin=sin,
    tan=tan,
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
    return ops.BaseConvert(arg, from_base, to_base).to_expr()


_integer_value_methods = dict(
    to_timestamp=_integer_to_timestamp,
    to_interval=_integer_to_interval,
    convert_base=convert_base,
)


mean = _agg_function('mean', ops.Mean, True)
cummean = _unary_op('cummean', ops.CumulativeMean)

sum = _agg_function('sum', ops.Sum, True)
cumsum = _unary_op('cumsum', ops.CumulativeSum)


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
    expr = ops.StandardDev(arg, how, where).to_expr()
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
    expr = ops.Variance(arg, how, where).to_expr()
    expr = expr.name('var')
    return expr


def correlation(left, right, where=None, how='sample'):
    """
    Compute correlation of two numeric array

    Parameters
    ----------
    how : {'sample', 'pop'}, default 'sample'

    Returns
    -------
    corr : double scalar
    """
    expr = ops.Correlation(left, right, how, where).to_expr()
    return expr


def covariance(left, right, where=None, how='sample'):
    """
    Compute covariance of two numeric array

    Parameters
    ----------
    how : {'sample', 'pop'}, default 'sample'

    Returns
    -------
    cov : double scalar
    """
    expr = ops.Covariance(left, right, how, where).to_expr()
    return expr


_numeric_column_methods = dict(
    mean=mean,
    cummean=cummean,
    sum=sum,
    cumsum=cumsum,
    quantile=quantile,
    std=std,
    var=variance,
    corr=correlation,
    cov=covariance,
    bucket=bucket,
    histogram=histogram,
    summary=_numeric_summary,
)

_floating_value_methods = dict(
    isnan=_unary_op('isnull', ops.IsNan), isinf=_unary_op('isinf', ops.IsInf)
)

_add_methods(ir.NumericValue, _numeric_value_methods)
_add_methods(ir.IntegerValue, _integer_value_methods)
_add_methods(ir.FloatingValue, _floating_value_methods)

_add_methods(ir.NumericColumn, _numeric_column_methods)

# ----------------------------------------------------------------------
# GeoSpatial API


def geo_area(arg):
    """
    Compute area of a geo spatial data

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    area : double scalar
    """
    op = ops.GeoArea(arg)
    return op.to_expr()


def geo_as_binary(arg):
    """
    Get the geometry as well-known bytes (WKB) without the SRID data.

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    wkb : binary
    """
    op = ops.GeoAsBinary(arg)
    return op.to_expr()


def geo_as_ewkt(arg):
    """
    Get the geometry as well-known text (WKT) with the SRID data.

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    wkt : string
    """
    op = ops.GeoAsEWKT(arg)
    return op.to_expr()


def geo_as_text(arg):
    """
    Get the geometry as well-known text (WKT) without the SRID data.

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    wkt : string
    """
    op = ops.GeoAsText(arg)
    return op.to_expr()


def geo_as_ewkb(arg):
    """
    Get the geometry as well-known bytes (WKB) with the SRID data.

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    wkb : binary
    """
    op = ops.GeoAsEWKB(arg)
    return op.to_expr()


def geo_contains(left, right):
    """
    Check if the first geometry contains the second one

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    contains : bool scalar
    """
    op = ops.GeoContains(left, right)
    return op.to_expr()


def geo_contains_properly(left, right):
    """
    Check if the first geometry contains the second one,
    with no common border points.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    contains_properly : bool scalar
    """
    op = ops.GeoContainsProperly(left, right)
    return op.to_expr()


def geo_covers(left, right):
    """
    Check if the first geometry covers the second one.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    covers : bool scalar
    """
    op = ops.GeoCovers(left, right)
    return op.to_expr()


def geo_covered_by(left, right):
    """
    Check if the first geometry is covered by the second one.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    covered_by : bool scalar
    """
    op = ops.GeoCoveredBy(left, right)
    return op.to_expr()


def geo_crosses(left, right):
    """
    Check if the geometries have some, but not all, interior points in common.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    crosses : bool scalar
    """
    op = ops.GeoCrosses(left, right)
    return op.to_expr()


def geo_d_fully_within(left, right, distance):
    """
    Check if the first geometry is fully within a specified distance from
    the second one.

    Parameters
    ----------
    left : geometry
    right : geometry
    distance: double

    Returns
    -------
    d_fully_within : bool scalar
    """
    op = ops.GeoDFullyWithin(left, right, distance)
    return op.to_expr()


def geo_disjoint(left, right):
    """
    Check if the geometries have no points in common.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    disjoint : bool scalar
    """
    op = ops.GeoDisjoint(left, right)
    return op.to_expr()


def geo_d_within(left, right, distance):
    """
    Check if the first geometry is within a specified distance from
    the second one.

    Parameters
    ----------
    left : geometry
    right : geometry
    distance: double

    Returns
    -------
    d_within : bool scalar
    """
    op = ops.GeoDWithin(left, right, distance)
    return op.to_expr()


def geo_equals(left, right):
    """
    Check if the geometries are the same.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    equals : bool scalar
    """
    op = ops.GeoEquals(left, right)
    return op.to_expr()


def geo_geometry_n(arg, n):
    """
    Get the 1-based Nth geometry of a multi geometry.

    Parameters
    ----------
    arg : geometry
    n : integer

    Returns
    -------
    geom : geometry scalar
    """
    op = ops.GeoGeometryN(arg, n)
    return op.to_expr()


def geo_geometry_type(arg):
    """
    Get the type of a geometry.

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    type : string scalar
    """
    op = ops.GeoGeometryType(arg)
    return op.to_expr()


def geo_intersects(left, right):
    """
    Check if the geometries share any points.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    intersects : bool scalar
    """
    op = ops.GeoIntersects(left, right)
    return op.to_expr()


def geo_is_valid(arg):
    """
    Check if the geometry is valid.

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    valid : bool scalar
    """
    op = ops.GeoIsValid(arg)
    return op.to_expr()


def geo_line_locate_point(left, right):
    """
    Locate the distance a point falls along the length of a line.

    Returns a float between zero and one representing the location of the
    closest point on the linestring to the given point, as a fraction of the
    total 2d line length.

    Parameters
    ----------
    left : linestring
    right: point

    Returns
    -------
    distance: float scalar
    """
    op = ops.GeoLineLocatePoint(left, right)
    return op.to_expr()


def geo_line_merge(arg):
    """
    Merge a MultiLineString into a LineString.

    Returns a (set of) LineString(s) formed by sewing together the
    constituent line work of a MultiLineString. If a geometry other than
    a LineString or MultiLineString is given, this will return an empty
    geometry collection.

    Parameters
    ----------
    arg : (multi)linestring

    Returns
    -------
    merged: geometry scalar
    """
    op = ops.GeoLineMerge(arg)
    return op.to_expr()


def geo_line_substring(arg, start, end):
    """
    Clip a substring from a LineString.

    Returns a linestring that is a substring of the input one, starting
    and ending at the given fractions of the total 2d length. The second
    and third arguments are floating point values between zero and one.
    This only works with linestrings.

    Parameters
    ----------
    arg: linestring
    start: float
    end: float

    Returns
    -------
    substring: linestring scalar
    """
    op = ops.GeoLineSubstring(arg, start, end)
    return op.to_expr()


def geo_ordering_equals(left, right):
    """
    Check if two geometries are equal and have the same point ordering.

    Returns true if the two geometries are equal and the coordinates
    are in the same order.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    ordering_equals : bool scalar
    """
    op = ops.GeoOrderingEquals(left, right)
    return op.to_expr()


def geo_overlaps(left, right):
    """
    Check if the geometries share space, are of the same dimension,
    but are not completely contained by each other.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    overlaps : bool scalar
    """
    op = ops.GeoOverlaps(left, right)
    return op.to_expr()


def geo_point(
    left: Union[NumericValue, int, float],
    right: Union[NumericValue, int, float],
) -> ops.GeoPoint:
    """
    Return a point constructed on the fly from the provided coordinate values.
    Constant coordinates result in construction of a POINT literal.

    Parameters
    ----------
    left : NumericValue, integer or float
    right : NumericValue, integer or float

    Returns
    -------
    point
    """
    op = ops.GeoPoint(left, right)
    return op.to_expr()


def geo_touches(left, right):
    """
    Check if the geometries have at least one point in common,
    but do not intersect.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    touches : bool scalar
    """
    op = ops.GeoTouches(left, right)
    return op.to_expr()


def geo_distance(left, right):
    """
    Compute distance between two geo spatial data

    Parameters
    ----------
    left : geometry or geography
    right : geometry or geography

    Returns
    -------
    distance : double scalar
    """
    op = ops.GeoDistance(left, right)
    return op.to_expr()


def geo_length(arg):
    """
    Compute length of a geo spatial data

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    length : double scalar
    """
    op = ops.GeoLength(arg)
    return op.to_expr()


def geo_perimeter(arg):
    """
    Compute perimeter of a geo spatial data

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    perimeter : double scalar
    """
    op = ops.GeoPerimeter(arg)
    return op.to_expr()


def geo_max_distance(left, right):
    """Returns the 2-dimensional maximum distance between two geometries in
    projected units. If g1 and g2 is the same geometry the function will
    return the distance between the two vertices most far from each other
    in that geometry

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    MaxDistance : double scalar
    """
    op = ops.GeoMaxDistance(left, right)
    return op.to_expr()


def geo_unary_union(arg):
    """
    Aggregate a set of geometries into a union.

    This corresponds to the aggregate version of the PostGIS ST_Union.
    We give it a different name (following the corresponding method
    in GeoPandas) to avoid name conflicts with the non-aggregate version.

    Parameters
    ----------
    arg : geometry column

    Returns
    -------
    union : geometry scalar
    """
    expr = ops.GeoUnaryUnion(arg).to_expr()
    expr = expr.name('union')
    return expr


def geo_union(left, right):
    """
    Merge two geometries into a union geometry.

    Returns the pointwise union of the two geometries.
    This corresponds to the non-aggregate version the PostGIS ST_Union.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    union : geometry scalar
    """
    op = ops.GeoUnion(left, right)
    return op.to_expr()


def geo_x(arg):
    """Return the X coordinate of the point, or NULL if not available.
    Input must be a point

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    X : double scalar
    """
    op = ops.GeoX(arg)
    return op.to_expr()


def geo_y(arg):
    """Return the Y coordinate of the point, or NULL if not available.
    Input must be a point

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    Y : double scalar
    """
    op = ops.GeoY(arg)
    return op.to_expr()


def geo_x_min(arg):
    """Returns Y minima of a geometry

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    XMin : double scalar
    """
    op = ops.GeoXMin(arg)
    return op.to_expr()


def geo_x_max(arg):
    """Returns X maxima of a geometry

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    XMax : double scalar
    """
    op = ops.GeoXMax(arg)
    return op.to_expr()


def geo_y_min(arg):
    """Returns Y minima of a geometry

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    YMin : double scalar
    """
    op = ops.GeoYMin(arg)
    return op.to_expr()


def geo_y_max(arg):
    """Returns Y maxima of a geometry

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    YMax : double scalar
    """
    op = ops.GeoYMax(arg)
    return op.to_expr()


def geo_start_point(arg):
    """Returns the first point of a LINESTRING geometry as a POINT or
    NULL if the input parameter is not a LINESTRING

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    Point : geometry scalar
    """
    op = ops.GeoStartPoint(arg)
    return op.to_expr()


def geo_end_point(arg):
    """Returns the last point of a LINESTRING geometry as a POINT or
    NULL if the input parameter is not a LINESTRING

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    EndPoint : geometry scalar
    """
    op = ops.GeoEndPoint(arg)
    return op.to_expr()


def geo_point_n(arg, n):
    """Return the Nth point in a single linestring in the geometry.
    Negative values are counted backwards from the end of the LineString,
    so that -1 is the last point. Returns NULL if there is no linestring in
    the geometry

    Parameters
    ----------
    arg : geometry
    n : integer

    Returns
    -------
    PointN : geometry scalar
    """
    op = ops.GeoPointN(arg, n)
    return op.to_expr()


def geo_n_points(arg):
    """Return the number of points in a geometry. Works for all geometries

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    NPoints : double scalar
    """
    op = ops.GeoNPoints(arg)
    return op.to_expr()


def geo_n_rings(arg):
    """If the geometry is a polygon or multi-polygon returns the number of
    rings. It counts the outer rings as well

    Parameters
    ----------
    arg : geometry or geography

    Returns
    -------
    NRings : double scalar
    """
    op = ops.GeoNRings(arg)
    return op.to_expr()


def geo_srid(arg):
    """Returns the spatial reference identifier for the ST_Geometry

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    SRID : Integer scalar
    """
    op = ops.GeoSRID(arg)
    return op.to_expr()


def geo_set_srid(arg, srid):
    """Set the spatial reference identifier for the ST_Geometry

    Parameters
    ----------
    arg : geometry
    srid : integer

    Returns
    -------
    SetSRID : geometry
    """
    op = ops.GeoSetSRID(arg, srid)
    return op.to_expr()


def geo_buffer(arg, radius):
    """Returns a geometry that represents all points whose distance from this
    Geometry is less than or equal to distance. Calculations are in the
    Spatial Reference System of this Geometry.

    Parameters
    ----------
    arg : geometry
    radius: double

    Returns
    -------
    buffer : geometry scalar
    """
    op = ops.GeoBuffer(arg, radius)
    return op.to_expr()


def geo_centroid(arg):
    """Returns the centroid of the geometry.

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    centroid : geometry scalar
    """
    op = ops.GeoCentroid(arg)
    return op.to_expr()


def geo_envelope(arg):
    """Returns a geometry representing the bounding box of the arg.

    Parameters
    ----------
    arg : geometry

    Returns
    -------
    envelope : geometry scalar
    """
    op = ops.GeoEnvelope(arg)
    return op.to_expr()


def geo_within(left, right):
    """
    Check if the first geometry is completely inside of the second.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    within : bool scalar
    """
    op = ops.GeoWithin(left, right)
    return op.to_expr()


def geo_azimuth(left, right):
    """
    Check if the geometries have at least one point in common,
    but do not intersect.

    Parameters
    ----------
    left : point
    right : point

    Returns
    -------
    azimuth : float scalar
    """
    op = ops.GeoAzimuth(left, right)
    return op.to_expr()


def geo_intersection(left, right):
    """
    Return the intersection of two geometries.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    intersection : geometry scalar
    """
    op = ops.GeoIntersection(left, right)
    return op.to_expr()


def geo_difference(left, right):
    """
    Return the difference of two geometries.

    Parameters
    ----------
    left : geometry
    right : geometry

    Returns
    -------
    difference : geometry scalar
    """
    op = ops.GeoDifference(left, right)
    return op.to_expr()


def geo_simplify(arg, tolerance, preserve_collapsed):
    """
    Simplify a given geometry.

    Parameters
    ----------
    arg : geometry
    tolerance: float
    preserved_collapsed: boolean

    Returns
    -------
    simplified : geometry scalar
    """
    op = ops.GeoSimplify(arg, tolerance, preserve_collapsed)
    return op.to_expr()


def geo_transform(arg, srid):
    """
    Transform a geometry into a new SRID.

    Parameters
    ----------
    arg : geometry
    srid: integer

    Returns
    -------
    transformed : geometry scalar
    """
    op = ops.GeoTransform(arg, srid)
    return op.to_expr()


_geospatial_value_methods = dict(
    area=geo_area,
    as_binary=geo_as_binary,
    as_ewkb=geo_as_ewkb,
    as_ewkt=geo_as_ewkt,
    as_text=geo_as_text,
    azimuth=geo_azimuth,
    buffer=geo_buffer,
    centroid=geo_centroid,
    contains=geo_contains,
    contains_properly=geo_contains_properly,
    covers=geo_covers,
    covered_by=geo_covered_by,
    crosses=geo_crosses,
    d_fully_within=geo_d_fully_within,
    difference=geo_difference,
    disjoint=geo_disjoint,
    distance=geo_distance,
    d_within=geo_d_within,
    end_point=geo_end_point,
    envelope=geo_envelope,
    equals=geo_equals,
    geometry_n=geo_geometry_n,
    geometry_type=geo_geometry_type,
    intersection=geo_intersection,
    intersects=geo_intersects,
    is_valid=geo_is_valid,
    line_locate_point=geo_line_locate_point,
    line_merge=geo_line_merge,
    line_substring=geo_line_substring,
    length=geo_length,
    max_distance=geo_max_distance,
    n_points=geo_n_points,
    n_rings=geo_n_rings,
    ordering_equals=geo_ordering_equals,
    overlaps=geo_overlaps,
    perimeter=geo_perimeter,
    point_n=geo_point_n,
    set_srid=geo_set_srid,
    simplify=geo_simplify,
    srid=geo_srid,
    start_point=geo_start_point,
    touches=geo_touches,
    transform=geo_transform,
    union=geo_union,
    within=geo_within,
    x=geo_x,
    x_max=geo_x_max,
    x_min=geo_x_min,
    y=geo_y,
    y_max=geo_y_max,
    y_min=geo_y_min,
)
_geospatial_column_methods = dict(unary_union=geo_unary_union)

_add_methods(ir.GeoSpatialValue, _geospatial_value_methods)
_add_methods(ir.GeoSpatialColumn, _geospatial_column_methods)

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
    case = ops.SearchedCaseBuilder()
    return case.when(arg, true_expr).else_(false_expr).end()


_boolean_value_methods = dict(
    ifelse=ifelse,
    __and__=_boolean_binary_op('__and__', ops.And),
    __or__=_boolean_binary_op('__or__', ops.Or),
    __xor__=_boolean_binary_op('__xor__', ops.Xor),
    __rand__=_boolean_binary_rop('__rand__', ops.And),
    __ror__=_boolean_binary_rop('__ror__', ops.Or),
    __rxor__=_boolean_binary_rop('__rxor__', ops.Xor),
    __invert__=_boolean_unary_op('__invert__', ops.Not),
)


_boolean_column_methods = dict(
    any=_unary_op('any', ops.Any),
    notany=_unary_op('notany', ops.NotAny),
    all=_unary_op('all', ops.All),
    notall=_unary_op('notany', ops.NotAll),
    cumany=_unary_op('cumany', ops.CumulativeAny),
    cumall=_unary_op('cumall', ops.CumulativeAll),
)


_add_methods(ir.BooleanValue, _boolean_value_methods)
_add_methods(ir.BooleanColumn, _boolean_column_methods)


# ---------------------------------------------------------------------
# Binary API


def hashbytes(arg, how='sha256'):
    """
    Compute a binary hash value for the indicated value expression.

    Parameters
    ----------
    arg : binary or string value expression
    how : {'md5', 'sha1', 'sha256', 'sha512'}, default 'sha256'
      Hash algorithm to use

    Returns
    -------
    hash_value : binary expression
    """
    return ops.HashBytes(arg, how).to_expr()


_binary_value_methods = dict(hashbytes=hashbytes)
_add_methods(ir.BinaryValue, _binary_value_methods)


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
    Return up to nchars starting from end of each string.

    Returns
    -------
    substrings : type of caller
    """
    return ops.StrRight(self, nchars).to_expr()


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
    return ops.Repeat(self, n).to_expr()


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
    return ops.Translate(self, from_str, to_str).to_expr()


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
    return ops.StringFind(self, substr, start, end).to_expr()


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
    return ops.LPad(self, length, pad).to_expr()


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
    return ops.RPad(self, length, pad).to_expr()


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
    return ops.FindInSet(self, str_list).to_expr()


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
    return ops.StringJoin(self, strings).to_expr()


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
            ops.StringSQLLike(self, pattern).to_expr()
            for pattern in util.promote_list(patterns)
        ),
    )


def _string_ilike(self, patterns):
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
            ops.StringSQLILike(self, pattern).to_expr()
            for pattern in util.promote_list(patterns)
        ),
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
    return ops.RegexSearch(arg, pattern).to_expr()


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
    return ops.RegexExtract(arg, pattern, index).to_expr()


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
    return ops.RegexReplace(arg, pattern, replacement).to_expr()


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
    return ops.StringReplace(arg, pattern, replacement).to_expr()


def to_timestamp(arg, format_str, timezone=None):
    """
    Parses a string and returns a timestamp.

    Parameters
    ----------
    format_str : A format string potentially of the type '%Y-%m-%d'
    timezone : An optional string indicating the timezone,
        i.e. 'America/New_York'

    Examples
    --------
    >>> import ibis
    >>> date_as_str = ibis.literal('20170206')
    >>> result = date_as_str.to_timestamp('%Y%m%d')

    Returns
    -------
    parsed : TimestampValue
    """
    return ops.StringToTimestamp(arg, format_str, timezone).to_expr()


def parse_url(arg, extract, key=None):
    """
    Returns the portion of a URL corresponding to a part specified
    by 'extract'
    Can optionally specify a key to retrieve an associated value
    if extract parameter is 'QUERY'

    Parameters
    ----------
    extract : str
        One of {'PROTOCOL', 'HOST', 'PATH', 'REF', 'AUTHORITY', 'FILE',
            'USERINFO', 'QUERY'}
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
    return ops.ParseURL(arg, extract, key).to_expr()


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
    return ops.StringSplit(arg, delimiter).to_expr()


def _string_concat(*args):
    return ops.StringConcat(args).to_expr()


def _string_dunder_contains(arg, substr):
    raise TypeError('Use val.contains(arg)')


def _string_getitem(self, key):
    if isinstance(key, slice):
        start, stop, step = key.start, key.stop, key.step

        if step is not None and not isinstance(step, ir.Expr) and step != 1:
            raise ValueError('Step can only be 1')

        if not isinstance(start, ir.Expr):
            if start is not None and start < 0:
                raise ValueError(
                    'Negative slicing not yet supported, got start value of '
                    '{:d}'.format(start)
                )
            if start is None:
                start = 0

        if not isinstance(stop, ir.Expr):
            if stop is not None and stop < 0:
                raise ValueError(
                    'Negative slicing not yet supported, got stop value of '
                    '{:d}'.format(stop)
                )
            if stop is None:
                stop = self.length()

        return self.substr(start, stop - start)
    elif isinstance(key, int):
        return self.substr(key, 1)
    raise NotImplementedError(
        'string __getitem__[{}]'.format(type(key).__name__)
    )


_string_value_methods = dict(
    __getitem__=_string_getitem,
    length=_unary_op('length', ops.StringLength),
    lower=_unary_op('lower', ops.Lowercase),
    upper=_unary_op('upper', ops.Uppercase),
    reverse=_unary_op('reverse', ops.Reverse),
    ascii_str=_unary_op('ascii', ops.StringAscii),
    strip=_unary_op('strip', ops.Strip),
    lstrip=_unary_op('lstrip', ops.LStrip),
    rstrip=_unary_op('rstrip', ops.RStrip),
    capitalize=_unary_op('initcap', ops.Capitalize),
    convert_base=convert_base,
    __contains__=_string_dunder_contains,
    contains=_string_contains,
    hashbytes=hashbytes,
    like=_string_like,
    ilike=_string_ilike,
    rlike=re_search,
    replace=_string_replace,
    re_search=re_search,
    re_extract=regex_extract,
    re_replace=regex_replace,
    to_timestamp=to_timestamp,
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


_add_methods(ir.StringValue, _string_value_methods)


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
        if (start is not None and start < 0) or (
            stop is not None and stop < 0
        ):
            raise ValueError('negative slicing not yet supported')

        step = index.step

        if step is not None and step != 1:
            raise NotImplementedError('step can only be 1')

        op = ops.ArraySlice(array, start if start is not None else 0, stop)
    else:
        op = ops.ArrayIndex(array, index)
    return op.to_expr()


_array_column_methods = dict(
    length=_unary_op('length', ops.ArrayLength),
    __getitem__=_array_slice,
    __add__=_binop_expr('__add__', ops.ArrayConcat),
    __radd__=toolz.flip(_binop_expr('__radd__', ops.ArrayConcat)),
    __mul__=_binop_expr('__mul__', ops.ArrayRepeat),
    __rmul__=_binop_expr('__rmul__', ops.ArrayRepeat),
)

_add_methods(ir.ArrayValue, _array_column_methods)


# ---------------------------------------------------------------------
# Map API


def get(expr, key, default=None):
    """
    Return the mapped value for this key, or the default
    if the key does not exist

    Parameters
    ----------
    key : any
    default : any
    """
    return ops.MapValueOrDefaultForKey(expr, key, default).to_expr()


_map_column_methods = dict(
    get=get,
    length=_unary_op('length', ops.MapLength),
    __getitem__=_binop_expr('__getitem__', ops.MapValueForKey),
    keys=_unary_op('keys', ops.MapKeys),
    values=_unary_op('values', ops.MapValues),
    __add__=_binop_expr('__add__', ops.MapConcat),
    __radd__=toolz.flip(_binop_expr('__radd__', ops.MapConcat)),
)

_add_methods(ir.MapValue, _map_column_methods)

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
    return ops.StructField(expr, field_name).to_expr().name(field_name)


def _destructure(expr: StructColumn) -> DestructColumn:
    """ Destructure a ``Struct`` to create a destruct column.

    When assigned, a destruct column will destructured and assigned to multiple
    columns.

    Parameters
    ----------
    expr : StructColumn
        The struct column to destructure.

    Returns
    -------
    destruct_expr: ibis.expr.types.DestructColumn
        A destruct column expression.
    """
    # Set name to empty string here so that we can detect and error when
    # user set name for a destruct column.
    if isinstance(expr, StructScalar):
        return DestructScalar(expr._arg, expr._dtype).name("")
    elif isinstance(expr, StructColumn):
        return DestructColumn(expr._arg, expr._dtype).name("")
    elif isinstance(expr, StructValue):
        return DestructValue(expr._arg, expr._dtype).name("")
    else:
        raise AssertionError()


_struct_value_methods = dict(
    destructure=_destructure,
    __getattr__=_struct_get_field,
    __getitem__=_struct_get_field,
)

_add_methods(ir.StructValue, _struct_value_methods)


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
    return ops.TimestampTruncate(arg, unit).to_expr()


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
    return ops.Strftime(arg, format_str).to_expr()


def _timestamp_time(arg):
    """Return a Time node for a Timestamp.

    We can perform certain operations on this node w/o actually instantiating
    the underlying structure (which is inefficient in pandas/numpy)

    Returns
    -------
    TimeValue
    """
    return ops.Time(arg).to_expr()


def _timestamp_date(arg):
    """Return a Date for a Timestamp.

    Returns
    -------
    DateValue
    """
    return ops.Date(arg).to_expr()


def _timestamp_sub(left, right):
    right = as_value_expr(right)

    if isinstance(right, ir.TimestampValue):
        op = ops.TimestampDiff(left, right)
    else:
        op = ops.TimestampSub(left, right)  # let the operation validate

    return op.to_expr()


_timestamp_add = _binop_expr('__add__', ops.TimestampAdd)
_timestamp_radd = _binop_expr('__radd__', ops.TimestampAdd)


_day_of_week = property(
    lambda self: ops.DayOfWeekNode(self).to_expr(),
    doc="""\
Namespace expression containing methods for extracting information about the
day of the week of a TimestampValue or DateValue expression.

Returns
-------
DayOfWeek
    An namespace expression containing methods to use to extract information.
""",
)


_timestamp_value_methods = dict(
    strftime=_timestamp_strftime,
    year=_extract_field('year', ops.ExtractYear),
    month=_extract_field('month', ops.ExtractMonth),
    day=_extract_field('day', ops.ExtractDay),
    day_of_week=_day_of_week,
    day_of_year=_extract_field('day_of_year', ops.ExtractDayOfYear),
    quarter=_extract_field('quarter', ops.ExtractQuarter),
    epoch_seconds=_extract_field('epoch', ops.ExtractEpochSeconds),
    week_of_year=_extract_field('week_of_year', ops.ExtractWeekOfYear),
    hour=_extract_field('hour', ops.ExtractHour),
    minute=_extract_field('minute', ops.ExtractMinute),
    second=_extract_field('second', ops.ExtractSecond),
    millisecond=_extract_field('millisecond', ops.ExtractMillisecond),
    truncate=_timestamp_truncate,
    time=_timestamp_time,
    date=_timestamp_date,
    __sub__=_timestamp_sub,
    sub=_timestamp_sub,
    __add__=_timestamp_add,
    add=_timestamp_add,
    __radd__=_timestamp_radd,
    radd=_timestamp_radd,
    __rsub__=_timestamp_sub,
    rsub=_timestamp_sub,
)

_add_methods(ir.TimestampValue, _timestamp_value_methods)


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
    return ops.DateTruncate(arg, unit).to_expr()


def _date_sub(left, right):
    right = rlz.one_of([rlz.date, rlz.interval], right)

    if isinstance(right, ir.DateValue):
        op = ops.DateDiff(left, right)
    else:
        op = ops.DateSub(left, right)  # let the operation validate

    return op.to_expr()


_date_add = _binop_expr('__add__', ops.DateAdd)

_date_value_methods = dict(
    strftime=_timestamp_strftime,
    year=_extract_field('year', ops.ExtractYear),
    month=_extract_field('month', ops.ExtractMonth),
    day=_extract_field('day', ops.ExtractDay),
    day_of_week=_day_of_week,
    day_of_year=_extract_field('day_of_year', ops.ExtractDayOfYear),
    quarter=_extract_field('quarter', ops.ExtractQuarter),
    epoch_seconds=_extract_field('epoch', ops.ExtractEpochSeconds),
    week_of_year=_extract_field('week_of_year', ops.ExtractWeekOfYear),
    truncate=_date_truncate,
    __sub__=_date_sub,
    sub=_date_sub,
    __rsub__=_date_sub,
    rsub=_date_sub,
    __add__=_date_add,
    add=_date_add,
    __radd__=_date_add,
    radd=_date_add,
)

_add_methods(ir.DateValue, _date_value_methods)


def _to_unit(arg, target_unit):
    if arg._dtype.unit != target_unit:
        arg = util.convert_unit(arg, arg._dtype.unit, target_unit)
        arg.type().unit = target_unit
    return arg


def _interval_property(target_unit, name):
    return property(
        functools.partial(_to_unit, target_unit=target_unit),
        doc="""Extract the number of {0}s from an IntervalValue expression.

Returns
-------
IntegerValue
    The number of {0}s in the expression
""".format(
            name
        ),
    )


_interval_add = _binop_expr('__add__', ops.IntervalAdd)
_interval_radd = _binop_expr('__radd__', ops.IntervalAdd)
_interval_sub = _binop_expr('__sub__', ops.IntervalSubtract)
_interval_mul = _binop_expr('__mul__', ops.IntervalMultiply)
_interval_rmul = _binop_expr('__rmul__', ops.IntervalMultiply)
_interval_floordiv = _binop_expr('__floordiv__', ops.IntervalFloorDivide)

_interval_value_methods = dict(
    to_unit=_to_unit,
    years=_interval_property('Y', 'year'),
    quarters=_interval_property('Q', 'quarter'),
    months=_interval_property('M', 'month'),
    weeks=_interval_property('W', 'week'),
    days=_interval_property('D', 'day'),
    hours=_interval_property('h', 'hour'),
    minutes=_interval_property('m', 'minute'),
    seconds=_interval_property('s', 'second'),
    milliseconds=_interval_property('ms', 'millisecond'),
    microseconds=_interval_property('us', 'microsecond'),
    nanoseconds=_interval_property('ns', 'nanosecond'),
    __add__=_interval_add,
    add=_interval_add,
    __sub__=_interval_sub,
    sub=_interval_sub,
    __radd__=_interval_radd,
    radd=_interval_radd,
    __mul__=_interval_mul,
    mul=_interval_mul,
    __rmul__=_interval_rmul,
    rmul=_interval_rmul,
    __floordiv__=_interval_floordiv,
    floordiv=_interval_floordiv,
    __neg__=negate,
    negate=negate,
)

_add_methods(ir.IntervalValue, _interval_value_methods)


# ---------------------------------------------------------------------
# Time API


def between_time(arg, lower, upper, timezone=None):
    """Check if the input expr falls between the lower/upper bounds passed.
    Bounds are inclusive. All arguments must be comparable.

    Parameters
    ----------
    lower : str, datetime.time
    upper : str, datetime.time
    timezone : str, timezone, default None

    Returns
    -------
    BooleanValue
    """
    op = arg.op()
    if isinstance(op, ops.Time):
        # Here we pull out the first argument to the underlying Time operation
        # which is by definition (in _timestamp_value_methods) a
        # TimestampValue. We do this so that we can potentially specialize the
        # "between time" operation for timestamp_value_expr.time().between().
        # A similar mechanism is triggered when creating expressions like
        # t.column.distinct().count(), which is turned into t.column.nunique().
        arg = op.arg
        if timezone is not None:
            arg = arg.cast(dt.Timestamp(timezone=timezone))
        op = ops.BetweenTime(arg, lower, upper)
    else:
        op = ops.Between(arg, lower, upper)

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
    return ops.TimeTruncate(arg, unit).to_expr()


def _time_sub(left, right):
    right = as_value_expr(right)

    if isinstance(right, ir.TimeValue):
        op = ops.TimeDiff(left, right)
    else:
        op = ops.TimeSub(left, right)  # let the operation validate

    return op.to_expr()


_time_add = _binop_expr('__add__', ops.TimeAdd)


_time_value_methods = dict(
    between=between_time,
    truncate=_time_truncate,
    hour=_extract_field('hour', ops.ExtractHour),
    minute=_extract_field('minute', ops.ExtractMinute),
    second=_extract_field('second', ops.ExtractSecond),
    millisecond=_extract_field('millisecond', ops.ExtractMillisecond),
    __sub__=_time_sub,
    sub=_time_sub,
    __rsub__=_time_sub,
    rsub=_time_sub,
    __add__=_time_add,
    add=_time_add,
    __radd__=_time_add,
    radd=_time_add,
)

_add_methods(ir.TimeValue, _time_value_methods)


# ---------------------------------------------------------------------
# Decimal API

_decimal_value_methods = dict(
    precision=_unary_op('precision', ops.DecimalPrecision),
    scale=_unary_op('scale', ops.DecimalScale),
)


_add_methods(ir.DecimalValue, _decimal_value_methods)


# ----------------------------------------------------------------------
# Category API


_category_value_methods = dict(label=_analytics.category_label)

_add_methods(ir.CategoryValue, _category_value_methods)


# ---------------------------------------------------------------------
# Table API

_join_classes = {
    'inner': ops.InnerJoin,
    'left': ops.LeftJoin,
    'any_inner': ops.AnyInnerJoin,
    'any_left': ops.AnyLeftJoin,
    'outer': ops.OuterJoin,
    'right': ops.RightJoin,
    'left_semi': ops.LeftSemiJoin,
    'semi': ops.LeftSemiJoin,
    'anti': ops.LeftAntiJoin,
    'cross': ops.CrossJoin,
}


def join(left, right, predicates=(), how='inner'):
    """Perform a relational join between two tables. Does not resolve resulting
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
        Note that the schema is not materialized yet
    """
    klass = _join_classes[how.lower()]
    if isinstance(predicates, Expr):
        predicates = _L.flatten_predicate(predicates)

    op = klass(left, right, predicates)
    return op.to_expr()


def asof_join(left, right, predicates=(), by=(), tolerance=None):
    """Perform an asof join between two tables.  Similar to a left join
    except that the match is done on nearest key rather than equal keys.

    Optionally, match keys with 'by' before joining with predicates.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr
    predicates : join expression(s)
    by : string
        column to group by before joining
    tolerance : interval
        Amount of time to look behind when joining

    Returns
    -------
    joined : TableExpr
        Note that the schema is not materialized yet
    """
    return ops.AsOfJoin(left, right, predicates, by, tolerance).to_expr()


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
    op = ops.CrossJoin(*tables, **kwargs)
    return op.to_expr()


def _table_count(self):
    """
    Returns the computed number of rows in the table expression

    Returns
    -------
    count : Int64Scalar
    """
    return ops.Count(self, None).to_expr().name('count')


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
    return op.to_expr()


def _table_distinct(self):
    """
    Compute set of unique rows/tuples occurring in this table
    """
    op = ops.Distinct(self)
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
    op = ops.Limit(table, n, offset=offset)
    return op.to_expr()


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
    result = table.op().sort_by(table, sort_exprs)
    return result.to_expr()


def _table_union(left, right, distinct=False):
    """
    Form the table set union of two table expressions having identical
    schemas.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr
    distinct : boolean, default False
        Only union distinct rows not occurring in the calling table (this
        can be very expensive, be careful)

    Returns
    -------
    union : TableExpr
    """
    return ops.Union(left, right, distinct=distinct).to_expr()


def _table_intersect(left: TableExpr, right: TableExpr):
    """
    Form the table set intersect of two table expressions having identical
    schemas. An intersect returns only the common rows between the two tables.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr

    Returns
    -------
    intersection : TableExpr
    """
    return ops.Intersection(left, right).to_expr()


def _table_difference(left: TableExpr, right: TableExpr):
    """
    Form the table set difference of two table expressions having identical
    schemas. A set difference returns only the rows present in the left table
    that are not present in the right table

    Parameters
    ----------
    left : TableExpr
    right : TableExpr

    Returns
    -------
    difference : TableExpr
    """
    return ops.Difference(left, right).to_expr()


def _table_to_array(self):
    """
    Single column tables can be viewed as arrays.
    """
    op = ops.TableArrayView(self)
    return op.to_expr()


def _table_materialize(table):
    """
    Force schema resolution for a joined table, selecting all fields from
    all tables.
    """
    if table._is_materialized():
        return table

    op = ops.MaterializedJoin(table)
    return op.to_expr()


def _safe_get_name(expr):
    try:
        return expr.get_name()
    except com.ExpressionError:
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
        foo : float64
        bar : float64
    <BLANKLINE>
    Selection[table]
      table:
        Table: ref_0
      selections:
        Table: ref_0
        baz = Literal[int8]
          5
        qux = Add[float64*]
          left:
            foo = Column[float64*] 'foo' from table
              ref_0
          right:
            bar = Column[float64*] 'bar' from table
              ref_0

    Using the :meth:`ibis.expr.types.Expr.name` method to name the new columns

    >>> new_columns = [ibis.literal(5).name('baz',),
    ...                (table.foo + table.bar).name('qux')]
    >>> expr2 = table.mutate(new_columns)
    >>> expr.equals(expr2)
    True

    """
    exprs = [] if exprs is None else util.promote_list(exprs)
    exprs.extend(
        (expr(table) if util.is_function(expr) else as_value_expr(expr)).name(
            name
        )
        for name, expr in sorted(mutations.items(), key=operator.itemgetter(0))
    )

    for expr in exprs:
        if expr.get_name() and isinstance(expr, ir.DestructColumn):
            raise com.ExpressionError(
                f"Cannot name a destruct column: {expr.get_name()}"
            )

    by_name = collections.OrderedDict(
        (expr.get_name(), expr) for expr in exprs
    )
    columns = table.columns
    used = by_name.keys() & columns

    if used:
        proj_exprs = [
            by_name.get(column, table[column]) for column in columns
        ] + [expr for name, expr in by_name.items() if name not in used]
    else:
        proj_exprs = [table] + exprs
    return table.projection(proj_exprs)


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
        b : float64
    <BLANKLINE>
    Selection[table]
      table:
        Table: ref_0
      selections:
        a = Column[int64*] 'a' from table
          ref_0
        b_plus_1 = Add[float64*]
          left:
            b = Column[float64*] 'b' from table
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
        b : float64
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
        mean_b = WindowOp[float64*]
          mean_b = Mean[float64]
            b = Column[float64*] 'b' from table
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

    if isinstance(exprs, (Expr, str)):
        exprs = [exprs]

    projector = L.Projector(table, exprs)
    op = projector.get_result()
    return op.to_expr()


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
    new_view = ops.SelfReference(self)
    return new_view.to_expr()


def _table_drop(self, fields):
    if not fields:
        # no-op if nothing to be dropped
        return self

    schema = self.schema()
    field_set = frozenset(fields)
    missing_fields = field_set.difference(schema)

    if missing_fields:
        raise KeyError('Fields not in table: {0!s}'.format(missing_fields))

    return self[[field for field in schema if field not in field_set]]


def _rowid(self):
    """
    An autonumeric representing the row number of the results.

    It can be 0 or 1 indexed depending on the backend. Check the backend
    documentation.

    Note that this is different from the window function row number
    (even if they are conceptually the same), and different from row
    id in backends where it represents the physical location (e.g. Oracle
    or PostgreSQL's ctid).

    Returns
    -------
    ir.IntegerColumn

    Examples
    --------
    >>> my_table[my_table.rowid(), my_table.name].execute()
    1|Ibis
    2|pandas
    3|Dask
    """
    return ops.RowID().to_expr()


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
    intersect=_table_intersect,
    difference=_table_difference,
    view=_table_view,
    rowid=_rowid,
)


_add_methods(ir.TableExpr, _table_methods)


def prevent_rewrite(expr, client=None):
    """Prevent optimization from happening below `expr`.

    Parameters
    ----------
    expr : ir.TableExpr
        Any table expression whose optimization you want to prevent
    client : ibis.client.Client, optional, default None
        A client to use to create the SQLQueryResult operation. This is useful
        if you're compiling an expression that derives from an
        :class:`~ibis.expr.operations.UnboundTable` operation.

    Returns
    -------
    sql_query_result : ir.TableExpr
    """
    if client is None:
        (client,) = ibis.client.find_backends(expr)
    query = client.compile(expr)
    return ops.SQLQueryResult(query, expr.schema(), client).to_expr()
