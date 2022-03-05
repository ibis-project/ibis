"""Ibis expression API definitions."""

from __future__ import annotations

import collections
import datetime
import numbers
from typing import Iterable, Literal, Mapping, Sequence, TypeVar

import dateutil.parser
import pandas as pd

import ibis.common.exceptions as com
import ibis.expr.builders as bl
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
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
    array,
    literal,
    map,
    null,
    struct,
)
from ibis.expr.types.groupby import GroupedTableExpr  # noqa
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
    'array',
    'case',
    'coalesce',
    'cross_join',
    'cumulative_window',
    'date',
    'desc',
    'Expr',
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
    'map',
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
    'struct',
    'table',
    'time',
    'timestamp',
    'trailing_range_window',
    'trailing_window',
    'where',
    'window',
)


infer_dtype = dt.infer
infer_schema = sch.infer


NA = null()

T = TypeVar("T")


def param(type: dt.DataType) -> ir.ScalarExpr:
    """Create a deferred parameter of a given type.

    Parameters
    ----------
    type
        The type of the unbound parameter, e.g., double, int64, date, etc.

    Returns
    -------
    ScalarExpr
        A scalar expression backend by a parameter

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> start = ibis.param(dt.date)
    >>> end = ibis.param(dt.date)
    >>> schema = [('timestamp_col', 'timestamp'), ('value', 'double')]
    >>> t = ibis.table(schema)
    >>> predicates = [t.timestamp_col >= start, t.timestamp_col <= end]
    >>> expr = t.filter(predicates).value.sum()
    """
    return ops.ScalarParameter(dt.dtype(type)).to_expr()


def sequence(values: Sequence[T | None]) -> ir.ListExpr:
    """Wrap a list of Python values as an Ibis sequence type.

    Parameters
    ----------
    values
        Should all be None or the same type

    Returns
    -------
    ListExpr
        A list expression
    """
    return ops.ValueList(values).to_expr()


def schema(
    pairs: Iterable[tuple[str, dt.DataType]]
    | Mapping[str, dt.DataType]
    | None = None,
    names: Iterable[str] | None = None,
    types: Iterable[str | dt.DataType] | None = None,
) -> sch.Schema:
    """Validate and return an Schema object.

    Parameters
    ----------
    pairs
        List or dictionary of name, type pairs. Mutually exclusive with `names`
        and `types`.
    names
        Field names. Mutually exclusive with `pairs`.
    types
        Field types. Mutually exclusive with `pairs`.

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
    Schema
        An ibis schema
    """  # noqa: E501
    if pairs is not None:
        return Schema.from_dict(dict(pairs))
    else:
        return Schema(names, types)


_schema = schema


def table(schema: sch.Schema, name: str | None = None) -> ir.TableExpr:
    """Create an unbound table for build expressions without data.


    Parameters
    ----------
    schema
        A schema for the table
    name
        Name for the table

    Returns
    -------
    TableExpr
        An unbound table expression
    """
    if not isinstance(schema, Schema):
        schema = _schema(pairs=schema)

    node = ops.UnboundTable(schema, name=name)
    return node.to_expr()


def desc(expr: ir.ColumnExpr | str) -> ir.SortExpr | ops.DeferredSortKey:
    """Create a descending sort key from `expr` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('g', 'string')])
    >>> result = t.group_by('g').size('count').sort_by(ibis.desc('count'))

    Returns
    -------
    ops.DeferredSortKey
        A deferred sort key
    """
    if not isinstance(expr, Expr):
        return ops.DeferredSortKey(expr, ascending=False)
    else:
        return ops.SortKey(expr, ascending=False).to_expr()


def timestamp(
    value: str | numbers.Integral,
    timezone: str | None = None,
) -> ir.TimestampScalar:
    """Construct a timestamp literal if `value` is coercible to a timestamp.

    Parameters
    ----------
    value
        The value to use for constructing the timestamp
    timezone
        The timezone of the timestamp

    Returns
    -------
    TimestampScalar
        A timestamp expression
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
                "ibis.literal({value}).to_timestamp() to create a timestamp "
                "expression from an integer."
            ).format(value=value)
        )
    return literal(value, type=dt.Timestamp(timezone=timezone))


def date(value: str) -> ir.DateScalar:
    """Return a date literal if `value` is coercible to a date.

    Parameters
    ----------
    value
        Date string

    Returns
    -------
    DateScalar
        A date expression
    """
    if isinstance(value, str):
        value = pd.to_datetime(value).date()
    return literal(value, type=dt.date)


def time(value: str) -> ir.TimeScalar:
    """Return a time literal if `value` is coercible to a time.

    Parameters
    ----------
    value
        Time string

    Returns
    -------
    TimeScalar
        A time expression
    """
    if isinstance(value, str):
        value = pd.to_datetime(value).time()
    return literal(value, type=dt.time)


def interval(
    value: int | datetime.timedelta | None = None,
    unit: str = 's',
    years: int | None = None,
    quarters: int | None = None,
    months: int | None = None,
    weeks: int | None = None,
    days: int | None = None,
    hours: int | None = None,
    minutes: int | None = None,
    seconds: int | None = None,
    milliseconds: int | None = None,
    microseconds: int | None = None,
    nanoseconds: int | None = None,
) -> ir.IntervalScalar:
    """Return an interval literal expression.

    Parameters
    ----------
    value
        Interval value. If passed, must be combined with `unit`.
    unit
        Unit of `value`
    years
        Number of years
    quarters
        Number of quarters
    months
        Number of months
    weeks
        Number of weeks
    days
        Number of days
    hours
        Number of hours
    minutes
        Number of minutes
    seconds
        Number of seconds
    milliseconds
        Number of milliseconds
    microseconds
        Number of microseconds
    nanoseconds
        Number of nanoseconds

    Returns
    -------
    IntervalScalar
        An interval expression
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


def case() -> bl.SearchedCaseBuilder:
    """Begin constructing a case expression.

    Notes
    -----
    Use the `.when` method on the resulting object followed by .end to create a
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
    SearchedCaseBuilder
        A builder object to use for constructing a case expression.
    """
    return bl.SearchedCaseBuilder()


def now() -> ir.TimestampScalar:
    """Return an expression that will compute the current timestamp.

    Returns
    -------
    TimestampScalar
        A "now" expression
    """
    return ops.TimestampNow().to_expr()


def row_number() -> ir.IntegerColumn:
    """Return an analytic function expression for the current row number.

    Returns
    -------
    IntegerColumn
        A column expression enumerating rows
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


def negate(arg: ir.NumericValue) -> ir.NumericValue:
    """Negate a numeric expression.

    Parameters
    ----------
    arg
        A numeric value to negate

    Returns
    -------
    N
        A numeric value expression
    """
    op = arg.op()
    if hasattr(op, 'negate'):
        result = op.negate()
    else:
        result = ops.Negate(arg)

    return result.to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        try:
            other = rlz.any(other)
            op = klass(self, other)
            return op.to_expr()
        except (com.IbisTypeError, NotImplementedError):
            return NotImplemented

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary ops, like radd, etc.
    def f(self, other):
        other = rlz.any(other)
        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _boolean_binary_op(name, klass):
    def f(self, other):
        other = rlz.any(other)

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
        other = rlz.any(other)

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
    f.__doc__ = klass.__doc__
    return f


def _extract_field(name, klass):
    def f(self):
        expr = klass(self).to_expr()
        return expr.name(name)

    f.__name__ = name
    return f


# ---------------------------------------------------------------------
# Generic value API


def greatest(*args: ir.ValueExpr) -> ir.ValueExpr:
    """Compute the largest value among the supplied arguments.

    Parameters
    ----------
    args
        Arguments to choose from

    Returns
    -------
    ValueExpr
        Maximum of the passed arguments
    """
    op = ops.Greatest(args)
    return op.to_expr()


def least(*args: ir.ValueExpr) -> ir.ValueExpr:
    """Compute the smallest value among the supplied arguments.

    Parameters
    ----------
    args
        Arguments to choose from

    Returns
    -------
    ValueExpr
        Minimum of the passed arguments
    """
    op = ops.Least(args)
    return op.to_expr()


def where(
    boolean_expr: ir.BooleanValue,
    true_expr: ir.ValueExpr,
    false_null_expr: ir.ValueExpr,
) -> ir.ValueExpr:
    """Return `true_expr` if `boolean_expr` is `True` else `false_null_expr`.

    Parameters
    ----------
    boolean_expr
        A boolean expression
    true_expr
        Value returned if `boolean_expr` is `True`
    false_null_expr
        Value returned if `boolean_expr` is `False` or `NULL`

    Returns
    -------
    ir.ValueExpr
        An expression
    """
    op = ops.Where(boolean_expr, true_expr, false_null_expr)
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


coalesce = ir.ValueExpr.coalesce


def _numeric_summary(
    arg: ir.NumericColumn,
    exact_nunique: bool = False,
    prefix: str = "",
    suffix: str = "",
) -> list[ir.NumericScalar]:
    """Compute a set of summary metrics from the input numeric value expression.

    Parameters
    ----------
    arg
        Numeric expression
    exact_nunique
        Compute the exact number of distinct values. Typically slower if
        `True`.
    prefix
        String prefix for metric names
    suffix
        String suffix for metric names

    Returns
    -------
    list[ir.NumericScalar]
        Metrics list
    """
    if exact_nunique:
        unique_metric = arg.nunique().name('nunique')
    else:
        unique_metric = arg.approx_nunique().name('approx_nunique')

    metrics = [
        arg.count(),
        arg.isnull().sum().name('nulls'),
        arg.min(),
        arg.max(),
        arg.sum(),
        arg.mean(),
        unique_metric,
    ]
    metrics = [m.name(f"{prefix}{m.get_name()}{suffix}") for m in metrics]

    return metrics


# ---------------------------------------------------------------------
# Numeric API


def round(arg: ir.NumericValue, digits: int | None = None) -> ir.NumericValue:
    """Round values to an indicated number of decimal places.

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


def log(
    arg: ir.NumericValue, base: ir.NumericValue | None = None
) -> ir.NumericValue:
    """Return the logarithm using a specified base.

    Parameters
    ----------
    arg
        A numeric expression
    base
        The base of the logarithm. If `None`, base `e` is used.

    Returns
    -------
    NumericValue
        Logarithm of `arg` with base `base`
    """
    op = ops.Log(arg, base)
    return op.to_expr()


def clip(
    arg: ir.NumericValue,
    lower: ir.NumericValue | None = None,
    upper: ir.NumericValue | None = None,
) -> ir.NumericValue:
    """
    Trim values at input threshold(s).

    Parameters
    ----------
    arg
        Numeric expression
    lower
        Lower bound
    upper
        Upper bound

    Returns
    -------
    NumericValue
        Clipped input
    """
    if lower is None and upper is None:
        raise ValueError("at least one of lower and upper must be provided")

    op = ops.Clip(arg, lower, upper)
    return op.to_expr()


def quantile(
    arg: ir.NumericValue,
    quantile: ir.NumericValue,
    interpolation: Literal[
        'linear',
        'lower',
        'higher',
        'midpoint',
        'nearest',
    ] = 'linear',
) -> ir.NumericValue:
    """Return value at the given quantile.

    Parameters
    ----------
    arg
        Numeric expression
    quantile
        `0 <= quantile <= 1`, the quantile(s) to compute
    interpolation
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
    NumericValue
        Quantile of the input
    """
    if isinstance(quantile, collections.abc.Sequence):
        op = ops.MultiQuantile(
            arg, quantile=quantile, interpolation=interpolation
        )
    else:
        op = ops.Quantile(arg, quantile=quantile, interpolation=interpolation)
    return op.to_expr()


def _integer_to_timestamp(
    arg: ir.IntegerValue, unit: Literal['s', 'ms', 'us'] = 's'
) -> ir.TimestampValue:
    """Convert integral UNIX timestamp to a timestamp.

    Parameters
    ----------
    arg
        Integral UNIX timestamp
    unit
        The resolution of `arg`

    Returns
    -------
    TimestampValue
        `arg` converted to a timestamp
    """
    op = ops.TimestampFromUNIX(arg, unit)
    return op.to_expr()


def _integer_to_interval(
    arg: ir.IntegerValue,
    unit: Literal['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'] = 's',
) -> ir.IntervalValue:
    """
    Convert integer interval with the same inner type

    Parameters
    ----------
    arg
        Integer value
    unit
        Unit for the resulting interval

    Returns
    -------
    IntervalValue
        An interval in units of `unit`
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


_numeric_value_methods = {
    '__neg__': negate,
    'abs': abs,
    'ceil': ceil,
    'degrees': degrees,
    'deg2rad': radians,
    'floor': floor,
    'radians': radians,
    'rad2deg': degrees,
    'sign': sign,
    'exp': exp,
    'sqrt': sqrt,
    'log': log,
    'ln': ln,
    'log2': log2,
    'log10': log10,
    'round': round,
    'nullifzero': _unary_op('nullifzero', ops.NullIfZero),
    'zeroifnull': _unary_op('zeroifnull', ops.ZeroIfNull),
    'clip': clip,
    '__add__': add,
    'add': add,
    '__sub__': sub,
    'sub': sub,
    '__mul__': mul,
    'mul': mul,
    '__div__': div,
    '__truediv__': div,
    '__floordiv__': floordiv,
    'div': div,
    'floordiv': floordiv,
    '__rdiv__': rdiv,
    '__rtruediv__': rdiv,
    '__rfloordiv__': rfloordiv,
    'rdiv': rdiv,
    'rfloordiv': rfloordiv,
    '__pow__': pow,
    'pow': pow,
    '__radd__': add,
    'radd': add,
    '__rsub__': rsub,
    'rsub': rsub,
    '__rmul__': _rbinop_expr('__rmul__', ops.Multiply),
    '__rpow__': _rbinop_expr('__rpow__', ops.Power),
    '__mod__': mod,
    '__rmod__': _rbinop_expr('__rmod__', ops.Modulus),
    # trigonometric operations
    'acos': acos,
    'asin': asin,
    'atan': atan,
    'atan2': atan2,
    'cos': cos,
    'cot': cot,
    'sin': sin,
    'tan': tan,
}


_integer_value_methods = {
    'to_timestamp': _integer_to_timestamp,
    'to_interval': _integer_to_interval,
}


bit_and = _agg_function('bit_and', ops.BitAnd, True)
bit_or = _agg_function('bit_or', ops.BitOr, True)
bit_xor = _agg_function('bit_xor', ops.BitXor, True)

mean = _agg_function('mean', ops.Mean, True)
cummean = _unary_op('cummean', ops.CumulativeMean)

sum = _agg_function('sum', ops.Sum, True)
cumsum = _unary_op('cumsum', ops.CumulativeSum)


def std(
    arg: ir.NumericColumn,
    where: ir.BooleanValue | None = None,
    how: Literal['sample', 'pop'] = 'sample',
) -> ir.NumericScalar:
    """Return the standard deviation of a numeric column.

    Parameters
    ----------
    arg
        Numeric column
    how
        Sample or population standard deviation

    Returns
    -------
    NumericScalar
        Standard deviation of `arg`
    """
    expr = ops.StandardDev(arg, how=how, where=where).to_expr()
    expr = expr.name('std')
    return expr


def variance(
    arg: ir.NumericColumn,
    where: ir.BooleanValue | None = None,
    how: Literal['sample', 'pop'] = 'sample',
) -> ir.NumericScalar:
    """Return the variance of a numeric column.

    Parameters
    ----------
    arg
        Numeric column
    how
        Sample or population variance

    Returns
    -------
    NumericScalar
        Standard deviation of `arg`
    """
    expr = ops.Variance(arg, how=how, where=where).to_expr()
    expr = expr.name('var')
    return expr


def correlation(
    left: ir.NumericColumn,
    right: ir.NumericColumn,
    where: ir.BooleanValue | None = None,
    how: Literal['sample', 'pop'] = 'sample',
) -> ir.NumericScalar:
    """Return the correlation of two numeric columns.

    Parameters
    ----------
    left
        Numeric column
    right
        Numeric column
    how
        Population or sample correlation

    Returns
    -------
    NumericScalar
        The correlation of `left` and `right`
    """
    expr = ops.Correlation(left, right, how=how, where=where).to_expr()
    return expr


def covariance(
    left: ir.NumericColumn,
    right: ir.NumericColumn,
    where: ir.BooleanValue | None = None,
    how: Literal['sample', 'pop'] = 'sample',
):
    """Return the covariance of two numeric columns.

    Parameters
    ----------
    left
        Numeric column
    right
        Numeric column
    how
        Population or sample covariance

    Returns
    -------
    NumericScalar
        The covariance of `left` and `right`
    """
    expr = ops.Covariance(left, right, how=how, where=where).to_expr()
    return expr


def bucket(
    arg: ir.NumericValue,
    buckets: Sequence[int],
    closed: Literal['left', 'right'] = 'left',
    close_extreme: bool = True,
    include_under: bool = False,
    include_over: bool = False,
) -> ir.CategoryColumn:
    """
    Compute a discrete binning of a numeric array

    Parameters
    ----------
    arg
        Numeric array expression
    buckets
        List of buckets
    closed
        Which side of each interval is closed. For example:

        ```python
        buckets = [0, 100, 200]
        closed = 'left': 100 falls in 2nd bucket
        closed = 'right': 100 falls in 1st bucket
        ```
    close_extreme
        Whether the extreme values fall in the last bucket

    Returns
    -------
    CategoryColumn
        A categorical column expression
    """
    op = ops.Bucket(
        arg,
        buckets,
        closed=closed,
        close_extreme=close_extreme,
        include_under=include_under,
        include_over=include_over,
    )
    return op.to_expr()


def histogram(
    arg: ir.NumericColumn,
    nbins: int | None = None,
    binwidth: float | None = None,
    base: float | None = None,
    closed: Literal['left', 'right'] = 'left',
    aux_hash: str | None = None,
) -> ir.CategoryColumn:
    """Compute a histogram with fixed width bins.

    Parameters
    ----------
    arg
        Numeric column
    nbins
        If supplied, will be used to compute the binwidth
    binwidth
        If not supplied, computed from the data (actual max and min values)
    base
        Histogram base
    closed
        Which side of each interval is closed
    aux_hash
        Auxiliary hash value to add to bucket names

    Returns
    -------
    CategoryColumn
        Coded value expression
    """
    op = ops.Histogram(
        arg, nbins, binwidth, base, closed=closed, aux_hash=aux_hash
    )
    return op.to_expr()


def category_label(
    arg: ir.CategoryValue,
    labels: Sequence[str],
    nulls: str | None = None,
) -> ir.StringValue:
    """Format a known number of categories as strings.

    Parameters
    ----------
    arg
        A category value
    labels
        Labels to use for formatting categories
    nulls
        How to label any null values among the categories

    Returns
    -------
    StringValue
        Labeled categories
    """
    op = ops.CategoryLabel(arg, labels, nulls)
    return op.to_expr()


_numeric_column_methods = {
    'mean': mean,
    'cummean': cummean,
    'sum': sum,
    'cumsum': cumsum,
    'quantile': quantile,
    'std': std,
    'var': variance,
    'corr': correlation,
    'cov': covariance,
    'bucket': bucket,
    'histogram': histogram,
    'summary': _numeric_summary,
}

_integer_column_methods = {
    'bit_and': bit_and,
    'bit_or': bit_or,
    'bit_xor': bit_xor,
}

_floating_value_methods = {
    'isnan': _unary_op('isnull', ops.IsNan),
    'isinf': _unary_op('isinf', ops.IsInf),
}

_add_methods(ir.NumericValue, _numeric_value_methods)
_add_methods(ir.IntegerValue, _integer_value_methods)
_add_methods(ir.FloatingValue, _floating_value_methods)

_add_methods(ir.NumericColumn, _numeric_column_methods)
_add_methods(ir.IntegerColumn, _integer_column_methods)

# ----------------------------------------------------------------------
# GeoSpatial API


def geo_area(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Compute the area of a geospatial value.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    FloatingValue
        The area of `arg`
    """
    op = ops.GeoArea(arg)
    return op.to_expr()


def geo_as_binary(arg: ir.GeoSpatialValue) -> ir.BinaryValue:
    """Get the geometry as well-known bytes (WKB) without the SRID data.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    BinaryValue
        Binary value
    """
    op = ops.GeoAsBinary(arg)
    return op.to_expr()


def geo_as_ewkt(arg: ir.GeoSpatialValue) -> ir.StringValue:
    """Get the geometry as well-known text (WKT) with the SRID data.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    StringValue
        String value
    """
    op = ops.GeoAsEWKT(arg)
    return op.to_expr()


def geo_as_text(arg: ir.GeoSpatialValue) -> ir.StringValue:
    """Get the geometry as well-known text (WKT) without the SRID data.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    StringValue
        String value
    """
    op = ops.GeoAsText(arg)
    return op.to_expr()


def geo_as_ewkb(arg: ir.GeoSpatialValue) -> ir.BinaryValue:
    """Get the geometry as well-known bytes (WKB) with the SRID data.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    BinaryValue
        WKB value
    """
    op = ops.GeoAsEWKB(arg)
    return op.to_expr()


def geo_contains(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the `left` geometry contains the `right` one.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether left contains right
    """
    op = ops.GeoContains(left, right)
    return op.to_expr()


def geo_contains_properly(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """
    Check if the first geometry contains the second one,
    with no common border points.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether left contains right, properly.
    """
    op = ops.GeoContainsProperly(left, right)
    return op.to_expr()


def geo_covers(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the first geometry covers the second one.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` covers `right`
    """
    op = ops.GeoCovers(left, right)
    return op.to_expr()


def geo_covered_by(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the first geometry is covered by the second one.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` is covered by `right`
    """
    op = ops.GeoCoveredBy(left, right)
    return op.to_expr()


def geo_crosses(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries have at least one interior point in common.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` and `right` have at least one common interior point.
    """
    op = ops.GeoCrosses(left, right)
    return op.to_expr()


def geo_d_fully_within(
    left: ir.GeoSpatialValue,
    right: ir.GeoSpatialValue,
    distance: ir.FloatingValue,
) -> ir.BooleanValue:
    """Check if the `left` is entirely within `distance` from `right`.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry
    distance
        Distance to check

    Returns
    -------
    BooleanValue
        Whether `left` is within a specified distance from `right`.
    """
    op = ops.GeoDFullyWithin(left, right, distance)
    return op.to_expr()


def geo_disjoint(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries have no points in common.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` and `right` are disjoin
    """
    op = ops.GeoDisjoint(left, right)
    return op.to_expr()


def geo_d_within(
    left: ir.GeoSpatialValue,
    right: ir.GeoSpatialValue,
    distance: ir.FloatingValue,
) -> ir.BooleanValue:
    """Check if `left` is partially within `distance` from `right`.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry
    distance
        Distance to check

    Returns
    -------
    BooleanValue
        Whether `left` is partially within `distance` from `right`.
    """
    op = ops.GeoDWithin(left, right, distance)
    return op.to_expr()


def geo_equals(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries are equal.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` equals `right`
    """
    op = ops.GeoEquals(left, right)
    return op.to_expr()


def geo_geometry_n(
    arg: ir.GeoSpatialValue, n: int | ir.IntegerValue
) -> ir.GeoSpatialValue:
    """Get the 1-based Nth geometry of a multi geometry.

    Parameters
    ----------
    arg
        Geometry expression
    n
        Nth geometry index

    Returns
    -------
    GeoSpatialValue
        Geometry value
    """
    op = ops.GeoGeometryN(arg, n)
    return op.to_expr()


def geo_geometry_type(arg: ir.GeoSpatialValue) -> ir.StringValue:
    """Get the type of a geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    StringValue
        String representing the type of `arg`.
    """
    op = ops.GeoGeometryType(arg)
    return op.to_expr()


def geo_intersects(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries share any points.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` intersects `right`
    """
    op = ops.GeoIntersects(left, right)
    return op.to_expr()


def geo_is_valid(arg: ir.GeoSpatialValue) -> ir.BooleanValue:
    """Check if the geometry is valid.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    BooleanValue
        Whether `arg` is valid
    """
    op = ops.GeoIsValid(arg)
    return op.to_expr()


def geo_line_locate_point(
    left: ir.LineStringValue, right: ir.PointValue
) -> ir.FloatingValue:
    """Locate the distance a point falls along the length of a line.

    Returns a float between zero and one representing the location of the
    closest point on the linestring to the given point, as a fraction of the
    total 2d line length.

    Parameters
    ----------
    left
        Linestring geometry
    right
        Point geometry

    Returns
    -------
    FloatingValue
        Fraction of the total line length
    """
    op = ops.GeoLineLocatePoint(left, right)
    return op.to_expr()


def geo_line_merge(arg: ir.GeoSpatialValue) -> ir.GeoSpatialValue:
    """Merge a `MultiLineString` into a `LineString`.

    Returns a (set of) LineString(s) formed by sewing together the
    constituent line work of a MultiLineString. If a geometry other than
    a LineString or MultiLineString is given, this will return an empty
    geometry collection.

    Parameters
    ----------
    arg
        Multiline string

    Returns
    -------
    ir.GeoSpatialValue
        Merged linestrings
    """
    op = ops.GeoLineMerge(arg)
    return op.to_expr()


def geo_line_substring(
    arg: ir.LineStringValue, start: ir.FloatingValue, end: ir.FloatingValue
) -> ir.LineStringValue:
    """Clip a substring from a LineString.

    Returns a linestring that is a substring of the input one, starting
    and ending at the given fractions of the total 2d length. The second
    and third arguments are floating point values between zero and one.
    This only works with linestrings.

    Parameters
    ----------
    arg
        Linestring value
    start
        Start value
    end
        End value

    Returns
    -------
    LineStringValue
        Clipped linestring
    """
    op = ops.GeoLineSubstring(arg, start, end)
    return op.to_expr()


def geo_ordering_equals(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if two geometries are equal and have the same point ordering.

    Returns true if the two geometries are equal and the coordinates
    are in the same order.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether points and orderings are equal.
    """
    op = ops.GeoOrderingEquals(left, right)
    return op.to_expr()


def geo_overlaps(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries share space, have the same dimension, and are
    not completely contained by each other.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Overlaps indicator
    """
    op = ops.GeoOverlaps(left, right)
    return op.to_expr()


def geo_point(
    left: NumericValue | int | float,
    right: NumericValue | int | float,
) -> ir.PointValue:
    """Return a point constructed from the coordinate values.

    Constant coordinates result in construction of a POINT literal.

    Parameters
    ----------
    left
        X coordinate
    right
        Y coordinate

    Returns
    -------
    PointValue
        Points
    """
    op = ops.GeoPoint(left, right)
    return op.to_expr()


def geo_touches(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the geometries have at least one point in common, but do not
    intersect.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether left and right are touching
    """
    op = ops.GeoTouches(left, right)
    return op.to_expr()


def geo_distance(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.FloatingValue:
    """Compute the distance between two geospatial expressions.

    Parameters
    ----------
    left
        Left geometry or geography
    right
        Right geometry or geography

    Returns
    -------
    FloatingValue
        Distance between `left` and `right`
    """
    op = ops.GeoDistance(left, right)
    return op.to_expr()


def geo_length(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Compute the length of a geospatial expression.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    FloatingValue
        Length of `arg`
    """
    op = ops.GeoLength(arg)
    return op.to_expr()


def geo_perimeter(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Compute the perimeter of a geospatial expression.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    FloatingValue
        Perimeter of `arg`
    """
    op = ops.GeoPerimeter(arg)
    return op.to_expr()


def geo_max_distance(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.FloatingValue:
    """Returns the 2-dimensional maximum distance between two geometries in
    projected units.

    If `left` and `right` are the same geometry the function will return the
    distance between the two vertices most far from each other in that
    geometry.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    FloatingValue
        Maximum distance
    """
    op = ops.GeoMaxDistance(left, right)
    return op.to_expr()


def geo_unary_union(arg: ir.GeoSpatialValue) -> ir.GeoSpatialScalar:
    """Aggregate a set of geometries into a union.

    This corresponds to the aggregate version of the PostGIS ST_Union.
    We give it a different name (following the corresponding method
    in GeoPandas) to avoid name conflicts with the non-aggregate version.

    Parameters
    ----------
    arg
        Geometry expression column

    Returns
    -------
    GeoSpatialScalar
        Union of geometries
    """
    expr = ops.GeoUnaryUnion(arg).to_expr()
    expr = expr.name('union')
    return expr


def geo_union(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.GeoSpatialValue:
    """Merge two geometries into a union geometry.

    Returns the pointwise union of the two geometries.
    This corresponds to the non-aggregate version the PostGIS ST_Union.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    GeoSpatialValue
        Union of geometries
    """
    op = ops.GeoUnion(left, right)
    return op.to_expr()


def geo_x(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the X coordinate of `arg`, or NULL if not available.

    Input must be a point.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        X coordinate of `arg`
    """
    op = ops.GeoX(arg)
    return op.to_expr()


def geo_y(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the Y coordinate of `arg`, or NULL if not available.

    Input must be a point.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        Y coordinate of `arg`
    """
    op = ops.GeoY(arg)
    return op.to_expr()


def geo_x_min(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the X minima of a geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        X minima
    """
    op = ops.GeoXMin(arg)
    return op.to_expr()


def geo_x_max(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the X maxima of a geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        X maxima
    """
    op = ops.GeoXMax(arg)
    return op.to_expr()


def geo_y_min(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the Y minima of a geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        Y minima
    """
    op = ops.GeoYMin(arg)
    return op.to_expr()


def geo_y_max(arg: ir.GeoSpatialValue) -> ir.FloatingValue:
    """Return the Y maxima of a geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    FloatingValue
        Y maxima
    YMax : double scalar
    """
    op = ops.GeoYMax(arg)
    return op.to_expr()


def geo_start_point(arg: ir.GeoSpatialValue) -> ir.PointValue:
    """Return the first point of a `LINESTRING` geometry as a `POINT`.

    Return NULL if the input parameter is not a `LINESTRING`

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    PointValue
        Start point
    """
    op = ops.GeoStartPoint(arg)
    return op.to_expr()


def geo_end_point(arg: ir.GeoSpatialValue) -> ir.PointValue:
    """Return the last point of a `LINESTRING` geometry as a `POINT`.

    Return NULL if the input parameter is not a LINESTRING

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    PointValue
        End point
    """
    op = ops.GeoEndPoint(arg)
    return op.to_expr()


def geo_point_n(arg: ir.GeoSpatialValue, n: ir.IntegerValue) -> ir.PointValue:
    """Return the Nth point in a single linestring in the geometry.
    Negative values are counted backwards from the end of the LineString,
    so that -1 is the last point. Returns NULL if there is no linestring in
    the geometry

    Parameters
    ----------
    arg
        Geometry expression
    n
        Nth point index

    Returns
    -------
    PointValue
        Nth point in `arg`
    """
    op = ops.GeoPointN(arg, n)
    return op.to_expr()


def geo_n_points(arg: ir.GeoSpatialValue) -> ir.IntegerValue:
    """Return the number of points in a geometry. Works for all geometries

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    IntegerValue
        Number of points
    """
    op = ops.GeoNPoints(arg)
    return op.to_expr()


def geo_n_rings(arg: ir.GeoSpatialValue) -> ir.IntegerValue:
    """Return the number of rings for polygons and multipolygons.

    Outer rings are counted as well.

    Parameters
    ----------
    arg
        Geometry or geography

    Returns
    -------
    IntegerValue
        Number of rings
    """
    op = ops.GeoNRings(arg)
    return op.to_expr()


def geo_srid(arg: ir.GeoSpatialValue) -> ir.IntegerValue:
    """Return the spatial reference identifier for the ST_Geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    IntegerValue
        SRID
    """
    op = ops.GeoSRID(arg)
    return op.to_expr()


def geo_set_srid(
    arg: ir.GeoSpatialValue, srid: ir.IntegerValue
) -> ir.GeoSpatialValue:
    """Set the spatial reference identifier for the ST_Geometry

    Parameters
    ----------
    arg
        Geometry expression
    srid
        SRID integer value

    Returns
    -------
    GeoSpatialValue
        `arg` with SRID set to `srid`
    """
    op = ops.GeoSetSRID(arg, srid)
    return op.to_expr()


def geo_buffer(
    arg: ir.GeoSpatialValue, radius: float | ir.FloatingValue
) -> ir.GeoSpatialValue:
    """Returns a geometry that represents all points whose distance from this
    Geometry is less than or equal to distance. Calculations are in the
    Spatial Reference System of this Geometry.

    Parameters
    ----------
    arg
        Geometry expression
    radius
        Floating expression

    Returns
    -------
    ir.GeoSpatialValue
        Geometry expression
    """
    op = ops.GeoBuffer(arg, radius)
    return op.to_expr()


def geo_centroid(arg: ir.GeoSpatialValue) -> ir.PointValue:
    """Returns the centroid of the geometry.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    PointValue
        The centroid
    """
    op = ops.GeoCentroid(arg)
    return op.to_expr()


def geo_envelope(arg: ir.GeoSpatialValue) -> ir.PolygonValue:
    """Returns a geometry representing the bounding box of the arg.

    Parameters
    ----------
    arg
        Geometry expression

    Returns
    -------
    PolygonValue
        A polygon
    """
    op = ops.GeoEnvelope(arg)
    return op.to_expr()


def geo_within(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.BooleanValue:
    """Check if the first geometry is completely inside of the second.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    BooleanValue
        Whether `left` is in `right`.
    """
    op = ops.GeoWithin(left, right)
    return op.to_expr()


def geo_azimuth(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.FloatingValue:
    """Return the angle in radians from the horizontal of the vector defined by
    `left` and `right`.

    Angle is computed clockwise from down-to-up on the clock:
    12=0; 3=PI/2; 6=PI; 9=3PI/2.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    FloatingValue
        azimuth
    """
    op = ops.GeoAzimuth(left, right)
    return op.to_expr()


def geo_intersection(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.GeoSpatialValue:
    """Return the intersection of two geometries.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    GeoSpatialValue
        Intersection of `left` and `right`
    """
    op = ops.GeoIntersection(left, right)
    return op.to_expr()


def geo_difference(
    left: ir.GeoSpatialValue, right: ir.GeoSpatialValue
) -> ir.GeoSpatialValue:
    """Return the difference of two geometries.

    Parameters
    ----------
    left
        Left geometry
    right
        Right geometry

    Returns
    -------
    GeoSpatialValue
        Difference of `left` and `right`
    """
    op = ops.GeoDifference(left, right)
    return op.to_expr()


def geo_simplify(
    arg: ir.GeoSpatialValue,
    tolerance: ir.FloatingValue,
    preserve_collapsed: ir.BooleanValue,
) -> ir.GeoSpatialValue:
    """Simplify a given geometry.

    Parameters
    ----------
    arg
        Geometry expression
    tolerance
        Tolerance
    preserve_collapsed
        Whether to preserve collapsed geometries

    Returns
    -------
    GeoSpatialValue
        Simplified geometry
    """
    op = ops.GeoSimplify(arg, tolerance, preserve_collapsed)
    return op.to_expr()


def geo_transform(
    arg: ir.GeoSpatialValue, srid: ir.IntegerValue
) -> ir.GeoSpatialValue:
    """Transform a geometry into a new SRID.

    Parameters
    ----------
    arg
        Geometry expression
    srid
        Integer expression

    Returns
    -------
    GeoSpatialValue
        Transformed geometry
    """
    op = ops.GeoTransform(arg, srid)
    return op.to_expr()


_geospatial_value_methods = {
    'area': geo_area,
    'as_binary': geo_as_binary,
    'as_ewkb': geo_as_ewkb,
    'as_ewkt': geo_as_ewkt,
    'as_text': geo_as_text,
    'azimuth': geo_azimuth,
    'buffer': geo_buffer,
    'centroid': geo_centroid,
    'contains': geo_contains,
    'contains_properly': geo_contains_properly,
    'covers': geo_covers,
    'covered_by': geo_covered_by,
    'crosses': geo_crosses,
    'd_fully_within': geo_d_fully_within,
    'difference': geo_difference,
    'disjoint': geo_disjoint,
    'distance': geo_distance,
    'd_within': geo_d_within,
    'end_point': geo_end_point,
    'envelope': geo_envelope,
    'geo_equals': geo_equals,
    'geometry_n': geo_geometry_n,
    'geometry_type': geo_geometry_type,
    'intersection': geo_intersection,
    'intersects': geo_intersects,
    'is_valid': geo_is_valid,
    'line_locate_point': geo_line_locate_point,
    'line_merge': geo_line_merge,
    'line_substring': geo_line_substring,
    'length': geo_length,
    'max_distance': geo_max_distance,
    'n_points': geo_n_points,
    'n_rings': geo_n_rings,
    'ordering_equals': geo_ordering_equals,
    'overlaps': geo_overlaps,
    'perimeter': geo_perimeter,
    'point_n': geo_point_n,
    'set_srid': geo_set_srid,
    'simplify': geo_simplify,
    'srid': geo_srid,
    'start_point': geo_start_point,
    'touches': geo_touches,
    'transform': geo_transform,
    'union': geo_union,
    'within': geo_within,
    'x': geo_x,
    'x_max': geo_x_max,
    'x_min': geo_x_min,
    'y': geo_y,
    'y_max': geo_y_max,
    'y_min': geo_y_min,
}
_geospatial_column_methods = {'unary_union': geo_unary_union}

_add_methods(ir.GeoSpatialValue, _geospatial_value_methods)
_add_methods(ir.GeoSpatialColumn, _geospatial_column_methods)

# ----------------------------------------------------------------------
# Boolean API


# TODO: logical binary operators for BooleanValue


def ifelse(
    arg: ir.ValueExpr, true_expr: ir.ValueExpr, false_expr: ir.ValueExpr
) -> ir.ValueExpr:
    """Construct a ternary conditional expression.

    Examples
    --------
    bool_expr.ifelse(0, 1)
    e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END

    Returns
    -------
    ValueExpr
        The value of `true_expr` if `arg` is `True` else `false_expr`
    """
    # Result will be the result of promotion of true/false exprs. These
    # might be conflicting types; same type resolution as case expressions
    # must be used.
    case = bl.SearchedCaseBuilder()
    return case.when(arg, true_expr).else_(false_expr).end()


_boolean_value_methods = {
    'ifelse': ifelse,
    '__and__': _boolean_binary_op('__and__', ops.And),
    '__or__': _boolean_binary_op('__or__', ops.Or),
    '__xor__': _boolean_binary_op('__xor__', ops.Xor),
    '__rand__': _boolean_binary_rop('__rand__', ops.And),
    '__ror__': _boolean_binary_rop('__ror__', ops.Or),
    '__rxor__': _boolean_binary_rop('__rxor__', ops.Xor),
    '__invert__': _boolean_unary_op('__invert__', ops.Not),
}


_boolean_column_methods = {
    'any': _unary_op('any', ops.Any),
    'notany': _unary_op('notany', ops.NotAny),
    'all': _unary_op('all', ops.All),
    'notall': _unary_op('notany', ops.NotAll),
    'cumany': _unary_op('cumany', ops.CumulativeAny),
    'cumall': _unary_op('cumall', ops.CumulativeAll),
}


_add_methods(ir.BooleanValue, _boolean_value_methods)
_add_methods(ir.BooleanColumn, _boolean_column_methods)


# ---------------------------------------------------------------------
# Decimal API


def _precision(arg: ir.DecimalValue) -> ir.IntegerValue:
    """Return the precision of `arg`.

    Parameters
    ----------
    arg
        Decimal expression

    Returns
    -------
    IntegerValue
        The precision of `arg`.
    """
    return ops.DecimalPrecision(arg).to_expr()


def _scale(arg: ir.DecimalValue) -> ir.IntegerValue:
    """Return the scale of `arg`.

    Parameters
    ----------
    arg
        Decimal expression

    Returns
    -------
    IntegerValue
        The scale of `arg`.
    """
    return ops.DecimalScale(arg).to_expr()


_decimal_value_methods = {
    'precision': _precision,
    'scale': _scale,
}


_add_methods(ir.DecimalValue, _decimal_value_methods)


# ----------------------------------------------------------------------
# Category API


_category_value_methods = {'label': category_label}

_add_methods(ir.CategoryValue, _category_value_methods)

prevent_rewrite = ir.TableExpr.prevent_rewrite
aggregate = ir.TableExpr.aggregate
cross_join = ir.TableExpr.cross_join
join = ir.TableExpr.join
asof_join = ir.TableExpr.asof_join
