"""Ibis expression API definitions."""

from __future__ import annotations

import datetime
import functools
from typing import Iterable, Mapping, Sequence
from typing import Tuple as _Tuple
from typing import TypeVar
from typing import Union as _Union

import dateutil.parser
import numpy as np
import pandas as pd

import ibis.expr.builders as bl
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import connect
from ibis.expr.deferred import Deferred
from ibis.expr.random import random
from ibis.expr.schema import Schema
from ibis.expr.types import (  # noqa: F401
    ArrayColumn,
    ArrayScalar,
    ArrayValue,
    BooleanColumn,
    BooleanScalar,
    BooleanValue,
    CategoryScalar,
    CategoryValue,
    Column,
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
    Scalar,
    StringColumn,
    StringScalar,
    StringValue,
    StructColumn,
    StructScalar,
    StructValue,
    Table,
    TimeColumn,
    TimeScalar,
    TimestampColumn,
    TimestampScalar,
    TimestampValue,
    TimeValue,
    Value,
    array,
    literal,
    map,
    null,
    struct,
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
    'array',
    'case',
    'coalesce',
    'connect',
    'cross_join',
    'cumulative_window',
    'date',
    'desc',
    'asc',
    'e',
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
    '_',
)


infer_dtype = dt.infer
infer_schema = sch.infer


NA = null()

T = TypeVar("T")

negate = ir.NumericValue.negate

SupportsSchema = TypeVar(
    "SupportsSchema",
    Iterable[_Tuple[str, _Union[str, dt.DataType]]],
    Mapping[str, dt.DataType],
    sch.Schema,
)


def param(type: dt.DataType) -> ir.Scalar:
    """Create a deferred parameter of a given type.

    Parameters
    ----------
    type
        The type of the unbound parameter, e.g., double, int64, date, etc.

    Returns
    -------
    Scalar
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


def sequence(values: Sequence[T | None]) -> ir.ValueList:
    """Wrap a list of Python values as an Ibis sequence type.

    Parameters
    ----------
    values
        Should all be None or the same type

    Returns
    -------
    ValueList
        A list expression
    """
    return ops.ValueList(values).to_expr()


def schema(
    pairs: SupportsSchema | None = None,
    names: Iterable[str] | None = None,
    types: Iterable[str | dt.DataType] | None = None,
) -> sch.Schema:
    """Validate and return an Schema object.

    Parameters
    ----------
    pairs
        List or dictionary of name, type pairs. Mutually exclusive with `names`
        and `types` arguments.
    names
        Field names. Mutually exclusive with `pairs`.
    types
        Field types. Mutually exclusive with `pairs`.

    Examples
    --------
    >>> from ibis import schema, Schema
    >>> sc = schema([('foo', 'string'),
    ...              ('bar', 'int64'),
    ...              ('baz', 'boolean')])
    >>> sc = schema(names=['foo', 'bar', 'baz'],
    ...             types=['string', 'int64', 'boolean'])
    >>> sc = schema(dict(foo="string"))
    >>> sc = schema(Schema(['foo'], ['string']))  # no-op

    Returns
    -------
    Schema
        An ibis schema
    """  # noqa: E501
    if pairs is not None:
        return sch.schema(pairs)
    else:
        return sch.schema(names, types)


def table(
    schema: SupportsSchema,
    name: str | None = None,
) -> ir.Table:
    """Create an unbound table for building expressions without data.

    Parameters
    ----------
    schema
        A schema for the table
    name
        Name for the table. One is generated if this value is `None`.

    Returns
    -------
    Table
        An unbound table expression
    """
    node = ops.UnboundTable(sch.schema(schema), name=name)
    return node.to_expr()


def desc(expr: ir.Column | str) -> ir.SortExpr | ops.DeferredSortKey:
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


def asc(expr: ir.Column | str) -> ir.SortExpr | ops.DeferredSortKey:
    """Create a ascending sort key from `asc` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('g', 'string')])
    >>> result = t.group_by('g').size('count').sort_by(ibis.asc('count'))

    Returns
    -------
    ops.DeferredSortKey
        A deferred sort key
    """
    if not isinstance(expr, Expr):
        return ops.DeferredSortKey(expr)
    else:
        return ops.SortKey(expr).to_expr()


@functools.singledispatch
def timestamp(
    value,
    *args,
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
    raise NotImplementedError(f'cannot convert {type(value)} to timestamp')


@timestamp.register(np.integer)
@timestamp.register(np.floating)
@timestamp.register(int)
@timestamp.register(float)
def _timestamp_from_ymdhms(
    value, *args, timezone: str | None = None
) -> ir.TimestampScalar:
    if timezone:
        raise NotImplementedError('timestamp timezone not implemented')

    if not args:  # only one value
        raise TypeError(f"Use ibis.literal({value}).to_timestamp")

    # pass through to datetime constructor
    return ops.TimestampFromYMDHMS(value, *args).to_expr()


@timestamp.register(pd.Timestamp)
def _timestamp_from_timestamp(
    value, timezone: str | None = None
) -> ir.TimestampScalar:
    return literal(value, type=dt.Timestamp(timezone=timezone))


@timestamp.register(datetime.datetime)
def _timestamp_from_datetime(
    value, timezone: str | None = None
) -> ir.TimestampScalar:
    return literal(value, type=dt.Timestamp(timezone=timezone))


@timestamp.register(str)
def _timestamp_from_str(
    value: str, timezone: str | None = None
) -> ir.TimestampScalar:
    try:
        value = pd.Timestamp(value, tz=timezone)
    except pd.errors.OutOfBoundsDatetime:
        value = dateutil.parser.parse(value)
    dtype = dt.Timestamp(
        timezone=timezone if timezone is not None else value.tzname()
    )
    return literal(value, type=dtype)


@functools.singledispatch
def date(value) -> DateValue:
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
    raise NotImplementedError()


@date.register(str)
def _date_from_str(value: str) -> ir.DateScalar:
    return literal(pd.to_datetime(value).date(), type=dt.date)


@date.register(pd.Timestamp)
def _date_from_timestamp(value) -> ir.DateScalar:
    return literal(value, type=dt.date)


@date.register(IntegerColumn)
@date.register(int)
def _date_from_int(year, month, day) -> ir.DateScalar:
    return ops.DateFromYMD(year, month, day).to_expr()


@date.register(StringValue)
def _date_from_string(value: StringValue) -> DateValue:
    return value.cast(dt.date)


@functools.singledispatch
def time(value) -> TimeValue:
    return literal(value, type=dt.time)


@time.register(str)
def _time_from_str(value: str) -> ir.TimeScalar:
    return literal(pd.to_datetime(value).time(), type=dt.time)


@time.register(IntegerColumn)
@time.register(int)
def _time_from_int(hours, mins, secs) -> ir.TimeScalar:
    return ops.TimeFromHMS(hours, mins, secs).to_expr()


@time.register(StringValue)
def _time_from_string(value: StringValue) -> TimeValue:
    return value.cast(dt.time)


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
    type = dt.Interval(unit, value_type=value_type)

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


def where(
    boolean_expr: ir.BooleanValue,
    true_expr: ir.Value,
    false_null_expr: ir.Value,
) -> ir.Value:
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
    ir.Value
        An expression
    """
    op = ops.Where(boolean_expr, true_expr, false_null_expr)
    return op.to_expr()


coalesce = ir.Value.coalesce
greatest = ir.Value.greatest
least = ir.Value.least


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


geo_area = ir.GeoSpatialValue.area
geo_as_binary = ir.GeoSpatialValue.as_binary
geo_as_ewkb = ir.GeoSpatialValue.as_ewkb
geo_as_ewkt = ir.GeoSpatialValue.as_ewkt
geo_as_text = ir.GeoSpatialValue.as_text
geo_azimuth = ir.GeoSpatialValue.azimuth
geo_buffer = ir.GeoSpatialValue.buffer
geo_centroid = ir.GeoSpatialValue.centroid
geo_contains = ir.GeoSpatialValue.contains
geo_contains_properly = ir.GeoSpatialValue.contains_properly
geo_covers = ir.GeoSpatialValue.covers
geo_covered_by = ir.GeoSpatialValue.covered_by
geo_crosses = ir.GeoSpatialValue.crosses
geo_d_fully_within = ir.GeoSpatialValue.d_fully_within
geo_difference = ir.GeoSpatialValue.difference
geo_disjoint = ir.GeoSpatialValue.disjoint
geo_distance = ir.GeoSpatialValue.distance
geo_d_within = ir.GeoSpatialValue.d_within
geo_end_point = ir.GeoSpatialValue.end_point
geo_envelope = ir.GeoSpatialValue.envelope
geo_equals = ir.GeoSpatialValue.geo_equals
geo_geometry_n = ir.GeoSpatialValue.geometry_n
geo_geometry_type = ir.GeoSpatialValue.geometry_type
geo_intersection = ir.GeoSpatialValue.intersection
geo_intersects = ir.GeoSpatialValue.intersects
geo_is_valid = ir.GeoSpatialValue.is_valid
geo_line_locate_point = ir.GeoSpatialValue.line_locate_point
geo_line_merge = ir.GeoSpatialValue.line_merge
geo_line_substring = ir.GeoSpatialValue.line_substring
geo_length = ir.GeoSpatialValue.length
geo_max_distance = ir.GeoSpatialValue.max_distance
geo_n_points = ir.GeoSpatialValue.n_points
geo_n_rings = ir.GeoSpatialValue.n_rings
geo_ordering_equals = ir.GeoSpatialValue.ordering_equals
geo_overlaps = ir.GeoSpatialValue.overlaps
geo_perimeter = ir.GeoSpatialValue.perimeter
geo_point = ir.NumericValue.point
geo_point_n = ir.GeoSpatialValue.point_n
geo_set_srid = ir.GeoSpatialValue.set_srid
geo_simplify = ir.GeoSpatialValue.simplify
geo_srid = ir.GeoSpatialValue.srid
geo_start_point = ir.GeoSpatialValue.start_point
geo_touches = ir.GeoSpatialValue.touches
geo_transform = ir.GeoSpatialValue.transform
geo_union = ir.GeoSpatialValue.union
geo_within = ir.GeoSpatialValue.within
geo_x = ir.GeoSpatialValue.x
geo_x_max = ir.GeoSpatialValue.x_max
geo_x_min = ir.GeoSpatialValue.x_min
geo_y = ir.GeoSpatialValue.y
geo_y_max = ir.GeoSpatialValue.y_max
geo_y_min = ir.GeoSpatialValue.y_min
geo_unary_union = ir.GeoSpatialColumn.unary_union

ifelse = ir.BooleanValue.ifelse

# ----------------------------------------------------------------------
# Category API


_category_value_methods = {'label': category_label}

_add_methods(ir.CategoryValue, _category_value_methods)

prevent_rewrite = ir.Table.prevent_rewrite
aggregate = ir.Table.aggregate
cross_join = ir.Table.cross_join
join = ir.Table.join
asof_join = ir.Table.asof_join

_ = Deferred()
