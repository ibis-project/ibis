"""Ibis expression API definitions."""

from __future__ import annotations

import datetime
import functools
import itertools
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, TypeVar
from typing import Tuple as _Tuple
from typing import Union as _Union

import dateutil.parser
import numpy as np

import ibis.expr.builders as bl
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, connect
from ibis.expr import selectors
from ibis.expr.decompile import decompile
from ibis.expr.deferred import Deferred
from ibis.expr.schema import Schema
from ibis.expr.sql import parse_sql, show_sql, to_sql
from ibis.expr.types import (
    DateValue,
    Expr,
    IntegerColumn,
    StringValue,
    Table,
    TimeValue,
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
from ibis.util import experimental

if TYPE_CHECKING:
    import pandas as pd

__all__ = (
    'aggregate',
    'and_',
    'array',
    'asc',
    'case',
    'coalesce',
    'connect',
    'cross_join',
    'cumulative_window',
    'date',
    'desc',
    'decompile',
    'difference',
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
    'get_backend',
    'greatest',
    'ifelse',
    'infer_dtype',
    'infer_schema',
    'intersect',
    'interval',
    'join',
    'least',
    'literal',
    'map',
    'memtable',
    'NA',
    'negate',
    'now',
    'null',
    'or_',
    'param',
    'parse_sql',
    'pi',
    'random',
    'range_window',
    'read_csv',
    'read_json',
    'read_parquet',
    'row_number',
    'rows_with_max_lookback',
    'schema',
    'Schema',
    'selectors',
    'sequence',
    'set_backend',
    'show_sql',
    'to_sql',
    'struct',
    'table',
    'time',
    'timestamp',
    'trailing_range_window',
    'trailing_window',
    'union',
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


def _deferred(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, Deferred):
            method = getattr(self, fn.__name__)
            return method(*args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


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
    >>> start = ibis.param('date')
    >>> end = ibis.param('date')
    >>> schema = dict(timestamp_col='timestamp', value='double')
    >>> t = ibis.table(schema, name='t')
    >>> predicates = [t.timestamp_col >= start, t.timestamp_col <= end]
    >>> t.filter(predicates).value.sum()
    r0 := UnboundTable: t
      timestamp_col timestamp
      value         float64
    r1 := Selection[r0]
      predicates:
        r0.timestamp_col >= $(date)
        r0.timestamp_col <= $(date)
    sum: Sum(r1.value)
    """
    return ops.ScalarParameter(type).to_expr()


# TODO(kszucs): should be deprecated
def sequence(values: Sequence[T | None]) -> ir.List:
    """Wrap a list of Python values as an Ibis sequence type.

    Parameters
    ----------
    values
        Should all be None or the same type

    Returns
    -------
    List
        A list expression
    """
    return rlz.tuple_of(rlz.any, values)


def schema(
    pairs: SupportsSchema | None = None,
    names: Iterable[str] | None = None,
    types: Iterable[str | dt.DataType] | None = None,
) -> sch.Schema:
    """Validate and return a [`Schema`][ibis.expr.schema.Schema] object.

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
    """
    if pairs is not None:
        return sch.schema(pairs)
    else:
        return sch.schema(names, types)


def table(
    schema: SupportsSchema | None = None,
    name: str | None = None,
) -> ir.Table:
    """Create a table literal or an abstract table without data.

    Parameters
    ----------
    schema
        A schema for the table
    name
        Name for the table. One is generated if this value is `None`.

    Returns
    -------
    Table
        A table expression

    Examples
    --------
    Create a table with no data backing it

    >>> t = ibis.table(schema=dict(a="int", b="string"))
    >>> t
    UnboundTable: unbound_table_0
      a int64
      b string
    """
    if schema is not None:
        schema = sch.schema(schema)
    return ops.UnboundTable(schema=schema, name=name).to_expr()


@functools.singledispatch
def memtable(
    data,
    *,
    columns: Iterable[str] | None = None,
    schema: SupportsSchema | None = None,
    name: str | None = None,
) -> Table:
    """Construct an ibis table expression from in-memory data.

    Parameters
    ----------
    data
        Any data accepted by the `pandas.DataFrame` constructor.

        The use of `DataFrame` underneath should **not** be relied upon and is
        free to change across non-major releases.
    columns
        Optional [`Iterable`][typing.Iterable] of [`str`][str] column names.
    schema
        Optional [`Schema`][ibis.expr.schema.Schema]. The functions use `data`
        to infer a schema if not passed.
    name
        Optional name of the table.

    Returns
    -------
    Table
        A table expression backed by in-memory data.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.memtable([{"a": 1}, {"a": 2}])
    >>> t
    PandasInMemoryTable
      data:
        DataFrameProxy:
             a
          0  1
          1  2

    >>> t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])
    >>> t
    PandasInMemoryTable
      data:
        DataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data and pass
    `columns`

    >>> t = ibis.memtable([(1, "foo"), (2, "baz")], columns=["a", "b"])
    >>> t
    PandasInMemoryTable
      data:
        DataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data. Ibis
    generates column names if none are provided.

    >>> t = ibis.memtable([(1, "foo"), (2, "baz")])
    >>> t
    PandasInMemoryTable
      data:
        DataFrameProxy:
             col0 col1
          0     1  foo
          1     2  baz
    """
    import pandas as pd

    if columns is not None and schema is not None:
        raise NotImplementedError(
            "passing `columns` and schema` is ambiguous; "
            "pass one or the other but not both"
        )
    df = pd.DataFrame(data, columns=columns)
    if df.columns.inferred_type != "string":
        cols = df.columns
        newcols = getattr(
            schema,
            "names",
            (f"col{i:d}" for i in range(len(cols))),
        )
        df = df.rename(columns=dict(zip(cols, newcols)))
    return _memtable_from_dataframe(df, name=name, schema=schema)


_gen_memtable_name = (f"_ibis_memtable{i:d}" for i in itertools.count())


def _memtable_from_dataframe(
    df: pd.DataFrame,
    *,
    name: str | None = None,
    schema: SupportsSchema | None = None,
) -> Table:
    from ibis.backends.pandas.client import DataFrameProxy, PandasInMemoryTable

    op = PandasInMemoryTable(
        name=name if name is not None else next(_gen_memtable_name),
        schema=sch.infer(df) if schema is None else schema,
        data=DataFrameProxy(df),
    )
    return op.to_expr()


def _sort_order(expr, order: Literal["desc", "asc"]):
    method = operator.methodcaller(order)
    if isinstance(expr, str):
        value = _[expr]
    elif isinstance(expr, Deferred):
        value = expr
    elif callable(expr):
        value = expr(_)
    else:
        value = expr
    return method(value)


def desc(expr: ir.Column | str) -> ir.Value:
    """Create a descending sort key from `expr` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table(dict(g='string'), name='t')
    >>> t.group_by('g').size('count').order_by(ibis.desc('count'))
    r0 := UnboundTable: t
      g string
    r1 := Aggregation[r0]
      metrics:
        count: Count(t)
      by:
        g: r0.g
    Selection[r1]
      sort_keys:
        desc|r1.count

    Returns
    -------
    ir.ValueExpr
        An expression
    """
    return _sort_order(expr, "desc")


def asc(expr: ir.Column | str) -> ir.Value:
    """Create a ascending sort key from `asc` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table(dict(g='string'), name='t')
    >>> t.group_by('g').size('count').order_by(ibis.asc('count'))
    r0 := UnboundTable: t
      g string
    r1 := Aggregation[r0]
      metrics:
        count: Count(t)
      by:
        g: r0.g
    Selection[r1]
      sort_keys:
        asc|r1.count

    Returns
    -------
    ir.ValueExpr
        An expression
    """
    return _sort_order(expr, "asc")


def and_(*predicates: ir.BooleanValue) -> ir.BooleanValue:
    """Combine multiple predicates using `&`.

    Parameters
    ----------
    predicates
        Boolean value expressions

    Returns
    -------
    BooleanValue
        A new predicate that evaluates to True if all composing predicates are
        True. If no predicates were provided, returns True.
    """
    if not predicates:
        return literal(True)
    return functools.reduce(operator.and_, predicates)


def or_(*predicates: ir.BooleanValue) -> ir.BooleanValue:
    """Combine multiple predicates using `|`.

    Parameters
    ----------
    predicates
        Boolean value expressions

    Returns
    -------
    BooleanValue
        A new predicate that evaluates to True if any composing predicates are
        True. If no predicates were provided, returns False.
    """
    if not predicates:
        return literal(False)
    return functools.reduce(operator.or_, predicates)


def random() -> ir.FloatingScalar:
    """Return a random floating point number in the range [0.0, 1.0).

    Similar to [`random.random`][random.random] in the Python standard library.

    Returns
    -------
    FloatingScalar
        Random float value expression
    """
    op = ops.RandomScalar()
    return op.to_expr()


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
    args
        Additional arguments used when constructing a timestamp
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
@timestamp.register(ir.IntegerValue)
def _timestamp_from_ymdhms(
    value, *args, timezone: str | None = None
) -> ir.TimestampScalar:
    if timezone:
        raise NotImplementedError('timestamp timezone not implemented')

    if not args:  # only one value
        raise TypeError(f"Use ibis.literal({value}).to_timestamp")

    # pass through to datetime constructor
    return ops.TimestampFromYMDHMS(value, *args).to_expr()


@timestamp.register(datetime.datetime)
def _timestamp_from_datetime(value, timezone: str | None = None) -> ir.TimestampScalar:
    return literal(value, type=dt.Timestamp(timezone=timezone))


@timestamp.register(str)
def _timestamp_from_str(value: str, timezone: str | None = None) -> ir.TimestampScalar:
    import pandas as pd

    try:
        value = pd.Timestamp(value, tz=timezone)
    except pd.errors.OutOfBoundsDatetime:
        value = dateutil.parser.parse(value)
    dtype = dt.Timestamp(timezone=timezone if timezone is not None else value.tzname())
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
    import pandas as pd

    return literal(pd.to_datetime(value).date(), type=dt.date)


@date.register(datetime.datetime)
def _date_from_timestamp(value) -> ir.DateScalar:
    return literal(value, type=dt.date)


@date.register(IntegerColumn)
@date.register(int)
def _date_from_int(year, month, day) -> ir.DateScalar:
    return ops.DateFromYMD(year, month, day).to_expr()


@date.register(StringValue)
def _date_from_string(value: StringValue) -> DateValue:
    return value.cast(dt.date)


@date.register(Deferred)
def _date_from_deferred(value: Deferred) -> Deferred:
    return value.date()


@functools.singledispatch
def time(value) -> TimeValue:
    return literal(value, type=dt.time)


@time.register(str)
def _time_from_str(value: str) -> ir.TimeScalar:
    import pandas as pd

    return literal(pd.to_datetime(value).time(), type=dt.time)


@time.register(IntegerColumn)
@time.register(int)
def _time_from_int(hours, mins, secs) -> ir.TimeScalar:
    return ops.TimeFromHMS(hours, mins, secs).to_expr()


@time.register(StringValue)
def _time_from_string(value: StringValue) -> TimeValue:
    return value.cast(dt.time)


@time.register(Deferred)
def _time_from_deferred(value: Deferred) -> Deferred:
    return value.time()


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

    return literal(value, type=type)


def case() -> bl.SearchedCaseBuilder:
    """Begin constructing a case expression.

    Use the `.when` method on the resulting object followed by `.end` to create a
    complete case.

    Examples
    --------
    >>> import ibis
    >>> cond1 = ibis.literal(1) == 1
    >>> cond2 = ibis.literal(2) == 1
    >>> expr = ibis.case().when(cond1, 3).when(cond2, 4).end()
    SearchedCase(cases=(1 == 1, 2 == 1), results=(3, 4)), default=Cast(None, to=int8))

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
        An expression representing the current timestamp.
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


def read_csv(sources: str | Path | Sequence[str | Path], **kwargs: Any) -> ir.Table:
    """Lazily load a CSV or set of CSVs.

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.  Supports CSV and TSV files.
    kwargs
        DuckDB-specific keyword arguments for the file type.

        * CSV/TSV: https://duckdb.org/docs/data/csv#parameters.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> batting = ibis.read_csv("ci/ibis-testing-data/batting.csv")
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_csv(sources, **kwargs)


@experimental
def read_json(sources: str | Path | Sequence[str | Path], **kwargs: Any) -> ir.Table:
    """Lazily load newline-delimited JSON data.

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    kwargs
        DuckDB-specific keyword arguments for the file type.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> t = ibis.read_json("data.json")
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_json(sources, **kwargs)


def read_parquet(sources: str | Path | Sequence[str | Path], **kwargs: Any) -> ir.Table:
    """Lazily load a parquet file or set of parquet files.

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    kwargs
        DuckDB-specific keyword arguments for the file type.

        * Parquet: https://duckdb.org/docs/data/parquet

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> batting = ibis.read_parquet("ci/ibis-testing-data/parquet/batting/batting.parquet")
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_parquet(sources, **kwargs)


def set_backend(backend: str | BaseBackend) -> None:
    """Set the default Ibis backend.

    Parameters
    ----------
    backend
        May be a backend name or URL, or an existing backend instance.

    Examples
    --------
    May pass the backend as a name:
    >>> ibis.set_backend("polars")

    Or as a URI:
    >>> ibis.set_backend("postgres://user:password@hostname:5432")

    Or as an existing backend instance:
    >>> ibis.set_backend(ibis.duckdb.connect())
    """
    import ibis

    if isinstance(backend, str) and backend.isidentifier():
        try:
            backend_type = getattr(ibis, backend)
        except AttributeError:
            pass
        else:
            backend = backend_type.connect()
    if isinstance(backend, str):
        backend = ibis.connect(backend)

    ibis.options.default_backend = backend


def get_backend(expr: Expr | None = None) -> BaseBackend:
    """Get the current Ibis backend to use for a given expression.

    Parameters
    ----------
    expr
        An expression to get the backend from. If not passed, the default
        backend is returned.

    Returns
    -------
    BaseBackend
        The Ibis backend.
    """
    if expr is None:
        from ibis.config import _default_backend

        return _default_backend()
    return expr._find_backend(use_default=True)


e = ops.E().to_expr()

pi = ops.Pi().to_expr()


geo_area = _deferred(ir.GeoSpatialValue.area)
geo_as_binary = _deferred(ir.GeoSpatialValue.as_binary)
geo_as_ewkb = _deferred(ir.GeoSpatialValue.as_ewkb)
geo_as_ewkt = _deferred(ir.GeoSpatialValue.as_ewkt)
geo_as_text = _deferred(ir.GeoSpatialValue.as_text)
geo_azimuth = _deferred(ir.GeoSpatialValue.azimuth)
geo_buffer = _deferred(ir.GeoSpatialValue.buffer)
geo_centroid = _deferred(ir.GeoSpatialValue.centroid)
geo_contains = _deferred(ir.GeoSpatialValue.contains)
geo_contains_properly = _deferred(ir.GeoSpatialValue.contains_properly)
geo_covers = _deferred(ir.GeoSpatialValue.covers)
geo_covered_by = _deferred(ir.GeoSpatialValue.covered_by)
geo_crosses = _deferred(ir.GeoSpatialValue.crosses)
geo_d_fully_within = _deferred(ir.GeoSpatialValue.d_fully_within)
geo_difference = _deferred(ir.GeoSpatialValue.difference)
geo_disjoint = _deferred(ir.GeoSpatialValue.disjoint)
geo_distance = _deferred(ir.GeoSpatialValue.distance)
geo_d_within = _deferred(ir.GeoSpatialValue.d_within)
geo_end_point = _deferred(ir.GeoSpatialValue.end_point)
geo_envelope = _deferred(ir.GeoSpatialValue.envelope)
geo_equals = _deferred(ir.GeoSpatialValue.geo_equals)
geo_geometry_n = _deferred(ir.GeoSpatialValue.geometry_n)
geo_geometry_type = _deferred(ir.GeoSpatialValue.geometry_type)
geo_intersection = _deferred(ir.GeoSpatialValue.intersection)
geo_intersects = _deferred(ir.GeoSpatialValue.intersects)
geo_is_valid = _deferred(ir.GeoSpatialValue.is_valid)
geo_line_locate_point = _deferred(ir.GeoSpatialValue.line_locate_point)
geo_line_merge = _deferred(ir.GeoSpatialValue.line_merge)
geo_line_substring = _deferred(ir.GeoSpatialValue.line_substring)
geo_length = _deferred(ir.GeoSpatialValue.length)
geo_max_distance = _deferred(ir.GeoSpatialValue.max_distance)
geo_n_points = _deferred(ir.GeoSpatialValue.n_points)
geo_n_rings = _deferred(ir.GeoSpatialValue.n_rings)
geo_ordering_equals = _deferred(ir.GeoSpatialValue.ordering_equals)
geo_overlaps = _deferred(ir.GeoSpatialValue.overlaps)
geo_perimeter = _deferred(ir.GeoSpatialValue.perimeter)
geo_point = _deferred(ir.NumericValue.point)
geo_point_n = _deferred(ir.GeoSpatialValue.point_n)
geo_set_srid = _deferred(ir.GeoSpatialValue.set_srid)
geo_simplify = _deferred(ir.GeoSpatialValue.simplify)
geo_srid = _deferred(ir.GeoSpatialValue.srid)
geo_start_point = _deferred(ir.GeoSpatialValue.start_point)
geo_touches = _deferred(ir.GeoSpatialValue.touches)
geo_transform = _deferred(ir.GeoSpatialValue.transform)
geo_union = _deferred(ir.GeoSpatialValue.union)
geo_within = _deferred(ir.GeoSpatialValue.within)
geo_x = _deferred(ir.GeoSpatialValue.x)
geo_x_max = _deferred(ir.GeoSpatialValue.x_max)
geo_x_min = _deferred(ir.GeoSpatialValue.x_min)
geo_y = _deferred(ir.GeoSpatialValue.y)
geo_y_max = _deferred(ir.GeoSpatialValue.y_max)
geo_y_min = _deferred(ir.GeoSpatialValue.y_min)
geo_unary_union = _deferred(ir.GeoSpatialColumn.unary_union)

where = ifelse = _deferred(ir.BooleanValue.ifelse)
coalesce = _deferred(ir.Value.coalesce)
greatest = _deferred(ir.Value.greatest)
least = _deferred(ir.Value.least)
category_label = _deferred(ir.CategoryValue.label)

aggregate = ir.Table.aggregate
cross_join = ir.Table.cross_join
join = ir.Table.join
asof_join = ir.Table.asof_join

union = ir.Table.union
intersect = ir.Table.intersect
difference = ir.Table.difference

_ = Deferred()
