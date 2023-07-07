"""Ibis expression API definitions."""

from __future__ import annotations

import datetime
import functools
import numbers
import operator
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Sequence, TypeVar

import ibis.expr.builders as bl
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import selectors, util
from ibis.backends.base import BaseBackend, connect
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import IbisInputError
from ibis.common.temporal import normalize_datetime, normalize_timezone
from ibis.expr.decompile import decompile
from ibis.expr.deferred import Deferred
from ibis.expr.schema import Schema
from ibis.expr.sql import parse_sql, show_sql, to_sql
from ibis.expr.types import (
    DateValue,
    Expr,
    Table,
    TimeValue,
    array,
    literal,
    map,
    null,
    struct,
)
from ibis.util import experimental

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa

    from ibis.common.typing import SupportsSchema

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
    'deferred',
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
    'read_delta',
    'read_json',
    'read_parquet',
    'row_number',
    'rows_window',
    'rows_with_max_lookback',
    'schema',
    'Schema',
    'selectors',
    'set_backend',
    'show_sql',
    'struct',
    'to_sql',
    'table',
    'time',
    'timestamp',
    'trailing_range_window',
    'trailing_window',
    'union',
    'where',
    'window',
    'preceding',
    'following',
    '_',
)


infer_dtype = dt.infer
infer_schema = sch.infer


NA = null()
"""The NULL scalar.

Examples
--------
>>> import ibis
>>> my_null = ibis.NA
>>> my_null.isnull()
True
"""

T = TypeVar("T")

negate = ir.NumericValue.negate


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
    Sum(value): Sum(r1.value)
    """
    return ops.ScalarParameter(type).to_expr()


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

    Returns
    -------
    Schema
        An ibis schema

    Examples
    --------
    >>> from ibis import schema, Schema
    >>> sc = schema([('foo', 'string'),
    ...              ('bar', 'int64'),
    ...              ('baz', 'boolean')])
    >>> sc = schema(names=['foo', 'bar', 'baz'],
    ...             types=['string', 'int64', 'boolean'])
    >>> sc = schema(dict(foo="string"))
    >>> sc = schema(Schema(dict(foo="string")))  # no-op
    """
    if pairs is not None:
        return sch.schema(pairs)

    # validate lengths of names and types are the same
    if len(names) != len(types):
        raise ValueError('Schema names and types must have the same length')

    return sch.Schema.from_tuples(zip(names, types))


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

    >>> import ibis
    >>> ibis.options.interactive
    False
    >>> t = ibis.table(schema=dict(a="int", b="string"), name="t")
    >>> t
    UnboundTable: t
      a int64
      b string
    """
    if isinstance(schema, type) and name is None:
        name = schema.__name__
    return ops.UnboundTable(schema=schema, name=name).to_expr()


@lazy_singledispatch
def _memtable(
    data,
    *,
    columns: Iterable[str] | None = None,
    schema: SupportsSchema | None = None,
    name: str | None = None,
):
    raise NotImplementedError(type(data))


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
        Any data accepted by the `pandas.DataFrame` constructor or a `pyarrow.Table`.

        Examples of acceptable objects are a `pandas.DataFrame`, a `pyarrow.Table`,
        a list of dicts of non-ibis Python objects, etc.
        `ibis` objects, like `MapValue`, will result in an error.

        Do not depend on the underlying storage type (e.g., pyarrow.Table), it's subject
        to change across non-major releases.
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
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a
          0  1
          1  2

    >>> t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data and pass
    `columns`

    >>> t = ibis.memtable([(1, "foo"), (2, "baz")], columns=["a", "b"])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data. Ibis
    generates column names if none are provided.

    >>> t = ibis.memtable([(1, "foo"), (2, "baz")])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             col0 col1
          0     1  foo
          1     2  baz
    """
    if columns is not None and schema is not None:
        raise NotImplementedError(
            "passing `columns` and schema` is ambiguous; "
            "pass one or the other but not both"
        )
    return _memtable(data, name=name, schema=schema, columns=columns)


@_memtable.register("pyarrow.Table")
def _memtable_from_pyarrow_table(
    data: pa.Table,
    *,
    name: str | None = None,
    schema: SupportsSchema | None = None,
    columns: Iterable[str] | None = None,
):
    from ibis.expr.operations.relations import PyArrowTableProxy

    if columns is not None:
        assert schema is None, "if `columns` is not `None` then `schema` must be `None`"
        schema = sch.Schema(dict(zip(columns, sch.infer(data).values())))
    return ops.InMemoryTable(
        name=name if name is not None else util.gen_name("pyarrow_memtable"),
        schema=sch.infer(data) if schema is None else schema,
        data=PyArrowTableProxy(data),
    ).to_expr()


@_memtable.register(object)
def _memtable_from_dataframe(
    data: pd.DataFrame | Any,
    *,
    name: str | None = None,
    schema: SupportsSchema | None = None,
    columns: Iterable[str] | None = None,
) -> Table:
    import pandas as pd

    from ibis.expr.operations.relations import PandasDataFrameProxy

    df = pd.DataFrame(data, columns=columns)
    if df.columns.inferred_type != "string":
        cols = df.columns
        newcols = getattr(
            schema,
            "names",
            (f"col{i:d}" for i in range(len(cols))),
        )
        df = df.rename(columns=dict(zip(cols, newcols)))
    op = ops.InMemoryTable(
        name=name if name is not None else util.gen_name("pandas_memtable"),
        schema=sch.infer(df) if schema is None else schema,
        data=PandasDataFrameProxy(df),
    )
    return op.to_expr()


def _deferred_method_call(expr, method_name):
    method = operator.methodcaller(method_name)
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
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t[["species", "year"]].order_by(ibis.desc("year")).head()
    ┏━━━━━━━━━┳━━━━━━━┓
    ┃ species ┃ year  ┃
    ┡━━━━━━━━━╇━━━━━━━┩
    │ string  │ int64 │
    ├─────────┼───────┤
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    └─────────┴───────┘

    Returns
    -------
    ir.ValueExpr
        An expression
    """
    return _deferred_method_call(expr, "desc")


def asc(expr: ir.Column | str) -> ir.Value:
    """Create a ascending sort key from `asc` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t[["species", "year"]].order_by(ibis.asc("year")).head()
    ┏━━━━━━━━━┳━━━━━━━┓
    ┃ species ┃ year  ┃
    ┡━━━━━━━━━╇━━━━━━━┩
    │ string  │ int64 │
    ├─────────┼───────┤
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    └─────────┴───────┘

    Returns
    -------
    ir.ValueExpr
        An expression
    """
    return _deferred_method_call(expr, "asc")


def preceding(value) -> ir.Value:
    return ops.WindowBoundary(value, preceding=True).to_expr()


def following(value) -> ir.Value:
    return ops.WindowBoundary(value, preceding=False).to_expr()


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
    return ops.RandomScalar().to_expr()


def timestamp(value, *args, timezone: str | None = None) -> ir.TimestampScalar:
    """Return a timestamp literal if `value` is coercible to a timestamp.

    Parameters
    ----------
    value
        Timestamp string, datetime object or numeric value
    args
        Additional arguments if `value` is numeric
    timezone
        Timezone name

    Returns
    -------
    TimestampScalar
        A timestamp expression

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.timestamp("2021-01-01 00:00:00")
    Timestamp('2021-01-01 00:00:00')
    """
    if isinstance(value, (numbers.Real, ir.IntegerValue)):
        if timezone:
            raise NotImplementedError('timestamp timezone not implemented')
        if not args:
            raise TypeError(f"Use ibis.literal({value}).to_timestamp() instead")
        return ops.TimestampFromYMDHMS(value, *args).to_expr()
    elif isinstance(value, (Deferred, ir.Expr)):
        # TODO(kszucs): could call .cast(dt.timestamp) for certain value expressions
        raise NotImplementedError(
            "`ibis.timestamp` isn't implemented for expression inputs"
        )
    else:
        value = normalize_datetime(value)
        tzinfo = normalize_timezone(timezone or value.tzinfo)
        timezone = tzinfo.tzname(value) if tzinfo is not None else None
        return literal(value, type=dt.Timestamp(timezone=timezone))


def date(value, *args) -> DateValue:
    """Return a date literal if `value` is coercible to a date.

    Parameters
    ----------
    value
        Date string, datetime object or numeric value
    args
        Month and day if `value` is a year

    Returns
    -------
    DateScalar
        A date expression
    """
    if isinstance(value, (numbers.Real, ir.IntegerValue)):
        year, month, day = value, *args
        return ops.DateFromYMD(year, month, day).to_expr()
    elif isinstance(value, ir.StringValue):
        return value.cast(dt.date)
    elif isinstance(value, Deferred):
        return value.date()
    else:
        return literal(value, type=dt.date)


def time(value, *args) -> TimeValue:
    """Return a time literal if `value` is coercible to a time.

    Parameters
    ----------
    value
        Time string
    args
        Minutes, seconds if `value` is an hour

    Returns
    -------
    TimeScalar
        A time expression

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.time("00:00:00")
    datetime.time(0, 0)
    >>> ibis.time(12, 15, 30)
    datetime.time(12, 15, 30)
    """
    if isinstance(value, (numbers.Real, ir.IntegerValue)):
        hours, mins, secs = value, *args
        return ops.TimeFromHMS(hours, mins, secs).to_expr()
    elif isinstance(value, ir.StringValue):
        return value.cast(dt.time)
    elif isinstance(value, Deferred):
        return value.time()
    else:
        return literal(value, type=dt.time)


def interval(
    value: int | datetime.timedelta | None = None,
    unit: str = 's',
    *,
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
        Interval value.
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
    keyword_value_unit = [
        ("nanoseconds", nanoseconds, "ns"),
        ("microseconds", microseconds, "us"),
        ("milliseconds", milliseconds, "ms"),
        ("seconds", seconds, "s"),
        ("minutes", minutes, "m"),
        ("hours", hours, "h"),
        ("days", days, "D"),
        ("weeks", weeks, "W"),
        ("months", months, "M"),
        ("quarters", quarters, "Q"),
        ("years", years, "Y"),
    ]
    if value is not None:
        for kw, v, _ in keyword_value_unit:
            if v is not None:
                raise TypeError(f"Cannot provide both 'value' and '{kw}'")
        if isinstance(value, datetime.timedelta):
            components = [
                (value.microseconds, "us"),
                (value.seconds, "s"),
                (value.days, "D"),
            ]
            components = [(v, u) for v, u in components if v]
        elif isinstance(value, int):
            components = [(value, unit)]
        else:
            raise TypeError("value must be an integer or timedelta")
    else:
        components = [(v, u) for _, v, u in keyword_value_unit if v is not None]

    # If no components, default to 0 s
    if not components:
        components.append((0, "s"))

    intervals = [literal(v, type=dt.Interval(u)) for v, u in components]
    return functools.reduce(operator.add, intervals)


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
    >>> expr
    SearchedCase(...)

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


def read_csv(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a CSV or set of CSVs.

    This function delegates to the `read_csv` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.  Supports CSV and TSV files.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * CSV/TSV: https://duckdb.org/docs/data/csv#parameters.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import ibis
    >>> t = ibis.read_csv("path/to/data.csv", table_name="my_csv_table")  # doctest: +SKIP
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_csv(sources, table_name=table_name, **kwargs)


@experimental
def read_json(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load newline-delimited JSON data.

    This function delegates to the `read_json` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. See
        https://duckdb.org/docs/extensions/json.html for details.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> lines = '''
    ... {"a": 1, "b": "d"}
    ... {"a": 2, "b": null}
    ... {"a": null, "b": "f"}
    ... '''
    >>> with open("/tmp/lines.json", mode="w") as f:
    ...     _ = f.write(lines)
    >>> t = ibis.read_json("/tmp/lines.json")
    >>> t
    ┏━━━━━━━┳━━━━━━━━┓
    ┃ a     ┃ b      ┃
    ┡━━━━━━━╇━━━━━━━━┩
    │ int64 │ string │
    ├───────┼────────┤
    │     1 │ d      │
    │     2 │ NULL   │
    │  NULL │ f      │
    └───────┴────────┘
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_json(sources, table_name=table_name, **kwargs)


def read_parquet(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a parquet file or set of parquet files.

    This function delegates to the `read_parquet` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * Parquet: https://duckdb.org/docs/data/parquet

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import ibis
    >>> t = ibis.read_parquet("path/to/data.parquet", table_name="my_parquet_table")  # doctest: +SKIP
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_parquet(sources, table_name=table_name, **kwargs)


def read_delta(
    source: str | Path, table_name: str | None = None, **kwargs: Any
) -> ir.Table:
    """Lazily load a Delta Lake table.

    Parameters
    ----------
    source
        A filesystem path or URL.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import ibis
    >>> t = ibis.read_delta("path/to/delta", table_name="my_table")  # doctest: +SKIP
    """
    from ibis.config import _default_backend

    con = _default_backend()
    return con.read_delta(source, table_name=table_name, **kwargs)


def set_backend(backend: str | BaseBackend) -> None:
    """Set the default Ibis backend.

    Parameters
    ----------
    backend
        May be a backend name or URL, or an existing backend instance.

    Examples
    --------
    You can pass the backend as a name:

    >>> import ibis
    >>> ibis.set_backend("polars")

    Or as a URI

    >>> ibis.set_backend("postgres://user:password@hostname:5432")  # doctest: +SKIP

    Or as an existing backend instance

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


class RowsWithMaxLookback(NamedTuple):
    rows: int
    max_lookback: ir.IntervalValue


def rows_with_max_lookback(
    rows: int | np.integer, max_lookback: ir.IntervalValue
) -> RowsWithMaxLookback:
    """Create a bound preceding value for use with trailing window functions.

    Parameters
    ----------
    rows
        Number of rows
    max_lookback
        Maximum lookback in time

    Returns
    -------
    RowsWithMaxLookback
        A named tuple of rows and maximum look-back in time.
    """
    return RowsWithMaxLookback(rows, max_lookback)


def window(
    preceding=None,
    following=None,
    order_by=None,
    group_by=None,
    *,
    rows=None,
    range=None,
    between=None,
):
    """Create a window clause for use with window functions.

    The `ROWS` window clause includes peer rows based on differences in row
    **number** whereas `RANGE` includes rows based on the differences in row
    **value** of a single `order_by` expression.

    All window frame bounds are inclusive.

    Parameters
    ----------
    preceding
        Number of preceding rows in the window
    following
        Number of following rows in the window
    group_by
        Grouping key
    order_by
        Ordering key
    rows
        Whether to use the `ROWS` window clause
    range
        Whether to use the `RANGE` window clause
    between
        Automatically infer the window kind based on the boundaries

    Returns
    -------
    Window
        A window frame
    """
    if isinstance(preceding, RowsWithMaxLookback):
        max_lookback = preceding.max_lookback
        preceding = preceding.rows
    else:
        max_lookback = None

    has_rows = rows is not None
    has_range = range is not None
    has_between = between is not None
    has_preceding_following = preceding is not None or following is not None
    if has_rows + has_range + has_between + has_preceding_following > 1:
        raise IbisInputError(
            "Must only specify either `rows`, `range`, `between` or `preceding`/`following`"
        )

    builder = (
        bl.LegacyWindowBuilder()
        .group_by(group_by)
        .order_by(order_by)
        .lookback(max_lookback)
    )
    if has_rows:
        return builder.rows(*rows)
    elif has_range:
        return builder.range(*range)
    elif has_between:
        return builder.between(*between)
    elif has_preceding_following:
        return builder.preceding_following(preceding, following)
    else:
        return builder


def rows_window(preceding=None, following=None, group_by=None, order_by=None):
    """Create a rows-based window clause for use with window functions.

    This ROWS window clause aggregates rows based upon differences in row
    number.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    preceding
        Number of preceding rows in the window
    following
        Number of following rows in the window
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    if isinstance(preceding, RowsWithMaxLookback):
        max_lookback = preceding.max_lookback
        preceding = preceding.rows
    else:
        max_lookback = None

    return (
        bl.LegacyWindowBuilder()
        .group_by(group_by)
        .order_by(order_by)
        .lookback(max_lookback)
        .preceding_following(preceding, following, how="rows")
    )


def range_window(preceding=None, following=None, group_by=None, order_by=None):
    """Create a range-based window clause for use with window functions.

    This RANGE window clause aggregates rows based upon differences in the
    value of the order-by expression.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    preceding
        Number of preceding rows in the window
    following
        Number of following rows in the window
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return (
        bl.LegacyWindowBuilder()
        .group_by(group_by)
        .order_by(order_by)
        .preceding_following(preceding, following, how="range")
    )


def cumulative_window(group_by=None, order_by=None):
    """Create a cumulative window for use with window functions.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return window(rows=(None, 0), group_by=group_by, order_by=order_by)


def trailing_window(preceding, group_by=None, order_by=None):
    """Create a trailing window for use with window functions.

    Parameters
    ----------
    preceding
        The number of preceding rows
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return window(
        preceding=preceding, following=0, group_by=group_by, order_by=order_by
    )


def trailing_rows_window(preceding, group_by=None, order_by=None):
    """Create a trailing window for use with aggregate window functions.

    Parameters
    ----------
    preceding
        The number of preceding rows
    group_by
        Grouping key
    order_by
        Ordering key

    Returns
    -------
    Window
        A window frame
    """
    return rows_window(
        preceding=preceding, following=0, group_by=group_by, order_by=order_by
    )


def trailing_range_window(preceding, order_by, group_by=None):
    """Create a trailing range window for use with window functions.

    Parameters
    ----------
    preceding
        A value expression
    order_by
        Ordering key
    group_by
        Grouping key

    Returns
    -------
    Window
        A window frame
    """
    return range_window(
        preceding=preceding, following=0, group_by=group_by, order_by=order_by
    )


def union(table: ir.Table, *rest: ir.Table, distinct: bool = False):
    """Compute the set union of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table
        A table expression
    *rest
        Additional table expressions
    distinct
        Only return distinct rows

    Returns
    -------
    Table
        A new table containing the union of all input tables.

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t1 = ibis.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = ibis.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> ibis.union(t1, t2)  # union all by default
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    │     2 │
    │     3 │
    └───────┘
    >>> ibis.union(t1, t2, distinct=True).order_by("a")
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    │     3 │
    └───────┘
    """
    return table.union(*rest, distinct=distinct) if rest else table


def intersect(table: ir.Table, *rest: ir.Table, distinct: bool = True):
    """Compute the set intersection of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table
        A table expression
    *rest
        Additional table expressions
    distinct
        Only return distinct rows

    Returns
    -------
    Table
        A new table containing the intersection of all input tables.

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t1 = ibis.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = ibis.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> ibis.intersect(t1, t2)
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    └───────┘
    """
    return table.intersect(*rest, distinct=distinct) if rest else table


def difference(table: ir.Table, *rest: ir.Table, distinct: bool = True):
    """Compute the set difference of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table:
        A table expression
    *rest:
        Additional table expressions
    distinct
        Only diff distinct rows not occurring in the calling table

    Returns
    -------
    Table
        The rows present in `self` that are not present in `tables`.

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> t1 = ibis.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = ibis.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> ibis.difference(t1, t2)
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    └───────┘
    """
    return table.difference(*rest, distinct=distinct) if rest else table


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
ifelse = _deferred(ir.BooleanValue.ifelse)
"""Construct a ternary conditional expression.

Parameters
----------
true_expr : ir.Value
    Expression to return if `self` evaluates to `True`
false_expr : ir.Value
    Expression to return if `self` evaluates to `False` or `NULL`

Returns
-------
Value : ir.Value
    The value of `true_expr` if `arg` is `True` else `false_expr`

Examples
--------
>>> import ibis
>>> ibis.options.interactive = True
>>> t = ibis.memtable({"is_person": [True, False, True, None]})
>>> ibis.ifelse(t.is_person, "yes", "no")
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Where(is_person, 'yes', 'no') ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ string                        │
├───────────────────────────────┤
│ yes                           │
│ no                            │
│ yes                           │
│ no                            │
└───────────────────────────────┘
"""
where = _deferred(ir.BooleanValue.ifelse)
"""Construct a ternary conditional expression.

Parameters
----------
true_expr : ir.Value
    Expression to return if `self` evaluates to `True`
false_expr : ir.Value
    Expression to return if `self` evaluates to `False` or `NULL`

Returns
-------
Value : ir.Value
    The value of `true_expr` if `arg` is `True` else `false_expr`

Examples
--------
>>> import ibis
>>> ibis.options.interactive = True
>>> t = ibis.memtable({"is_person": [True, False, True, None]})
>>> ibis.where(t.is_person, "yes", "no")
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Where(is_person, 'yes', 'no') ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ string                        │
├───────────────────────────────┤
│ yes                           │
│ no                            │
│ yes                           │
│ no                            │
└───────────────────────────────┘
"""
coalesce = _deferred(ir.Value.coalesce)
greatest = _deferred(ir.Value.greatest)
least = _deferred(ir.Value.least)
category_label = _deferred(ir.IntegerColumn.label)

aggregate = ir.Table.aggregate
cross_join = ir.Table.cross_join
join = ir.Table.join
asof_join = ir.Table.asof_join

_ = deferred = Deferred()
"""Deferred expression object.

Use this object to refer to a previous table expression in a chain of
expressions.

!!! note "`_` may conflict with other idioms in Python"

    See https://github.com/ibis-project/ibis/issues/4704 for details.

    Use `from ibis import deferred as <NAME>` to assign a different name to
    the deferred object builder.

Examples
--------
>>> from ibis import _
>>> t = ibis.table(dict(key="int", value="float"), name="t")
>>> expr = t.group_by(key=_.key - 1).agg(total=_.value.sum())
>>> expr.schema()
ibis.Schema {
  key    int64
  total  float64
}
"""
