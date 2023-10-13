from __future__ import annotations

import collections
import datetime
import decimal
import enum
import ipaddress
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import toolz
from public import public

import ibis.expr.datatypes as dt
from ibis.common.collections import frozendict
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import IbisTypeError, InputTypeError
from ibis.common.numeric import normalize_decimal
from ibis.common.temporal import (
    IntervalUnit,
    normalize_datetime,
    normalize_timedelta,
    normalize_timezone,
)
from ibis.expr.datatypes.cast import highest_precedence


@lazy_singledispatch
def infer(value: Any) -> dt.DataType:
    """Infer the corresponding ibis dtype for a python object."""
    raise InputTypeError(
        f"Unable to infer datatype of value {value!r} with type {type(value)}"
    )


# TODO(kszucs): support NamedTuples and dataclasses instead of OrderedDict
# which should trigger infer_map instead
@infer.register(collections.OrderedDict)
def infer_struct(value: Mapping[str, Any]) -> dt.Struct:
    """Infer the [`Struct`](./datatypes.qmd#ibis.expr.datatypes.Struct) type of `value`."""
    if not value:
        raise TypeError("Empty struct type not supported")
    fields = {name: infer(val) for name, val in value.items()}
    return dt.Struct(fields)


@infer.register(collections.abc.Mapping)
def infer_map(value: Mapping[Any, Any]) -> dt.Map:
    """Infer the [`Map`](./datatypes.qmd#ibis.expr.datatypes.Map) type of `value`."""
    if not value:
        return dt.Map(dt.null, dt.null)
    try:
        return dt.Map(
            highest_precedence(map(infer, value.keys())),
            highest_precedence(map(infer, value.values())),
        )
    except IbisTypeError:
        return dt.Struct(toolz.valmap(infer, value, factory=type(value)))


@infer.register((list, tuple, set, frozenset))
def infer_list(values: Sequence[Any]) -> dt.Array:
    """Infer the [`Array`](./datatypes.qmd#ibis.expr.datatypes.Array) type of `value`."""
    if not values:
        return dt.Array(dt.null)
    return dt.Array(highest_precedence(map(infer, values)))


@infer.register(datetime.time)
def infer_time(value: datetime.time) -> dt.Time:
    return dt.time


@infer.register(datetime.date)
def infer_date(value: datetime.date) -> dt.Date:
    return dt.date


@infer.register(datetime.datetime)
def infer_timestamp(value: datetime.datetime) -> dt.Timestamp:
    if value.tzinfo:
        return dt.Timestamp(timezone=str(value.tzinfo))
    else:
        return dt.timestamp


@infer.register(datetime.timedelta)
def infer_interval(value: datetime.timedelta) -> dt.Interval:
    # datetime.timedelta only stores days, seconds, and microseconds internally
    if value.days:
        if value.seconds or value.microseconds:
            raise ValueError(
                "Unable to infer interval type from mixed units, "
                "use ibis.interval(timedelta) instead"
            )
        else:
            return dt.Interval(IntervalUnit.DAY)
    elif value.seconds:
        if value.microseconds:
            return dt.Interval(IntervalUnit.MICROSECOND)
        else:
            return dt.Interval(IntervalUnit.SECOND)
    elif value.microseconds:
        return dt.Interval(IntervalUnit.MICROSECOND)
    else:
        raise ValueError("Unable to infer interval type from zero value")


@infer.register(str)
def infer_string(value: str) -> dt.String:
    return dt.string


@infer.register(bytes)
def infer_bytes(value: bytes) -> dt.Binary:
    return dt.binary


@infer.register(float)
def infer_floating(value: float) -> dt.Float64:
    return dt.float64


@infer.register(int)
def infer_integer(value: int, prefer_unsigned: bool = False) -> dt.Integer:
    types = (dt.uint8, dt.uint16, dt.uint32, dt.uint64) if prefer_unsigned else ()
    types += (dt.int8, dt.int16, dt.int32, dt.int64)
    for dtype in types:
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype
    return dt.uint64 if prefer_unsigned else dt.int64


@infer.register(enum.Enum)
def infer_enum(_: enum.Enum) -> dt.String:
    return dt.string


@infer.register(decimal.Decimal)
def infer_decimal(value: decimal.Decimal) -> dt.Decimal:
    """Infer the [`Decimal`](./datatypes.qmd#ibis.expr.datatypes.Decimal) type of `value`."""
    return dt.decimal


@infer.register(bool)
def infer_boolean(value: bool) -> dt.Boolean:
    return dt.boolean


@infer.register((type(None), dt.Null))
def infer_null(value: dt.Null | None) -> dt.Null:
    return dt.null


@infer.register((ipaddress.IPv4Address, ipaddress.IPv6Address))
def infer_ipaddr(
    _: ipaddress.IPv4Address | ipaddress.IPv6Address | None,
) -> dt.INET:
    return dt.inet


@infer.register("numpy.generic")
def infer_numpy_scalar(value):
    from ibis.formats.numpy import NumpyType

    return NumpyType.to_ibis(value.dtype)


@infer.register("pandas.Timestamp")
def infer_pandas_timestamp(value):
    if value.tz is not None:
        return dt.Timestamp(timezone=str(value.tz))
    else:
        return dt.timestamp


@infer.register("pandas.Timedelta")
def infer_interval_pandas(value) -> dt.Interval:
    # pandas Timedelta has more granularity
    units = {"D": "d", "H": "h", "T": "m", "S": "s", "L": "ms", "U": "us", "N": "ns"}
    unit = units[value.resolution_string]
    return dt.Interval(unit)


@infer.register("numpy.ndarray")
@infer.register("pandas.Series")
def infer_numpy_array(value):
    from ibis.formats.numpy import NumpyType
    from ibis.formats.pyarrow import PyArrowData

    if value.dtype.kind == "O":
        value_dtype = PyArrowData.infer_column(value)
    else:
        value_dtype = NumpyType.to_ibis(value.dtype)

    return dt.Array(value_dtype)


@infer.register("shapely.geometry.Point")
def infer_shapely_point(value) -> dt.Point:
    return dt.point


@infer.register("shapely.geometry.LineString")
def infer_shapely_linestring(value) -> dt.LineString:
    return dt.linestring


@infer.register("shapely.geometry.Polygon")
def infer_shapely_polygon(value) -> dt.Polygon:
    return dt.polygon


@infer.register("shapely.geometry.MultiLineString")
def infer_shapely_multilinestring(value) -> dt.MultiLineString:
    return dt.multilinestring


@infer.register("shapely.geometry.MultiPoint")
def infer_shapely_multipoint(value) -> dt.MultiPoint:
    return dt.multipoint


@infer.register("shapely.geometry.MultiPolygon")
def infer_shapely_multipolygon(value) -> dt.MultiPolygon:
    return dt.multipolygon


# lock the dispatcher to prevent adding new implementations
del infer.register


@public
class _WellKnownText:
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


# TODO(kszucs): should raise ValueError instead of TypeError
def normalize(typ, value):
    """Ensure that the Python type underlying a literal resolves to a single type."""

    dtype = dt.dtype(typ)
    if value is None:
        if not dtype.nullable:
            raise TypeError(f"Cannot convert `None` to non-nullable type {typ!r}")
        return None

    if dtype.is_boolean():
        try:
            return bool(value)
        except ValueError:
            raise TypeError(f"Unable to normalize {value!r} to {dtype!r}")
    elif dtype.is_integer():
        try:
            value = int(value)
        except ValueError:
            raise TypeError(f"Unable to normalize {value!r} to {dtype!r}")
        if value not in dtype.bounds:
            raise TypeError(
                f"Value {value} is out of bounds for type {dtype!r} "
                f"(bounds: {dtype.bounds})"
            )
        else:
            return value
    elif dtype.is_floating():
        try:
            return float(value)
        except ValueError:
            raise TypeError(f"Unable to normalize {value!r} to {dtype!r}")
    elif dtype.is_json():
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError:
                raise TypeError(f"Invalid JSON string: {value!r}")
            else:
                return value
        else:
            return json.dumps(value)
    elif dtype.is_binary():
        return bytes(value)
    elif dtype.is_string() or dtype.is_macaddr() or dtype.is_inet():
        return str(value)
    elif dtype.is_decimal():
        return normalize_decimal(value, precision=dtype.precision, scale=dtype.scale)
    elif dtype.is_uuid():
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
    elif dtype.is_array():
        return tuple(normalize(dtype.value_type, item) for item in value)
    elif dtype.is_map():
        return frozendict({k: normalize(dtype.value_type, v) for k, v in value.items()})
    elif dtype.is_struct():
        if not isinstance(value, Mapping):
            raise TypeError(f"Unable to normalize {dtype} from non-mapping {value!r}")
        if missing_keys := (dtype.keys() - value.keys()):
            raise TypeError(
                f"Unable to normalize {value!r} to {dtype} because of missing keys {missing_keys!r}"
            )
        return frozendict({k: normalize(t, value[k]) for k, t in dtype.items()})
    elif dtype.is_geospatial():
        if isinstance(value, (tuple, list)):
            if dtype.is_point():
                return tuple(normalize(dt.float64, item) for item in value)
            elif dtype.is_linestring() or dtype.is_multipoint():
                return tuple(normalize(dt.point, item) for item in value)
            elif dtype.is_polygon() or dtype.is_multilinestring():
                return tuple(normalize(dt.linestring, item) for item in value)
            elif dtype.is_multipolygon():
                return tuple(normalize(dt.polygon, item) for item in value)
        elif isinstance(value, _WellKnownText):
            return value
        return _WellKnownText(value.wkt)
    elif dtype.is_date():
        return normalize_datetime(value).date()
    elif dtype.is_time():
        return normalize_datetime(value).time()
    elif dtype.is_timestamp():
        value = normalize_datetime(value)
        tzinfo = normalize_timezone(dtype.timezone)
        if tzinfo is None:
            return value
        elif value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=tzinfo)
        else:
            return value.astimezone(tzinfo)
    elif dtype.is_interval():
        return normalize_timedelta(value, dtype.unit)
    else:
        raise TypeError(f"Unable to normalize {value!r} to {dtype!r}")


public(infer=infer, normalize=normalize)
