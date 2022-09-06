from __future__ import annotations

import collections
import datetime
import decimal
import enum
import ipaddress
import uuid
from typing import (
    AbstractSet,
    Any,
    Mapping,
    NamedTuple,
    Sequence,
    SupportsFloat,
)

import numpy as np
import pandas as pd
import toolz
from multipledispatch import Dispatcher
from public import public

import ibis.expr.datatypes.core as dt
from ibis.common.exceptions import IbisTypeError, InputTypeError
from ibis.expr.datatypes.cast import highest_precedence
from ibis.util import frozendict

try:
    import shapely.geometry as geo

    IS_SHAPELY_AVAILABLE = True
except ImportError:
    IS_SHAPELY_AVAILABLE = False


infer = Dispatcher("infer")
normalize = Dispatcher(
    "normalize",
    doc="""\
Ensure that the Python type underlying an
:class:`~ibis.expr.operations.generic.Literal` resolves to a single acceptable
type regardless of the input value.

Parameters
----------
typ : DataType
value :

Returns
-------
value
    the input ``value`` normalized to the expected type
""",
)


@infer.register(object)
def infer_dtype_default(value: Any) -> dt.DataType:
    """Default implementation of :func:`~ibis.expr.datatypes.infer`."""
    raise InputTypeError(value)


@infer.register(collections.OrderedDict)
def infer_struct(value: Mapping[str, Any]) -> dt.Struct:
    """Infer the [`Struct`][ibis.expr.datatypes.Struct] type of `value`."""
    if not value:
        raise TypeError('Empty struct type not supported')
    return dt.Struct(list(value.keys()), list(map(infer, value.values())))


@infer.register(collections.abc.Mapping)
def infer_map(value: Mapping[Any, Any]) -> dt.Map:
    """Infer the [`Map`][ibis.expr.datatypes.Map] type of `value`."""
    if not value:
        return dt.Map(dt.null, dt.null)
    try:
        return dt.Map(
            highest_precedence(map(infer, value.keys())),
            highest_precedence(map(infer, value.values())),
        )
    except IbisTypeError:
        return dt.Struct.from_dict(
            toolz.valmap(infer, value, factory=type(value))
        )


@infer.register((list, tuple))
def infer_list(values: Sequence[Any]) -> dt.Array:
    """Infer the [`Array`][ibis.expr.datatypes.Array] type of `value`."""
    if not values:
        return dt.Array(dt.null)
    return dt.Array(highest_precedence(map(infer, values)))


@infer.register((set, frozenset))
def infer_set(values: set) -> dt.Set:
    """Infer the [`Set`][ibis.expr.datatypes.Set] type of `value`."""
    if not values:
        return dt.Set(dt.null)
    return dt.Set(highest_precedence(map(infer, values)))


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


def _get_timedelta_units(
    timedelta: datetime.timedelta | pd.Timedelta,
) -> list[str]:
    # pandas Timedelta has more granularity
    if isinstance(timedelta, pd.Timedelta):
        unit_fields = timedelta.components._fields
        base_object = timedelta.components
    # datetime.timedelta only stores days, seconds, and microseconds internally
    else:
        unit_fields = ['days', 'seconds', 'microseconds']
        base_object = timedelta

    return [field for field in unit_fields if getattr(base_object, field) > 0]


@infer.register(datetime.timedelta)
def infer_interval(value: datetime.timedelta) -> dt.Interval:
    time_units = _get_timedelta_units(value)
    # we can attempt a conversion in the simplest case, i.e. there is exactly
    # one unit (e.g. pd.Timedelta('2 days') vs. pd.Timedelta('2 days 3 hours')
    if len(time_units) == 1:
        unit = time_units[0]
        return dt.Interval(unit)
    else:
        return dt.interval


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
    types = (
        (dt.uint8, dt.uint16, dt.uint32, dt.uint64) if prefer_unsigned else ()
    )
    types += (dt.int8, dt.int16, dt.int32, dt.int64)
    for dtype in types:
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype
    return dt.uint64 if prefer_unsigned else dt.int64


@infer.register(enum.Enum)
def infer_enum(value: enum.Enum) -> dt.Enum:
    return dt.Enum(infer(value.name), infer(value.value))


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


@normalize.register(dt.DataType, object)
def normalize_default(typ: dt.DataType, value: object) -> object:
    return value


@normalize.register(dt.Integer, (int, float, np.integer, np.floating))
def normalize_int(typ: dt.Integer, value: float) -> float:
    return int(value)


@normalize.register(
    dt.Floating, (int, float, np.integer, np.floating, SupportsFloat)
)
def normalize_float(typ: dt.Floating, value: float) -> float:
    return float(value)


@normalize.register(dt.UUID, str)
def normalize_str_to_uuid(typ: dt.UUID, value: str) -> uuid.UUID:
    return uuid.UUID(value)


@normalize.register(dt.String, uuid.UUID)
def normalize_uuid_to_str(typ: dt.String, value: uuid.UUID) -> str:
    return str(value)


@normalize.register(dt.Decimal, int)
def normalize_int_to_decimal(typ: dt.Decimal, value: int) -> decimal.Decimal:
    return decimal.Decimal(value).scaleb(-typ.scale)


@normalize.register(dt.Array, (tuple, list, np.ndarray))
def normalize_array_to_tuple(typ: dt.Array, values: Sequence) -> tuple:
    return tuple(normalize(typ.value_type, item) for item in values)


@normalize.register(dt.Set, (set, frozenset))
def normalize_set_to_frozenset(typ: dt.Set, values: AbstractSet) -> frozenset:
    return frozenset(normalize(typ.value_type, item) for item in values)


@normalize.register(dt.Map, dict)
def normalize_map_to_frozendict(
    typ: dt.Map, values: Mapping
) -> decimal.Decimal:
    values = {k: normalize(typ.value_type, v) for k, v in values.items()}
    return frozendict(values)


@normalize.register(dt.Struct, dict)
def normalize_struct_to_frozendict(
    typ: dt.Struct, values: Mapping
) -> decimal.Decimal:
    value_types = typ.pairs
    values = {
        k: normalize(typ[k], v) for k, v in values.items() if k in value_types
    }
    return frozendict(values)


@normalize.register(dt.Point, (tuple, list))
def normalize_point_to_tuple(typ: dt.Point, values: Sequence) -> tuple:
    return tuple(normalize(dt.float64, item) for item in values)


@normalize.register((dt.LineString, dt.MultiPoint), (tuple, list))
def normalize_linestring_to_tuple(
    typ: dt.LineString, values: Sequence
) -> tuple:
    return tuple(normalize(dt.point, item) for item in values)


@normalize.register((dt.Polygon, dt.MultiLineString), (tuple, list))
def normalize_polygon_to_tuple(typ: dt.Polygon, values: Sequence) -> tuple:
    return tuple(normalize(dt.linestring, item) for item in values)


@normalize.register(dt.MultiPolygon, (tuple, list))
def normalize_multipolygon_to_tuple(
    typ: dt.MultiPolygon, values: Sequence
) -> tuple:
    return tuple(normalize(dt.polygon, item) for item in values)


@public
class _WellKnownText(NamedTuple):
    text: str

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


if IS_SHAPELY_AVAILABLE:

    @infer.register(geo.Point)
    def infer_shapely_point(value: geo.Point) -> dt.Point:
        return dt.point

    @infer.register(geo.LineString)
    def infer_shapely_linestring(value: geo.LineString) -> dt.LineString:
        return dt.linestring

    @infer.register(geo.Polygon)
    def infer_shapely_polygon(value: geo.Polygon) -> dt.Polygon:
        return dt.polygon

    @infer.register(geo.MultiLineString)
    def infer_shapely_multilinestring(
        value: geo.MultiLineString,
    ) -> dt.MultiLineString:
        return dt.multilinestring

    @infer.register(geo.MultiPoint)
    def infer_shapely_multipoint(value: geo.MultiPoint) -> dt.MultiPoint:
        return dt.multipoint

    @infer.register(geo.MultiPolygon)
    def infer_shapely_multipolygon(value: geo.MultiPolygon) -> dt.MultiPolygon:
        return dt.multipolygon

    @normalize.register(dt.GeoSpatial, geo.base.BaseGeometry)
    def normalize_geom_to_wkt(
        typ: dt.GeoSpatial, base_geom: geo.base.BaseGeometry
    ) -> _WellKnownText:
        return _WellKnownText(base_geom.wkt)


public(infer=infer, normalize=normalize)
