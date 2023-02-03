from __future__ import annotations

import collections
import datetime
import decimal
import enum
import ipaddress
import uuid
from functools import partial
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Sequence

import dateutil.parser
import numpy as np
import pytz
import toolz
from public import public

import ibis.expr.datatypes.core as dt
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import IbisTypeError, InputTypeError
from ibis.expr.datatypes.cast import highest_precedence
from ibis.util import frozendict

if TYPE_CHECKING:
    import pandas as pd


@lazy_singledispatch
def infer(value: Any) -> dt.DataType:
    """Infer the corresponding ibis dtype for a python object."""
    raise InputTypeError(value)


# TODO(kszucs): support NamedTuples and dataclasses instead of OrderedDict
# which should trigger infer_map instead
@infer.register(collections.OrderedDict)
def infer_struct(value: Mapping[str, Any]) -> dt.Struct:
    """Infer the [`Struct`][ibis.expr.datatypes.Struct] type of `value`."""
    if not value:
        raise TypeError('Empty struct type not supported')
    fields = {name: infer(val) for name, val in value.items()}
    return dt.Struct(fields)


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
        return dt.Struct(toolz.valmap(infer, value, factory=type(value)))


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
    import pandas as pd

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
    types = (dt.uint8, dt.uint16, dt.uint32, dt.uint64) if prefer_unsigned else ()
    types += (dt.int8, dt.int16, dt.int32, dt.int64)
    for dtype in types:
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype
    return dt.uint64 if prefer_unsigned else dt.int64


@infer.register(enum.Enum)
def infer_enum(_: enum.Enum) -> dt.String:
    return dt.string


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


@public
class _WellKnownText(NamedTuple):
    text: str

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


def _infer_object_array_dtype(x):
    import pandas as pd
    from pandas.api.types import infer_dtype

    classifier = infer_dtype(x, skipna=True)
    if classifier == "mixed":
        value = x.iloc[0] if isinstance(x, pd.Series) else x[0]
        if isinstance(value, (np.ndarray, pd.Series, Sequence, Mapping)):
            return infer(value)
        else:
            return dt.binary
    else:
        return {
            'string': dt.string,
            'bytes': dt.string,
            'floating': dt.float64,
            'integer': dt.int64,
            'mixed-integer': dt.binary,
            'mixed-integer-float': dt.float64,
            'decimal': dt.float64,
            'complex': dt.binary,
            'categorical': dt.category,
            'boolean': dt.boolean,
            'datetime64': dt.timestamp,
            'datetime': dt.timestamp,
            'date': dt.date,
            'timedelta64': dt.interval,
            'timedelta': dt.interval,
            'time': dt.time,
            'period': dt.binary,
            'empty': dt.binary,
            'unicode': dt.string,
        }[classifier]


@infer.register(np.generic)
def infer_numpy_scalar(value):
    return dt.dtype(value.dtype)


@infer.register(np.ndarray)
def infer_numpy_array(value):
    np_dtype = value.dtype
    if np_dtype.type == np.object_:
        return dt.Array(_infer_object_array_dtype(value))
    elif np_dtype.type == np.str_:
        return dt.Array(dt.string)
    return dt.Array(dt.dtype(np_dtype))


@infer.register("pandas.Series")
def infer_pandas_series(value):
    if value.dtype == np.object_:
        value_dtype = _infer_object_array_dtype(value)
    else:
        value_dtype = dt.dtype(value.dtype)

    return dt.Array(value_dtype)


@infer.register("pandas.Timestamp")
def infer_pandas_timestamp(value):
    if value.tz is not None:
        return dt.Timestamp(timezone=str(value.tz))
    else:
        return dt.timestamp


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


def normalize(typ, value):
    """Ensure that the Python type underlying a literal resolves to a single type."""
    if value is None:
        if not typ.nullable:
            raise TypeError("Cannot convert `None` to non-nullable type {typ!r}")
        return None

    if typ.is_boolean():
        return bool(value)
    elif typ.is_integer():
        return int(value)
    elif typ.is_floating():
        return float(value)
    elif typ.is_string() and not typ.is_json():
        return str(value)
    elif typ.is_decimal():
        out = decimal.Decimal(value)
        if isinstance(value, int):
            return out.scaleb(-typ.scale)
        return out
    elif typ.is_uuid():
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
    elif typ.is_array():
        return tuple(normalize(typ.value_type, item) for item in value)
    elif typ.is_set():
        return frozenset(normalize(typ.value_type, item) for item in value)
    elif typ.is_map():
        return frozendict({k: normalize(typ.value_type, v) for k, v in value.items()})
    elif typ.is_struct():
        return frozendict(
            {k: normalize(typ[k], v) for k, v in value.items() if k in typ.fields}
        )
    elif typ.is_geospatial():
        if isinstance(value, (tuple, list)):
            if typ.is_point():
                return tuple(normalize(dt.float64, item) for item in value)
            elif typ.is_linestring() or typ.is_multipoint():
                return tuple(normalize(dt.point, item) for item in value)
            elif typ.is_polygon() or typ.is_multilinestring():
                return tuple(normalize(dt.linestring, item) for item in value)
            elif typ.is_multipolygon():
                return tuple(normalize(dt.polygon, item) for item in value)
        return _WellKnownText(value.wkt)
    elif (is_timestamp := typ.is_timestamp()) or typ.is_date():
        import pandas as pd

        converter = (
            partial(_convert_timezone, tz=typ.timezone)
            if is_timestamp
            else methodcaller("date")
        )

        if isinstance(value, str):
            return converter(dateutil.parser.parse(value))
        elif isinstance(value, pd.Timestamp):
            return converter(value.to_pydatetime())
        elif isinstance(value, datetime.datetime):
            return converter(value)
        elif isinstance(value, datetime.date):
            return converter(
                datetime.datetime(year=value.year, month=value.month, day=value.day)
            )
        elif isinstance(value, np.datetime64):
            original_value = value
            raw_value = value.item()
            if isinstance(raw_value, int):
                unit, _ = np.datetime_data(original_value)
                return converter(pd.Timestamp(raw_value, unit=unit).to_pydatetime())
            elif isinstance(raw_value, datetime.datetime):
                return converter(raw_value)
            elif is_timestamp:
                return datetime.datetime(raw_value.year, raw_value.month, raw_value.day)
            else:
                return raw_value

        raise TypeError(
            f"Unsupported {'timestamp' if is_timestamp else 'date'} literal type: {type(value)}"
        )
    else:
        return value


def _convert_timezone(value: datetime.datetime, *, tz: str | None) -> datetime.datetime:
    return value if tz is None else value.astimezone(tz=pytz.timezone(tz))


public(infer=infer, normalize=normalize)
