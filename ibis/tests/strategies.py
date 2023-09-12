from __future__ import annotations

import warnings

import hypothesis as h
import hypothesis.extra.pandas as past
import hypothesis.extra.pytz as tzst
import hypothesis.strategies as st

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.temporal import IntervalUnit

# Strategies for generating ibis datatypes

_nullable = st.booleans()

null_dtype = st.just(dt.null)


def boolean_dtype(nullable=_nullable):
    return st.builds(dt.Boolean, nullable=nullable)


def signed_integer_dtypes(nullable=_nullable):
    return st.one_of(
        st.builds(dt.Int8, nullable=nullable),
        st.builds(dt.Int16, nullable=nullable),
        st.builds(dt.Int32, nullable=nullable),
        st.builds(dt.Int64, nullable=nullable),
    )


def unsigned_integer_dtypes(nullable=_nullable):
    return st.one_of(
        st.builds(dt.UInt8, nullable=nullable),
        st.builds(dt.UInt16, nullable=nullable),
        st.builds(dt.UInt32, nullable=nullable),
        st.builds(dt.UInt64, nullable=nullable),
    )


def integer_dtypes(nullable=_nullable):
    return st.one_of(
        signed_integer_dtypes(nullable=nullable),
        unsigned_integer_dtypes(nullable=nullable),
    )


def floating_dtypes(nullable=_nullable):
    return st.one_of(
        st.builds(dt.Float16, nullable=nullable),
        st.builds(dt.Float32, nullable=nullable),
        st.builds(dt.Float64, nullable=nullable),
    )


@st.composite
def decimal_dtypes(draw, nullable=_nullable):
    number = st.integers(min_value=1, max_value=38)
    precision, scale = draw(number), draw(number)
    h.assume(precision >= scale)
    return dt.Decimal(precision, scale, nullable=draw(nullable))


def numeric_dtypes(nullable=_nullable):
    return st.one_of(
        integer_dtypes(nullable=nullable),
        floating_dtypes(nullable=nullable),
        decimal_dtypes(nullable=nullable),
    )


def string_dtype(nullable=_nullable):
    return st.builds(dt.String, nullable=nullable)


def binary_dtype(nullable=_nullable):
    return st.builds(dt.Binary, nullable=nullable)


def json_dtype(nullable=_nullable):
    return st.builds(dt.JSON, nullable=nullable)


def inet_dtype(nullable=_nullable):
    return st.builds(dt.INET, nullable=nullable)


def macaddr_dtype(nullable=_nullable):
    return st.builds(dt.MACADDR, nullable=nullable)


def uuid_dtype(nullable=_nullable):
    return st.builds(dt.UUID, nullable=nullable)


def string_like_dtypes(nullable=_nullable):
    return st.one_of(
        string_dtype(nullable=nullable),
        binary_dtype(nullable=nullable),
        json_dtype(nullable=nullable),
        inet_dtype(nullable=nullable),
        macaddr_dtype(nullable=nullable),
        uuid_dtype(nullable=nullable),
    )


def date_dtype(nullable=_nullable):
    return st.builds(dt.Date, nullable=nullable)


def time_dtype(nullable=_nullable):
    return st.builds(dt.Time, nullable=nullable)


_timezone = st.none() | tzst.timezones().map(str)
_interval = st.sampled_from(list(IntervalUnit))
_timestamp_scale = st.none() | st.integers(min_value=0, max_value=9)


def timestamp_dtype(scale=_timestamp_scale, timezone=_timezone, nullable=_nullable):
    return st.builds(dt.Timestamp, scale=scale, timezone=timezone, nullable=nullable)


def interval_dtype(interval=_interval, nullable=_nullable):
    return st.builds(dt.Interval, unit=interval, nullable=nullable)


def temporal_dtypes(timezone=_timezone, interval=_interval, nullable=_nullable):
    return st.one_of(
        date_dtype(nullable=nullable),
        time_dtype(nullable=nullable),
        timestamp_dtype(timezone=timezone, nullable=nullable),
    )


def primitive_dtypes(nullable=_nullable):
    return st.one_of(
        null_dtype,
        boolean_dtype(nullable=nullable),
        integer_dtypes(nullable=nullable),
        floating_dtypes(nullable=nullable),
        date_dtype(nullable=nullable),
        time_dtype(nullable=nullable),
    )


_item_strategy = primitive_dtypes()


def array_dtypes(value_type=_item_strategy, nullable=_nullable):
    return st.builds(dt.Array, value_type=value_type, nullable=nullable)


def map_dtypes(key_type=_item_strategy, value_type=_item_strategy, nullable=_nullable):
    return st.builds(
        dt.Map, key_type=key_type, value_type=value_type, nullable=nullable
    )


_any_text = st.text()


@st.composite
def struct_dtypes(
    draw,
    types=_item_strategy,
    names=_any_text,
    num_fields=st.integers(min_value=0, max_value=20),  # noqa: B008
    nullable=_nullable,
):
    num_fields = draw(num_fields)
    names = draw(st.lists(names, min_size=num_fields, max_size=num_fields))
    types = draw(st.lists(types, min_size=num_fields, max_size=num_fields))
    fields = dict(zip(names, types))
    return dt.Struct(fields, nullable=draw(nullable))


def geometry_dtypes(nullable=_nullable):
    return st.builds(dt.GeoSpatial, geotype=st.just("geometry"), nullable=nullable)


def geography_dtypes(nullable=_nullable):
    return st.builds(dt.GeoSpatial, geotype=st.just("geography"), nullable=nullable)


def geospatial_dtypes(nullable=_nullable):
    return st.one_of(
        st.builds(dt.Point, nullable=nullable),
        st.builds(dt.LineString, nullable=nullable),
        st.builds(dt.Polygon, nullable=nullable),
        st.builds(dt.MultiPoint, nullable=nullable),
        st.builds(dt.MultiLineString, nullable=nullable),
        st.builds(dt.MultiPolygon, nullable=nullable),
        geometry_dtypes(nullable=nullable),
        geography_dtypes(nullable=nullable),
    )


def variadic_dtypes(nullable=_nullable):
    return st.one_of(
        string_dtype(nullable=nullable),
        binary_dtype(nullable=nullable),
        json_dtype(nullable=nullable),
        array_dtypes(nullable=nullable),
        map_dtypes(nullable=nullable),
    )


def all_dtypes(nullable=_nullable):
    recursive = st.deferred(
        lambda: (
            primitive_dtypes(nullable=nullable)
            | string_like_dtypes(nullable=nullable)
            | temporal_dtypes(nullable=nullable)
            | interval_dtype(nullable=nullable)
            | geospatial_dtypes(nullable=nullable)
            | variadic_dtypes(nullable=nullable)
            | struct_dtypes(nullable=nullable)
            | array_dtypes(recursive, nullable=nullable)
            | map_dtypes(recursive, recursive, nullable=nullable)
            | struct_dtypes(recursive, nullable=nullable)
        )
    )
    return recursive


# Strategies for generating schema


@st.composite
def schema(draw, item_strategy=_item_strategy, max_size=20):
    num_fields = draw(st.integers(min_value=0, max_value=max_size))
    names = draw(
        st.lists(st.text(), min_size=num_fields, max_size=num_fields, unique=True)
    )
    types = draw(st.lists(item_strategy, min_size=num_fields, max_size=num_fields))
    fields = dict(zip(names, types))
    return sch.Schema(fields)


all_schema = schema(all_dtypes)


# Strategies for generating in memory tables holding data


@st.composite
def memtable(draw, schema=schema(primitive_dtypes)):  # noqa: B008
    schema = draw(schema)

    columns = [past.column(name, dtype=dtype) for name, dtype in schema.to_pandas()]
    dataframe = past.data_frames(columns=columns)

    with warnings.catch_warnings():
        # TODO(cpcloud): pandas 2.1.0 junk
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = draw(dataframe)
    return ibis.memtable(df)
