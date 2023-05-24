from __future__ import annotations

import hypothesis as h
import hypothesis.extra.pandas as past
import hypothesis.extra.pytz as tzst
import hypothesis.strategies as st

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.temporal import IntervalUnit

# pyarrow also has hypothesis strategies various pyarrow objects

# Strategies for generating datatypes

nullable = st.booleans()

null_dtype = st.just(dt.null)
boolean_dtype = st.builds(dt.Boolean, nullable=nullable)

int8_dtype = st.builds(dt.Int8, nullable=nullable)
int16_dtype = st.builds(dt.Int16, nullable=nullable)
int32_dtype = st.builds(dt.Int32, nullable=nullable)
int64_dtype = st.builds(dt.Int64, nullable=nullable)
uint8_dtype = st.builds(dt.UInt8, nullable=nullable)
uint16_dtype = st.builds(dt.UInt16, nullable=nullable)
uint32_dtype = st.builds(dt.UInt32, nullable=nullable)
uint64_dtype = st.builds(dt.UInt64, nullable=nullable)
float16_dtype = st.builds(dt.Float16, nullable=nullable)
float32_dtype = st.builds(dt.Float32, nullable=nullable)
float64_dtype = st.builds(dt.Float64, nullable=nullable)


@st.composite
def decimal_dtype(draw):
    number = st.integers(min_value=1, max_value=38)
    precision, scale = draw(number), draw(number)
    h.assume(precision >= scale)
    return dt.Decimal(precision, scale, nullable=draw(nullable))


signed_integer_dtypes = st.one_of(int8_dtype, int16_dtype, int32_dtype, int64_dtype)
unsigned_integer_dtypes = st.one_of(
    uint8_dtype, uint16_dtype, uint32_dtype, uint64_dtype
)
integer_dtypes = st.one_of(signed_integer_dtypes, unsigned_integer_dtypes)
floating_dtypes = st.one_of(float16_dtype, float32_dtype, float64_dtype)
numeric_dtypes = st.one_of(integer_dtypes, floating_dtypes, decimal_dtype())

date_dtype = st.builds(dt.Date, nullable=nullable)
time_dtype = st.builds(dt.Time, nullable=nullable)
timestamp_dtype = st.builds(
    dt.Timestamp, timezone=st.none() | tzst.timezones().map(str), nullable=nullable
)
interval_unit = st.sampled_from(list(IntervalUnit))
interval_dtype = st.builds(dt.Interval, unit=interval_unit, nullable=nullable)
temporal_dtypes = st.one_of(
    date_dtype,
    time_dtype,
    timestamp_dtype,
    # interval_dtype
)

primitive_dtypes = st.one_of(
    null_dtype,
    boolean_dtype,
    integer_dtypes,
    floating_dtypes,
    date_dtype,
    time_dtype,
)


def array_dtypes(item_strategy=primitive_dtypes):
    return st.builds(dt.Array, value_type=item_strategy, nullable=nullable)


def map_dtypes(key_strategy=primitive_dtypes, value_strategy=primitive_dtypes):
    return st.builds(
        dt.Map, key_type=key_strategy, value_type=value_strategy, nullable=nullable
    )


@st.composite
def struct_dtypes(
    draw,
    item_strategy=primitive_dtypes,
    num_fields=st.integers(min_value=0, max_value=20),  # noqa: B008
):
    num_fields = draw(num_fields)
    names = draw(st.lists(st.text(), min_size=num_fields, max_size=num_fields))
    types = draw(st.lists(item_strategy, min_size=num_fields, max_size=num_fields))
    fields = dict(zip(names, types))
    return dt.Struct(fields, nullable=draw(nullable))


point_dtype = st.builds(dt.Point, nullable=nullable)
linestring_dtype = st.builds(dt.LineString, nullable=nullable)
polygon_dtype = st.builds(dt.Polygon, nullable=nullable)
multipoint_dtype = st.builds(dt.MultiPoint, nullable=nullable)
multilinestring_dtype = st.builds(dt.MultiLineString, nullable=nullable)
multipolygon_dtype = st.builds(dt.MultiPolygon, nullable=nullable)
geometry_dtype = st.builds(
    dt.GeoSpatial, geotype=st.just("geometry"), nullable=nullable
)
geography_dtype = st.builds(
    dt.GeoSpatial, geotype=st.just("geography"), nullable=nullable
)
geospatial_dtypes = st.one_of(
    point_dtype,
    linestring_dtype,
    polygon_dtype,
    multipoint_dtype,
    multilinestring_dtype,
    multipolygon_dtype,
    geometry_dtype,
    geography_dtype,
)

string_dtype = st.builds(dt.String, nullable=nullable)
binary_dtype = st.builds(dt.Binary, nullable=nullable)
json_dtype = st.builds(dt.JSON, nullable=nullable)
inet_dtype = st.builds(dt.INET, nullable=nullable)
macaddr_dtype = st.builds(dt.MACADDR, nullable=nullable)
uuid_dtype = st.builds(dt.UUID, nullable=nullable)

variadic_dtypes = st.one_of(
    string_dtype,
    binary_dtype,
    json_dtype,
    inet_dtype,
    macaddr_dtype,
    array_dtypes(),
    map_dtypes(),
)

all_dtypes = st.deferred(
    lambda: (
        primitive_dtypes
        | interval_dtype
        | uuid_dtype
        | geospatial_dtypes
        | variadic_dtypes
        | struct_dtypes()
        | array_dtypes(all_dtypes)
        | map_dtypes(all_dtypes, all_dtypes)
        | struct_dtypes(all_dtypes)
    )
)


# Strategies for generating schema


@st.composite
def schema(draw, item_strategy=primitive_dtypes, max_size=20):
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

    df = draw(dataframe)
    return ibis.memtable(df)
