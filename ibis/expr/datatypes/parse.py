from __future__ import annotations

import functools

import parsy
from public import public

import ibis.expr.datatypes.core as dt
from ibis.common.exceptions import IbisTypeError
from ibis.common.parsing import (
    COLON,
    COMMA,
    FIELD,
    LANGLE,
    LPAREN,
    NUMBER,
    PRECISION,
    RANGLE,
    RAW_NUMBER,
    RAW_STRING,
    RPAREN,
    SCALE,
    SEMICOLON,
    SINGLE_DIGIT,
    spaceless,
    spaceless_string,
)


@public
@functools.lru_cache(maxsize=100)
def parse(text: str) -> dt.DataType:
    """Parse a type from a [`str`][str] `text`.

    The default `maxsize` parameter for caching is chosen to cache the most
    commonly used types--there are about 30--along with some capacity for less
    common but repeatedly-used complex types.

    Parameters
    ----------
    text
        The type string to parse

    Examples
    --------
    Parse an array type from a string

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> dt.parse("array<int64>")
    Array(value_type=Int64(nullable=True), nullable=True)

    You can avoid parsing altogether by constructing objects directly

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> ty = dt.parse("array<int64>")
    >>> ty == dt.Array(dt.int64)
    True
    """

    srid = NUMBER
    geotype = spaceless_string("geography") | spaceless_string("geometry")

    @parsy.generate
    def srid_geotype():
        yield SEMICOLON
        sr = yield srid
        yield COLON
        gt = yield geotype
        return (gt, sr)

    @parsy.generate
    def geotype_part():
        yield COLON
        gt = yield geotype
        return (gt, None)

    @parsy.generate
    def srid_part():
        yield SEMICOLON
        sr = yield srid
        return (None, sr)

    def geotype_parser(name, type):
        name_parser = spaceless_string(name)
        geosubtype_parser = srid_geotype | geotype_part | srid_part

        @parsy.generate
        def parser():
            yield name_parser
            gt_sr = yield geosubtype_parser.optional()
            return type(*gt_sr) if gt_sr is not None else type()

        return parser

    primitive = (
        spaceless_string("boolean").result(dt.boolean)  # docprecated
        | spaceless_string("bool").result(dt.boolean)
        | spaceless_string("int8").result(dt.int8)
        | spaceless_string("int16").result(dt.int16)
        | spaceless_string("int32").result(dt.int32)
        | spaceless_string("int64").result(dt.int64)
        | spaceless_string("uint8").result(dt.uint8)
        | spaceless_string("uint16").result(dt.uint16)
        | spaceless_string("uint32").result(dt.uint32)
        | spaceless_string("uint64").result(dt.uint64)
        | spaceless_string("halffloat").result(dt.float16)  # docprecated
        | spaceless_string("double").result(dt.float64)  # docprecated
        | spaceless_string("float16").result(dt.float16)
        | spaceless_string("float32").result(dt.float32)
        | spaceless_string("float64").result(dt.float64)
        | spaceless_string("float").result(dt.float64)
        | spaceless_string("string").result(dt.string)
        | spaceless_string("binary").result(dt.binary)  # docprecated
        | spaceless_string("bytes").result(dt.binary)
        | spaceless_string("timestamp").result(dt.Timestamp())
        | spaceless_string("time").result(dt.time)
        | spaceless_string("date").result(dt.date)
        | spaceless_string("category").result(dt.category)
        | spaceless_string("geometry").result(dt.GeoSpatial(geotype='geometry'))
        | spaceless_string("geography").result(dt.GeoSpatial(geotype='geography'))
        | spaceless_string("null").result(dt.null)
        | geotype_parser("linestring", dt.LineString)
        | geotype_parser("polygon", dt.Polygon)
        | geotype_parser("point", dt.Point)
        | geotype_parser("multilinestring", dt.MultiLineString)
        | geotype_parser("multipolygon", dt.MultiPolygon)
        | geotype_parser("multipoint", dt.MultiPoint)
    )

    @parsy.generate
    def varchar_or_char():
        yield spaceless_string("varchar", "char").then(
            LPAREN.then(RAW_NUMBER).skip(RPAREN).optional()
        )
        return dt.String()

    @parsy.generate
    def decimal():
        yield spaceless_string("decimal")
        precision, scale = (
            yield LPAREN.then(
                parsy.seq(spaceless(PRECISION).skip(COMMA), spaceless(SCALE))
            )
            .skip(RPAREN)
            .optional()
        ) or (None, None)
        return dt.Decimal(precision=precision, scale=scale)

    parened_string = LPAREN.then(RAW_STRING).skip(RPAREN)
    timestamp_scale = SINGLE_DIGIT.map(int)

    timestamp_tz_args = (
        LPAREN.then(
            parsy.seq(timezone=RAW_STRING, scale=COMMA.then(timestamp_scale).optional())
        )
        .skip(RPAREN)
        .combine_dict(dict)
    )

    timestamp_no_tz_args = LPAREN.then(parsy.seq(scale=timestamp_scale).skip(RPAREN))

    timestamp = spaceless_string("timestamp").then(
        parsy.alt(timestamp_tz_args, timestamp_no_tz_args)
        .optional(default={})
        .combine_dict(dt.Timestamp)
    )

    @parsy.generate
    def angle_type():
        yield LANGLE
        value_type = yield ty
        yield RANGLE
        return value_type

    @parsy.generate
    def interval():
        yield spaceless_string("interval")
        value_type = yield angle_type.optional()
        unit = yield parened_string.optional()
        return dt.Interval(
            value_type=value_type,
            unit=unit if unit is not None else "s",
        )

    @parsy.generate
    def array():
        yield spaceless_string("array")
        value_type = yield angle_type
        return dt.Array(value_type)

    @parsy.generate
    def set():
        yield spaceless_string("set")
        value_type = yield angle_type
        return dt.Set(value_type)

    @parsy.generate
    def map():
        yield spaceless_string("map")
        yield LANGLE
        key_type = yield primitive
        yield COMMA
        value_type = yield ty
        yield RANGLE
        return dt.Map(key_type, value_type)

    spaceless_field = spaceless(FIELD)

    @parsy.generate
    def struct():
        yield spaceless_string("struct")
        yield LANGLE
        field_names_types = yield (
            parsy.seq(spaceless_field.skip(COLON), ty)
            .combine(lambda field, ty: (field, ty))
            .sep_by(COMMA)
        )
        yield RANGLE
        return dt.Struct.from_tuples(field_names_types)

    @parsy.generate
    def nullable():
        yield spaceless_string("!")
        parsed_ty = yield ty
        return parsed_ty(nullable=False)

    ty = (
        nullable
        | timestamp
        | primitive
        | decimal
        | varchar_or_char
        | interval
        | array
        | set
        | map
        | struct
        | spaceless_string("json").result(dt.json)
        | spaceless_string("uuid").result(dt.uuid)
        | spaceless_string("macaddr").result(dt.macaddr)
        | spaceless_string("inet").result(dt.inet)
        | spaceless_string("geography").result(dt.geography)
        | spaceless_string("geometry").result(dt.geometry)
        | spaceless_string("int").result(dt.int64)
        | spaceless_string("str").result(dt.string)
    )

    return ty.parse(text)


@dt.dtype.register(str)
def from_string(value: str) -> dt.DataType:
    try:
        return parse(value)
    except SyntaxError:
        raise IbisTypeError(f'{value!r} cannot be parsed as a datatype')
