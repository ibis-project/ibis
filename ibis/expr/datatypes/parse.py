from __future__ import annotations

import ast
import functools
import re
from operator import methodcaller

import parsy
from public import public

import ibis.expr.datatypes.core as dt

_STRING_REGEX = (
    """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""
)

SPACES = parsy.regex(r"\s*", re.MULTILINE)


def spaceless(parser):
    return SPACES.then(parser).skip(SPACES)


def spaceless_string(*strings: str):
    return spaceless(
        parsy.alt(*(parsy.string(string, transform=str.lower) for string in strings))
    )


SINGLE_DIGIT = parsy.decimal_digit
RAW_NUMBER = SINGLE_DIGIT.at_least(1).concat()
PRECISION = SCALE = NUMBER = LENGTH = RAW_NUMBER.map(int)
TEMPORAL_SCALE = SINGLE_DIGIT.map(int)

LPAREN = spaceless_string("(")
RPAREN = spaceless_string(")")

LBRACKET = spaceless_string("[")
RBRACKET = spaceless_string("]")

LANGLE = spaceless_string("<")
RANGLE = spaceless_string(">")

COMMA = spaceless_string(",")
COLON = spaceless_string(":")
SEMICOLON = spaceless_string(";")

RAW_STRING = parsy.regex(_STRING_REGEX).map(ast.literal_eval)
FIELD = parsy.regex("[a-zA-Z_0-9]+") | parsy.string("")


@public
@functools.lru_cache(maxsize=100)
def parse(
    text: str, default_decimal_parameters: tuple[int | None, int | None] = (None, None)
) -> dt.DataType:
    """Parse a type from a [](`str`) `text`.

    The default `maxsize` parameter for caching is chosen to cache the most
    commonly used types--there are about 30--along with some capacity for less
    common but repeatedly-used complex types.

    Parameters
    ----------
    text
        The type string to parse
    default_decimal_parameters
        Default precision and scale for decimal types

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
    geotype = spaceless_string("geography", "geometry")

    srid_geotype = SEMICOLON.then(parsy.seq(srid=NUMBER.skip(COLON), geotype=geotype))
    geotype_part = COLON.then(parsy.seq(geotype=geotype))
    srid_part = SEMICOLON.then(parsy.seq(srid=NUMBER))

    def geotype_parser(typ: type[dt.DataType]) -> dt.DataType:
        return spaceless_string(typ.__name__.lower()).then(
            (srid_geotype | geotype_part | srid_part).optional(dict()).combine_dict(typ)
        )

    primitive = (
        spaceless_string("boolean", "bool").result(dt.boolean)
        | spaceless_string("halffloat", "float16").result(dt.float16)
        | spaceless_string("float32").result(dt.float32)
        | spaceless_string("double", "float64", "float").result(dt.float64)
        | spaceless_string(
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "string",
            "binary",
            "timestamp",
            "time",
            "date",
            "null",
        ).map(functools.partial(getattr, dt))
        | spaceless_string("bytes").result(dt.binary)
        | geotype.map(dt.GeoSpatial)
        | geotype_parser(dt.LineString)
        | geotype_parser(dt.Polygon)
        | geotype_parser(dt.Point)
        | geotype_parser(dt.MultiLineString)
        | geotype_parser(dt.MultiPolygon)
        | geotype_parser(dt.MultiPoint)
    )

    varchar_or_char = (
        spaceless_string("varchar", "char")
        .then(LPAREN.then(RAW_NUMBER).skip(RPAREN).optional())
        .result(dt.string)
    )

    decimal = spaceless_string("decimal").then(
        parsy.seq(
            LPAREN.then(spaceless(PRECISION)).skip(COMMA), spaceless(SCALE).skip(RPAREN)
        )
        .optional(default_decimal_parameters)
        .combine(dt.Decimal)
    )

    bignumeric = spaceless_string("bignumeric", "bigdecimal").then(
        parsy.seq(
            LPAREN.then(spaceless(PRECISION)).skip(COMMA), spaceless(SCALE).skip(RPAREN)
        )
        .optional((76, 38))
        .combine(dt.Decimal)
    )

    parened_string = LPAREN.then(RAW_STRING).skip(RPAREN)
    timestamp_scale = SINGLE_DIGIT.map(int)

    timestamp_tz_args = LPAREN.then(
        parsy.seq(timezone=RAW_STRING, scale=COMMA.then(timestamp_scale).optional())
    ).skip(RPAREN)

    timestamp_no_tz_args = LPAREN.then(parsy.seq(scale=timestamp_scale).skip(RPAREN))

    timestamp = spaceless_string("timestamp").then(
        (timestamp_tz_args | timestamp_no_tz_args)
        .optional({})
        .combine_dict(dt.Timestamp)
    )

    interval = spaceless_string("interval").then(
        parsy.seq(unit=parened_string.optional("s")).combine_dict(dt.Interval)
    )

    ty = parsy.forward_declaration()
    angle_type = LANGLE.then(ty).skip(RANGLE)
    array = spaceless_string("array").then(angle_type).map(dt.Array)

    map = (
        spaceless_string("map")
        .then(LANGLE)
        .then(parsy.seq(ty, COMMA.then(ty)).combine(dt.Map))
        .skip(RANGLE)
    )

    struct = (
        spaceless_string("struct")
        .then(LANGLE)
        .then(parsy.seq(spaceless(FIELD).skip(COLON), ty).sep_by(COMMA))
        .skip(RANGLE)
        .map(dt.Struct.from_tuples)
    )

    nullable = spaceless_string("!").then(ty).map(methodcaller("copy", nullable=False))

    ty.become(
        nullable
        | timestamp
        | primitive
        | decimal
        | bignumeric
        | varchar_or_char
        | interval
        | array
        | map
        | struct
        | spaceless_string("json", "uuid", "macaddr", "inet").map(
            functools.partial(getattr, dt)
        )
        | spaceless_string("int").result(dt.int64)
        | spaceless_string("str").result(dt.string)
    )

    return ty.parse(text)
