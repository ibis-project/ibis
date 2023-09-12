from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import ibis.expr.datatypes as dt
from ibis.common import exceptions as ex

if TYPE_CHECKING:
    import ibis.expr.operations as ops

# TODO(kszucs): move this module to the base sql backend

NumberType = TypeVar("NumberType", int, float)
# Geometry primitives (2D)
PointType = Iterable[NumberType]
LineStringType = list[PointType]
PolygonType = list[LineStringType]
# Multipart geometries (2D)
MultiPointType = list[PointType]
MultiLineStringType = list[LineStringType]
MultiPolygonType = list[PolygonType]


def _format_point_value(value: PointType) -> str:
    """Convert a iterable with a point to text."""
    return " ".join(str(v) for v in value)


def _format_linestring_value(value: LineStringType, nested=False) -> str:
    """Convert a iterable with a linestring to text."""
    template = "({})" if nested else "{}"
    if not isinstance(value[0], (tuple, list)):
        msg = "{} structure expected: LineStringType".format(
            "Data" if not nested else "Inner data"
        )
        raise ex.IbisInputError(msg)
    return template.format(", ".join(_format_point_value(point) for point in value))


def _format_polygon_value(value: PolygonType, nested=False) -> str:
    """Convert a iterable with a polygon to text."""
    template = "({})" if nested else "{}"
    if not isinstance(value[0][0], (tuple, list)):
        msg = "{} data structure expected: PolygonType".format(
            "Data" if not nested else "Inner data"
        )
        raise ex.IbisInputError(msg)

    return template.format(
        ", ".join(_format_linestring_value(line, nested=True) for line in value)
    )


def _format_multipoint_value(value: MultiPointType) -> str:
    """Convert a iterable with a multipoint to text."""
    if not isinstance(value[0], (tuple, list)):
        raise ex.IbisInputError("Data structure expected: MultiPointType")
    return ", ".join(f"({_format_point_value(point)})" for point in value)


def _format_multilinestring_value(value: MultiLineStringType) -> str:
    """Convert a iterable with a multilinestring to text."""
    if not isinstance(value[0][0], (tuple, list)):
        raise ex.IbisInputError("Data structure expected: MultiLineStringType")
    return ", ".join(f"({_format_linestring_value(line)})" for line in value)


def _format_multipolygon_value(value: MultiPolygonType) -> str:
    """Convert a iterable with a multipolygon to text."""
    if not isinstance(value[0][0], (tuple, list)):
        raise ex.IbisInputError("Data structure expected: MultiPolygonType")
    return ", ".join(_format_polygon_value(polygon, nested=True) for polygon in value)


def _format_geo_metadata(op, value: str, inline_metadata: bool = False) -> str:
    """Format a geometry/geography text when it is necessary."""
    srid = op.args[1].srid
    geotype = op.args[1].geotype

    if inline_metadata:
        value = "'{}{}'{}".format(
            f"SRID={srid};" if srid else "",
            value,
            f"::{geotype}" if geotype else "",
        )
        return value

    geofunc = "ST_GeogFromText" if geotype == "geography" else "ST_GeomFromText"

    value = repr(value)
    if srid:
        value += f", {srid}"

    return f"{geofunc}({value})"


def translate_point(value: Iterable) -> str:
    """Translate a point to WKT."""
    return f"POINT ({_format_point_value(value)})"


def translate_linestring(value: list) -> str:
    """Translate a linestring to WKT."""
    return f"LINESTRING ({_format_linestring_value(value)})"


def translate_polygon(value: list) -> str:
    """Translate a polygon to WKT."""
    return f"POLYGON ({_format_polygon_value(value)})"


def translate_multilinestring(value: list) -> str:
    """Translate a multilinestring to WKT."""
    return f"MULTILINESTRING ({_format_multilinestring_value(value)})"


def translate_multipoint(value: list) -> str:
    """Translate a multipoint to WKT."""
    return f"MULTIPOINT ({_format_multipoint_value(value)})"


def translate_multipolygon(value: list) -> str:
    """Translate a multipolygon to WKT."""
    return f"MULTIPOLYGON ({_format_multipolygon_value(value)})"


def translate_literal(op: ops.Literal, inline_metadata: bool = False) -> str:
    value = op.value
    dtype = op.dtype

    if isinstance(value, dt._WellKnownText):
        result = value.text
    elif dtype.is_point():
        result = translate_point(value)
    elif dtype.is_linestring():
        result = translate_linestring(value)
    elif dtype.is_polygon():
        result = translate_polygon(value)
    elif dtype.is_multilinestring():
        result = translate_multilinestring(value)
    elif dtype.is_multipoint():
        result = translate_multipoint(value)
    elif dtype.is_multipolygon():
        result = translate_multipolygon(value)
    else:
        raise ex.UnboundExpressionError("Geo Spatial type not supported.")
    return _format_geo_metadata(op, result, inline_metadata)
