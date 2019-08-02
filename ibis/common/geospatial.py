from typing import Iterable

import ibis.expr.types as ir
from ibis.common import exceptions as ex


def _format_point_value(value: Iterable) -> str:
    """Convert a iterable with a point to text."""
    return ' '.join(str(v) for v in value)


def _format_linestring_value(value: Iterable) -> str:
    """Convert a iterable with a linestring to text."""
    return ', '.join(
        _format_point_value(point) for point in value
    )


def _format_polygon_value(value: Iterable) -> str:
    """Convert a iterable with a polygon to text."""
    return ', '.join(
        '({})'.format(_format_linestring_value(line)) for line in value
    )


def _format_multipolygon_value(value: Iterable) -> str:
    """Convert a iterable with a multipolygon to text."""
    return ', '.join(
        '({})'.format(_format_polygon_value(polygon)) for polygon in value
    )


def _format_geo_metadata(op, value: str, inline_metadata: bool = False) -> str:
    """Format a geometry/geography text when it is necessary."""
    srid = op.args[1].srid
    geotype = op.args[1].geotype

    if inline_metadata and srid:
        value = "'SRID={};{}'".format(srid, value)
    else:
        value = "'{}'".format(value)

    if geotype not in ('geometry', 'geography'):
        if inline_metadata and srid:
            return value

    geofunc = 'ST_GeomFromText' if geotype == 'geometry' else 'ST_GeogFromText'

    if srid and not inline_metadata:
        value += ', {}'.format(srid)

    return "{}({})".format(geofunc, value)


def translate_point(value: Iterable) -> str:
    """Translate a point to WKT."""
    return "POINT ({})".format(_format_point_value(value))


def translate_linestring(value: Iterable) -> str:
    """Translate a linestring to WKT."""
    return "LINESTRING ({})".format(_format_linestring_value(value))


def translate_polygon(value: Iterable) -> str:
    """Translate a polygon to WKT."""
    return "POLYGON ({})".format(_format_polygon_value(value))


def translate_multipolygon(value: Iterable) -> str:
    """Translate a multipolygon to WKT."""
    return "MULTIPOLYGON ({})".format(_format_multipolygon_value(value))


def translate_literal(expr, inline_metadata: bool = False) -> str:
    op = expr.op()
    value = op.value

    if isinstance(expr, ir.PointScalar):
        result = translate_point(value)
    elif isinstance(expr, ir.LineStringScalar):
        result = translate_linestring(value)
    elif isinstance(expr, ir.PolygonScalar):
        result = translate_polygon(value)
    elif isinstance(expr, ir.MultiPolygonScalar):
        result = translate_multipolygon(value)
    else:
        raise ex.UnboundExpressionError('Geo Spatial type not supported.')
    return _format_geo_metadata(op, result, inline_metadata)
