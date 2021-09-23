from typing import Iterable, List, TypeVar

import ibis.expr.types as ir
from ibis.common import exceptions as ex

IS_SHAPELY_AVAILABLE = False
try:
    import shapely

    IS_SHAPELY_AVAILABLE = True
except ImportError:
    ...

NumberType = TypeVar('NumberType', int, float)
# Geometry primitives (2D)
PointType = Iterable[NumberType]
LineStringType = List[PointType]
PolygonType = List[LineStringType]
# Multipart geometries (2D)
MultiPointType = List[PointType]
MultiLineStringType = List[LineStringType]
MultiPolygonType = List[PolygonType]


def _format_point_value(value: PointType) -> str:
    """Convert a iterable with a point to text."""
    return ' '.join(str(v) for v in value)


def _format_linestring_value(value: LineStringType, nested=False) -> str:
    """Convert a iterable with a linestring to text."""
    template = '({})' if nested else '{}'
    if not isinstance(value[0], (tuple, list)):
        msg = '{} structure expected: LineStringType'.format(
            'Data' if not nested else 'Inner data'
        )
        raise ex.IbisInputError(msg)
    return template.format(
        ', '.join(_format_point_value(point) for point in value)
    )


def _format_polygon_value(value: PolygonType, nested=False) -> str:
    """Convert a iterable with a polygon to text."""
    template = '({})' if nested else '{}'
    if not isinstance(value[0][0], (tuple, list)):
        msg = '{} data structure expected: PolygonType'.format(
            'Data' if not nested else 'Inner data'
        )
        raise ex.IbisInputError(msg)

    return template.format(
        ', '.join(
            _format_linestring_value(line, nested=True) for line in value
        )
    )


def _format_multipoint_value(value: MultiPointType) -> str:
    """Convert a iterable with a multipoint to text."""
    if not isinstance(value[0], (tuple, list)):
        raise ex.IbisInputError('Data structure expected: MultiPointType')
    return ', '.join(f'({_format_point_value(point)})' for point in value)


def _format_multilinestring_value(value: MultiLineStringType) -> str:
    """Convert a iterable with a multilinestring to text."""
    if not isinstance(value[0][0], (tuple, list)):
        raise ex.IbisInputError('Data structure expected: MultiLineStringType')
    return ', '.join(f'({_format_linestring_value(line)})' for line in value)


def _format_multipolygon_value(value: MultiPolygonType) -> str:
    """Convert a iterable with a multipolygon to text."""
    if not isinstance(value[0][0], (tuple, list)):
        raise ex.IbisInputError('Data structure expected: MultiPolygonType')
    return ', '.join(
        _format_polygon_value(polygon, nested=True) for polygon in value
    )


def _format_geo_metadata(op, value: str, inline_metadata: bool = False) -> str:
    """Format a geometry/geography text when it is necessary."""
    srid = op.args[1].srid
    geotype = op.args[1].geotype

    if inline_metadata:
        value = "'{}{}'{}".format(
            f'SRID={srid};' if srid else '',
            value,
            f'::{geotype}' if geotype else '',
        )
        return value

    geofunc = (
        'ST_GeogFromText' if geotype == 'geography' else 'ST_GeomFromText'
    )

    value = repr(value)
    if srid:
        value += f', {srid}'

    return f"{geofunc}({value})"


def translate_point(value: Iterable) -> str:
    """Translate a point to WKT."""
    return f"POINT ({_format_point_value(value)})"


def translate_linestring(value: List) -> str:
    """Translate a linestring to WKT."""
    return f"LINESTRING ({_format_linestring_value(value)})"


def translate_polygon(value: List) -> str:
    """Translate a polygon to WKT."""
    return f"POLYGON ({_format_polygon_value(value)})"


def translate_multilinestring(value: List) -> str:
    """Translate a multilinestring to WKT."""
    return f"MULTILINESTRING ({_format_multilinestring_value(value)})"


def translate_multipoint(value: List) -> str:
    """Translate a multipoint to WKT."""
    return f"MULTIPOINT ({_format_multipoint_value(value)})"


def translate_multipolygon(value: List) -> str:
    """Translate a multipolygon to WKT."""
    return f"MULTIPOLYGON ({_format_multipolygon_value(value)})"


def translate_literal(expr, inline_metadata: bool = False) -> str:
    op = expr.op()
    value = op.value

    if IS_SHAPELY_AVAILABLE and isinstance(
        value, shapely.geometry.base.BaseGeometry
    ):
        result = value.wkt
    elif isinstance(expr, ir.PointScalar):
        result = translate_point(value)
    elif isinstance(expr, ir.LineStringScalar):
        result = translate_linestring(value)
    elif isinstance(expr, ir.PolygonScalar):
        result = translate_polygon(value)
    elif isinstance(expr, ir.MultiLineStringScalar):
        result = translate_multilinestring(value)
    elif isinstance(expr, ir.MultiPointScalar):
        result = translate_multipoint(value)
    elif isinstance(expr, ir.MultiPolygonScalar):
        result = translate_multipolygon(value)
    else:
        raise ex.UnboundExpressionError('Geo Spatial type not supported.')
    return _format_geo_metadata(op, result, inline_metadata)
