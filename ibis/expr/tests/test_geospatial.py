""" Tests for geo spatial data types"""
import pytest

import ibis


@pytest.mark.parametrize(
    'modifier', ['', ';4326', ';4326:geometry', ';4326:geography']
)
def test_geo_literals_smoke(modifier):
    """Smoke tests for geo spatial literals"""
    point = (0, 1)
    ibis.literal(point, type='point{}'.format(modifier))

    linestring = [(0, 1), (2, 3)]
    ibis.literal(linestring, type='linestring{}'.format(modifier))

    polygon = [
        ((0, 0), (4, 0), (4, 4), (0, 4), (0, 0)),
        ((1, 1), (2, 1), (2, 2), (1, 2), (1, 1)),
    ]
    ibis.literal(polygon, type='polygon{}'.format(modifier))

    point1 = (0, 1)
    point2 = (2, 3)
    multipoint = [point1, point2]
    ibis.literal(multipoint, type='multipoint{}'.format(modifier))

    linestring1 = [(0, 1), (2, 3)]
    linestring2 = [(4, 5), (6, 7)]
    multilinestring = [linestring1, linestring2]
    ibis.literal(multilinestring, type='multilinestring{}'.format(modifier))

    polygon1 = (
        (0, 0), (1, 0), (0.5, 1), (0, 0)
    )
    polygon2 = (
        ((35, 10), (45, 45), (15, 40), (10, 20), (35, 10)),
        ((20, 30), (35, 35), (30, 20), (20, 30))
    )
    multipolygon = [polygon1, polygon2]
    ibis.literal(multipolygon, type='multipolygon{}'.format(modifier))


def test_geo_ops_smoke(geo_table):
    """Smoke tests for geo spatial operations."""
    t = geo_table

    # alias for fields
    point = t.geo_point
    linestring = t.geo_linestring
    polygon = t.geo_polygon
    multipolygon = t.geo_multipolygon
    # TODO: the mock tests don't support multipoint and multilinestring yet,
    # but once they do, add some more tests here.

    # test ops
    point.srid()
    point.set_srid(4326)
    point.x()
    point.y()

    linestring.contains(point)
    linestring.end_point()
    linestring.length()
    linestring.max_distance(point)
    linestring.point_n(1)
    linestring.start_point()
    linestring.x_max()
    linestring.x_min()
    linestring.y_max()
    linestring.y_min()

    polygon.area()
    polygon.perimeter()

    multipolygon.n_points()
    multipolygon.n_rings()
