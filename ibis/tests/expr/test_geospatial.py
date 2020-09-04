""" Tests for geo spatial data types"""
import pytest

import ibis
from ibis.tests.expr.mocks import (
    GeoMockConnectionOmniSciDB,
    GeoMockConnectionPostGIS,
)

pytest.importorskip('geoalchemy2')
pytest.importorskip('shapely')
pytest.importorskip('geopandas')

pytest.mark.postgis
pytest.mark.omniscidb

mock_omniscidb = GeoMockConnectionOmniSciDB()
mock_postgis = GeoMockConnectionPostGIS()


@pytest.mark.parametrize('backend', [mock_postgis])
@pytest.mark.parametrize(
    'modifier',
    [
        {},
        {'srid': '4326'},
        {'srid': '4326', 'geo_type': 'geometry'},
        {'srid': '4326', 'geo_type': 'geography'},
    ],
)
@pytest.mark.parametrize(
    'shape,value,expected',
    [
        # Geometry primitives (2D)
        ('point', (30, 10), 'POINT (30 10)'),
        (
            'linestring',
            ((30, 10), (10, 30), (40, 40)),
            'LINESTRING (30 10, 10 30, 40 40)',
        ),
        (
            'polygon',
            (
                ((35, 10), (45, 45), (15, 40), (10, 20), (35, 10)),
                ((20, 30), (35, 35), (30, 20), (20, 30)),
            ),
            (
                'POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), '
                + '(20 30, 35 35, 30 20, 20 30))'
            ),
        ),
        (
            'polygon',
            (((30, 10), (40, 40), (20, 40), (10, 20), (30, 10)),),
            'POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))',
        ),
        # Multipart geometries (2D)
        (
            'multipoint',
            ((10, 40), (40, 30), (20, 20), (30, 10)),
            'MULTIPOINT ((10 40), (40 30), (20 20), (30 10))',
        ),
        (
            'multilinestring',
            (
                ((10, 10), (20, 20), (10, 40)),
                ((40, 40), (30, 30), (40, 20), (30, 10)),
            ),
            (
                'MULTILINESTRING ((10 10, 20 20, 10 40), '
                + '(40 40, 30 30, 40 20, 30 10))'
            ),
        ),
        (
            'multipolygon',
            (
                (((40, 40), (20, 45), (45, 30), (40, 40)),),
                (
                    (
                        (20, 35),
                        (10, 30),
                        (10, 10),
                        (30, 5),
                        (45, 20),
                        (20, 35),
                    ),
                    ((30, 20), (20, 15), (20, 25), (30, 20)),
                ),
            ),
            (
                'MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), '
                + '((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), '
                + '(30 20, 20 15, 20 25, 30 20)))'
            ),
        ),
    ],
)
def test_geo_literals_smoke(backend, shape, value, modifier, expected):
    """Smoke tests for geo spatial literals"""
    expr_type = '{}{}{}'.format(
        shape,
        ';{}'.format(modifier['srid']) if 'srid' in modifier else '',
        ':{}'.format(modifier['geo_type']) if 'geo_type' in modifier else '',
    )
    expr = ibis.literal(value, type=expr_type).name('tmp')
    result_expected = "SELECT '{}{}'{}".format(
        'SRID={};'.format(modifier['srid']) if 'srid' in modifier else '',
        expected,
        '::{}'.format(modifier['geo_type']) if 'geo_type' in modifier else '',
    )

    assert str(backend.compile(expr)) == result_expected


@pytest.mark.parametrize(
    'fn_expr',
    [
        pytest.param(lambda t: t.geo_point.srid(), id='point_srid'),
        pytest.param(
            lambda t: t.geo_point.set_srid(4326), id='point_set_srid'
        ),
        pytest.param(lambda t: t.geo_point.x(), id='point_x'),
        pytest.param(lambda t: t.geo_point.y(), id='point_y'),
        pytest.param(
            lambda t: t.geo_linestring.contains(t.geo_point),
            id='linestring_contains',
        ),
        pytest.param(
            lambda t: t.geo_linestring.end_point(), id='linestring_end_point'
        ),
        pytest.param(
            lambda t: t.geo_linestring.length(), id='linestring_length'
        ),
        pytest.param(
            lambda t: t.geo_linestring.max_distance(t.geo_point),
            id='linestring_max_distance',
        ),
        pytest.param(
            lambda t: t.geo_linestring.point_n(1), id='linestring_point_n'
        ),
        pytest.param(
            lambda t: t.geo_linestring.start_point(),
            id='linestring_start_point',
        ),
        pytest.param(
            lambda t: t.geo_linestring.x_max(), id='linestring_x_max'
        ),
        pytest.param(
            lambda t: t.geo_linestring.x_min(), id='linestring_x_min'
        ),
        pytest.param(
            lambda t: t.geo_linestring.y_max(), id='linestring_y_max'
        ),
        pytest.param(
            lambda t: t.geo_linestring.y_min(), id='linestring_y_min'
        ),
        pytest.param(lambda t: t.geo_polygon.area(), id='polygon_area'),
        pytest.param(
            lambda t: t.geo_polygon.perimeter(), id='polygon_perimeter'
        ),
        pytest.param(
            lambda t: t.geo_multipolygon.n_points(), id='multipolygon_n_points'
        ),
        pytest.param(
            lambda t: t.geo_multipolygon.n_rings(), id='multipolygon_n_rings'
        ),
        # TODO: the mock tests don't support multipoint and multilinestring
        #       yet, but once they do, add some more tests here.
    ],
)
@pytest.mark.parametrize('backend', [mock_omniscidb, mock_postgis])
@pytest.mark.xfail_unsupported
def test_geo_ops_smoke(backend, fn_expr):
    """Smoke tests for geo spatial operations."""
    geo_table = backend.table('geo')
    assert fn_expr(geo_table).compile() != ''
