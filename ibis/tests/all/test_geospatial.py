""" Tests for geo spatial data types"""
import sys

import numpy as np
import pytest
from numpy import testing
from pytest import param

import ibis
from ibis.tests.backends import MapD, PostgreSQL

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6),
    reason='Geo Spatial support available just for Python >= 3.6',
)
geopandas = pytest.importorskip('geopandas')
shapely_wkt = pytest.importorskip('shapely.wkt')

# geo literals declaration
point_geom_0 = ibis.literal((0, 0), type='point;4326:geometry').name('p')
point_geom_1 = ibis.literal((1, 1), type='point;4326:geometry').name('p')
point_geom_2 = ibis.literal((2, 2), type='point;4326:geometry').name('p')
point_geog_0 = ibis.literal((0, 0), type='point;4326:geography').name('p')
point_geog_1 = ibis.literal((1, 1), type='point;4326:geography').name('p')
point_geog_2 = ibis.literal((2, 2), type='point;4326:geography').name('p')
polygon_0 = ibis.literal(
    (((30, 10), (40, 40), (20, 40), (10, 20), (30, 10)),),
    type='polygon;4326:geometry',
)

# add here backends that support geo spatial types
all_db_geo_supported = [MapD, PostgreSQL]


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        param(lambda t: t['geo_linestring'].length(), [1.41] * 5, id='length'),
        param(lambda t: t['geo_point'].x(), [0, 1, 2, 3, 4], id='x'),
        param(lambda t: t['geo_point'].y(), [0, 1, 2, 3, 4], id='y'),
        param(
            lambda t: t['geo_linestring'].x_min(), [0, 1, 2, 3, 4], id='x_min'
        ),
        param(
            lambda t: t['geo_linestring'].x_max(), [1, 2, 3, 4, 5], id='x_max'
        ),
        param(
            lambda t: t['geo_linestring'].y_min(), [0, 1, 2, 3, 4], id='y_min'
        ),
        param(
            lambda t: t['geo_linestring'].y_max(), [1, 2, 3, 4, 5], id='y_max'
        ),
        param(
            lambda t: t['geo_multipolygon'].n_rings(),
            [2, 3, 1, 1, 1],
            id='n_rings',
        ),
        param(
            lambda t: t['geo_polygon'].set_srid(4326).perimeter(),
            [96.34, 114.36, 10.24, 10.24, 10.24],
            id='perimeter',
            marks=pytest.mark.skip_backends(
                [PostgreSQL], reason='TODO: fix different results issue'
            ),
        ),
        param(
            lambda t: t['geo_multipolygon'].n_points(),
            [7, 11, 5, 5, 5],
            id='n_points',
            marks=pytest.mark.skip_backends(
                [PostgreSQL], reason='TODO: fix different results issue'
            ),
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_geo_spatial_unops(backend, geo, expr_fn, expected):
    """Testing for geo spatial unary operations."""
    expr = expr_fn(geo)
    result = expr.execute()
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        param(
            lambda t: t['geo_linestring'].contains(point_geom_1),
            {
                'mapd': [False] * 5,
                'postgres': [False] * 5,  # not contains the border
            },
            id='contains',
        ),
        param(
            lambda t: t['geo_linestring'].distance(point_geom_0),
            {
                'mapd': [0.0, 1.41, 2.82, 4.24, 5.66],
                'postgres': [0.0, 1.41, 2.82, 4.24, 5.66],
            },
            id='distance',
        ),
        param(
            lambda t: t['geo_linestring'].max_distance(point_geom_0),
            {
                'mapd': [1.41, 2.82, 4.24, 5.66, 7.08],
                'postgres': [1.41, 2.82, 4.24, 5.66, 7.08],
            },
            id='max_distance',
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_geo_spatial_binops(backend, geo, expr_fn, expected):
    """Testing for geo spatial binary operations."""
    expr = expr_fn(geo)
    result = expr.execute()
    testing.assert_almost_equal(result, expected[backend.name], decimal=2)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        param(
            lambda t: t['geo_linestring'].end_point(),
            [False, False, True, True, True],
            id='end_point',
        ),
        param(
            lambda t: t['geo_linestring'].point_n(1),
            [False, False, True, True, True],
            id='point_n',
        ),
        param(
            lambda t: t['geo_linestring'].start_point(),
            [False, False, True, True, True],
            id='start_point',
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_get_point(backend, geo, expr_fn, expected):
    """Testing for geo spatial get point operations."""
    arg = expr_fn(geo)
    expr = geo['geo_polygon'].contains(arg)
    result = geo[geo, expr.name('tmp')].execute()['tmp']
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(('arg', 'expected'), [(polygon_0, [550.0] * 5)])
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_area(backend, geo, arg, expected):
    """Testing for geo spatial area operation."""
    expr = geo[geo.id, arg.area().name('tmp')]
    result = expr.execute()['tmp']
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_geom_2.srid(), {'mapd': 4326, 'postgres': 4326}),
        (lambda t: point_geom_0.srid(), {'mapd': 4326, 'postgres': 4326}),
        (lambda t: t.geo_point.srid(), {'mapd': 0, 'postgres': 4326}),
        (lambda t: t.geo_linestring.srid(), {'mapd': 0, 'postgres': 4326}),
        (lambda t: t.geo_polygon.srid(), {'mapd': 0, 'postgres': 4326}),
        (lambda t: t.geo_multipolygon.srid(), {'mapd': 0, 'postgres': 4326}),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_srid(backend, geo, condition, expected):
    """Testing for geo spatial srid operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected[backend.name])


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_geom_0.set_srid(4326).srid(), 4326),
        (lambda t: point_geom_0.set_srid(4326).set_srid(0).srid(), 0),
        (lambda t: t.geo_point.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_linestring.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_polygon.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_multipolygon.set_srid(4326).srid(), 4326),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_set_srid(backend, geo, condition, expected):
    """Testing for geo spatial set_srid operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_geom_0.transform(900913).srid(), 900913),
        (lambda t: point_geom_2.transform(900913).srid(), 900913),
        (
            lambda t: t.geo_point.set_srid(4326).transform(900913).srid(),
            900913,
        ),
        (
            lambda t: t.geo_linestring.set_srid(4326).transform(900913).srid(),
            900913,
        ),
        (
            lambda t: t.geo_polygon.set_srid(4326).transform(900913).srid(),
            900913,
        ),
        (
            lambda t: t.geo_multipolygon.set_srid(4326)
            .transform(900913)
            .srid(),
            900913,
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_transform(backend, geo, condition, expected):
    """Testing for geo spatial transform operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    'expr_fn',
    [
        param(lambda t: t.geo_point.set_srid(4326), id='geom_geo_point'),
        param(lambda t: point_geom_0, id='point_geom_0'),
        param(lambda t: point_geom_1, id='point_geom_1'),
        param(lambda t: point_geom_2, id='point_geom_2'),
        param(lambda t: point_geog_0, id='point_geog_0'),
        param(lambda t: point_geog_1, id='point_geog_1'),
        param(lambda t: point_geog_2, id='point_geog_2'),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_cast_geography(backend, geo, expr_fn):
    """Testing for geo spatial transform operation."""
    p = expr_fn(geo).cast('geography')
    expr = geo[geo.id, p.distance(p).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    # distance from a point to a same point should be 0
    assert np.all(result == 0)


@pytest.mark.parametrize(
    'expr_fn',
    [
        param(lambda t: t.geo_point.set_srid(4326), id='t_geo_point'),
        param(lambda t: point_geom_0, id='point_geom_0'),
        param(lambda t: point_geom_1, id='point_geom_1'),
        param(lambda t: point_geom_2, id='point_geom_2'),
        param(lambda t: point_geog_0, id='point_geog_0'),
        param(lambda t: point_geog_1, id='point_geog_1'),
        param(lambda t: point_geog_2, id='point_geog_2'),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_cast_geometry(backend, geo, expr_fn):
    """Testing for geo spatial transform operation."""
    p = expr_fn(geo).cast('geometry')
    expr = geo[geo.id, p.distance(p).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    # distance from a point to a same point should be 0
    assert np.all(result == 0)


@pytest.mark.only_on_backends(all_db_geo_supported)
def test_geo_dataframe(backend, geo):
    """Testing for geo dataframe output."""
    assert isinstance(geo.execute(), geopandas.geodataframe.GeoDataFrame)
