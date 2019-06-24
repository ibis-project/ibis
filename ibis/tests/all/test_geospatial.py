""" Tests for geo spatial data types"""
from inspect import isfunction

import numpy as np
import pytest
from numpy import testing

import ibis
from ibis.tests.backends import MapD

# geo literals declaration
point_0 = ibis.literal((0, 0), type='point:geometry').name('p')
point_1 = ibis.literal((1, 1), type='point:geometry').name('p')
point_2 = ibis.literal((2, 2), type='point;4326:geometry').name('p')
point_3 = ibis.literal((1, 1), type='point:geography').name('p')
point_4 = ibis.literal((2, 2), type='point;4326:geography').name('p')
polygon_0 = ibis.literal(
    (
        ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)),
        ((0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1), (0.1, 0)),
    ),
    type='polygon',
)

# add here backends that support geo spatial types
all_db_geo_supported = [MapD]


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t['geo_linestring'].length(), [1.41] * 5),
        (lambda t: t['geo_polygon'].perimeter(), [5.66] * 5),
        (lambda t: t['geo_point'].x(), [0, 1, 2, 3, 4]),
        (lambda t: t['geo_point'].y(), [0, 1, 2, 3, 4]),
        (lambda t: t['geo_linestring'].x_min(), [0, 1, 2, 3, 4]),
        (lambda t: t['geo_linestring'].x_max(), [1, 2, 3, 4, 5]),
        (lambda t: t['geo_linestring'].y_min(), [0, 1, 2, 3, 4]),
        (lambda t: t['geo_linestring'].y_max(), [1, 2, 3, 4, 5]),
        (lambda t: t['geo_multipolygon'].n_points(), [12] * 5),
        (lambda t: t['geo_multipolygon'].n_rings(), [4] * 5),
        (lambda t: t['geo_point'].srid(), [0] * 5),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_geo_spatial_unops(backend, geo, expr_fn, expected):
    """Testing for geo spatial unary operations."""
    expr = expr_fn(geo)
    result = expr.execute()
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ('fn', 'arg_left', 'arg_right', 'expected'),
    [
        (
            'contains',
            lambda t: t['geo_linestring'],
            point_1,
            [True, True, False, False, False],
        ),
        (
            'distance',
            lambda t: t['geo_linestring'],
            point_0,
            [0.0, 1.41, 2.82, 4.24, 5.66],
        ),
        (
            'max_distance',
            lambda t: t['geo_linestring'],
            point_0,
            [1.41, 2.82, 4.24, 5.66, 7.08],
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_geo_spatial_binops(backend, geo, fn, arg_left, arg_right, expected):
    """Testing for geo spatial binary operations."""
    left = arg_left(geo) if isfunction(arg_left) else arg_left
    right = arg_right(geo) if isfunction(arg_right) else arg_right
    expr = getattr(left, fn)(right)
    result = expr.execute()
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t['geo_linestring'].end_point(), [False] * 5),
        (lambda t: t['geo_linestring'].point_n(1), [False] * 5),
        (lambda t: t['geo_linestring'].start_point(), [False] * 5),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_get_point(backend, geo, expr_fn, expected):
    """Testing for geo spatial get point operations."""
    # a geo spatial data does not contain its boundary
    arg = expr_fn(geo)
    expr = geo['geo_polygon'].contains(arg)
    result = geo[geo, expr.name('tmp')].execute()['tmp']
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(('arg', 'expected'), [(polygon_0, [1.98] * 5)])
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_area(backend, geo, arg, expected):
    """Testing for geo spatial area operation."""
    expr = geo[geo.id, arg.area().name('tmp')]
    result = expr.execute()['tmp']
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_2.srid(), 4326),
        (lambda t: point_0.srid(), 0),
        (lambda t: t.geo_point.srid(), 0),
        (lambda t: t.geo_linestring.srid(), 0),
        (lambda t: t.geo_polygon.srid(), 0),
        (lambda t: t.geo_multipolygon.srid(), 0),
    ]
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_srid(backend, geo, condition, expected):
    """Testing for geo spatial srid operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_0.set_srid(4326).srid(), 4326),
        (lambda t: point_0.set_srid(4326).set_srid(0).srid(), 0),
        (lambda t: t.geo_point.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_linestring.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_polygon.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_multipolygon.set_srid(4326).srid(), 4326),
    ]
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_set_srid(backend, geo, condition, expected):
    """Testing for geo spatial set_srid operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    ('condition', 'expected'),
    [
        (lambda t: point_0.set_srid(4326).transform(900913).srid(),
         900913),
        (lambda t: point_2.transform(900913).srid(),
         900913),
        (lambda t: t.geo_point.set_srid(4326).transform(900913).srid(),
         900913),
        (lambda t: t.geo_linestring.set_srid(4326).transform(900913).srid(),
         900913),
        (lambda t: t.geo_polygon.set_srid(4326).transform(900913).srid(),
         900913),
        (lambda t: t.geo_multipolygon.set_srid(4326).transform(900913).srid(),
         900913),
    ]
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
        lambda t: t.geo_point.set_srid(4326),
        lambda t: point_0.set_srid(4326),
        lambda t: point_1.set_srid(4326),
        lambda t: point_2,
        lambda t: point_3.set_srid(4326),
        lambda t: point_4,
    ]
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
        lambda t: t.geo_point.set_srid(4326),
        lambda t: point_0.set_srid(4326),
        lambda t: point_1.set_srid(4326),
        lambda t: point_2,
        lambda t: point_3.set_srid(4326),
        lambda t: point_4,
    ]
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
