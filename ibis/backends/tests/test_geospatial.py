""" Tests for geo spatial data types"""
import numpy as np
import pytest
from numpy import testing
from pytest import param

import ibis

pytestmark = pytest.mark.geo


# TODO find a way to just run for the backends that support geo, without
# skipping if dependencies are missing
shapely = pytest.importorskip('shapely')

# geo literals declaration
point_0 = ibis.literal((0, 0), type='point').name('tmp')
point_0_4326 = ibis.literal((0, 0), type='point;4326').name('tmp')

point_geom_0 = ibis.literal((0, 0), type='point;4326:geometry').name('p')
point_geom_1 = ibis.literal((1, 1), type='point;4326:geometry').name('p')
point_geom_0_srid0 = ibis.literal((0, 0), type='point;0:geometry').name('p')
point_geom_1_srid0 = ibis.literal((1, 1), type='point;0:geometry').name('p')
point_geom_2 = ibis.literal((2, 2), type='point;4326:geometry').name('p')
point_geog_0 = ibis.literal((0, 0), type='point;4326:geography').name('p')
point_geog_1 = ibis.literal((1, 1), type='point;4326:geography').name('p')
point_geog_2 = ibis.literal((2, 2), type='point;4326:geography').name('p')
polygon_0 = ibis.literal(
    (
        ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)),
        ((0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1), (0.1, 0)),
    ),
    type='polygon;4326:geometry',
).name('p')

# add here backends that support geo spatial types
all_db_geo_supported = ['omniscidb', 'postgres']

# test input data with shapely geometries
shp_point_0 = shapely.geometry.Point(0, 0)
shp_point_1 = shapely.geometry.Point(1, 1)
shp_point_2 = shapely.geometry.Point(2, 2)

shp_linestring_0 = shapely.geometry.LineString(
    [shp_point_0, shp_point_1, shp_point_2]
)
shp_linestring_1 = shapely.geometry.LineString(
    [shp_point_2, shp_point_1, shp_point_0]
)
shp_polygon_0 = shapely.geometry.Polygon(shp_linestring_0)
shp_multilinestring_0 = shapely.geometry.MultiLineString(
    [shp_linestring_0, shp_linestring_1]
)
shp_multipoint_0 = shapely.geometry.MultiPoint(
    [shp_point_0, shp_point_1, shp_point_2]
)
shp_multipolygon_0 = shapely.geometry.MultiPolygon([shp_polygon_0])


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (
            point_0,
            {
                'omniscidb': "ST_GeomFromText('POINT (0 0)')",
                'postgres': "'POINT (0 0)'",
            },
        ),
        # TODO: add SRID to the literals. OmniSciDB and PostGIS handle in a
        #       different way that.
        (
            point_0_4326,
            {
                'omniscidb': "ST_GeomFromText('POINT (0 0)', 4326)",
                'postgres': "'SRID=4326;POINT (0 0)'",
            },
        ),
        (
            point_geom_0,
            {
                'omniscidb': "ST_GeomFromText('POINT (0 0)', 4326)",
                'postgres': "'SRID=4326;POINT (0 0)'::geometry",
            },
        ),
        (
            point_geom_1,
            {
                'omniscidb': "ST_GeomFromText('POINT (1 1)', 4326)",
                'postgres': "'SRID=4326;POINT (1 1)'::geometry",
            },
        ),
        (
            point_geom_2,
            {
                'omniscidb': "ST_GeomFromText('POINT (2 2)', 4326)",
                'postgres': "'SRID=4326;POINT (2 2)'::geometry",
            },
        ),
        (
            point_geog_0,
            {
                'omniscidb': "ST_GeogFromText('POINT (0 0)', 4326)",
                'postgres': "'SRID=4326;POINT (0 0)'::geography",
            },
        ),
        (
            point_geog_1,
            {
                'omniscidb': "ST_GeogFromText('POINT (1 1)', 4326)",
                'postgres': "'SRID=4326;POINT (1 1)'::geography",
            },
        ),
        (
            point_geog_2,
            {
                'omniscidb': "ST_GeogFromText('POINT (2 2)', 4326)",
                'postgres': "'SRID=4326;POINT (2 2)'::geography",
            },
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
def test_literal_geospatial_explicit(backend, con, expr, expected):
    result = str(con.compile(expr))
    result_expected = f"SELECT {expected[backend.name()]} AS tmp"
    # use `in` op because if name is specified omniscidb doesn't compile
    # with alias but postgresql does. but if name is not provided,
    # omniscidb uses tmp as a default alias but postgres doesn't use alias
    assert result in result_expected


@pytest.mark.parametrize(
    ('shp', 'expected'),
    [
        (
            shp_point_0,
            {
                'omniscidb': "ST_GeomFromText('POINT (0 0)')",
                'postgres': "'POINT (0 0)'",
            },
        ),
        (
            shp_point_1,
            {
                'omniscidb': "ST_GeomFromText('POINT (1 1)')",
                'postgres': "'POINT (1 1)'",
            },
        ),
        (
            shp_point_2,
            {
                'omniscidb': "ST_GeomFromText('POINT (2 2)')",
                'postgres': "'POINT (2 2)'",
            },
        ),
        (
            shp_linestring_0,
            {
                'omniscidb': "ST_GeomFromText('LINESTRING (0 0, 1 1, 2 2)')",
                'postgres': "'LINESTRING (0 0, 1 1, 2 2)'",
            },
        ),
        (
            shp_linestring_1,
            {
                'omniscidb': "ST_GeomFromText('LINESTRING (2 2, 1 1, 0 0)')",
                'postgres': "'LINESTRING (2 2, 1 1, 0 0)'",
            },
        ),
        (
            shp_polygon_0,
            {
                'omniscidb': (
                    "ST_GeomFromText('POLYGON ((0 0, 1 1, 2 2, 0 0))')"
                ),
                'postgres': "'POLYGON ((0 0, 1 1, 2 2, 0 0))'",
            },
        ),
        (
            shp_multipolygon_0,
            {
                'omniscidb': (
                    "ST_GeomFromText('MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))')"
                ),
                'postgres': "'MULTIPOLYGON (((0 0, 1 1, 2 2, 0 0)))'",
            },
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_literal_geospatial_inferred(backend, con, shp, expected):
    result = str(con.compile(ibis.literal(shp)))
    result_expected = f"SELECT {expected[backend.name()]} AS tmp"
    # use `in` op because if name is specified omniscidb doesn't compile
    # with alias but postgresql does. but if name is not provided,
    # omniscidb uses tmp as a default alias but postgres doesn't use alias
    assert result in result_expected


@pytest.mark.parametrize(
    ('shp', 'expected'),
    [
        (
            shp_multilinestring_0,
            {
                'postgres': (
                    "'MULTILINESTRING ((0 0, 1 1, 2 2), (2 2, 1 1, 0 0))'"
                ),
                'omniscidb': (
                    "ST_GeomFromText('MULTILINESTRING ((0 0, 1 1, 2 2),"
                    + " (2 2, 1 1, 0 0))')"
                ),
            },
        ),
        (
            shp_multipoint_0,
            {
                'postgres': "'MULTIPOINT (0 0, 1 1, 2 2)'",
                'omniscidb': "ST_GeomFromText('MULTIPOINT (0 0, 1 1, 2 2)')",
            },
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_literal_multi_geospatial_inferred(backend, con, shp, expected):
    result = str(con.compile(ibis.literal(shp)))
    result_expected = f"SELECT {expected[backend.name()]} AS tmp"
    assert result in result_expected


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
                ['postgres'], reason='TODO: fix different results issue'
            ),
        ),
        param(
            lambda t: t['geo_multipolygon'].n_points(),
            [7, 11, 5, 5, 5],
            id='n_points',
            marks=pytest.mark.skip_backends(
                ['postgres'], reason='TODO: fix different results issue'
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
            lambda t: t['geo_linestring'].contains(point_geom_1_srid0),
            {
                'omniscidb': [True, True, False, False, False],
                'postgres': [False] * 5,  # not contains the border
            },
            id='contains',
        ),
        param(
            lambda t: t['geo_linestring'].disjoint(point_geom_0_srid0),
            {
                'omniscidb': [False, True, True, True, True],
                'postgres': [False, True, True, True, True],
            },
            id='disjoint',
        ),
        param(
            lambda t: t['geo_point'].d_within(point_geom_1_srid0, 2.0),
            {
                'omniscidb': [True, True, True, False, False],
                'postgres': [True, True, True, False, False],
            },
            id='d_within',
        ),
        param(
            lambda t: t['geo_point'].d_fully_within(t['geo_linestring'], 2.0),
            {
                'omniscidb': [True, True, True, True, True],
                'postgres': [True, True, True, True, True],
            },
            id='d_fully_within',
        ),
        param(
            lambda t: t['geo_linestring'].intersects(point_geom_0_srid0),
            {
                'omniscidb': [True, False, False, False, False],
                'postgres': [True, False, False, False, False],
            },
            id='intersects',
        ),
        param(
            lambda t: t['geo_linestring'].distance(point_geom_0_srid0),
            {
                'omniscidb': [0.0, 1.41, 2.82, 4.24, 5.66],
                'postgres': [0.0, 1.41, 2.82, 4.24, 5.66],
            },
            id='distance',
        ),
        param(
            lambda t: t['geo_linestring'].max_distance(point_geom_0_srid0),
            {
                'omniscidb': [1.41, 2.82, 4.24, 5.66, 7.08],
                'postgres': [1.41, 2.82, 4.24, 5.66, 7.08],
            },
            id='max_distance',
        ),
        param(
            lambda t: t.geo_polygon.contains(ibis.geo_point(30, 10)),
            {
                'omniscidb': [True, False, False, False, False],
                'postgres': [True, False, False, False, False],
            },
            id='point',
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_geo_spatial_binops(backend, geo, expr_fn, expected):
    """Testing for geo spatial binary operations."""
    expr = expr_fn(geo)
    result = expr.execute()
    testing.assert_almost_equal(result, expected[backend.name()], decimal=2)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        param(
            lambda t: t['geo_linestring'].end_point(),
            [True, True, True, True, True],
            id='end_point',
        ),
        param(
            lambda t: t['geo_linestring'].point_n(1),
            [True, True, True, True, True],
            id='point_n',
        ),
        param(
            lambda t: t['geo_linestring'].start_point(),
            [True, True, True, True, True],
            id='start_point',
        ),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_get_point(backend, geo, expr_fn, expected):
    """Testing for geo spatial get point operations."""
    arg = expr_fn(geo)
    # Note: there is a difference in how OmnisciDB and PostGIS consider
    # boundaries with the contains predicate. Work around this by adding a
    # small buffer.
    expr = geo['geo_linestring'].buffer(0.01).contains(arg)
    result = geo[geo, expr.name('tmp')].execute()['tmp']
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(('arg', 'expected'), [(polygon_0, [1.98] * 5)])
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
        (lambda t: point_geom_2.srid(), {'omniscidb': 4326, 'postgres': 4326}),
        (lambda t: point_geom_0.srid(), {'omniscidb': 4326, 'postgres': 4326}),
        (lambda t: t.geo_point.srid(), {'omniscidb': 0, 'postgres': 0}),
        (lambda t: t.geo_linestring.srid(), {'omniscidb': 0, 'postgres': 0}),
        (lambda t: t.geo_polygon.srid(), {'omniscidb': 0, 'postgres': 0}),
        (lambda t: t.geo_multipolygon.srid(), {'omniscidb': 0, 'postgres': 0}),
    ],
)
@pytest.mark.only_on_backends(all_db_geo_supported)
@pytest.mark.xfail_unsupported
def test_srid(backend, geo, condition, expected):
    """Testing for geo spatial srid operation."""
    expr = geo[geo.id, condition(geo).name('tmp')]
    result = expr.execute()['tmp'][[0]]
    assert np.all(result == expected[backend.name()])


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
    import geopandas

    assert isinstance(geo.execute(), geopandas.geodataframe.GeoDataFrame)


@pytest.mark.only_on_backends(['postgres'])
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

    assert str(backend.api.compile(expr)) == result_expected


@pytest.mark.only_on_backends(all_db_geo_supported)
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
@pytest.mark.xfail_unsupported
def test_geo_ops_smoke(backend, fn_expr):
    """Smoke tests for geo spatial operations."""
    geo_table = backend.geo
    assert fn_expr(geo_table).compile() != ''


@pytest.mark.only_on_backends(["postgres"])
def test_geo_equals(backend, geo):
    # Fix https://github.com/ibis-project/ibis/pull/2956
    expr = geo.mutate(
        [
            geo.geo_point.y().name('Location_Latitude'),
            geo.geo_point.y().name('Latitude'),
        ]
    )

    result = str(expr.compile().compile())

    assert result == (
        'SELECT t0.id, ST_AsEWKB(t0.geo_point) AS geo_point, '
        'ST_AsEWKB(t0.geo_linestring) AS geo_linestring, '
        'ST_AsEWKB(t0.geo_polygon) AS geo_polygon, '
        'ST_AsEWKB(t0.geo_multipolygon) AS geo_multipolygon, '
        'ST_Y(t0.geo_point) AS "Location_Latitude", '
        'ST_Y(t0.geo_point) AS "Latitude" \n'
        'FROM geo AS t0'
    )

    # simple test using ==
    expected = 'SELECT t0.geo_point = t0.geo_point AS tmp \nFROM geo AS t0'
    expr = geo.geo_point == geo.geo_point
    assert str(expr.compile().compile()) == expected
    assert expr.execute().all()

    # using geo_equals
    expected = (
        'SELECT ST_Equals(t0.geo_point, t0.geo_point) AS tmp \nFROM geo AS t0'
    )
    expr = geo.geo_point.geo_equals(geo.geo_point)
    assert str(expr.compile().compile()) == expected
    assert expr.execute().all()

    # equals returns a boolean object
    assert geo.geo_point.equals(geo.geo_point)
