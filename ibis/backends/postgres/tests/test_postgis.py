import pandas.util.testing as tm
import pytest
from numpy import testing

gp = pytest.importorskip('geopandas')
sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.postgis,
    pytest.mark.postgres_extensions,
]


def test_load_geodata(con):
    t = con.table('geo')
    result = t.execute()
    assert isinstance(result, gp.GeoDataFrame)


def test_empty_select(geotable):
    expr = geotable[geotable.geo_point.equals(geotable.geo_linestring)]
    result = expr.execute()
    assert len(result) == 0


def test_select_point_geodata(geotable):
    expr = geotable['geo_point']
    sqla_expr = expr.compile()
    compiled = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = "SELECT ST_AsEWKB(t0.geo_point) AS geo_point \nFROM geo AS t0"
    assert compiled == expected
    data = expr.execute()
    assert data.geom_type.iloc[0] == 'Point'


def test_select_linestring_geodata(geotable):
    expr = geotable['geo_linestring']
    sqla_expr = expr.compile()
    compiled = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT ST_AsEWKB(t0.geo_linestring) AS geo_linestring \n"
        "FROM geo AS t0"
    )
    assert compiled == expected
    data = expr.execute()
    assert data.geom_type.iloc[0] == 'LineString'


def test_select_polygon_geodata(geotable):
    expr = geotable['geo_polygon']
    sqla_expr = expr.compile()
    compiled = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT ST_AsEWKB(t0.geo_polygon) AS geo_polygon \n" "FROM geo AS t0"
    )
    assert compiled == expected
    data = expr.execute()
    assert data.geom_type.iloc[0] == 'Polygon'


def test_select_multipolygon_geodata(geotable):
    expr = geotable['geo_multipolygon']
    sqla_expr = expr.compile()
    compiled = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT ST_AsEWKB(t0.geo_multipolygon) AS geo_multipolygon \n"
        "FROM geo AS t0"
    )
    assert compiled == expected
    data = expr.execute()
    assert data.geom_type.iloc[0] == 'MultiPolygon'


def test_geo_area(geotable, gdf):
    expr = geotable.geo_multipolygon.area()
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_multipolygon).area
    tm.assert_series_equal(result, expected, check_names=False)


def test_geo_buffer(geotable, gdf):
    expr = geotable.geo_linestring.buffer(1.0)
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_linestring).buffer(1.0)
    tm.assert_series_equal(
        result.area, expected.area, check_names=False, check_less_precise=2
    )


def test_geo_contains(geotable):
    expr = geotable.geo_point.buffer(1.0).contains(geotable.geo_point)
    assert expr.execute().all()


def test_geo_contains_properly(geotable):
    expr = geotable.geo_point.buffer(1.0).contains_properly(geotable.geo_point)
    assert expr.execute().all()


def test_geo_covers(geotable):
    expr = geotable.geo_point.buffer(1.0).covers(geotable.geo_point)
    assert expr.execute().all()


def test_geo_covered_by(geotable):
    expr = geotable.geo_point.covered_by(geotable.geo_point.buffer(1.0))
    assert expr.execute().all()


def test_geo_d_fully_within(geotable):
    expr = geotable.geo_point.d_fully_within(
        geotable.geo_point.buffer(1.0), 2.0
    )
    assert expr.execute().all()


def test_geo_d_within(geotable):
    expr = geotable.geo_point.d_within(geotable.geo_point.buffer(1.0), 1.0)
    assert expr.execute().all()


def test_geo_end_point(geotable, gdf):
    expr = geotable.geo_linestring.end_point()
    result = expr.execute()
    end_point = gdf.apply(
        lambda x: x.geo_linestring.interpolate(1, True), axis=1
    )
    for a, b in zip(result, end_point):
        assert a.equals(b)


def test_geo_envelope(geotable, gdf):
    expr = geotable.geo_linestring.buffer(1.0).envelope()
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_linestring).buffer(1.0).envelope
    tm.assert_series_equal(result.area, expected.area, check_names=False)


def test_geo_within(geotable):
    expr = geotable.geo_point.within(geotable.geo_point.buffer(1.0))
    assert expr.execute().all()


def test_geo_disjoint(geotable):
    expr = geotable.geo_point.disjoint(geotable.geo_point)
    assert not expr.execute().any()


def test_geo_equals(geotable):
    expr = geotable.geo_point.equals(geotable.geo_point)
    assert expr.execute().all()


def test_geo_geometry_n(geotable, gdf):
    expr = geotable.geo_multipolygon.geometry_n(1)  # PostGIS is one-indexed.
    result = expr.execute()
    first_polygon = gdf.apply(lambda x: list(x.geo_multipolygon)[0], axis=1)
    for a, b in zip(result, first_polygon):
        assert a.equals(b)


def test_geo_geometry_type(geotable):
    expr = geotable.geo_point.geometry_type()
    assert (expr.execute() == 'ST_Point').all()
    expr = geotable.geo_multipolygon.geometry_type()
    assert (expr.execute() == 'ST_MultiPolygon').all()


def test_geo_intersects(geotable):
    expr = geotable.geo_point.intersects(geotable.geo_point.buffer(1.0))
    assert expr.execute().all()


def test_geo_is_valid(geotable):
    expr = geotable.geo_point.is_valid()
    assert expr.execute().all()


def test_geo_line_locate_point(geotable):
    expr = geotable.geo_linestring.line_locate_point(geotable.geo_point)
    assert (expr.execute() == 0).all()


def test_geo_line_merge(geotable, gdf):
    expr = geotable.geo_linestring.line_merge()
    expected = gp.GeoSeries(gdf.geo_linestring)
    tm.assert_series_equal(expr.execute().length, expected.length)


def test_geo_line_substring(geotable, gdf):
    expr = geotable.geo_linestring.line_substring(0.25, 0.75)
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_linestring)
    tm.assert_series_equal(expected.length / 2.0, result.length)


def test_geo_ordering_equals(geotable):
    expr = geotable.geo_point.ordering_equals(geotable.geo_point)
    assert expr.execute().all()


def test_geo_overlaps(geotable):
    expr = geotable.geo_point.overlaps(geotable.geo_point.buffer(1.0))
    assert not expr.execute().any()


def test_geo_touches(geotable):
    expr = geotable.geo_point.touches(geotable.geo_linestring)
    assert expr.execute().all()


def test_geo_distance(geotable, gdf):
    expr = geotable.geo_point.distance(geotable.geo_multipolygon.centroid())
    result = expr.execute()
    expected = gdf.geo_point.distance(
        gp.GeoSeries(gdf.geo_multipolygon).centroid
    )
    tm.assert_series_equal(result, expected, check_names=False)


def test_geo_length(geotable, gdf):
    expr = geotable.geo_linestring.length()
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_linestring).length
    tm.assert_series_equal(result, expected, check_names=False)


def test_geo_n_points(geotable):
    expr = geotable.geo_linestring.n_points()
    result = expr.execute()
    assert (result == 2).all()


def test_geo_perimeter(geotable):
    expr = geotable.geo_multipolygon.perimeter()
    result = expr.execute()
    # Geopandas doesn't implement perimeter, so we do a simpler check.
    assert (result > 0.0).all()


def test_geo_srid(geotable):
    expr = geotable.geo_linestring.srid()
    result = expr.execute()
    assert (result == 0).all()


def test_geo_start_point(geotable, gdf):
    expr = geotable.geo_linestring.start_point()
    result = expr.execute()
    start_point = gdf.apply(
        lambda x: x.geo_linestring.interpolate(0, True), axis=1
    )
    for a, b in zip(result, start_point):
        assert a.equals(b)


def test_geo_difference(geotable, gdf):
    expr = (
        geotable.geo_linestring.buffer(1.0)
        .difference(geotable.geo_point.buffer(0.5))
        .area()
    )
    result = expr.execute()
    expected = (
        gp.GeoSeries(gdf.geo_linestring)
        .buffer(1.0)
        .difference(gp.GeoSeries(gdf.geo_point).buffer(0.5))
        .area
    )
    tm.assert_series_equal(
        result, expected, check_names=False, check_less_precise=2
    )


def test_geo_intersection(geotable, gdf):
    expr = (
        geotable.geo_linestring.buffer(1.0)
        .intersection(geotable.geo_point.buffer(0.5))
        .area()
    )
    result = expr.execute()
    expected = (
        gp.GeoSeries(gdf.geo_linestring)
        .buffer(1.0)
        .intersection(gp.GeoSeries(gdf.geo_point).buffer(0.5))
        .area
    )
    tm.assert_series_equal(
        result, expected, check_names=False, check_less_precise=2
    )


def test_geo_unary_union(geotable, gdf):
    expr = geotable.geo_polygon.unary_union().area()
    expected = gp.GeoSeries(gdf.geo_polygon).unary_union.area
    testing.assert_almost_equal(expr.execute(), expected, decimal=2)


def test_geo_union(geotable, gdf):
    expr = geotable.geo_polygon.union(geotable.geo_multipolygon).area()
    expected = (
        gp.GeoSeries(gdf.geo_polygon)
        .union(gp.GeoSeries(gdf.geo_multipolygon))
        .area
    )
    tm.assert_series_equal(
        expr.execute(), expected, check_names=False, check_less_precise=2
    )


def test_geo_x(geotable, gdf):
    expr = geotable.geo_point.x()
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_point).x
    tm.assert_series_equal(result, expected, check_names=False)


def test_geo_y(geotable, gdf):
    expr = geotable.geo_point.y()
    result = expr.execute()
    expected = gp.GeoSeries(gdf.geo_point).y
    tm.assert_series_equal(result, expected, check_names=False)
