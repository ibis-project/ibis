"""Tests for geo spatial data types."""
from __future__ import annotations

import numpy as np
import pytest
from numpy import testing
from pytest import param

import ibis
import ibis.expr.datatypes as dt

pytestmark = pytest.mark.geospatial


# TODO find a way to just run for the backends that support geo, without
# skipping if dependencies are missing
pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

# geo literals declaration
point_0 = ibis.literal((0, 0), type="point").name("p")
point_0_4326 = ibis.literal((0, 0), type="point;4326").name("p")

point_geom_0 = ibis.literal((0, 0), type="point;4326:geometry").name("p")
point_geom_1 = ibis.literal((1, 1), type="point;4326:geometry").name("p")
point_geom_0_srid0 = ibis.literal((0, 0), type="point;0:geometry").name("p")
point_geom_1_srid0 = ibis.literal((1, 1), type="point;0:geometry").name("p")
point_geom_2 = ibis.literal((2, 2), type="point;4326:geometry").name("p")
point_geog_0 = ibis.literal((0, 0), type="point;4326:geography").name("p")
point_geog_1 = ibis.literal((1, 1), type="point;4326:geography").name("p")
point_geog_2 = ibis.literal((2, 2), type="point;4326:geography").name("p")
polygon_0 = ibis.literal(
    (
        ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)),
        ((0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1), (0.1, 0)),
    ),
    type="polygon;4326:geometry",
).name("p")

# test input data with shapely geometries
shp_point_0 = shapely.geometry.Point(0, 0)
shp_point_1 = shapely.geometry.Point(1, 1)
shp_point_2 = shapely.geometry.Point(2, 2)

shp_linestring_0 = shapely.geometry.LineString([shp_point_0, shp_point_1, shp_point_2])
shp_linestring_1 = shapely.geometry.LineString([shp_point_2, shp_point_1, shp_point_0])
shp_polygon_0 = shapely.geometry.Polygon(shp_linestring_0)
shp_multilinestring_0 = shapely.geometry.MultiLineString(
    [shp_linestring_0, shp_linestring_1]
)
shp_multipoint_0 = shapely.geometry.MultiPoint([shp_point_0, shp_point_1, shp_point_2])
shp_multipolygon_0 = shapely.geometry.MultiPolygon([shp_polygon_0])


@pytest.mark.parametrize(
    "expr",
    [
        point_0,
        point_0_4326,
        point_geom_0,
        point_geom_1,
        point_geom_2,
        point_geog_0,
        point_geog_1,
        point_geog_2,
    ],
)
def test_literal_geospatial_explicit(con, expr, snapshot):
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "shp",
    [
        shp_point_0,
        shp_point_1,
        shp_point_2,
        shp_linestring_0,
        shp_linestring_1,
        shp_polygon_0,
        shp_multipolygon_0,
        shp_multilinestring_0,
        param(
            shp_multipoint_0,
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="Bug-fix change in GEOS 3.12 see shapely issue #1992",
            ),
        ),
    ],
)
def test_literal_geospatial_inferred(con, shp, snapshot):
    snapshot.assert_match(con.compile(ibis.literal(shp)), "out.sql")


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        param(lambda t: t["geo_linestring"].length(), [1.41] * 5, id="length"),
        param(lambda t: t["geo_point"].x(), [0, 1, 2, 3, 4], id="x"),
        param(lambda t: t["geo_point"].y(), [0, 1, 2, 3, 4], id="y"),
        param(
            lambda t: t["geo_linestring"].x_min(),
            [0, 1, 2, 3, 4],
            id="x_min",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_linestring"].x_max(),
            [1, 2, 3, 4, 5],
            id="x_max",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_linestring"].y_min(),
            [0, 1, 2, 3, 4],
            id="y_min",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_linestring"].y_max(),
            [1, 2, 3, 4, 5],
            id="y_max",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_multipolygon"].n_rings(),
            [2, 3, 1, 1, 1],
            id="n_rings",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_polygon"].set_srid(4326).perimeter(),
            [96.34, 114.36, 10.24, 10.24, 10.24],
            id="perimeter",
            marks=pytest.mark.notimpl(
                ["postgres"], reason="TODO: fix different results issue"
            ),
        ),
        param(
            lambda t: t["geo_multipolygon"].n_points(),
            [7, 11, 5, 5, 5],
            id="n_points",
            marks=pytest.mark.notimpl(
                ["postgres"], reason="TODO: fix different results issue"
            ),
        ),
    ],
)
def test_geo_spatial_unops(geotable, expr_fn, expected):
    """Testing for geo spatial unary operations."""
    expr = expr_fn(geotable)
    result = expr.execute()
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        param(
            lambda t: t["geo_linestring"].contains(point_geom_1_srid0),
            [False] * 5,  # does not contain the border
            id="contains",
        ),
        param(
            lambda t: t["geo_linestring"].disjoint(point_geom_0_srid0),
            [False, True, True, True, True],
            id="disjoint",
        ),
        param(
            lambda t: t["geo_point"].d_within(point_geom_1_srid0, 2.0),
            [True, True, True, False, False],
            id="d_within",
        ),
        param(
            lambda t: t["geo_point"].d_fully_within(t["geo_linestring"], 2.0),
            [True, True, True, True, True],
            id="d_fully_within",
        ),
        param(
            lambda t: t["geo_linestring"].intersects(point_geom_0_srid0),
            [True, False, False, False, False],
            id="intersects",
        ),
        param(
            lambda t: t["geo_linestring"].distance(point_geom_0_srid0),
            [0.0, 1.41, 2.82, 4.24, 5.66],
            id="distance",
        ),
        param(
            lambda t: t["geo_linestring"].max_distance(point_geom_0_srid0),
            [1.41, 2.82, 4.24, 5.66, 7.08],
            id="max_distance",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t.geo_polygon.contains(ibis.literal(30).point(10)),
            [True, False, False, False, False],
            id="point",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
    ],
)
def test_geo_spatial_binops(geotable, expr_fn, expected):
    """Testing for geo spatial binary operations."""
    expr = expr_fn(geotable)
    result = expr.execute()
    testing.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        param(
            lambda t: t["geo_linestring"].end_point(),
            [True, True, True, True, True],
            id="end_point",
        ),
        param(
            lambda t: t["geo_linestring"].point_n(1),
            [True, True, True, True, True],
            id="point_n",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        param(
            lambda t: t["geo_linestring"].start_point(),
            [True, True, True, True, True],
            id="start_point",
        ),
    ],
)
def test_get_point(geotable, expr_fn, expected):
    """Testing for geo spatial get point operations."""
    arg = expr_fn(geotable)
    # NB: there is a difference in how OmnisciDB and PostGIS consider
    # boundaries with the contains predicate. Work around this by adding a
    # small buffer.
    expr = geotable["geo_linestring"].buffer(0.01).contains(arg)
    result = geotable[geotable, expr.name("tmp")].execute()["tmp"]
    testing.assert_almost_equal(result, expected, decimal=2)


def test_area(con, geotable):
    """Testing for geo spatial area operation."""
    expr = geotable.select("id", tmp=polygon_0.area())
    result = expr.execute()["tmp"].values
    expected = np.array([con.execute(polygon_0).area] * len(result))
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (lambda _: point_geom_2.srid(), 4326),
        (lambda _: point_geom_0.srid(), 4326),
        (lambda t: t.geo_point.srid(), 0),
        (lambda t: t.geo_linestring.srid(), 0),
        (lambda t: t.geo_polygon.srid(), 0),
        (lambda t: t.geo_multipolygon.srid(), 0),
    ],
)
def test_srid(geotable, condition, expected):
    """Testing for geo spatial srid operation."""
    expr = geotable[geotable.id, condition(geotable).name("tmp")]
    result = expr.execute()["tmp"][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (lambda _: point_geom_0.set_srid(4326).srid(), 4326),
        (lambda _: point_geom_0.set_srid(4326).set_srid(0).srid(), 0),
        (lambda t: t.geo_point.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_linestring.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_polygon.set_srid(4326).srid(), 4326),
        (lambda t: t.geo_multipolygon.set_srid(4326).srid(), 4326),
    ],
)
def test_set_srid(geotable, condition, expected):
    """Testing for geo spatial set_srid operation."""
    expr = geotable[geotable.id, condition(geotable).name("tmp")]
    result = expr.execute()["tmp"][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    ("condition", "expected"),
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
            lambda t: t.geo_multipolygon.set_srid(4326).transform(900913).srid(),
            900913,
        ),
    ],
)
def test_transform(geotable, condition, expected):
    """Testing for geo spatial transform operation."""
    expr = geotable[geotable.id, condition(geotable).name("tmp")]
    result = expr.execute()["tmp"][[0]]
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.geo_point.set_srid(4326), id="geom_geo_point"),
        param(lambda t: point_geom_0, id="point_geom_0"),
        param(lambda t: point_geom_1, id="point_geom_1"),
        param(lambda t: point_geom_2, id="point_geom_2"),
        param(lambda t: point_geog_0, id="point_geog_0"),
        param(lambda t: point_geog_1, id="point_geog_1"),
        param(lambda t: point_geog_2, id="point_geog_2"),
    ],
)
def test_cast_geography(geotable, expr_fn):
    """Testing for geo spatial transform operation."""
    p = expr_fn(geotable).cast("geography")
    expr = geotable[geotable.id, p.distance(p).name("tmp")]
    result = expr.execute()["tmp"][[0]]
    # distance from a point to a same point should be 0
    assert np.all(result == 0)


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.geo_point.set_srid(4326), id="t_geo_point"),
        param(lambda _: point_geom_0, id="point_geom_0"),
        param(lambda _: point_geom_1, id="point_geom_1"),
        param(lambda _: point_geom_2, id="point_geom_2"),
        param(lambda _: point_geog_0, id="point_geog_0"),
        param(lambda _: point_geog_1, id="point_geog_1"),
        param(lambda _: point_geog_2, id="point_geog_2"),
    ],
)
def test_cast_geometry(geotable, expr_fn):
    """Testing for geo spatial transform operation."""
    p = expr_fn(geotable).cast("geometry")
    expr = geotable[geotable.id, p.distance(p).name("tmp")]
    result = expr.execute()["tmp"][[0]]
    # distance from a point to a same point should be 0
    assert np.all(result == 0)


def test_geo_dataframe(geotable):
    """Testing for geo dataframe output."""
    import geopandas

    assert isinstance(geotable.execute(), geopandas.geodataframe.GeoDataFrame)


@pytest.mark.parametrize(
    "modifier",
    [
        param({}, id="none"),
        param({"srid": 4326}, id="srid"),
        param({"srid": 4326, "geotype": "geometry"}, id="geometry"),
        param({"srid": 4326, "geotype": "geography"}, id="geography"),
    ],
)
@pytest.mark.parametrize(
    ("shape", "value"),
    [
        # Geometry primitives (2D)
        param("point", (30, 10), id="point"),
        param("linestring", ((30, 10), (10, 30), (40, 40)), id="linestring"),
        param(
            "polygon",
            (
                ((35, 10), (45, 45), (15, 40), (10, 20), (35, 10)),
                ((20, 30), (35, 35), (30, 20), (20, 30)),
            ),
            id="polygon",
        ),
        param(
            "polygon",
            (((30, 10), (40, 40), (20, 40), (10, 20), (30, 10)),),
            id="polygon_single",
        ),
        # Multipart geometries (2D)
        param(
            "multipoint",
            ((10, 40), (40, 30), (20, 20), (30, 10)),
            id="multipoint",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="Bug-fix change in GEOS 3.12 see shapely issue #1992",
            ),
        ),
        param(
            "multilinestring",
            (((10, 10), (20, 20), (10, 40)), ((40, 40), (30, 30), (40, 20), (30, 10))),
            id="multilinestring",
        ),
        param(
            "multipolygon",
            (
                (((40, 40), (20, 45), (45, 30), (40, 40)),),
                (
                    ((20, 35), (10, 30), (10, 10), (30, 5), (45, 20), (20, 35)),
                    ((30, 20), (20, 15), (20, 25), (30, 20)),
                ),
            ),
            id="multipolygon",
        ),
    ],
)
def test_geo_literals_smoke(con, shape, value, modifier, snapshot):
    """Smoke tests for geo spatial literals."""
    expr = ibis.literal(value, type=getattr(dt, shape).copy(**modifier))
    snapshot.assert_match(con.compile(expr), "out.sql")


@pytest.mark.parametrize(
    "fn_expr",
    [
        pytest.param(lambda t: t.geo_point.srid(), id="point_srid"),
        pytest.param(lambda t: t.geo_point.set_srid(4326), id="point_set_srid"),
        pytest.param(lambda t: t.geo_point.x(), id="point_x"),
        pytest.param(lambda t: t.geo_point.y(), id="point_y"),
        pytest.param(
            lambda t: t.geo_linestring.contains(t.geo_point),
            id="linestring_contains",
        ),
        pytest.param(lambda t: t.geo_linestring.end_point(), id="linestring_end_point"),
        pytest.param(lambda t: t.geo_linestring.length(), id="linestring_length"),
        pytest.param(
            lambda t: t.geo_linestring.max_distance(t.geo_point),
            id="linestring_max_distance",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(
            lambda t: t.geo_linestring.point_n(1),
            id="linestring_point_n",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(
            lambda t: t.geo_linestring.start_point(),
            id="linestring_start_point",
        ),
        pytest.param(
            lambda t: t.geo_linestring.x_max(),
            id="linestring_x_max",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(
            lambda t: t.geo_linestring.x_min(),
            id="linestring_x_min",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(
            lambda t: t.geo_linestring.y_max(),
            id="linestring_y_max",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(
            lambda t: t.geo_linestring.y_min(),
            id="linestring_y_min",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
        pytest.param(lambda t: t.geo_polygon.area(), id="polygon_area"),
        pytest.param(lambda t: t.geo_polygon.perimeter(), id="polygon_perimeter"),
        pytest.param(
            lambda t: t.geo_multipolygon.n_points(), id="multipolygon_n_points"
        ),
        pytest.param(
            lambda t: t.geo_multipolygon.n_rings(),
            id="multipolygon_n_rings",
            marks=pytest.mark.notimpl(["postgres"]),
        ),
    ],
)
def test_geo_ops_smoke(geotable, fn_expr, snapshot):
    """Smoke tests for geo spatial operations."""
    snapshot.assert_match(fn_expr(geotable).compile(), "out.sql")


def test_geo_equals(geotable, snapshot):
    # Fix https://github.com/ibis-project/ibis/pull/2956
    expr = geotable.mutate(
        Location_Latitude=geotable.geo_point.y(), Latitude=geotable.geo_point.y()
    )

    snapshot.assert_match(expr.compile(), "out1.sql")

    # simple test using ==
    expr = geotable.geo_point == geotable.geo_point
    snapshot.assert_match(expr.compile(), "out2.sql")
    result = expr.execute()
    assert not result.empty
    assert result.all()

    # using geo_equals
    expr = geotable.geo_point.geo_equals(geotable.geo_point).name("tmp")
    snapshot.assert_match(expr.compile(), "out3.sql")
    result = expr.execute()
    assert not result.empty
    assert result.all()

    # equals returns a boolean object
    assert geotable.geo_point.equals(geotable.geo_point)
