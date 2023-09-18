"""Tests for geo spatial data types."""
from __future__ import annotations

import numpy as np
import pytest
from numpy import testing
from pytest import param

import ibis

pytestmark = pytest.mark.geospatial


# TODO find a way to just run for the backends that support geo, without
# skipping if dependencies are missing
pytest.importorskip("geoalchemy2")
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
    ("expr", "expected"),
    [
        (point_0, "'POINT (0.0 0.0)'"),
        (point_0_4326, "'SRID=4326;POINT (0.0 0.0)'"),
        (point_geom_0, "'SRID=4326;POINT (0.0 0.0)'::geometry"),
        (point_geom_1, "'SRID=4326;POINT (1.0 1.0)'::geometry"),
        (point_geom_2, "'SRID=4326;POINT (2.0 2.0)'::geometry"),
        (point_geog_0, "'SRID=4326;POINT (0.0 0.0)'::geography"),
        (point_geog_1, "'SRID=4326;POINT (1.0 1.0)'::geography"),
        (point_geog_2, "'SRID=4326;POINT (2.0 2.0)'::geography"),
    ],
)
def test_literal_geospatial_explicit(con, expr, expected):
    result = str(con.compile(expr))
    assert result == f"SELECT {expected} AS p"


@pytest.mark.parametrize(
    ("shp", "expected"),
    [
        (shp_point_0, "(0 0)"),
        (shp_point_1, "(1 1)"),
        (shp_point_2, "(2 2)"),
        (shp_linestring_0, "(0 0, 1 1, 2 2)"),
        (shp_linestring_1, "(2 2, 1 1, 0 0)"),
        (shp_polygon_0, "((0 0, 1 1, 2 2, 0 0))"),
        (shp_multipolygon_0, "(((0 0, 1 1, 2 2, 0 0)))"),
        (shp_multilinestring_0, "((0 0, 1 1, 2 2), (2 2, 1 1, 0 0))"),
        (shp_multipoint_0, "(0 0, 1 1, 2 2)"),
    ],
)
def test_literal_geospatial_inferred(con, shp, expected):
    result = str(con.compile(ibis.literal(shp).name("result")))
    name = type(shp).__name__.upper()
    pair = f"{name} {expected}"
    assert result == f"SELECT {pair!r} AS result"


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


@pytest.mark.parametrize(("arg", "expected"), [(polygon_0, [1.98] * 5)])
def test_area(geotable, arg, expected):
    """Testing for geo spatial area operation."""
    expr = geotable[geotable.id, arg.area().name("tmp")]
    result = expr.execute()["tmp"]
    testing.assert_almost_equal(result, expected, decimal=2)


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
        {},
        {"srid": "4326"},
        {"srid": "4326", "geo_type": "geometry"},
        {"srid": "4326", "geo_type": "geography"},
    ],
)
@pytest.mark.parametrize(
    ("shape", "value", "expected"),
    [
        # Geometry primitives (2D)
        param("point", (30, 10), "(30.0 10.0)", id="point"),
        param(
            "linestring",
            ((30, 10), (10, 30), (40, 40)),
            "(30.0 10.0, 10.0 30.0, 40.0 40.0)",
            id="linestring",
        ),
        param(
            "polygon",
            (
                ((35, 10), (45, 45), (15, 40), (10, 20), (35, 10)),
                ((20, 30), (35, 35), (30, 20), (20, 30)),
            ),
            (
                "((35.0 10.0, 45.0 45.0, 15.0 40.0, 10.0 20.0, 35.0 10.0), "
                "(20.0 30.0, 35.0 35.0, 30.0 20.0, 20.0 30.0))"
            ),
            id="polygon",
        ),
        param(
            "polygon",
            (((30, 10), (40, 40), (20, 40), (10, 20), (30, 10)),),
            "((30.0 10.0, 40.0 40.0, 20.0 40.0, 10.0 20.0, 30.0 10.0))",
            id="polygon_single",
        ),
        # Multipart geometries (2D)
        param(
            "multipoint",
            ((10, 40), (40, 30), (20, 20), (30, 10)),
            "((10.0 40.0), (40.0 30.0), (20.0 20.0), (30.0 10.0))",
            id="multipoint",
        ),
        param(
            "multilinestring",
            (
                ((10, 10), (20, 20), (10, 40)),
                ((40, 40), (30, 30), (40, 20), (30, 10)),
            ),
            (
                "((10.0 10.0, 20.0 20.0, 10.0 40.0), "
                "(40.0 40.0, 30.0 30.0, 40.0 20.0, 30.0 10.0))"
            ),
            id="multilinestring",
        ),
        param(
            "multipolygon",
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
                "(((40.0 40.0, 20.0 45.0, 45.0 30.0, 40.0 40.0)), "
                "((20.0 35.0, 10.0 30.0, 10.0 10.0, 30.0 5.0, 45.0 20.0, 20.0 35.0), "
                "(30.0 20.0, 20.0 15.0, 20.0 25.0, 30.0 20.0)))"
            ),
            id="multipolygon",
        ),
    ],
)
def test_geo_literals_smoke(con, shape, value, modifier, expected):
    """Smoke tests for geo spatial literals."""
    srid = f";{modifier['srid']}" if "srid" in modifier else ""
    geo_type = f":{modifier['geo_type']}" if "geo_type" in modifier else ""
    expr_type = f"{shape.upper()} {srid}{geo_type}"
    expr = ibis.literal(value, type=expr_type).name("tmp")
    prefix = f"SRID={modifier['srid']};" if "srid" in modifier else ""
    suffix = f"::{modifier['geo_type']}" if "geo_type" in modifier else ""

    result = str(con.compile(expr))
    expected = f"SELECT '{prefix}{shape.upper()} {expected}'{suffix} AS tmp"
    assert result == expected


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
        # TODO: the mock tests don't support multipoint and multilinestring
        #       yet, but once they do, add some more tests here.
    ],
)
def test_geo_ops_smoke(geotable, fn_expr):
    """Smoke tests for geo spatial operations."""
    assert str(fn_expr(geotable).compile())


def test_geo_equals(geotable):
    # Fix https://github.com/ibis-project/ibis/pull/2956
    expr = geotable.mutate(
        [
            geotable.geo_point.y().name("Location_Latitude"),
            geotable.geo_point.y().name("Latitude"),
        ]
    )

    result = str(expr.compile().compile())

    assert result == (
        "SELECT t0.id, ST_AsEWKB(t0.geo_point) AS geo_point, "
        "ST_AsEWKB(t0.geo_linestring) AS geo_linestring, "
        "ST_AsEWKB(t0.geo_polygon) AS geo_polygon, "
        "ST_AsEWKB(t0.geo_multipolygon) AS geo_multipolygon, "
        'ST_Y(t0.geo_point) AS "Location_Latitude", '
        'ST_Y(t0.geo_point) AS "Latitude" \n'
        "FROM geo AS t0"
    )

    # simple test using ==
    expected = "SELECT t0.geo_point = t0.geo_point AS tmp \nFROM geo AS t0"
    expr = geotable.geo_point == geotable.geo_point
    assert str(expr.name("tmp").compile().compile()) == expected
    assert expr.execute().all()

    # using geo_equals
    expected = "SELECT ST_Equals(t0.geo_point, t0.geo_point) AS tmp \nFROM geo AS t0"
    expr = geotable.geo_point.geo_equals(geotable.geo_point).name("tmp")
    assert str(expr.compile().compile()) == expected
    assert expr.execute().all()

    # equals returns a boolean object
    assert geotable.geo_point.equals(geotable.geo_point)
