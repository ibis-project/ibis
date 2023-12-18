from __future__ import annotations

import numpy.testing as npt
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis

gpd = pytest.importorskip("geopandas")
gtm = pytest.importorskip("geopandas.testing")
shapely = pytest.importorskip("shapely")


def test_geospatial_point(zones, zones_gdf):
    coord = zones.x_cent.point(zones.y_cent).name("coord")
    # this returns GeometryArray
    gp_coord = gpd.points_from_xy(zones_gdf.x_cent, zones_gdf.y_cent)

    npt.assert_array_equal(coord.to_pandas().values, gp_coord)


# this functions are not implemented in geopandas
@pytest.mark.parametrize(
    ("operation", "keywords"),
    [
        param("as_text", {}, id="as_text"),
        param("n_points", {}, id="n_points"),
    ],
)
def test_geospatial_unary_snapshot(operation, keywords, snapshot):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = getattr(t.geom, operation)(**keywords).name("tmp")
    snapshot.assert_match(ibis.to_sql(expr), "out.sql")


def test_geospatial_dwithin(snapshot):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = t.geom.d_within(t.geom, 3.0).name("tmp")

    snapshot.assert_match(ibis.to_sql(expr), "out.sql")


# geospatial unary functions that return a non-geometry series
# we can test using pd.testing (tm)
@pytest.mark.parametrize(
    ("op", "keywords", "gp_op"),
    [
        param("area", {}, "area", id="area"),
        param("is_valid", {}, "is_valid", id="is_valid"),
        param(
            "geometry_type",
            {},
            "geom_type",
            id="geometry_type",
            marks=pytest.mark.xfail(raises=pa.lib.ArrowTypeError),
        ),
    ],
)
def test_geospatial_unary_tm(op, keywords, gp_op, zones, zones_gdf):
    expr = getattr(zones.geom, op)(**keywords).name("tmp")
    gp_expr = getattr(zones_gdf.geometry, gp_op)

    tm.assert_series_equal(expr.to_pandas(), gp_expr, check_names=False)


@pytest.mark.parametrize(
    ("op", "keywords", "gp_op"),
    [
        param("x", {}, "x", id="x_coord"),
        param("y", {}, "y", id="y_coord"),
    ],
)
def test_geospatial_xy(op, keywords, gp_op, zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid

    expr = getattr(cen, op)(**keywords).name("tmp")
    gp_expr = getattr(gp_cen, gp_op)

    tm.assert_series_equal(expr.to_pandas(), gp_expr, check_names=False)


def test_geospatial_length(lines, lines_gdf):
    # note: ST_LENGTH returns 0 for the case of polygon
    # or multi polygon while pandas geopandas returns the perimeter.
    length = lines.geom.length().name("length")
    gp_length = lines_gdf.geometry.length

    tm.assert_series_equal(length.to_pandas(), gp_length, check_names=False)


# geospatial binary functions that return a non-geometry series
# we can test using pd.testing (tm)
@pytest.mark.parametrize(
    ("op", "gp_op"),
    [
        param("contains", "contains", id="contains"),
        param("geo_equals", "geom_equals", id="geo_eqs"),
        param("covers", "covers", id="covers"),
        param("covered_by", "covered_by", id="covered_by"),
        param("crosses", "crosses", id="crosses"),
        param("disjoint", "disjoint", id="disjoint"),
        param("distance", "distance", id="distance"),
        param("intersects", "intersects", id="intersects"),
        param("overlaps", "overlaps", id="overlaps"),
        param("touches", "touches", id="touches"),
        param("within", "within", id="within"),
    ],
)
def test_geospatial_binary_tm(op, gp_op, zones, zones_gdf):
    expr = getattr(zones.geom, op)(zones.geom).name("tmp")
    gp_func = getattr(zones_gdf.geometry, gp_op)(zones_gdf.geometry)

    tm.assert_series_equal(expr.to_pandas(), gp_func, check_names=False)


# geospatial unary functions that return a geometry series
# we can test using gpd.testing (gtm)
@pytest.mark.parametrize(
    ("op", "gp_op"),
    [
        param("centroid", "centroid", id="centroid"),
        param("envelope", "envelope", id="envelope"),
    ],
)
def test_geospatial_unary_gtm(op, gp_op, zones, zones_gdf):
    expr = getattr(zones.geom, op)().name("tmp")
    gp_expr = getattr(zones_gdf.geometry, gp_op)

    gtm.assert_geoseries_equal(expr.to_pandas(), gp_expr, check_crs=False)


# geospatial binary functions that return a geometry series
# we can test using gpd.testing (gtm)
@pytest.mark.parametrize(
    ("op", "gp_op"),
    [
        param("difference", "difference", id="difference"),
        param("intersection", "intersection", id="intersection"),
        param("union", "union", id=""),
    ],
)
def test_geospatial_binary_gtm(op, gp_op, zones, zones_gdf):
    expr = getattr(zones.geom, op)(zones.geom).name("tmp")
    gp_func = getattr(zones_gdf.geometry, gp_op)(zones_gdf.geometry)

    gtm.assert_geoseries_equal(expr.to_pandas(), gp_func, check_crs=False)


def test_geospatial_end_point(lines, lines_gdf):
    epoint = lines.geom.end_point().name("end_point")
    # geopandas does not have end_point this is a work around to get it
    gp_epoint = lines_gdf.geometry.boundary.explode(index_parts=True).xs(1, level=1)

    gtm.assert_geoseries_equal(epoint.to_pandas(), gp_epoint, check_crs=False)


def test_geospatial_start_point(lines, lines_gdf):
    spoint = lines.geom.start_point().name("start_point")
    # geopandas does not have start_point this is a work around to get it
    gp_spoint = lines_gdf.geometry.boundary.explode(index_parts=True).xs(0, level=1)

    gtm.assert_geoseries_equal(spoint.to_pandas(), gp_spoint, check_crs=False)


# this one takes a bit longer than the rest.
def test_geospatial_unary_union(zones, zones_gdf):
    unary_union = zones.geom.unary_union().name("unary_union")
    # this returns a shapely geometry object
    gp_unary_union = zones_gdf.geometry.unary_union

    # using set_precision because https://github.com/duckdb/duckdb_spatial/issues/189
    assert shapely.equals(
        shapely.set_precision(unary_union.to_pandas(), grid_size=1e-7),
        shapely.set_precision(gp_unary_union, grid_size=1e-7),
    )


def test_geospatial_buffer_point(zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid

    buffer = cen.buffer(100.0)
    # geopandas resolution default is 16, while duckdb is 8.
    gp_buffer = gp_cen.buffer(100.0, resolution=8)

    gtm.assert_geoseries_equal(buffer.to_pandas(), gp_buffer, check_crs=False)


def test_geospatial_buffer(zones, zones_gdf):
    buffer = zones.geom.buffer(100.0)
    # geopandas resolution default is 16, while duckdb is 8.
    gp_buffer = zones_gdf.geometry.buffer(100.0, resolution=8)

    gtm.assert_geoseries_equal(buffer.to_pandas(), gp_buffer, check_crs=False)


# using a smaller dataset for time purposes
def test_geospatial_convert(geotable, gdf):
    # geotable is fabricated but let's say the
    # data is in CRS: EPSG:2263
    # let's transform to EPSG:4326 (latitude-longitude projection)
    geo_ll = geotable.geom.convert("EPSG:2263", "EPSG:4326")

    gdf.crs = "EPSG:2263"
    gdf_ll = gdf.geometry.to_crs(crs=4326)

    gtm.assert_geoseries_equal(
        geo_ll.to_pandas(), gdf_ll, check_less_precise=True, check_crs=False
    )
