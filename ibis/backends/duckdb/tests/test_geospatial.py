from __future__ import annotations

import geopandas as gpd
import numpy.testing as npt
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis


def test_geospatial_point(zones, zones_gdf):
    coord = zones.x_cent.point(zones.y_cent).name("coord")
    # this returns GeometryArray
    gp_coord = gpd.points_from_xy(zones_gdf.x_cent, zones_gdf.y_cent)

    npt.assert_array_equal(gpd.array.from_wkb(coord.to_pandas().values), gp_coord)


def test_geospatial_as_text(snapshot):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = t.geom.as_text().name("tmp")

    snapshot.assert_match(ibis.to_sql(expr), "out.sql")


def test_geospatial_area(zones, zones_gdf):
    gp_area = zones_gdf.geometry.area
    area = zones.geom.area().name("area")

    tm.assert_series_equal(area.to_pandas(), gp_area, check_names=False)


def test_geospatial_centroid(zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid

    tm.assert_series_equal(gpd.GeoSeries.from_wkb(cen.to_pandas().values), gp_cen)


def test_geospatial_contains(zones, zones_gdf):
    cont = zones.geom.contains(zones.geom).name("contains")
    gp_cont = zones_gdf.geometry.contains(zones_gdf.geometry)

    tm.assert_series_equal(cont.to_pandas(), gp_cont, check_names=False)


def test_geospatial_covers(zones, zones_gdf):
    # Note that ST_Covers(A, B) = ST_CoveredBy(B, A)
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    covers = zones.geom.covers(zones.geom).name("covers")
    gp_covers = zones_gdf.geometry.covers(zones_gdf.geometry)

    tm.assert_series_equal(covers.to_pandas(), gp_covers, check_names=False)


def test_geospatial_covered_by(zones, zones_gdf):
    # Note that  ST_CoveredBy(A, B) = ST_Covers(B,A)
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    coverby = zones.geom.covered_by(zones.geom).name("coverby")
    gp_coverby = zones_gdf.geometry.covered_by(zones_gdf.geometry)

    tm.assert_series_equal(coverby.to_pandas(), gp_coverby, check_names=False)


def test_geospatial_crosses(zones, zones_gdf):
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    crosses = zones.geom.crosses(zones.geom).name("crosses")
    gp_crosses = zones_gdf.geometry.crosses(zones_gdf.geometry)

    tm.assert_series_equal(crosses.to_pandas(), gp_crosses, check_names=False)


def test_geospatial_diff(zones, zones_gdf):
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    diff = zones.geom.difference(zones.geom).name("diff")
    gp_diff = zones_gdf.geometry.difference(zones_gdf.geometry)

    tm.assert_series_equal(gpd.GeoSeries.from_wkb(diff.to_pandas().values), gp_diff)


def test_geospatial_disjoint(zones, zones_gdf):
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    disj = zones.geom.disjoint(zones.geom).name("disj")
    gp_disj = zones_gdf.geometry.disjoint(zones_gdf.geometry)

    tm.assert_series_equal(disj.to_pandas(), gp_disj, check_names=False)


def test_geospatial_distance(zones, zones_gdf):
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    dist = zones.geom.distance(zones.geom).name("dist")
    gp_dist = zones_gdf.geometry.distance(zones_gdf.geometry)

    tm.assert_series_equal(dist.to_pandas(), gp_dist, check_names=False)


def test_geospatial_dwithin(snapshot):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = t.geom.d_within(t.geom, 3.0).name("tmp")

    snapshot.assert_match(ibis.to_sql(expr), "out.sql")


# for end and start point need to have a different dataset that contains
# lines or rings types to test this one.
def test_geospatial_end_point(zones):
    # This operation only applies to linesstrings, it gets the
    # end point of a line or ring if we implement ST_MakeLine or
    # ST_ExteriorRing
    # when xfail fixed, this should failed on polygon type
    epoint = zones.geom.end_point().name("end_point")
    # this returns a series of None because the geom column has only Polygons
    epoint.to_pandas()


def test_geospatial_start_point(zones):
    start_point = zones.geom.start_point().name("start_point")
    # this returns a series of None because the geom column has only Polygons
    start_point.to_pandas()


def test_geospatial_envelope(zones, zones_gdf):
    envelope = zones.geom.envelope().name("envelope")
    gp_envelope = zones_gdf.geometry.envelope

    tm.assert_series_equal(
        gpd.GeoSeries.from_wkb(envelope.to_pandas().values), gp_envelope
    )


def test_geospatial_equals(zones, zones_gdf):
    eqs = zones.geom.geo_equals(zones.geom).name("geo_eq")
    gp_eqs = zones_gdf.geometry.geom_equals(zones_gdf.geometry)

    tm.assert_series_equal(eqs.to_pandas(), gp_eqs, check_names=False)


@pytest.mark.xfail(raises=pa.lib.ArrowTypeError)
def test_geospatial_geometry_type(zones, zones_gdf):
    geom_type = zones.geom.geometry_type().name("geom_type")
    gp_geom_type = zones_gdf.geometry.geom_type

    tm.assert_series_equal(geom_type.to_pandas(), gp_geom_type)


def test_geospatial_intersection(zones, zones_gdf):
    intersection = zones.geom.intersection(zones.geom).name("intersection")
    gp_intersection = zones_gdf.geometry.intersection(zones_gdf.geometry)

    tm.assert_series_equal(
        gpd.GeoSeries.from_wkb(intersection.to_pandas().values), gp_intersection
    )


def test_geospatial_intersects(zones, zones_gdf):
    intersects = zones.geom.intersects(zones.geom).name("intersects")
    gp_intersects = zones_gdf.geometry.intersects(zones_gdf.geometry)

    tm.assert_series_equal(intersects.to_pandas(), gp_intersects, check_names=False)


def test_geospatial_is_valid(zones, zones_gdf):
    is_valid = zones.geom.is_valid().name("is_valid")
    gp_is_valid = zones_gdf.geometry.is_valid

    tm.assert_series_equal(is_valid.to_pandas(), gp_is_valid, check_names=False)


# FIX ME: SEE https://postgis.net/docs/ST_Length.html
# ST_LENGTH returns 0 for the case of polygon or multi polygon while pandas geopandas returns the perimeter.
# https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.length.html

# def test_geospatial_length(zones, zones_gdf):
#     length = zones.geom.length().name("length")
#     gp_length = zones_gdf.geometry.length

#     tm.assert_series_equal(length.to_pandas(), gp_length, check_names=False)


# not implemented in geopandas
def test_geospatial_npoints(snapshot):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = t.geom.n_points().name("tmp")

    snapshot.assert_match(ibis.to_sql(expr), "out.sql")


def test_geospatial_overlaps(zones, zones_gdf):
    overlaps = zones.geom.overlaps(zones.geom).name("overlaps")
    gp_overlaps = zones_gdf.geometry.overlaps(zones_gdf.geometry)

    tm.assert_series_equal(overlaps.to_pandas(), gp_overlaps, check_names=False)


def test_geospatial_touches(zones, zones_gdf):
    touches = zones.geom.touches(zones.geom).name("touches")
    gp_touches = zones_gdf.geometry.touches(zones_gdf.geometry)

    tm.assert_series_equal(touches.to_pandas(), gp_touches, check_names=False)


def test_geospatial_union(zones, zones_gdf):
    union = zones.geom.union(zones.geom).name("union")
    gp_union = zones_gdf.geometry.union(zones_gdf.geometry)

    tm.assert_series_equal(gpd.GeoSeries.from_wkb(union.to_pandas().values), gp_union)


def test_geospatial_within(zones, zones_gdf):
    within = zones.geom.within(zones.geom).name("within")
    gp_within = zones_gdf.geometry.within(zones_gdf.geometry)

    tm.assert_series_equal(within.to_pandas(), gp_within, check_names=False)


def test_geospatial_x(zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid
    # Get x coord
    x = cen.x().name("x")
    tm.assert_series_equal(x.to_pandas(), gp_cen.x, check_names=False)


def test_geospatial_y(zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid
    # Get y coord
    y = cen.y().name("y")
    tm.assert_series_equal(y.to_pandas(), gp_cen.y, check_names=False)


# def test_geospatial_buffer()
# GeoBUffer in alchemy supports 2 arguments, but duckdb is a unary?
# actually docs are wrong, it takes 2 or 3 args
# looks like we have fixed_arity(sa.func.ST_Buffer, 2)
