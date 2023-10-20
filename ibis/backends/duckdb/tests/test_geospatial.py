from __future__ import annotations

import pyarrow as pa
import pytest
import sqlalchemy as sa

# import ibis

# how to load geo extension here, do I need to create a con
# to_sql = ibis.duckdb.compile


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_point(con):
    zones = con.tables.zones
    coord = zones.x_cent.point(zones.y_cent).name("coord")
    coord.to_pandas()


def test_geospatial_as_text(con):
    # there is no geopandas available to convert to text
    # do snapshot test
    zones = con.tables.zones
    at = zones.geom.as_text().name("as_text")
    at.to_pandas()
    # ASK HERE
    # t = ibis.table([("geom", "geometry")], name="t")
    # expr = t.geom.as_text().name("tmp")
    # snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_area(con, zones_gdf):
    zones = con.tables.zones
    gp_area = zones_gdf.geometry.area
    area = zones.geom.area().name("area")

    assert all(area.to_pandas() == gp_area)


# def test_geospatial_buffer()
# GeoBUffer in alchemy supports 2 arguments, but duckdb is a unary?
# actually docs are wrong, it takes 2 or 3 args
# looks like we have fixed_arity(sa.func.ST_Buffer, 2)


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_centroid(con):
    zones = con.tables.zones
    cen = zones.geom.centroid().name("centroid")
    cen.to_pandas()


## ????
def test_geospatial_contains(con, zones_gdf):
    zones = con.tables.zones
    # using same geom because of issue to generate geojason
    # with 2 geom cols on duckdb
    gp_cont = zones_gdf.geometry.contains(zones_gdf.geometry)
    cont = zones.geom.contains(zones.geom).name("contains")
    cont.to_pandas()
    assert all(cont.to_pandas() == gp_cont)


def test_geospatial_covers(con):
    # Note that ST_Covers(A, B) = ST_CoveredBy(B, A)
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    covers = zones.geom.covers(zones.geom).name("covers")
    covers.to_pandas()


def test_geospatial_covered_by(con):
    # Note that  ST_CoveredBy(A, B) = ST_Covers(B,A)
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    coverby = zones.geom.covered_by(zones.geom).name("coverby")
    coverby.to_pandas()


def test_geospatial_crosses(con):
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    crosses = zones.geom.crosses(zones.geom).name("crosses")
    crosses.to_pandas()


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_diff(con):
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    diff = zones.geom.difference(zones.geom).name("diff")
    diff.to_pandas()


def test_geospatial_disjoint(con):
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    disj = zones.geom.disjoint(zones.geom).name("disj")
    disj.to_pandas()


def test_geospatial_distance(con):
    zones = con.tables.zones
    # using same geom because of issue to generate geojason with 2 geom cols on duckdb
    dist = zones.geom.distance(zones.geom).name("dist")
    dist.to_pandas()


# #try this one after I merge master.
# def test_geospatial_dwithin(con):
#     zones = con.tables.zones
#     #using same geom because of issue to generate geojason with 2 geom cols on duckdb
#     dwithin = zones.geom.d_within(zones.geom, 3.0).name("dwithin")
#     dwithin.to_pandas()


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_end_point(con):
    zones = con.tables.zones
    epoint = zones.geom.end_point().name("end_point")
    epoint.to_pandas()


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_envelope(con):
    zones = con.tables.zones
    envelope = zones.geom.envelope().name("envelope")
    envelope.to_pandas()


def test_geospatial_equals(con):
    zones = con.tables.zones
    eqs = zones.geom.geo_equals(zones.geom).name("geo_eq")
    eqs.to_pandas()


@pytest.mark.xfail(raises=pa.lib.ArrowTypeError)
def test_geospatial_geometry_type(con):
    zones = con.tables.zones
    geom_type = zones.geom.geometry_type().name("geom_type")
    geom_type.to_pandas()  # breaks here


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_intersection(con):
    zones = con.tables.zones
    intersection = zones.geom.intersection(zones.geom).name("intersection")
    intersection.to_pandas()


def test_geospatial_intersects(con):
    zones = con.tables.zones
    intersects = zones.geom.intersects(zones.geom).name("intersects")
    intersects.to_pandas()


def test_geospatial_is_valid(con):
    zones = con.tables.zones
    is_valid = zones.geom.is_valid().name("is_valid")
    is_valid.to_pandas()


def test_geospatial_length(con):
    zones = con.tables.zones
    length = zones.geom.length().name("length")
    length.to_pandas()


def test_geospatial_npoints(con):
    zones = con.tables.zones
    npoints = zones.geom.n_points()
    npoints.to_pandas()


def test_geospatial_overlaps(con):
    zones = con.tables.zones
    overlaps = zones.geom.overlaps(zones.geom).name("overlaps")
    overlaps.to_pandas()


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_start_point(con):
    zones = con.tables.zones
    start_point = zones.geom.start_point().name("start_point")
    start_point.to_pandas()


def test_geospatial_touches(con):
    zones = con.tables.zones
    touches = zones.geom.touches(zones.geom).name("touches")
    touches.to_pandas()


@pytest.mark.xfail(raises=sa.exc.ProgrammingError, reason="ST_AsEWKB")
def test_geospatial_union(con):
    zones = con.tables.zones
    union = zones.geom.union(zones.geom).name("union")
    union.to_pandas()


def test_geospatial_within(con):
    zones = con.tables.zones
    within = zones.geom.within(zones.geom).name("within")
    within.to_pandas()


def test_geospatial_x(con):
    # we need to have a point type column to test this, need to figure geojson problem
    # duckdb.duckdb.InvalidInputException: Invalid Input Error: ST_X/ST_Y only supports POINT geometries
    # try to create it with geojson only accepts one geometry column
    zones = con.tables.zones
    # work around: get centroids to get Point() like things
    cen = zones.geom.centroid().name("centroid")
    # Get x coord
    x = cen.x().name("x")
    x.to_pandas()


def test_geospatial_y(con):
    # we need to have a point type column to test this, need to figure geojson problem
    # duckdb.duckdb.InvalidInputException: Invalid Input Error: ST_X/ST_Y only supports POINT geometries
    # try to create it with geojson only accepts one geometry column
    zones = con.tables.zones
    # work around: get centroids to get Point() like things
    cen = zones.geom.centroid().name("centroid")
    # Get x coord
    y = cen.y().name("y")
    y.to_pandas()
