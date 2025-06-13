from __future__ import annotations

import os
from operator import attrgetter, methodcaller

import numpy.testing as npt
import pandas.testing as tm
import pytest
from packaging.version import parse as vparse
from pytest import param

import ibis
from ibis.conftest import LINUX, MACOS, SANDBOXED, WINDOWS

duckdb = pytest.importorskip("duckdb")
gpd = pytest.importorskip("geopandas")
gtm = pytest.importorskip("geopandas.testing")
shapely = pytest.importorskip("shapely")

Point = shapely.Point


def test_geospatial_point(zones, zones_gdf):
    coord = zones.x_cent.point(zones.y_cent).name("coord")
    # this returns GeometryArray
    gp_coord = gpd.points_from_xy(zones_gdf.x_cent, zones_gdf.y_cent)

    npt.assert_array_equal(coord.to_pandas().values, gp_coord)


# this functions are not implemented in geopandas
@pytest.mark.parametrize("operation", ["as_text", "n_points"])
def test_geospatial_unary_snapshot(operation, assert_sql):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = getattr(t.geom, operation)().name("tmp")
    assert_sql(expr)


def test_geospatial_dwithin(assert_sql):
    t = ibis.table([("geom", "geometry")], name="t")
    expr = t.geom.d_within(t.geom, distance=3.0).name("tmp")
    assert_sql(expr)


# geospatial unary functions that return a non-geometry series
# we can test using pd.testing (tm)
@pytest.mark.parametrize(
    ("op", "gp_op"),
    [
        param("area", "area", id="area"),
        param("is_valid", "is_valid", id="is_valid"),
        param(
            "geometry_type",
            "geom_type",
            id="geometry_type",
            marks=pytest.mark.xfail(
                raises=AssertionError, reason="capitalization is different"
            ),
        ),
    ],
)
def test_geospatial_unary_tm(op, gp_op, zones, zones_gdf):
    expr = getattr(zones.geom, op)().name("tmp")
    gp_expr = getattr(zones_gdf.geometry, gp_op)

    tm.assert_series_equal(expr.to_pandas(), gp_expr, check_names=False)


@pytest.mark.parametrize("op", ["x", "y"], ids=["x_coord", "y_coord"])
def test_geospatial_xy(op, zones, zones_gdf):
    cen = zones.geom.centroid().name("centroid")
    gp_cen = zones_gdf.geometry.centroid

    expr = getattr(cen, op)().name("tmp")
    gp_expr = getattr(gp_cen, op)

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
@pytest.mark.parametrize("op", ["centroid", "envelope"])
def test_geospatial_unary_gtm(op, zones, zones_gdf):
    expr = getattr(zones.geom, op)().name("tmp")
    gp_expr = getattr(zones_gdf.geometry, op)

    gtm.assert_geoseries_equal(expr.to_pandas(), gp_expr, check_crs=False)


# geospatial binary functions that return a geometry series
# we can test using gpd.testing (gtm)
@pytest.mark.parametrize("op", ["difference", "intersection", "union"])
def test_geospatial_binary_gtm(op, zones, zones_gdf):
    expr = getattr(zones.geom, op)(zones.geom).name("tmp")
    gp_func = getattr(zones_gdf.geometry, op)(zones_gdf.geometry)

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
@pytest.mark.parametrize(
    "expected_func",
    [
        param(
            attrgetter("unary_union"),
            marks=pytest.mark.xfail(
                condition=vparse(gpd.__version__) >= vparse("1"),
                raises=Warning,
                reason="unary_union property is deprecated",
            ),
            id="version<1",
        ),
        param(
            methodcaller("union_all"),
            marks=pytest.mark.xfail(
                condition=(
                    vparse(duckdb.__version__) < vparse("1.1.1")
                    and (
                        vparse(gpd.__version__) < vparse("1")
                        or vparse(shapely.__version__) >= vparse("2.0.5")
                    )
                ),
                raises=(AttributeError, AssertionError),
                reason="union_all doesn't exist; shapely 2.0.5 results in a different value for union_all",
            ),
            id="version>=1",
        ),
    ],
)
def test_geospatial_unary_union(zones, zones_gdf, expected_func):
    unary_union = zones.geom.unary_union().name("unary_union")
    # this returns a shapely geometry object
    gp_unary_union = expected_func(zones_gdf.geometry)

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


@pytest.mark.xfail_version(
    duckdb=["shapely>=2.1.0"], raises=AssertionError, reason="numerics are different"
)
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


def test_geospatial_flip_coordinates(geotable):
    flipped = geotable.geom.flip_coordinates()

    # flipped coords
    point = shapely.Point(40, -100)
    line_string = shapely.LineString([[0, 0], [1, 1], [1, 2], [2, 2]])
    polygon = shapely.Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0)))

    d = {
        "name": ["Point", "LineString", "Polygon"],
        "geometry": [point, line_string, polygon],
    }

    flipped_gdf = gpd.GeoDataFrame(d)

    gtm.assert_geoseries_equal(
        flipped.to_pandas(), flipped_gdf.geometry, check_crs=False
    )


def test_create_table_geospatial_types(geotable, con):
    name = ibis.util.gen_name("geotable")

    # con = ibis.get_backend(geotable)

    t = con.create_table(name, geotable, temp=True)

    assert t.op().name in con.list_tables()
    assert any(map(methodcaller("is_geospatial"), t.schema().values()))


# geo literals declaration
point = ibis.literal((1, 0), type="point").name("p")
point_geom = ibis.literal((1, 0), type="point:geometry").name("p")


@pytest.mark.parametrize("expr", [point, point_geom])
def test_literal_geospatial_explicit(expr, assert_sql):
    assert_sql(expr)


# test input data with shapely geometries
shp_point_0 = shapely.Point(0, 0)
shp_point_1 = shapely.Point(1, 1)
shp_point_2 = shapely.Point(2, 2)

shp_linestring_0 = shapely.LineString([shp_point_0, shp_point_1, shp_point_2])
shp_linestring_1 = shapely.LineString([shp_point_2, shp_point_1, shp_point_0])
shp_polygon_0 = shapely.Polygon(shp_linestring_0)
shp_multilinestring_0 = shapely.MultiLineString([shp_linestring_0, shp_linestring_1])
shp_multipoint_0 = shapely.MultiPoint([shp_point_0, shp_point_1, shp_point_2])
shp_multipolygon_0 = shapely.MultiPolygon([shp_polygon_0])


@pytest.mark.parametrize(
    "shp",
    [
        param(shp_point_0, id="point"),
        param(shp_linestring_0, id="linestring"),
        param(shp_polygon_0, id="polygon"),
        param(shp_multipolygon_0, id="multipolygon"),
        param(shp_multilinestring_0, id="multilinestring"),
        param(shp_multipoint_0, id="multipoint"),
    ],
)
def test_literal_geospatial_inferred(con, shp):
    expr = ibis.literal(shp).name("result")
    result = con.compile(expr)
    assert str(shp) in result


@pytest.mark.skipif(
    (LINUX or MACOS) and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)
def test_load_geo_example(con):
    pytest.importorskip("pins")

    t = ibis.examples.zones.fetch(backend=con)
    assert t.geom.type().is_geospatial()


# For the next two tests we really want to ensure that
# load_extension("spatial") hasn't been run yet, so we create a new connection
# instead of using the con fixture.


@pytest.fixture(scope="session")
def geo_line_lit():
    return ibis.literal(shapely.LineString([[0, 0], [1, 0], [1, 1]]), type="geometry")


def test_geo_unop_geo_literals(con, geo_line_lit):
    """GeoSpatialUnOp operation on a geospatial literal"""
    expr = geo_line_lit.length()
    assert con.execute(expr) == 2


def test_geo_binop_geo_literals(con, geo_line_lit):
    """GeoSpatialBinOp operation on a geospatial literal"""
    expr = geo_line_lit.distance(shapely.Point(0, 0))
    assert con.execute(expr) == 0


def test_cast_wkb_to_geo(con):
    t = con.table("geo_wkb")
    geo_expr = t.geometry.cast("geometry")
    assert geo_expr.type().is_geospatial()
    assert isinstance(con.execute(geo_expr), gpd.GeoSeries)


def test_load_spatial_casting(data_dir, tmp_path_factory):
    # this directory is necessary because of Windows extension downloads race
    # condition
    con = ibis.duckdb.connect(extension_directory=tmp_path_factory.mktemp("extensions"))
    t = con.read_parquet(data_dir / "parquet" / "geo_wkb.parquet")

    geo_expr = t.limit(1).geometry.cast("geometry")

    assert geo_expr.type().is_geospatial()
    assert isinstance(con.execute(geo_expr), gpd.GeoSeries)


def test_geom_from_string(con):
    value = ibis.literal("POINT (1 2)")
    assert value.type().is_string()

    expr = value.cast("geometry")
    result = con.execute(expr)
    assert result == shapely.from_wkt("POINT (1 2)")


def no_roundtrip(
    *,
    reason: str,
    raises: type[Exception] | tuple[type[Exception], ...] = (
        duckdb.IOException,
        duckdb.NotImplementedException,
        duckdb.PermissionException,
    ),
):
    """Mark a test as expected to fail due to a reader/writer issue."""
    return pytest.mark.xfail(raises=raises, reason=reason)


@pytest.mark.parametrize(
    ("driver", "canonical_extension", "kwargs", "preproc"),
    [
        param("ESRI Shapefile", "shp", {}, None, id="shapefile"),
        param("MapInfo File", "mif", {}, None, id="mapinfo"),
        param(
            "S57",
            None,
            {},
            None,
            marks=no_roundtrip(
                reason="GDAL Error (1): Unable to load s57objectclasses.csv.  Unable to continue.",
            ),
            id="s57",
        ),
        param(
            "CSV",
            None,
            {"layer_creation_options": {"geometry_name": "geom", "geometry": "as_wkt"}},
            lambda t: t.mutate(geom=t.geom.as_text()),
            id="csv",
        ),
        ("GML", None, {}, None),
        param(
            "GPX",
            None,
            {},
            None,
            marks=no_roundtrip(reason="no geometry type specified"),
            id="gpx",
        ),
        param("KML", None, {}, None, id="kml"),
        param("GeoJSON", None, {}, None, id="geojson"),
        param("GeoJSONSeq", None, {}, None, id="geojsonseq"),
        param("OGR_GMT", "gmt", {}, None, id="gmt"),
        param("GPKG", None, {}, None, id="gpkg"),
        param("SQLite", None, {}, None, id="sqlite"),
        param(
            "WAsP",
            "map",
            {},
            None,
            marks=[
                no_roundtrip(reason="only linestrings are allowed"),
                pytest.mark.skipif(condition=WINDOWS, reason="hard crash on windows"),
            ],
            id="wasp",
        ),
        param(
            "OpenFileGDB",
            "gdb",
            {
                "layer_creation_options": {"geometry_name": "geom"},
                "geometry_type": "point",
            },
            None,
            id="gdb",
        ),
        param("FlatGeobuf", "fgb", {}, None, id="flatgeobuf"),
        param(
            "Geoconcept",
            "gxt",
            {"srs": "4326"},
            None,
            id="geoconcept",
            marks=no_roundtrip(reason="not entirely sure"),
        ),
        param(
            "PGDUMP",
            None,
            {"layer_creation_options": {"geometry_name": "geom"}},
            None,
            marks=no_roundtrip(reason="can only be read by postgres"),
            id="pgdump",
        ),
        param(
            "GPSBabel",
            None,
            {},
            None,
            id="gpsbabel",
            marks=no_roundtrip(
                reason="duckdb can't write this because it doesn't expose GDAL's dataset creation options"
            ),
        ),
        param("ODS", "ods", {}, lambda t: t.mutate(geom=t.geom.as_text()), id="ods"),
        param("XLSX", "xlsx", {}, lambda t: t.mutate(geom=t.geom.as_text()), id="xlsx"),
        param(
            "Selafin",
            None,
            {},
            None,
            marks=no_roundtrip(reason="duckdb wkb doesn't preserve the geometry type"),
            id="selafin",
        ),
        param("JML", None, {}, None, id="jml"),
        param("VDV", None, {}, lambda t: t.mutate(geom=t.geom.as_text()), id="vdv"),
        param(
            "MVT",
            "pbf",
            {},
            None,
            marks=no_roundtrip(reason="can't read the written file"),
            id="mvt",
        ),
        param("MapML", None, {}, None, id="mapml"),
        param(
            "PMTiles",
            None,
            {},
            None,
            id="pmtiles",
            marks=no_roundtrip(reason="row counts differ", raises=AssertionError),
        ),
        param("JSONFG", None, {}, None, id="jsonfg"),
    ],
)
def test_to_geo(con, driver, canonical_extension, kwargs, preproc, tmp_path):
    data = ibis.memtable({"x": [1, 3], "y": [2, 4]}).mutate(
        geom=lambda t: t.x.point(t.y)
    )

    if preproc is not None:
        data = preproc(data)

    ext = canonical_extension or driver.replace(" ", "_").lower()
    out = tmp_path / f"outfile.{ext}"

    con.to_geo(data, path=out, format=driver, **kwargs)
    dread = con.read_geo(out)

    assert dread.count().execute() == 2


GDAL_DATA = os.environ.get("GDAL_DATA")


@pytest.mark.parametrize(
    "driver",
    [
        param(
            "DGN",
            marks=pytest.mark.xfail(
                condition=GDAL_DATA is None,
                raises=duckdb.IOException,
                reason="GDAL_DATA not set",
            ),
        ),
        param(
            "DXF",
            marks=pytest.mark.xfail(
                condition=GDAL_DATA is None,
                raises=duckdb.IOException,
                reason="GDAL_DATA not set",
            ),
        ),
        "GEORSS",
    ],
)
def test_to_geo_geom_only(con, driver, tmp_path):
    data = ibis.memtable({"x": [1, 3], "y": [2, 4]}).select(
        geom=lambda t: t.x.point(t.y)
    )

    ext = driver.replace(" ", "_").lower()
    out = tmp_path / f"outfile.{ext}"

    con.to_geo(data, path=out, format=driver)
    dread = con.read_geo(out)

    assert dread.count().execute() == 2


def test_cache_geometry(con, monkeypatch):
    # ibis issue #10324

    # monkeypatching is necessary to ensure the correct backend is used for
    # caching
    monkeypatch.setattr(ibis.options, "default_backend", con)
    data = ibis.memtable({"x": [1], "y": [2]})
    data = data.select(geom=data.x.point(data.y)).cache()
    result = data.execute()
    assert result.at[0, "geom"] == shapely.Point(1, 2)
