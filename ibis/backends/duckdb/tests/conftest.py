from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.conftest import SANDBOXED

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ibis.backends.base import BaseBackend

TEST_TABLES_GEO = {
    "zones": ibis.schema(
        {
            "zone": "string",
            "LocationID": "int32",
            "borough": "string",
            "geom": "geometry",
            "x_cent": "float32",
            "y_cent": "float32",
        }
    ),
    "lines": ibis.schema(
        {
            "loc_id": "int32",
            "geom": "geometry",
        }
    ),
}


class TestConf(BackendTest):
    supports_map = True
    deps = "duckdb", "duckdb_engine"
    stateful = False
    supports_tpch = True

    def preload(self):
        if not SANDBOXED:
            self.connection._load_extensions(
                ["httpfs", "postgres_scanner", "sqlite_scanner", "spatial"]
            )

    @property
    def ddl_script(self) -> Iterator[str]:
        from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported

        parquet_dir = self.data_dir / "parquet"
        geojson_dir = self.data_dir / "geojson"
        for table in TEST_TABLES:
            yield (
                f"""
                CREATE OR REPLACE TABLE {table} AS
                SELECT * FROM read_parquet('{parquet_dir / f'{table}.parquet'}')
                """
            )
        if geospatial_supported and not SANDBOXED:
            for table in TEST_TABLES_GEO:
                yield (
                    f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM st_read('{geojson_dir / f'{table}.geojson'}')
                    """
                )
            yield (
                """
                CREATE or REPLACE TABLE geo (name VARCHAR, geom GEOMETRY);
                INSERT INTO geo VALUES
                    ('Point', ST_GeomFromText('POINT(-100 40)')),
                    ('Linestring', ST_GeomFromText('LINESTRING(0 0, 1 1, 2 1, 2 2)')),
                    ('Polygon', ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'));
                """
            )
        yield from super().ddl_script

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        # use an extension directory per test worker to prevent simultaneous
        # downloads
        extension_directory = tmpdir.getbasetemp().joinpath("duckdb_extensions")
        extension_directory.mkdir(exist_ok=True)
        return ibis.duckdb.connect(extension_directory=extension_directory, **kw)

    def load_tpch(self) -> None:
        with self.connection.begin() as con:
            con.exec_driver_sql("CALL dbgen(sf=0.1)")


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="session")
def gpd():
    pytest.importorskip("shapely")
    pytest.importorskip("geoalchemy2")
    return pytest.importorskip("geopandas")


@pytest.fixture(scope="session")
def zones(con, data_dir, gpd):
    return con.read_geo(data_dir / "geojson" / "zones.geojson")


@pytest.fixture(scope="session")
def lines(con, data_dir, gpd):
    return con.read_geo(data_dir / "geojson" / "lines.geojson")


@pytest.fixture(scope="session")
def zones_gdf(data_dir, gpd):
    return gpd.read_file(data_dir / "geojson" / "zones.geojson")


@pytest.fixture(scope="session")
def lines_gdf(data_dir, gpd):
    return gpd.read_file(data_dir / "geojson" / "lines.geojson")


@pytest.fixture(scope="session")
def geotable(con, gpd):
    return con.table("geo")


@pytest.fixture(scope="session")
def gdf(geotable):
    return geotable.execute()
