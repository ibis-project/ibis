from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import sqlglot as sg

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.conftest import SANDBOXED, WINDOWS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ibis.backends import BaseBackend

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

TEST_TABLE_GEO_PARQUET = {
    "geo_wkb": ibis.schema(
        {
            "name": "string",
            "geom": "binary",
        }
    ),
}


class TestConf(BackendTest):
    supports_map = True
    deps = ("duckdb",)
    stateful = False
    supports_tpch = True
    supports_tpcds = True
    driver_supports_multiple_statements = True

    def preload(self):
        if not SANDBOXED:
            self.connection._load_extensions(
                ["httpfs", "postgres_scanner", "sqlite_scanner", "spatial"]
            )

    @property
    def ddl_script(self) -> Iterator[str]:
        parquet_dir = self.data_dir / "parquet"
        geojson_dir = self.data_dir / "geojson"
        for table in TEST_TABLES:
            yield (
                f"""
                CREATE OR REPLACE TABLE {table} AS
                SELECT * FROM read_parquet('{parquet_dir / f"{table}.parquet"}')
                """
            )
        if not SANDBOXED:
            for table in TEST_TABLES_GEO:
                yield (
                    f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM st_read('{geojson_dir / f"{table}.geojson"}')
                    """
                )
            for table in TEST_TABLE_GEO_PARQUET:
                # the ops on this table will need the spatial extension
                yield (
                    f"""
                CREATE OR REPLACE TABLE {table} AS
                SELECT * FROM read_parquet('{parquet_dir / f"{table}.parquet"}')
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
        # downloads on windows
        #
        # avoid enabling on linux because this adds a lot of time to parallel
        # test runs due to each worker getting its own extensions directory
        if WINDOWS:
            extension_directory = tmpdir.getbasetemp().joinpath("duckdb_extensions")
            extension_directory.mkdir(exist_ok=True)
            kw["extension_directory"] = extension_directory

        kw["threads"] = 1 if worker_id != "master" else os.cpu_count() // 2
        return ibis.duckdb.connect(**kw)

    def _load_tpc(self, *, suite, scale_factor):
        con = self.connection
        schema = f"tpc{suite}"
        con.create_database(schema, force=True)
        parquet_dir = self.data_dir.joinpath(schema, f"sf={scale_factor}", "parquet")
        assert parquet_dir.exists(), parquet_dir
        for path in parquet_dir.glob("*.parquet"):
            table_name = path.with_suffix("").name
            # duckdb automatically infers the sf= as a hive partition so we
            # need to disable it
            con.create_table(
                table_name,
                con.read_parquet(path, hive_partitioning=False),
                database=schema,
            )

    def _transform_tpc_sql(self, parsed, *, suite, leaves):
        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table) and node.name in leaves:
                return node.__class__(
                    catalog=f"tpc{suite}",
                    **{k: v for k, v in node.args.items() if k != "catalog"},
                )
            return node

        return parsed.transform(add_catalog_and_schema)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection


@pytest.fixture(scope="session")
def gpd():
    pytest.importorskip("shapely")
    return pytest.importorskip("geopandas")


@pytest.fixture(scope="session")
def zones(con, data_dir):
    return con.read_geo(data_dir / "geojson" / "zones.geojson")


@pytest.fixture(scope="session")
def lines(con, data_dir):
    return con.read_geo(data_dir / "geojson" / "lines.geojson")


@pytest.fixture(scope="session")
def zones_gdf(data_dir, gpd):
    return gpd.read_file(data_dir / "geojson" / "zones.geojson")


@pytest.fixture(scope="session")
def lines_gdf(data_dir, gpd):
    return gpd.read_file(data_dir / "geojson" / "lines.geojson")


@pytest.fixture(scope="session")
def geotable(con):
    return con.table("geo")


@pytest.fixture(scope="session")
def gdf(geotable):
    return geotable.execute()
