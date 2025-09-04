from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# SingleStoreDB test connection parameters
SINGLESTOREDB_USER = os.environ.get("IBIS_TEST_SINGLESTOREDB_USER", "root")
SINGLESTOREDB_PASS = os.environ.get("IBIS_TEST_SINGLESTOREDB_PASSWORD", "ibis_testing")
SINGLESTOREDB_HOST = os.environ.get("IBIS_TEST_SINGLESTOREDB_HOST", "127.0.0.1")
SINGLESTOREDB_PORT = int(os.environ.get("IBIS_TEST_SINGLESTOREDB_PORT", "3307"))
SINGLESTOREDB_HTTP_PORT = int(
    os.environ.get("IBIS_TEST_SINGLESTOREDB_HTTP_PORT", "9089")
)
IBIS_TEST_SINGLESTOREDB_DB = os.environ.get(
    "IBIS_TEST_SINGLESTOREDB_DATABASE", "ibis_testing"
)


class TestConf(ServiceBackendTest):
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = True  # SingleStoreDB supports JSON arrays
    native_bool = False
    supports_structs = False  # May support in future via JSON
    rounding_method = "half_to_even"
    force_sort = True  # SingleStoreDB has non-deterministic row ordering
    service_name = "singlestoredb"
    deps = ("singlestoredb",)  # Primary dependency

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(self, **kwargs: Any) -> None:
        """Load test data into a SingleStoreDB backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        super()._load_data(**kwargs)

        # Check if we're using HTTP protocol by inspecting the connection
        is_http_protocol = (
            hasattr(self.connection, "_client")
            and "http" in self.connection._client.__class__.__module__
        )

        if is_http_protocol:
            # For HTTP protocol, use a MySQL connection for data loading since LOAD DATA LOCAL INFILE
            # is not supported over HTTP
            mysql_connection = ibis.singlestoredb.connect(
                host=SINGLESTOREDB_HOST,
                user=SINGLESTOREDB_USER,
                password=SINGLESTOREDB_PASS,
                database=IBIS_TEST_SINGLESTOREDB_DB,
                port=SINGLESTOREDB_PORT,  # Use MySQL port for data loading
                driver="mysql",
                local_infile=1,
                autocommit=True,
            )

        else:
            mysql_connection = self.connection

        with mysql_connection.begin() as cur:
            for table in TEST_TABLES:
                csv_path = self.data_dir / "csv" / f"{table}.csv"
                lines = [
                    f"LOAD DATA LOCAL INFILE {str(csv_path)!r}",
                    f"INTO TABLE {table}",
                    "FIELDS TERMINATED BY ','",
                    """OPTIONALLY ENCLOSED BY '"'""",
                    "NULL DEFINED BY ''",
                    "LINES TERMINATED BY '\\n'",
                    "IGNORE 1 LINES",
                ]
                cur.execute("\n".join(lines))

        if is_http_protocol:
            mysql_connection.disconnect()

    @staticmethod
    def connect(*, tmpdir, worker_id, driver=None, port=None, **kw):  # noqa: ARG004
        # Use provided port or default MySQL port
        connection_port = port if port is not None else SINGLESTOREDB_PORT
        # Only pass driver parameter if it's not None and not 'mysql' (default)
        driver_kwargs = {"driver": driver} if driver and driver != "mysql" else {}

        return ibis.singlestoredb.connect(
            host=SINGLESTOREDB_HOST,
            user=SINGLESTOREDB_USER,
            password=SINGLESTOREDB_PASS,
            database=IBIS_TEST_SINGLESTOREDB_DB,
            port=connection_port,
            local_infile=1,
            autocommit=True,
            **driver_kwargs,
            **kw,
        )


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("mysql", id="mysql", marks=pytest.mark.singlestoredb_mysql),
        pytest.param("http", id="http", marks=pytest.mark.singlestoredb_http),
    ],
)
def con(request, tmp_path_factory, data_dir, worker_id):
    driver = request.param
    port = SINGLESTOREDB_PORT if driver == "mysql" else SINGLESTOREDB_HTTP_PORT

    # Create a custom TestConf class for this specific connection
    class CustomTestConf(TestConf):
        @staticmethod
        def connect(*, tmpdir, worker_id, **kw):
            return TestConf.connect(
                tmpdir=tmpdir, worker_id=worker_id, driver=driver, port=port, **kw
            )

    with CustomTestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
