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
SINGLESTOREDB_PASS = os.environ.get("IBIS_TEST_SINGLESTOREDB_PASSWORD", "")
SINGLESTOREDB_HOST = os.environ.get("IBIS_TEST_SINGLESTOREDB_HOST", "localhost")
SINGLESTOREDB_PORT = int(os.environ.get("IBIS_TEST_SINGLESTOREDB_PORT", "3306"))
IBIS_TEST_SINGLESTOREDB_DB = os.environ.get(
    "IBIS_TEST_SINGLESTOREDB_DATABASE", "ibis-testing"
)


class TestConf(ServiceBackendTest):
    # SingleStoreDB has similar behavior to MySQL
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = True  # SingleStoreDB supports JSON arrays
    native_bool = False
    supports_structs = False  # May support in future via JSON
    rounding_method = "half_to_even"
    service_name = "singlestoredb"
    deps = ("singlestoredb",)  # Primary dependency, falls back to MySQLdb

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

        with self.connection.begin() as cur:
            for table in TEST_TABLES:
                csv_path = self.data_dir / "csv" / f"{table}.csv"
                lines = [
                    f"LOAD DATA LOCAL INFILE {str(csv_path)!r}",
                    f"INTO TABLE {table}",
                    "COLUMNS TERMINATED BY ','",
                    """OPTIONALLY ENCLOSED BY '"'""",
                    "LINES TERMINATED BY '\\n'",
                    "IGNORE 1 LINES",
                ]
                cur.execute("\\n".join(lines))

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.singlestoredb.connect(
            host=SINGLESTOREDB_HOST,
            user=SINGLESTOREDB_USER,
            password=SINGLESTOREDB_PASS,
            database=IBIS_TEST_SINGLESTOREDB_DB,
            port=SINGLESTOREDB_PORT,
            local_infile=1,
            autocommit=True,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
