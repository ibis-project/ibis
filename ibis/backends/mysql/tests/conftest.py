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

MYSQL_USER = os.environ.get("IBIS_TEST_MYSQL_USER", "ibis")
MYSQL_PASS = os.environ.get("IBIS_TEST_MYSQL_PASSWORD", "ibis")
MYSQL_HOST = os.environ.get("IBIS_TEST_MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("IBIS_TEST_MYSQL_PORT", 3306))
IBIS_TEST_MYSQL_DB = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis_testing")


class TestConf(ServiceBackendTest):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = supports_arrays_outside_of_select = False
    native_bool = False
    supports_structs = False
    rounding_method = "half_to_even"
    service_name = "mysql"
    deps = ("pymysql",)
    supports_window_operations = True

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(self, **kwargs: Any) -> None:
        """Load test data into a MySql backend instance.

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
                cur.execute("\n".join(lines))

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.mysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=IBIS_TEST_MYSQL_DB,
            port=MYSQL_PORT,
            local_infile=1,
            autocommit=True,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
