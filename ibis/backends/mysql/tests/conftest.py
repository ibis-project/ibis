from __future__ import annotations

import csv
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
MYSQL_PORT = int(os.environ.get("IBIS_TEST_MYSQL_PORT", "3306"))
IBIS_TEST_MYSQL_DB = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis-testing")


class TestConf(ServiceBackendTest):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    native_bool = False
    supports_structs = False
    rounding_method = "half_to_even"
    service_name = "mysql"
    deps = ("adbc_driver_manager",)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(self, **kwargs: Any) -> None:
        """Load test data into a MySQL backend instance."""
        super()._load_data(**kwargs)

        batch_size = 1000
        with self.connection.con.cursor() as cur:
            for table in TEST_TABLES:
                csv_path = self.data_dir / "csv" / f"{table}.csv"
                with open(csv_path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader)  # skip header
                    columns = ", ".join(f"`{col}`" for col in header)
                    batch = []
                    for row in reader:
                        parts = []
                        for v in row:
                            if v == "":
                                parts.append("NULL")
                            else:
                                escaped = v.replace("\\", "\\\\").replace("'", "\\'")
                                parts.append(f"'{escaped}'")
                        batch.append(f"({', '.join(parts)})")
                        if len(batch) >= batch_size:
                            values_sql = ", ".join(batch)
                            cur.execute(
                                f"INSERT INTO `{table}` ({columns}) VALUES {values_sql}"
                            )
                            batch = []
                    if batch:
                        values_sql = ", ".join(batch)
                        cur.execute(
                            f"INSERT INTO `{table}` ({columns}) VALUES {values_sql}"
                        )

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.mysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=IBIS_TEST_MYSQL_DB,
            port=MYSQL_PORT,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
