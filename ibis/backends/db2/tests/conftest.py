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

DB2_USER = os.environ.get("IBIS_TEST_DB2_USER", "db2inst1")
DB2_PASS = os.environ.get("IBIS_TEST_DB2_PASSWORD", "ibistest")
DB2_HOST = os.environ.get("IBIS_TEST_DB2_HOST", "localhost")
DB2_PORT = int(os.environ.get("IBIS_TEST_DB2_PORT", "50000"))
DB2_DATABASE = os.environ.get("IBIS_TEST_DB2_DATABASE", "ibistest")


def pytest_collection_modifyitems(items, config):  # noqa: ARG001
    # `notyet` becomes `xfail` in `pytest_runtest_call`, too late to prevent
    # execution. These tests need collection-time `xfail(run=False)`:
    # emoji fetching can segfault ibm_db, and test_table_info_large can hang.
    # This hook applies only when the Db2 test directory is collected.
    # ibm_db 3.2.9 corrupts non-BMP characters during parameter binding,
    # which can cause a segmentation fault when fetching the affected row.
    # Fixed upstream in #1065, but not available in our pinned version.
    # See https://github.com/ibmdb/python-ibmdb/issues/1064.
    for item in items:
        if (
            item.originalname == "test_string_methods_accents_and_emoji"
            and "[db2-" in item.name
        ):
            item.add_marker(
                pytest.mark.xfail(
                    run=False,
                    reason="ibm_db segfaults when fetching emoji",
                )
            )
        if item.originalname == "test_table_info_large" and "[db2]" in item.name:
            item.add_marker(
                pytest.mark.xfail(
                    run=False,
                    reason=(
                        "Db2 optimizer time grows roughly cubically with the "
                        "width of the UNION ALL that `info()` builds; 129 "
                        "columns does not plan within 15 minutes "
                        "(64 columns already takes ~100s)"
                    ),
                )
            )


class TestConf(ServiceBackendTest):
    check_dtype = False
    force_sort = True  # Db2 does not guarantee row order without ORDER BY
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_structs = False
    supports_json = False
    native_bool = False
    service_name = "db2"
    deps = ("ibm_db", "ibm_db_dbi")

    @staticmethod
    def format_table(name: str) -> str:
        return f'"{name}"'

    @property
    def test_files(self) -> Iterable[Path]:
        # Db2 data loading is done entirely in Python (_load_data).
        # There are no files to docker-cp into the container.
        return iter([])

    def _load_data(self, **_: Any) -> None:
        # The USER TEMPORARY tablespaces come from docker/db2/tablespaces.sh.
        import pyarrow.csv as pcsv

        # ibm_db_dbi does not support multi-statement batches, so run the DDL
        # one statement at a time and commit after each.
        for stmt in self.ddl_script:
            with self.connection._safe_raw_sql(stmt):
                pass
            self.connection.con.commit()

        for table, schema in TEST_TABLES.items():
            csv_path = self.data_dir / "csv" / f"{table}.csv"
            if not csv_path.exists():
                continue
            arrow_table = pcsv.read_csv(
                csv_path,
                convert_options=pcsv.ConvertOptions(
                    column_types=schema.to_pyarrow(), strings_can_be_null=True
                ),
            )
            self.connection.insert(table, arrow_table)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.db2.connect(
            host=DB2_HOST,
            database=DB2_DATABASE,
            user=DB2_USER,
            password=DB2_PASS,
            port=DB2_PORT,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
