from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

import ibis
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
    """Do not run shared tests that Db2 cannot complete or that crash the driver.

    This is a hook and not a `notyet` mark in the shared test files because
    `ibis/backends/conftest.py` only turns those marks into `xfail` while the
    test is already running (in `pytest_runtest_call`), so `run=False` has no
    effect and the test body still executes. These two tests cannot be run:
    one segfaults the interpreter and the other takes longer than the CI job
    allows.

    This hook only applies when `ibis/backends/db2/tests` is collected.
    Running a shared test file alone with `-m db2` skips it, and the emoji
    test then segfaults the driver. Run `pytest ibis -m db2` or include this
    directory in the path.
    """
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
    """Db2 backend test configuration.

    Data loading strategy
    ---------------------
    Db2 has no ``BULK INSERT``/``COPY FROM`` equivalent that works from outside
    the container, so we use ``ibm_db_dbi.executemany`` in Python instead.

    ``ibm_db_dbi.executemany`` requires plain ``list`` rows, not namedtuples.
    """

    check_dtype = False
    force_sort = True  # Db2 does not guarantee row order without ORDER BY
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_structs = False
    supports_json = False
    native_bool = False
    rounding_method = "half_to_even"
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
        """Load the standard ibis test tables into Db2.

        The USER TEMPORARY tablespaces needed for GLOBAL TEMPORARY tables are
        created by ``docker/db2/tablespaces.sh`` when the container starts.

        Steps
        -----
        1. Execute every DDL statement from ``ci/schema/db2.sql`` one at a
           time, committing after each — ibm_db_dbi does not support
           multi-statement batches.
        2. Bulk-insert the five CSV tables via ``executemany`` in batches of
           5 000 rows, committing each batch.
        """
        import pandas as pd

        # Step 1: DDL — uses self.ddl_script (BackendTest.ddl_script reads
        # ci/schema/db2.sql and splits on ";", same as every other backend).
        for stmt in self.ddl_script:
            with self.connection._safe_raw_sql(stmt):
                pass
            self.connection.con.commit()

        # Step 2: Bulk-insert CSV tables.
        csv_dir = self.data_dir / "csv"
        table_files = {
            "diamonds": "diamonds.csv",
            "astronauts": "astronauts.csv",
            "batting": "batting.csv",
            "awards_players": "awards_players.csv",
            "functional_alltypes": "functional_alltypes.csv",
        }
        raw = self.connection.con
        for table, filename in table_files.items():
            csv_path = csv_dir / filename
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Column names are double-quoted in the DDL so they are stored
            # verbatim.  Mirror that quoting in the INSERT statement.
            col_names = ", ".join(f'"{c}"' for c in df.columns)
            placeholders = ", ".join("?" for _ in df.columns)
            insert_sql = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})'

            # ibm_db_dbi.executemany requires plain lists (not namedtuples).
            # Replace NaN/NaT/pd.NA → None so Db2 stores SQL NULL correctly.
            df_obj = df.astype(object).where(pd.notnull(df), None)
            all_rows = [list(row) for row in df_obj.itertuples(index=False, name=None)]
            cursor = raw.cursor()
            try:
                batch_size = 5000
                for i in range(0, len(all_rows), batch_size):
                    cursor.executemany(insert_sql, all_rows[i : i + batch_size])
                    raw.commit()
            finally:
                cursor.close()

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
