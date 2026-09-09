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


class TestConf(ServiceBackendTest):
    """Db2 backend test configuration.

    Data loading strategy
    ---------------------
    Db2 has no ``BULK INSERT``/``COPY FROM`` equivalent that works from outside
    the container, so we use ``ibm_db_dbi.executemany`` in Python instead.

    ibm_db_dbi has two driver-level constraints we work around:
    1. ``executemany`` requires plain ``list`` rows, not namedtuples.
    2. ``executemany`` can raise ``CLI0125E Function sequence error`` on the
       same connection that ran DDL — we call ``self.connection._reconnect()``
       between DDL and bulk inserts to get a clean driver state.
    """

    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_structs = False
    supports_json = False
    native_bool = False
    rounding_method = "half_to_even"
    service_name = "db2"
    deps = ("ibm_db", "ibm_db_dbi")

    @property
    def test_files(self) -> Iterable[Path]:
        # Db2 data loading is done entirely in Python (_load_data).
        # There are no files to docker-cp into the container.
        return iter([])

    def _load_data(self, **_: Any) -> None:
        """Load the standard ibis test tables into Db2.

        Steps
        -----
        0. Create USER TEMPORARY tablespaces required for GLOBAL TEMPORARY
           tables (create_table(temp=True) and .cache()).
        1. Execute every DDL statement from ``ci/schema/db2.sql`` one at a
           time, committing after each — ibm_db_dbi does not support
           multi-statement batches.
        2. Reconnect (clears driver state left over from DDL, avoiding the
           ``CLI0125E Function sequence error`` on ``executemany``).
        3. Bulk-insert the five CSV tables via ``executemany`` in batches of
           5 000 rows, committing each batch.
        """
        import pandas as pd

        # Step 0: USER TEMPORARY tablespaces for GLOBAL TEMPORARY tables.
        #
        # DB2 will not use a REGULAR tablespace for CREATE GLOBAL TEMPORARY
        # TABLE — it requires a USER TEMPORARY tablespace (SQL0286N otherwise).
        # The CI error asked for page size >= 32768, which means functional_alltypes
        # (many wide VARCHAR columns) pushes the row width past 4K-page capacity,
        # so we need a 32K user temporary tablespace backed by a matching bufferpool.
        #
        # Regular tablespaces (IBIS_4K / IBIS_32K) are NOT recreated here —
        # VARCHAR(255) fits the default REGULAR tablespace without any extras.
        #
        # SQL0601N / -601 = object already exists → safe to ignore on re-runs.
        _user_temp_stmts = [
            "CREATE BUFFERPOOL IBIS_BP32K SIZE 250 PAGESIZE 32K",
            (
                "CREATE USER TEMPORARY TABLESPACE IBIS_TEMP_4K"
                " PAGESIZE 4K MANAGED BY AUTOMATIC STORAGE"
            ),
            (
                "CREATE USER TEMPORARY TABLESPACE IBIS_TEMP_32K"
                " PAGESIZE 32K MANAGED BY AUTOMATIC STORAGE"
                " BUFFERPOOL IBIS_BP32K"
            ),
        ]
        for stmt in _user_temp_stmts:
            try:
                with self.connection._safe_raw_sql(stmt):
                    pass
                self.connection._connection.commit()
            except Exception as e:  # noqa: PERF203
                err_str = str(e)
                if (
                    "SQL0601N" not in err_str
                    and "-601" not in err_str
                    and "already exists" not in err_str.lower()
                ):
                    raise

        # Step 1: DDL — uses self.ddl_script (BackendTest.ddl_script reads
        # ci/schema/db2.sql and splits on ";", same as every other backend).
        for stmt in self.ddl_script:
            with self.connection._safe_raw_sql(stmt):
                pass
            self.connection._connection.commit()

        # Step 2: Reconnect to get a clean connection for bulk inserts.
        self.connection._reconnect()

        # Step 3: Bulk-insert CSV tables.
        csv_dir = self.data_dir / "csv"
        table_files = {
            "diamonds": "diamonds.csv",
            "astronauts": "astronauts.csv",
            "batting": "batting.csv",
            "awards_players": "awards_players.csv",
            "functional_alltypes": "functional_alltypes.csv",
        }
        raw = self.connection._connection
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
            insert_sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"

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
            hostname=DB2_HOST,
            database=DB2_DATABASE,
            username=DB2_USER,
            password=DB2_PASS,
            port=DB2_PORT,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
