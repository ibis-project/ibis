from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import ServiceBackendTest


def pytest_collection_modifyitems(items, config):  # noqa: ARG001
    """Mark array tests to run serially and skip TPC benchmarks."""
    for item in items:
        # Skip TPC benchmark tests (mark to skip instead of deselecting to avoid dead fixture warnings)
        if "tpc/h/test_queries.py" in str(
            item.fspath
        ) or "tpc/ds/test_queries.py" in str(item.fspath):
            item.add_marker(pytest.mark.skip(reason="TPC benchmarks not supported"))
        # Mark array tests to run serially
        if "test_array.py" in str(item.fspath):
            item.add_marker(pytest.mark.xdist_group("materialize_array_serial"))
        # Mark source and subscribe tests to run serially (AUCTION source conflicts)
        if "test_sources.py" in str(item.fspath) or "test_subscribe.py" in str(
            item.fspath
        ):
            item.add_marker(pytest.mark.xdist_group("materialize_sources_serial"))


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# Get Materialize connection details from environment
MZ_USER = os.environ.get(
    "IBIS_TEST_MATERIALIZE_USER", os.environ.get("MZ_USER", "materialize")
)
MZ_PASS = os.environ.get(
    "IBIS_TEST_MATERIALIZE_PASSWORD", os.environ.get("MZ_PASSWORD", "")
)
MZ_HOST = os.environ.get(
    "IBIS_TEST_MATERIALIZE_HOST", os.environ.get("MZ_HOST", "localhost")
)
MZ_PORT = os.environ.get(
    "IBIS_TEST_MATERIALIZE_PORT", os.environ.get("MZ_PORT", "6875")
)
MZ_DATABASE = os.environ.get(
    "IBIS_TEST_MATERIALIZE_DATABASE", os.environ.get("MZ_DATABASE", "materialize")
)


class TestConf(ServiceBackendTest):
    """Test configuration for Materialize backend.

    Materialize is PostgreSQL-compatible but has some differences:
    - Temporary tables are supported (stored in mz_temp schema)
    - No UDF support
    - Streaming-optimized database
    """

    returned_timestamp_unit = "s"
    supports_structs = False  # Materialize doesn't support structs yet
    supports_map = False  # Materialize has limited map support, disable for stability
    rounding_method = "half_to_even"
    service_name = "materialize"
    data_volume = "/data"
    deps = ("psycopg",)  # Uses psycopg like PostgreSQL
    supports_python_udfs = False  # Materialize doesn't support UDFs
    supports_temporary_tables = (
        True  # Materialize supports temp tables in mz_temp schema
    )
    force_sort = True  # Streaming database - results are unordered without ORDER BY

    driver_supports_multiple_statements = (
        False  # Materialize cannot mix DDL and DML in same transaction
    )

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(self, **_: Any) -> None:
        """Load test data into Materialize.

        Materialize transactions cannot mix DDL and DML, so we execute each
        DDL statement separately outside of transactions.

        Loads CSV data using COPY FROM STDIN for efficient bulk loading.
        """
        # Execute each DDL statement separately
        for stmt in self.ddl_script:
            if stmt.strip():  # Skip empty statements
                with self.connection._safe_raw_sql(stmt):
                    pass

        # Load CSV files using COPY FROM STDIN
        # Note: CSV files must be downloaded first using `just download-data`
        # Materialize supports COPY FROM STDIN for efficient bulk loading

        con = self.connection.con
        # Cache list_tables() result to avoid repeated expensive calls
        existing_tables = set(self.connection.list_tables())
        for csv_file in self.test_files:
            table_name = csv_file.stem
            if table_name in existing_tables and csv_file.exists():
                # Get column list from schema
                schema = self.connection.get_schema(table_name)
                columns = list(schema.keys())
                col_list = ", ".join(f'"{c}"' for c in columns)

                # Use COPY FROM STDIN with CSV format (psycopg3 API)
                copy_sql = f'COPY "{table_name}" ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER true)'

                with con.cursor() as cur:
                    # Open CSV file and use copy() context manager for psycopg3
                    with open(csv_file) as f:
                        with cur.copy(copy_sql) as copy:
                            while data := f.read(8192):
                                copy.write(data)
                con.commit()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        """Connect to Materialize for testing."""
        return ibis.materialize.connect(
            host=MZ_HOST,
            port=int(MZ_PORT),
            user=MZ_USER,
            password=MZ_PASS,
            database=MZ_DATABASE,
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    """Session-scoped connection fixture."""
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection


@pytest.fixture(scope="module")
def alltypes(con):
    """Standard test table fixture."""
    return con.table("functional_alltypes")


@pytest.fixture
def temp_table(con) -> str:  # noqa: ARG001
    """
    Return a temporary table name.

    Materialize supports temporary tables in the mz_temp schema.
    They are automatically cleaned up at the end of the session.
    """
    from ibis.util import gen_name

    return gen_name("materialize_temp_table")


@pytest.fixture
def assert_sql(con):
    """Fixture for asserting SQL compilation."""

    def check_sql(expr):
        """Check that expression can be compiled to SQL using Materialize backend."""
        # Use the Materialize backend's compiler, not the generic postgres dialect
        sql = con.compile(expr)
        assert sql is not None
        assert isinstance(sql, str)
        assert len(sql) > 0
        return sql

    return check_sql
