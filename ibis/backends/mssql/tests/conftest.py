from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import TEST_TABLES, init_database
from ibis.backends.tests.base import BackendTest, RoundHalfToEven

MSSQL_USER = os.environ.get('IBIS_TEST_MSSQL_USER', 'sa')
MSSQL_PASS = os.environ.get('IBIS_TEST_MSSQL_PASSWORD', '1bis_Testing!')
MSSQL_HOST = os.environ.get('IBIS_TEST_MSSQL_HOST', 'localhost')
MSSQL_PORT = int(os.environ.get('IBIS_TEST_MSSQL_PORT', 1433))
IBIS_TEST_MSSQL_DB = os.environ.get('IBIS_TEST_MSSQL_DATABASE', 'ibis_testing')


class TestConf(BackendTest, RoundHalfToEven):
    # MSSQL has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_structs = False
    supports_arrays = False
    supports_json = False

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)

    @staticmethod
    def _load_data(
        data_dir: Path,
        script_dir: Path,
        user: str = MSSQL_USER,
        password: str = MSSQL_PASS,
        host: str = MSSQL_HOST,
        port: int = MSSQL_PORT,
        database: str = IBIS_TEST_MSSQL_DB,
        **_: Any,
    ) -> None:
        """Load test data into a MSSQL backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        with open(script_dir / 'schema' / 'mssql.sql') as schema:
            engine = init_database(
                url=sa.engine.make_url(
                    f"mssql+pymssql://{user}:{password}@{host}:{port:d}/{database}"
                ),
                database=database,
                schema=schema,
                isolation_level="AUTOCOMMIT",
            )

            futures = []
            with concurrent.futures.ThreadPoolExecutor() as e:
                for table in TEST_TABLES:
                    # /data is a volume mount to the ibis testing data
                    # used for snappy test data loading
                    # DataFrame.to_sql is unusably slow for loading CSVs
                    query = f"""
                    BULK INSERT {table}
                    FROM '/data/{table}.csv'
                    WITH (
                      FORMAT = 'CSV',
                      FIELDTERMINATOR = ',',
                      ROWTERMINATOR = '\\n',
                      FIRSTROW = 2
                    )
                    """
                    futures.append(e.submit(engine.execute, query))

                for future in concurrent.futures.as_completed(futures):
                    future.result()

    @staticmethod
    def connect(_: Path):
        return ibis.mssql.connect(
            host=MSSQL_HOST,
            user=MSSQL_USER,
            password=MSSQL_PASS,
            database=IBIS_TEST_MSSQL_DB,
            port=MSSQL_PORT,
        )


@pytest.fixture(scope='session')
def con():
    return ibis.mssql.connect(
        host=MSSQL_HOST,
        user=MSSQL_USER,
        password=MSSQL_PASS,
        database=IBIS_TEST_MSSQL_DB,
        port=MSSQL_PORT,
    )
