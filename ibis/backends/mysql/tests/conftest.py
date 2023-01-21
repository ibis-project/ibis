from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from packaging.version import parse as parse_version

import ibis
from ibis.backends.conftest import TEST_TABLES, init_database
from ibis.backends.tests.base import BackendTest, RoundHalfToEven

MYSQL_USER = os.environ.get('IBIS_TEST_MYSQL_USER', 'ibis')
MYSQL_PASS = os.environ.get('IBIS_TEST_MYSQL_PASSWORD', 'ibis')
MYSQL_HOST = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.environ.get('IBIS_TEST_MYSQL_PORT', 3306))
IBIS_TEST_MYSQL_DB = os.environ.get('IBIS_TEST_MYSQL_DATABASE', 'ibis_testing')


class TestConf(BackendTest, RoundHalfToEven):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    native_bool = False
    supports_structs = False

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)
        # mariadb supports window operations after version 10.2
        # but the sqlalchemy version string looks like:
        # 5.5.5.10.2.12.MariaDB.10.2.12+maria~jessie
        # or 10.4.12.MariaDB.1:10.4.12+maria~bionic
        # example of possible results:
        # https://github.com/sqlalchemy/sqlalchemy/blob/rel_1_3/
        # test/dialect/mysql/test_dialect.py#L244-L268
        con = self.connection
        if 'MariaDB' in str(con.version):
            # we might move this parsing step to the mysql client
            version_detail = con.con.dialect._parse_server_version(str(con.version))
            version = (
                version_detail[:3]
                if version_detail[3] == 'MariaDB'
                else version_detail[3:6]
            )
            self.__class__.supports_window_operations = version >= (10, 2)
        elif parse_version(con.version) >= parse_version('8.0'):
            # mysql supports window operations after version 8
            self.__class__.supports_window_operations = True

    @staticmethod
    def _load_data(
        data_dir: Path,
        script_dir: Path,
        user: str = MYSQL_USER,
        password: str = MYSQL_PASS,
        host: str = MYSQL_HOST,
        port: int = MYSQL_PORT,
        database: str = IBIS_TEST_MYSQL_DB,
        **_: Any,
    ) -> None:
        """Load test data into a MySql backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        with open(script_dir / 'schema' / 'mysql.sql') as schema:
            engine = init_database(
                url=sa.engine.make_url(
                    f"mysql+pymysql://{user}:{password}@{host}:{port:d}?local_infile=1",
                ),
                database=database,
                schema=schema,
                isolation_level="AUTOCOMMIT",
                recreate=False,
            )
            with engine.begin() as con:
                for table in TEST_TABLES:
                    csv_path = data_dir / f"{table}.csv"
                    lines = [
                        f"LOAD DATA LOCAL INFILE {str(csv_path)!r}",
                        f"INTO TABLE {table}",
                        "COLUMNS TERMINATED BY ','",
                        """OPTIONALLY ENCLOSED BY '"'""",
                        "LINES TERMINATED BY '\\n'",
                        "IGNORE 1 LINES",
                    ]
                    con.exec_driver_sql("\n".join(lines))

    @staticmethod
    def connect(_: Path):
        return ibis.mysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=IBIS_TEST_MYSQL_DB,
            port=MYSQL_PORT,
        )


@pytest.fixture(scope='session')
def con():
    return ibis.mysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=IBIS_TEST_MYSQL_DB,
        port=MYSQL_PORT,
    )
