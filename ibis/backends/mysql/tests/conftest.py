from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from packaging.version import parse as parse_version

import ibis
from ibis.backends.conftest import TEST_TABLES, init_database
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest, ServiceSpec

MYSQL_USER = os.environ.get('IBIS_TEST_MYSQL_USER', 'ibis')
MYSQL_PASS = os.environ.get('IBIS_TEST_MYSQL_PASSWORD', 'ibis')
MYSQL_HOST = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.environ.get('IBIS_TEST_MYSQL_PORT', 3306))
IBIS_TEST_MYSQL_DB = os.environ.get('IBIS_TEST_MYSQL_DATABASE', 'ibis_testing')


class TestConf(ServiceBackendTest, RoundHalfToEven):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    native_bool = False
    supports_structs = False

    @classmethod
    def service_spec(cls, data_dir: Path) -> ServiceSpec:
        return ServiceSpec(
            name=cls.name(),
            data_volume="/data",
            files=data_dir.joinpath("csv").glob("*.csv"),
        )

    def __init__(self, data_directory: Path) -> None:
        super().__init__(data_directory)
        con = self.connection
        with con.begin() as c:
            version = c.exec_driver_sql("SELECT VERSION()").scalar()

        # mariadb supports window operations after version 10.2
        # mysql supports window operations after version 8
        min_version = "10.2" if "MariaDB" in version else "8.0"
        self.__class__.supports_window_operations = parse_version(
            con.version
        ) >= parse_version(min_version)

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
                    csv_path = data_dir / "csv" / f"{table}.csv"
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
def setup_privs():
    engine = sa.create_engine(f"mysql+pymysql://root:@{MYSQL_HOST}:{MYSQL_PORT:d}")
    with engine.begin() as con:
        # allow the ibis user to use any database
        con.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS `test_schema`")
        con.exec_driver_sql(
            f"GRANT CREATE,SELECT,DROP ON `test_schema`.* TO `{MYSQL_USER}`@`%%`"
        )
    yield
    with engine.begin() as con:
        con.exec_driver_sql("DROP SCHEMA IF EXISTS `test_schema`")


@pytest.fixture(scope='session')
def con():
    return ibis.mysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=IBIS_TEST_MYSQL_DB,
        port=MYSQL_PORT,
    )


@pytest.fixture(scope='session')
def con_nodb():
    return ibis.mysql.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, port=MYSQL_PORT
    )
