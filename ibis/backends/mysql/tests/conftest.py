import os
from pathlib import Path

from pkg_resources import parse_version

import ibis
from ibis.backends.tests.base import BackendTest, RoundHalfToEven


class TestConf(BackendTest, RoundHalfToEven):
    # mysql has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    bool_is_int = True

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
            version_detail = con.con.dialect._parse_server_version(
                str(con.version)
            )
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
    def connect(data_directory: Path):
        user = os.environ.get('IBIS_TEST_MYSQL_USER', 'ibis')
        password = os.environ.get('IBIS_TEST_MYSQL_PASSWORD', 'ibis')
        host = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
        port = os.environ.get('IBIS_TEST_MYSQL_PORT', 3306)
        database = os.environ.get('IBIS_TEST_MYSQL_DATABASE', 'ibis_testing')
        return ibis.mysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
