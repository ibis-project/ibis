"""The MySQL backend."""

from __future__ import annotations

import re
from typing import Iterable, Literal

import sqlalchemy as sa

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mysql.compiler import MySQLCompiler
from ibis.backends.mysql.datatypes import _type_from_cursor_info


class Backend(BaseAlchemyBackend):
    name = 'mysql'
    compiler = MySQLCompiler

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        database: str | None = None,
        url: str | None = None,
        driver: Literal["pymysql"] = "pymysql",
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port
        database
            Database to connect to
        url
            Complete SQLAlchemy connection string. If passed, the other
            connection arguments are ignored.
        driver
            Python MySQL database driver

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> host = os.environ.get('IBIS_TEST_MYSQL_HOST', 'localhost')
        >>> user = os.environ.get('IBIS_TEST_MYSQL_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_MYSQL_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_MYSQL_DATABASE',
        ...                           'ibis_testing')
        >>> con = connect(
        ...     database=database,
        ...     host=host,
        ...     user=user,
        ...     password=password
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table('functional_alltypes')
        >>> t
        MySQLTable[table]
          name: functional_alltypes
          schema:
            index : int64
            Unnamed: 0 : int64
            id : int32
            bool_col : int8
            tinyint_col : int8
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32
        """
        if driver != 'pymysql':
            raise NotImplementedError('pymysql is currently the only supported driver')
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f'mysql+{driver}',
        )

        self.database_name = alchemy_url.database

        engine = sa.create_engine(alchemy_url, poolclass=sa.pool.StaticPool)

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                cur.execute("SET @@session.time_zone = 'UTC'")

        super().do_connect(engine)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        if (
            re.search(r"^\s*SELECT\s", query, flags=re.MULTILINE | re.IGNORECASE)
            is not None
        ):
            query = f"({query})"

        with self.begin() as con:
            result = con.exec_driver_sql(f"SELECT * FROM {query} _ LIMIT 0")
            cursor = result.cursor
            yield from (
                (field.name, _type_from_cursor_info(descr, field))
                for descr, field in zip(cursor.description, cursor._result.fields)
            )

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE VIEW {name} AS {definition}"
