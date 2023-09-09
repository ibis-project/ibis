"""The MySQL backend."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base import CanCreateDatabase
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mysql.compiler import MySQLCompiler
from ibis.backends.mysql.datatypes import MySQLDateTime, MySQLType

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ibis.expr.datatypes as dt


class Backend(BaseAlchemyBackend, CanCreateDatabase):
    name = "mysql"
    compiler = MySQLCompiler
    supports_create_or_replace = False

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        database: str | None = None,
        url: str | None = None,
        driver: Literal["pymysql"] = "pymysql",
        **kwargs,
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
        kwargs
            Additional keyword arguments passed to `connect_args` in
            `sqlalchemy.create_engine`. Use these to pass dialect specific
            arguments.

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> host = os.environ.get("IBIS_TEST_MYSQL_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MYSQL_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_MYSQL_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis_testing")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        MySQLTable[table]
          name: functional_alltypes
          schema:
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
        if driver != "pymysql":
            raise NotImplementedError("pymysql is currently the only supported driver")
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f"mysql+{driver}",
        )

        engine = sa.create_engine(
            alchemy_url, poolclass=sa.pool.StaticPool, connect_args=kwargs
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                try:
                    cur.execute("SET @@session.time_zone = 'UTC'")
                except sa.exc.OperationalError:
                    warnings.warn("Unable to set session timezone to UTC.")

        super().do_connect(engine)

    @property
    def current_database(self) -> str:
        return self._scalar_query(sa.select(sa.func.database()))

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if isinstance(column_info["type"], mysql.DATETIME):
                column_info["type"] = MySQLDateTime()
            if isinstance(column_info["type"], mysql.DOUBLE):
                column_info["type"] = mysql.DOUBLE(asdecimal=False)
            if isinstance(column_info["type"], mysql.FLOAT):
                column_info["type"] = mysql.FLOAT(asdecimal=False)

        return meta

    def list_databases(self, like: str | None = None) -> list[str]:
        # In MySQL, "database" and "schema" are synonymous
        databases = self.inspector.get_schema_names()
        return self._filter_with_like(databases, like)

    def _metadata(self, table: str) -> Iterable[tuple[str, dt.DataType]]:
        with self.begin() as con:
            result = con.exec_driver_sql(f"DESCRIBE {table}").mappings().all()

        for field in result:
            name = field["Field"]
            type_string = field["Type"]
            is_nullable = field["Null"] == "YES"
            yield name, MySQLType.from_string(type_string, nullable=is_nullable)

    def _get_schema_using_query(self, query: str):
        table = f"__ibis_mysql_metadata_{util.guid()}"

        with self.begin() as con:
            con.exec_driver_sql(f"CREATE TEMPORARY TABLE {table} AS {query}")
            result = con.exec_driver_sql(f"DESCRIBE {table}").mappings().all()
            con.exec_driver_sql(f"DROP TABLE {table}")

        fields = {}
        for field in result:
            name = field["Field"]
            type_string = field["Type"]
            is_nullable = field["Null"] == "YES"
            fields[name] = MySQLType.from_string(type_string, nullable=is_nullable)

        return sch.Schema(fields)

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE VIEW {name} AS {definition}"

    def create_database(self, name: str, force: bool = False) -> None:
        name = self._quote(name)
        if_exists = "IF NOT EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE DATABASE {if_exists}{name}")

    def drop_database(self, name: str, force: bool = False) -> None:
        name = self._quote(name)
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP DATABASE {if_exists}{name}")
