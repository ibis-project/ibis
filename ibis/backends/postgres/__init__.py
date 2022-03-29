"""PostgreSQL backend."""

from __future__ import annotations

import contextlib
from typing import Literal

import sqlalchemy as sa

import ibis.backends.duckdb.datatypes as ddb
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .compiler import PostgreSQLCompiler
from .udf import udf


class Backend(BaseAlchemyBackend):
    name = 'postgres'
    compiler = PostgreSQLCompiler

    def do_connect(
        self,
        host: str = 'localhost',
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        url: str | None = None,
        driver: Literal["psycopg2"] = "psycopg2",
    ) -> None:
        """Create an Ibis client connected to PostgreSQL database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number
        database
            Database to connect to
        url
            SQLAlchemy connection string.

            If passed, the other connection arguments are ignored.
        driver
            Database driver

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> import ibis
        >>> host = os.environ.get('IBIS_TEST_POSTGRES_HOST', 'localhost')
        >>> user = os.environ.get('IBIS_TEST_POSTGRES_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
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
        PostgreSQLTable[table]
          name: functional_alltypes
          schema:
            index : int64
            Unnamed: 0 : int64
            id : int32
            bool_col : boolean
            tinyint_col : int16
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
        if driver != 'psycopg2':
            raise NotImplementedError(
                'psycopg2 is currently the only supported driver'
            )
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f'postgresql+{driver}',
        )
        self.database_name = alchemy_url.database
        super().do_connect(sa.create_engine(alchemy_url))

    def list_databases(self, like=None):
        # http://dba.stackexchange.com/a/1304/58517
        databases = [
            row.datname
            for row in self.con.execute(
                'SELECT datname FROM pg_database WHERE NOT datistemplate'
            )
        ]
        return self._filter_with_like(databases, like)

    @util.deprecated(version='2.0', instead='use `list_databases`')
    def list_schemas(self, like=None):
        """List all the schemas in the current database."""
        # In Postgres we support schemas, which in other engines (e.g. MySQL)
        # are databases
        return super().list_databases(like)

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute('SHOW TIMEZONE').scalar()
            bind.execute('SET TIMEZONE = UTC')
            try:
                yield bind
            finally:
                bind.execute(f"SET TIMEZONE = '{previous_timezone}'")

    def udf(
        self,
        pyfunc,
        in_types,
        out_type,
        schema=None,
        replace=False,
        name=None,
        language="plpythonu",
    ):
        """Decorator that defines a PL/Python UDF in-database.

        Parameters
        ----------
        pyfunc
            Python function
        in_types
            Input types
        out_type
            Output type
        schema
            The postgres schema in which to define the UDF
        replace
            replace UDF in database if already exists
        name
            name for the UDF to be defined in database
        language
            Language extension to use for PL/Python

        Returns
        -------
        Callable
            A callable ibis expression

        Function that takes in ColumnExpr arguments and returns an instance
        inheriting from PostgresUDFNode
        """

        return udf(
            client=self,
            python_func=pyfunc,
            in_types=in_types,
            out_type=out_type,
            schema=schema,
            replace=replace,
            name=name,
            language=language,
        )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        raw_name = util.guid()
        name = self.con.dialect.identifier_preparer.quote_identifier(raw_name)
        type_info_sql = f"""\
SELECT
  attname,
  format_type(atttypid, atttypmod) AS type
FROM pg_attribute
WHERE attrelid = {raw_name!r}::regclass
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum
"""
        with self.con.connect() as con:
            con.execute(f"CREATE TEMPORARY VIEW {name} AS {query}")
            try:
                type_info = con.execute(type_info_sql).fetchall()
            finally:
                con.execute(f"DROP VIEW {name}")
        tuples = [(col, _get_type(typestr)) for col, typestr in type_info]
        return sch.Schema.from_tuples(tuples)


def _get_type(typestr: str) -> dt.DataType:
    try:
        return _type_mapping[typestr]
    except KeyError:
        return ddb.parse_type(typestr)


_type_mapping = {
    "boolean": dt.bool,
    "boolean[]": dt.Array(dt.bool),
    "bytea": dt.binary,
    "bytea[]": dt.Array(dt.binary),
    "character(1)": dt.string,
    "character(1)[]": dt.Array(dt.string),
    "bigint": dt.int64,
    "bigint[]": dt.Array(dt.int64),
    "smallint": dt.int16,
    "smallint[]": dt.Array(dt.int16),
    "integer": dt.int32,
    "integer[]": dt.Array(dt.int32),
    "text": dt.string,
    "text[]": dt.Array(dt.string),
    "json": dt.json,
    "json[]": dt.Array(dt.json),
    "point": dt.point,
    "point[]": dt.Array(dt.point),
    "polygon": dt.polygon,
    "polygon[]": dt.Array(dt.polygon),
    "line": dt.linestring,
    "line[]": dt.Array(dt.linestring),
    "real": dt.float32,
    "real[]": dt.Array(dt.float32),
    "double precision": dt.float64,
    "double precision[]": dt.Array(dt.float64),
    "macaddr8": dt.macaddr,
    "macaddr8[]": dt.Array(dt.macaddr),
    "macaddr": dt.macaddr,
    "macaddr[]": dt.Array(dt.macaddr),
    "inet": dt.inet,
    "inet[]": dt.Array(dt.inet),
    "character": dt.string,
    "character[]": dt.Array(dt.string),
    "character varying": dt.string,
    "character varying[]": dt.Array(dt.string),
    "date": dt.date,
    "date[]": dt.Array(dt.date),
    "time without time zone": dt.time,
    "time without time zone[]": dt.Array(dt.time),
    "timestamp without time zone": dt.timestamp,
    "timestamp without time zone[]": dt.Array(dt.timestamp),
    "timestamp with time zone": dt.Timestamp("UTC"),
    "timestamp with time zone[]": dt.Array(dt.Timestamp("UTC")),
    "interval": dt.interval,
    "interval[]": dt.Array(dt.interval),
    # NB: this isn"t correct, but we try not to fail
    "time with time zone": "time",
    "numeric": dt.decimal,
    "numeric[]": dt.Array(dt.decimal),
    "uuid": dt.uuid,
    "uuid[]": dt.Array(dt.uuid),
    "jsonb": dt.jsonb,
    "jsonb[]": dt.Array(dt.jsonb),
    "geometry": dt.geometry,
    "geometry[]": dt.Array(dt.geometry),
    "geography": dt.geography,
    "geography[]": dt.Array(dt.geography),
}
