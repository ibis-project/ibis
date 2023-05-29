"""PostgreSQL backend."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Literal

import sqlalchemy as sa

from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.postgres.compiler import PostgreSQLCompiler
from ibis.backends.postgres.datatypes import _get_type
from ibis.backends.postgres.udf import udf as _udf

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


# adapted from https://wiki.postgresql.org/wiki/First/last_%28aggregate%29
_CREATE_FIRST_LAST_AGGS_SQL = """\
CREATE OR REPLACE FUNCTION public._ibis_first_agg (anyelement, anyelement)
RETURNS anyelement
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
'SELECT $1';

CREATE OR REPLACE AGGREGATE public._ibis_first (anyelement) (
  SFUNC = public._ibis_first_agg,
  STYPE = anyelement,
  PARALLEL = safe
);

CREATE OR REPLACE FUNCTION public._ibis_last_agg (anyelement, anyelement)
RETURNS anyelement
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
'SELECT $2';

CREATE OR REPLACE AGGREGATE public._ibis_last (anyelement) (
  SFUNC = public._ibis_last_agg,
  STYPE = anyelement,
  PARALLEL = safe
);"""

_DROP_FIRST_LAST_AGGS_SQL = """\
DROP AGGREGATE IF EXISTS public._ibis_first(anyelement), public._ibis_last(anyelement);
DROP FUNCTION IF EXISTS public._ibis_first_agg(anyelement, anyelement), public._ibis_last_agg(anyelement, anyelement);"""


class Backend(BaseAlchemyBackend):
    name = 'postgres'
    compiler = PostgreSQLCompiler
    supports_create_or_replace = False

    def do_connect(
        self,
        host: str = 'localhost',
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
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
        schema
            PostgreSQL schema to use. If `None`, use the default `search_path`.
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
            raise NotImplementedError('psycopg2 is currently the only supported driver')

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

        connect_args = {}
        if schema is not None:
            connect_args["options"] = f"-csearch_path={schema}"

        engine = sa.create_engine(
            alchemy_url, connect_args=connect_args, poolclass=sa.pool.StaticPool
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                cur.execute("SET TIMEZONE = UTC")

        @sa.event.listens_for(engine, "before_execute")
        def receive_before_execute(
            conn, clauseelement, multiparams, params, execution_options
        ):
            with conn.connection.cursor() as cur:
                try:
                    cur.execute(_CREATE_FIRST_LAST_AGGS_SQL)
                except Exception as e:  # noqa: BLE001
                    # a user may not have permissions to create funtions and/or aggregates
                    warnings.warn(f"Unable to create first/last aggregates: {e}")

        @sa.event.listens_for(engine, "after_execute")
        def receive_after_execute(
            conn, clauseelement, multiparams, params, execution_options, result
        ):
            with conn.connection.cursor() as cur:
                try:
                    cur.execute(_DROP_FIRST_LAST_AGGS_SQL)
                except Exception as e:  # noqa: BLE001
                    # a user may not have permissions to drop funtions and/or aggregates
                    warnings.warn(f"Unable to drop first/last aggregates: {e}")

        super().do_connect(engine)

    def list_databases(self, like=None):
        with self.begin() as con:
            # http://dba.stackexchange.com/a/1304/58517
            databases = [
                row.datname
                for row in con.exec_driver_sql(
                    "SELECT datname FROM pg_database WHERE NOT datistemplate"
                ).mappings()
            ]
        return self._filter_with_like(databases, like)

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

        Function that takes in Column arguments and returns an instance
        inheriting from PostgresUDFNode
        """

        return _udf(
            client=self,
            python_func=pyfunc,
            in_types=in_types,
            out_type=out_type,
            schema=schema,
            replace=replace,
            name=name,
            language=language,
        )

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        name = util.gen_name("postgres_metadata")
        type_info_sql = """\
SELECT
  attname,
  format_type(atttypid, atttypmod) AS type
FROM pg_attribute
WHERE attrelid = CAST(:name AS regclass)
  AND attnum > 0
  AND NOT attisdropped
ORDER BY attnum"""
        if self.inspector.has_table(query):
            query = f"TABLE {query}"

        text = sa.text(type_info_sql).bindparams(name=name)
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE TEMPORARY VIEW {name} AS {query}")
            try:
                yield from (
                    (col, _get_type(typestr)) for col, typestr in con.execute(text)
                )
            finally:
                con.exec_driver_sql(f"DROP VIEW IF EXISTS {name}")

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"DROP VIEW IF EXISTS {name}"
        yield f"CREATE TEMPORARY VIEW {name} AS {definition}"
