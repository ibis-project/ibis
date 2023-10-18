"""PostgreSQL backend."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING, Callable, Literal

import sqlalchemy as sa

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sql.alchemy import AlchemyCanCreateSchema, BaseAlchemyBackend
from ibis.backends.postgres.compiler import PostgreSQLCompiler
from ibis.backends.postgres.datatypes import PostgresType
from ibis.common.exceptions import InvalidDecoratorError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import ibis.expr.datatypes as dt


def _verify_source_line(func_name: str, line: str):
    if line.startswith("@"):
        raise InvalidDecoratorError(func_name, line)
    return line


class Backend(BaseAlchemyBackend, AlchemyCanCreateSchema):
    name = "postgres"
    compiler = PostgreSQLCompiler
    supports_create_or_replace = False
    supports_python_udfs = True

    def do_connect(
        self,
        host: str | None = None,
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
        >>> host = os.environ.get("IBIS_TEST_POSTGRES_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_POSTGRES_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_POSTGRES_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_POSTGRES_DATABASE", "ibis_testing")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
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
        if driver != "psycopg2":
            raise NotImplementedError("psycopg2 is currently the only supported driver")

        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f"postgresql+{driver}",
        )

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

        super().do_connect(engine)

    def list_tables(self, like=None, database=None, schema=None):
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            (deprecated) The database to perform the list against.
        schema
            The schema to perform the list against.

            ::: {.callout-warning}
            ## `schema` refers to database hierarchy

            The `schema` parameter does **not** refer to the column names and
            types of `table`.
            :::
        """
        if database is not None:
            util.warn_deprecated(
                "database",
                instead="Use the `schema` keyword argument instead",
                as_of="7.1",
                removed_in="8.0",
            )
        schema = schema or database
        tables = self.inspector.get_table_names(schema=schema)
        views = self.inspector.get_view_names(schema=schema)
        return self._filter_with_like(tables + views, like)

    def list_databases(self, like=None) -> list[str]:
        # http://dba.stackexchange.com/a/1304/58517
        dbs = sa.table(
            "pg_database",
            sa.column("datname", sa.TEXT()),
            sa.column("datistemplate", sa.BOOLEAN()),
            schema="pg_catalog",
        )
        query = sa.select(dbs.c.datname).where(sa.not_(dbs.c.datistemplate))
        with self.begin() as con:
            databases = list(con.execute(query).scalars())

        return self._filter_with_like(databases, like)

    @property
    def current_database(self) -> str:
        return self._scalar_query(sa.select(sa.func.current_database()))

    @property
    def current_schema(self) -> str:
        return self._scalar_query(sa.select(sa.func.current_schema()))

    def function(self, name: str, *, schema: str | None = None) -> Callable:
        query = sa.text(
            """
SELECT
  n.nspname as schema,
  pg_catalog.pg_get_function_result(p.oid) as return_type,
  string_to_array(pg_catalog.pg_get_function_arguments(p.oid), ', ') as signature,
  CASE p.prokind
    WHEN 'a' THEN 'agg'
    WHEN 'w' THEN 'window'
    WHEN 'p' THEN 'proc'
    ELSE 'func'
  END as "Type"
FROM pg_catalog.pg_proc p
LEFT JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
WHERE p.proname = :name
"""
            + "AND n.nspname OPERATOR(pg_catalog.~) :schema COLLATE pg_catalog.default"
            * (schema is not None)
        ).bindparams(name=name, schema=f"^({schema})$")

        def split_name_type(arg: str) -> tuple[str, dt.DataType]:
            name, typ = arg.split(" ", 1)
            return name, PostgresType.from_string(typ)

        with self.begin() as con:
            rows = con.execute(query).mappings().fetchall()

        if not rows:
            name = f"{schema}.{name}" if schema else name
            raise exc.MissingUDFError(name)
        elif len(rows) > 1:
            raise exc.AmbiguousUDFError(name)

        [row] = rows
        return_type = PostgresType.from_string(row["return_type"])
        signature = list(map(split_name_type, row["signature"]))

        # dummy callable
        def fake_func(*args, **kwargs):
            ...

        fake_func.__name__ = name
        fake_func.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ
                )
                for name, typ in signature
            ],
            return_annotation=return_type,
        )
        fake_func.__annotations__ = {"return": return_type, **dict(signature)}
        op = ops.udf.scalar.builtin(fake_func, schema=schema)
        return op

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        config = udf_node.__config__
        func = udf_node.__func__
        func_name = func.__name__
        return dict(
            name=udf_node.__func_name__,
            ident=udf_node.__full_name__,
            signature=", ".join(
                f"{argname} {self._compile_type(arg.dtype)}"
                for argname, arg in zip(udf_node.argnames, udf_node.args)
            ),
            return_type=self._compile_type(udf_node.dtype),
            language=config.get("language", "plpython3u"),
            source="\n".join(
                _verify_source_line(func_name, line)
                for line in textwrap.dedent(inspect.getsource(func)).splitlines()
                if not line.strip().startswith("@udf")
            ),
            args=", ".join(udf_node.argnames),
        )

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
CREATE OR REPLACE FUNCTION {ident}({signature})
RETURNS {return_type}
LANGUAGE {language}
AS $$
{source}
return {name}({args})
$$""".format(
            **self._get_udf_source(udf_node)
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
                    (col, PostgresType.from_string(typestr))
                    for col, typestr in con.execute(text)
                )
            finally:
                con.exec_driver_sql(f"DROP VIEW IF EXISTS {name}")

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"DROP VIEW IF EXISTS {name}"
        yield f"CREATE TEMPORARY VIEW {name} AS {definition}"

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None and database != self.current_database:
            raise exc.UnsupportedOperationError(
                "Postgres does not support creating a schema in a different database"
            )
        if_not_exists = "IF NOT EXISTS " * force
        name = self._quote(name)
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA {if_not_exists}{name}")

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None and database != self.current_database:
            raise exc.UnsupportedOperationError(
                "Postgres does not support dropping a schema in a different database"
            )
        name = self._quote(name)
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP SCHEMA {if_exists}{name}")
